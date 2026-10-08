"""
Wordinator: context-aware spell corrector (two-pass: spellchecker + BERT-tiny).

Features:
  - tokenize(): prefix punctuation / core word / suffix punctuation + char offsets
  - Pass 1: pyspellchecker (edit distance 2), Zipf-sorted, RapidFuzz fallback
  - Pass 2: ONE batched BERT call (chunked) for all error positions
  - rank_candidates(): 0.40 edit + 0.45 BERT + 0.15 frequency
  - rebuild(): replaces words at original offsets (keeps spacing/punctuation/case)
  - Optional real-word error detection ({"real_word": true})
  - Input validation, thread-safe lazy model loading, graceful fallback
"""
import os
import re
import threading

from flask import Flask, jsonify, render_template, request
from rapidfuzz import fuzz, process
from spellchecker import SpellChecker
from wordfreq import top_n_list, zipf_frequency

MODEL_NAME = "prajjwal1/bert-tiny"
MAX_CHARS = 2000
MAX_TOKENS = 128          # BERT input length per masked sentence
BATCH_SIZE = 64           # masked sentences per forward pass
W_EDIT, W_BERT, W_FREQ = 0.40, 0.45, 0.15
BERT_TOP_K = 15           # BERT predictions inspected per mask
BERT_EXTRA_MAX = 5        # how many BERT words may join the candidate pool
RESEMBLE_MIN = 60         # RapidFuzz ratio (0-100) a BERT word needs vs the typo
RW_MIN_PROB = 0.05        # real-word: neighbour must be at least 5% likely
RW_MIN_RATIO = 50         # real-word: and 50x more likely than the written word

app = Flask(__name__, static_folder="static", template_folder="templates")

spell = SpellChecker(distance=2)
try:
    VOCAB = top_n_list("en", 50000)
except Exception as e:
    print(f"Warning: wordfreq VOCAB load failed ({e}); fuzzy fallback disabled")
    VOCAB = []

# ---------------------------------------------------------------------------
# Lazy, thread-safe model loading
# ---------------------------------------------------------------------------
_tokenizer = None
_model = None
_VOCAB_IDS = None
_load_lock = threading.Lock()


def get_mlm():
    """Return (tokenizer, model). Loads on first call; safe if two requests race."""
    global _tokenizer, _model, _VOCAB_IDS
    if _model is None:
        with _load_lock:
            if _model is None:  # re-check inside the lock
                print("Loading model (first request)...")
                from transformers import AutoModelForMaskedLM, AutoTokenizer
                tok = AutoTokenizer.from_pretrained(MODEL_NAME)
                mdl = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
                mdl.eval()
                _tokenizer = tok
                _VOCAB_IDS = tok.get_vocab()
                _model = mdl  # assigned last: other threads see it only when ready
                print("Model ready.")
    return _tokenizer, _model


# ---------------------------------------------------------------------------
# Step 1: tokenize (keeps character offsets)
# ---------------------------------------------------------------------------
_SPLIT_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)


def tokenize(text):
    """Split on whitespace, then into prefix punctuation, core word, suffix punctuation."""
    tokens = []
    for m in re.finditer(r"\S+", text):
        raw = m.group()
        prefix, core, suffix = _SPLIT_RE.match(raw).groups()
        start = m.start() + len(prefix)
        tokens.append({
            "raw": raw,
            "prefix": prefix,
            "core": core,
            "suffix": suffix,
            "clean": core.lower(),
            "is_word": core.isalpha(),   # only plain alphabetic words are checked
            "start": start,
            "end": start + len(core),
        })
    return tokens


# ---------------------------------------------------------------------------
# Step 2: Pass 1 candidates
# ---------------------------------------------------------------------------
def spell_candidates(word, n=6):
    """pyspellchecker candidates sorted by Zipf frequency; RapidFuzz if none."""
    cands = [c for c in (spell.candidates(word) or []) if c != word]
    cands.sort(key=lambda w: zipf_frequency(w, "en"), reverse=True)
    cands = cands[:n]
    if not cands and VOCAB:
        hits = process.extract(word, VOCAB, scorer=fuzz.ratio, limit=n, score_cutoff=60)
        cands = [h[0] for h in hits]
    return cands


# ---------------------------------------------------------------------------
# Step 3: Pass 2, batched BERT
# ---------------------------------------------------------------------------
def masked_sentence(tokens, spell_info, i):
    """Mask token i; every other misspelling is swapped for its pass-1 fix."""
    parts = []
    for j, t in enumerate(tokens):
        if j == i:
            parts.append(t["prefix"] + "[MASK]" + t["suffix"])
        elif j in spell_info:
            parts.append(t["prefix"] + spell_info[j]["top"] + t["suffix"])
        else:
            parts.append(t["raw"])
    return " ".join(parts)


def mask_distributions(sentences):
    """
    One batched forward pass (chunked). Returns a list aligned with `sentences`:
    a probability vector over BERT's vocab at the [MASK] position, or None.
    The LM head runs only at mask positions to keep memory small.
    """
    import torch

    tokenizer, model = get_mlm()
    out = [None] * len(sentences)
    for s in range(0, len(sentences), BATCH_SIZE):
        chunk = sentences[s:s + BATCH_SIZE]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_TOKENS)
        rows, cols = (enc["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)
        seen, sel_r, sel_c = set(), [], []
        for r, c in zip(rows.tolist(), cols.tolist()):
            if r not in seen:           # first mask per sentence
                seen.add(r)
                sel_r.append(r)
                sel_c.append(c)
        if not sel_r:
            continue
        with torch.no_grad():
            hidden = model.bert(**enc).last_hidden_state
            probs = torch.softmax(model.cls(hidden[sel_r, sel_c]), dim=-1)
        for k, r in enumerate(sel_r):
            out[s + r] = probs[k]
    return out


def bert_words(dist, word, tokenizer):
    """BERT's own top predictions: whole words only (no ##pieces) that resemble the typo."""
    _, idxs = dist.topk(BERT_TOP_K)
    words = []
    for tok in tokenizer.convert_ids_to_tokens(idxs.tolist()):
        if tok.startswith("##") or not tok.isalpha() or len(tok) < 2:
            continue
        if fuzz.ratio(word, tok) < RESEMBLE_MIN:
            continue
        words.append(tok)
        if len(words) == BERT_EXTRA_MAX:
            break
    return words


# ---------------------------------------------------------------------------
# Step 4: ranking
# ---------------------------------------------------------------------------
def combine_score(edit_sim, bert_prob, freq):
    """Zipf is on a 0-7 scale, so freq/7 normalises it to 0-1."""
    return W_EDIT * edit_sim + W_BERT * bert_prob + W_FREQ * min(freq / 7.0, 1.0)


def rank_candidates(word, spell_cands, bert_extra, dist, vocab_ids):
    """Score every candidate. bert_prob = candidate's share of the pool's BERT mass."""
    pool = list(dict.fromkeys(list(spell_cands) + list(bert_extra)))
    raw = {}
    for c in pool:
        cid = vocab_ids.get(c) if (dist is not None and vocab_ids) else None
        raw[c] = float(dist[cid].item()) if cid is not None else 0.0
    total = sum(raw.values())

    ranked = []
    for c in pool:
        edit_sim = fuzz.ratio(word, c) / 100.0
        bert_prob = raw[c] / total if total > 0 else 0.0
        freq = zipf_frequency(c, "en")
        in_spell, in_bert = c in spell_cands, c in bert_extra
        ranked.append({
            "word": c,
            "combined_score": round(combine_score(edit_sim, bert_prob, freq), 4),
            "edit_sim": round(edit_sim, 3),
            "bert_prob": round(bert_prob, 4),
            "freq": round(freq, 2),
            "source": "both" if in_spell and in_bert else "spell" if in_spell else "bert",
        })
    ranked.sort(key=lambda r: -r["combined_score"])
    return ranked


# ---------------------------------------------------------------------------
# Step 5: rebuild text in place
# ---------------------------------------------------------------------------
def apply_case(original, replacement):
    if len(original) > 1 and original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def rebuild(text, replacements):
    """
    replacements: list of (start, end, new_text) in ORIGINAL offsets.
    Applied right-to-left so earlier offsets never shift.
    """
    for start, end, new in sorted(replacements, key=lambda r: r[0], reverse=True):
        text = text[:start] + new + text[end:]
    return text


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------
def _plain(tok):
    return {"original": tok["raw"], "clean": tok["clean"], "is_error": False,
            "spell_suggestion": tok["clean"] or None,
            "bert_suggestion": tok["clean"] or None,
            "combined_suggestion": None, "agree": True, "candidates": []}


def analyse(text, real_word=False):
    tokens = tokenize(text)

    # Pass 1
    spell_info = {}
    for i, t in enumerate(tokens):
        if t["is_word"] and spell.unknown([t["clean"]]):
            cands = spell_candidates(t["clean"])
            spell_info[i] = {"top": cands[0] if cands else t["clean"], "cands": cands}

    # Pass 2: one list of masked sentences -> one batched BERT call
    jobs = [("err", i) for i in spell_info]
    if real_word:
        jobs += [("rw", i) for i, t in enumerate(tokens)
                 if t["is_word"] and i not in spell_info and len(t["clean"]) >= 3]
    sentences = [masked_sentence(tokens, spell_info, i) for _, i in jobs]

    dists, model_used = [None] * len(jobs), False
    if jobs:
        try:
            dists = mask_distributions(sentences)
            model_used = any(d is not None for d in dists)
        except Exception as e:  # model missing / inference error -> fallback
            print(f"BERT unavailable, using spell + frequency only: {e}")
            dists = [None] * len(jobs)
    dist_of = dict(zip(jobs, dists))
    tokenizer = get_mlm()[0] if model_used else None
    vocab_ids = _VOCAB_IDS or {}

    results, replacements = [], []
    for i, t in enumerate(tokens):
        if i in spell_info:
            word = t["clean"]
            spell_top, sp_cands = spell_info[i]["top"], spell_info[i]["cands"]
            dist = dist_of.get(("err", i))
            extra = bert_words(dist, word, tokenizer) if dist is not None else []
            ranked = rank_candidates(word, sp_cands, extra, dist, vocab_ids)
            combined = ranked[0]["word"] if ranked else spell_top
            bert_top = spell_top
            if ranked and any(r["bert_prob"] > 0 for r in ranked):
                bert_top = max(ranked, key=lambda r: r["bert_prob"])["word"]
            if combined != word:
                replacements.append((t["start"], t["end"], apply_case(t["core"], combined)))
            results.append({
                "original": t["raw"], "clean": word, "is_error": True,
                "spell_suggestion": spell_top, "bert_suggestion": bert_top,
                "combined_suggestion": combined, "agree": spell_top == bert_top,
                "candidates": ranked[:8],
            })
            continue

        rw = dist_of.get(("rw", i))
        flagged = _real_word_check(t, rw, tokenizer, vocab_ids) if rw is not None else None
        if flagged:
            replacements.append((t["start"], t["end"], apply_case(t["core"], flagged["best"])))
            results.append({
                "original": t["raw"], "clean": t["clean"], "is_error": True,
                "real_word": True, "spell_suggestion": t["clean"],
                "bert_suggestion": flagged["best"], "combined_suggestion": flagged["best"],
                "agree": False, "candidates": flagged["candidates"],
            })
        else:
            results.append(_plain(t))

    return results, replacements, model_used


def _real_word_check(tok, dist, tokenizer, vocab_ids):
    """Flag a correctly-spelled word if a one-edit neighbour fits far better."""
    word = tok["clean"]
    wid = vocab_ids.get(word)
    if wid is None:
        return None
    p_word = float(dist[wid].item())
    neighbours = [n for n in spell.known(spell.edit_distance_1(word))
                  if n != word and n.isalpha() and n in vocab_ids]
    scored = sorted(((float(dist[vocab_ids[n]].item()), n) for n in neighbours), reverse=True)
    if not scored:
        return None
    p_best, best = scored[0]
    if p_best >= RW_MIN_PROB and p_best >= RW_MIN_RATIO * max(p_word, 1e-9):
        cands = [{
            "word": n, "combined_score": round(combine_score(fuzz.ratio(word, n) / 100.0, p, zipf_frequency(n, "en")), 4),
            "edit_sim": round(fuzz.ratio(word, n) / 100.0, 3), "bert_prob": round(p, 4),
            "freq": round(zipf_frequency(n, "en"), 2), "source": "bert",
        } for p, n in scored[:5]]
        return {"best": best, "candidates": cands}
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/ping")
def ping():
    return jsonify({"status": "ok", "model_loaded": _model is not None})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/correct", methods=["POST"])
def api_correct():
    try:
        payload = request.get_json(silent=True) or {}
        text = payload.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "No text provided"}), 400
        if len(text) > MAX_CHARS:
            return jsonify({"error": f"Text too long (max {MAX_CHARS} characters)"}), 400
        real_word = bool(payload.get("real_word", False))

        results, replacements, model_used = analyse(text, real_word)
        return jsonify({
            "original": text,
            "corrected": rebuild(text, replacements),
            "tokens": results,
            "error_count": sum(1 for r in results if r["is_error"]),
            "disagreements": sum(1 for r in results if r["is_error"] and not r["agree"]),
            "model_used": model_used,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
