import nltk
import os

# Download required NLTK data if not present
for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger", "wordnet"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass
nltk.data.path.append("./nltk_data")

from flask import Flask, render_template, request, jsonify
from spellchecker import SpellChecker
from rapidfuzz import fuzz
from wordfreq import zipf_frequency, top_n_list
import re
import threading

app = Flask(__name__, static_folder="static", template_folder="templates")

# ─────────────────────────────────────────────
# Model initialisation (done once at startup)
# ─────────────────────────────────────────────
spell = SpellChecker(distance=2)
try:
    VOCAB = top_n_list("en", 50000)
except Exception as e:
    print(f"Warning: wordfreq VOCAB load failed ({e}), using empty list")
    VOCAB = []

# Lazy-load the model on first request instead of at startup.
# This avoids OOM kills on HF Spaces free tier during boot.
_mlm = None

def get_mlm():
    global _mlm
    if _mlm is None:
        print("Loading model (first request)…")
        from transformers import pipeline as hf_pipeline
        _mlm = hf_pipeline(
            "fill-mask",
            model="prajjwal1/bert-tiny",   # 17MB vs 260MB — much faster cold start
            top_k=15,
            device=-1
        )
        print("Model ready.")
    return _mlm

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
_PUNCT_RE = re.compile(r"[^\w\s]")

def tokenize(text):
    """Return list of (original_token, cleaned_token, start_idx) tuples."""
    tokens = []
    for m in re.finditer(r"\S+", text):
        raw = m.group()
        clean = _PUNCT_RE.sub("", raw).lower()
        tokens.append({"raw": raw, "clean": clean, "start": m.start()})
    return tokens


def spell_candidates(word, n=6):
    """Candidates from pyspellchecker, sorted by frequency."""
    cands = list(spell.candidates(word) or [])
    cands.sort(key=lambda w: zipf_frequency(w, "en"), reverse=True)
    return cands[:n]


def combine_score(word, edit_sim, bert_prob, freq):
    """
    Weighted combination:
      40 % edit similarity  (rewards words that look like the typo)
      45 % BERT context     (rewards words that fit the sentence)
      15 % word frequency   (breaks ties toward common words)
    Zipf frequency is on a 0-7 scale; normalise to 0-1.
    """
    freq_norm = min(freq / 7.0, 1.0)
    return 0.40 * edit_sim + 0.45 * bert_prob + 0.15 * freq_norm


# ─────────────────────────────────────────────
# Core analysis  (two-pass)
# ─────────────────────────────────────────────
def analyse(text):
    tokens = tokenize(text)

    # ── Pass 1: run spell checker on every token ──────────────────────
    # Build a map of  idx -> (spell_top, sp_cands)  for every error token.
    # This lets pass 2 substitute clean words around the current [MASK].
    spell_info = {}   # idx -> {"top": str, "cands": list}
    for i, tok in enumerate(tokens):
        word = tok["clean"]
        if word and word.isalpha() and spell.unknown([word]):
            cands = spell_candidates(word)
            spell_info[i] = {"top": cands[0] if cands else word, "cands": cands}

    # ── Pass 2: for each error token, give BERT a clean context ───────
    # Every OTHER misspelled word is swapped with its spell correction
    # before masking the current token, so BERT sees a coherent sentence.
    results = []
    for i, tok in enumerate(tokens):
        word = tok["clean"]

        # Non-alphabetic tokens (punctuation, numbers) – pass through
        if not word or not word.isalpha():
            results.append({
                "original": tok["raw"], "clean": word,
                "is_error": False, "spell_suggestion": None,
                "bert_suggestion": None, "agree": True, "candidates": []
            })
            continue

        # Correctly spelled token – pass through
        if i not in spell_info:
            results.append({
                "original": tok["raw"], "clean": word,
                "is_error": False, "spell_suggestion": word,
                "bert_suggestion": word, "agree": True, "candidates": []
            })
            continue

        spell_top = spell_info[i]["top"]
        sp_cands  = spell_info[i]["cands"]

        # Build a clean sentence: other errors → spell correction, this one → [MASK]
        clean_words = []
        for j, t in enumerate(tokens):
            if j == i:
                clean_words.append("[MASK]")
            elif j in spell_info:
                clean_words.append(spell_info[j]["top"])   # use corrected form
            else:
                clean_words.append(t["raw"])
        clean_ctx = " ".join(clean_words)

        # BERT fill-mask on the clean context
        bert_map = {}
        try:
            preds = get_mlm()(clean_ctx)
            bert_map = {p["token_str"].strip(): p["score"] for p in preds}
        except Exception as e:
            print(f"BERT inference error: {e}")

        bert_top = max(bert_map, key=bert_map.get) if bert_map else spell_top

        # ── Combine spell + BERT candidates and score them ────────────
        all_cands = set(sp_cands) | set(list(bert_map.keys())[:5])
        ranked = []
        for cand in all_cands:
            edit_sim  = fuzz.ratio(word, cand) / 100.0
            bert_prob = bert_map.get(cand, 0.0)
            freq      = zipf_frequency(cand, "en")
            score     = combine_score(cand, edit_sim, bert_prob, freq)
            ranked.append({
                "word": cand,
                "combined_score": round(score, 4),
                "edit_sim":  round(edit_sim, 3),
                "bert_prob": round(bert_prob, 4),
                "freq":      round(freq, 2),
                "source": ("both"  if cand in sp_cands and cand in bert_map
                           else "spell" if cand in sp_cands
                           else "bert")
            })

        ranked.sort(key=lambda x: -x["combined_score"])
        combined_top = ranked[0]["word"] if ranked else spell_top

        results.append({
            "original":            tok["raw"],
            "clean":               word,
            "is_error":            True,
            "spell_suggestion":    spell_top,
            "bert_suggestion":     bert_top,
            "combined_suggestion": combined_top,
            "agree":               spell_top == bert_top,
            "candidates":          ranked[:8]
        })

    return results


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/ping")
def ping():
    return jsonify({"status": "ok", "model_loaded": _mlm is not None})


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/correct", methods=["POST"])
def api_correct():
    try:
        payload  = request.json or {}
        text     = payload.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400

        analysis = analyse(text)

        corrected_words = []
        for item in analysis:
            if item["is_error"] and item.get("combined_suggestion"):
                corrected_words.append(item["combined_suggestion"])
            else:
                corrected_words.append(item["original"])
        corrected_text = " ".join(corrected_words)

        disagreements = [a for a in analysis if a["is_error"] and not a["agree"]]

        return jsonify({
            "original":        text,
            "corrected":       corrected_text,
            "tokens":          analysis,
            "error_count":     sum(1 for a in analysis if a["is_error"]),
            "disagreements":   len(disagreements),
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
