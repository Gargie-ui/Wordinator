import nltk
nltk.data.path.append("./nltk_data")

from flask import Flask, render_template, request, jsonify
from spellchecker import SpellChecker
from textblob import TextBlob
from rapidfuzz import process, fuzz
from wordfreq import zipf_frequency, top_n_list

app = Flask(__name__, static_folder="static", template_folder="templates")

# initialize spellchecker
spell = SpellChecker(distance=2)  # distance can be tuned

# Build a candidate vocabulary (top English words by frequency)
# wordfreq provides ordered useful words; we take a decent-sized list
VOCAB = top_n_list("en", 50000)  # ~50k common words

def fuzzy_candidates(word, n=6, score_cutoff=60):
    """Return fuzzy matches from our VOCAB using rapidfuzz."""
    if not word:
        return []
    results = process.extract(
        word,
        VOCAB,
        scorer=fuzz.ratio,
        limit=n
    )
    # results: list of tuples (match, score, index)
    return [{"word": r[0], "score": int(r[1]), "freq_score": round(zipf_frequency(r[0], "en"), 2)} 
            for r in results if r[1] >= score_cutoff]

def per_word_spell_suggestions(text):
    """Return per-word corrections with candidates."""
    words = text.split()
    suggestions = []
    for w in words:
        lowered = w.strip()
        if not lowered:
            continue
        # simple correction from pyspellchecker
        corrected = spell.correction(lowered)
        # If correction is same as input, still compute fuzzy candidates
        fuzzy = fuzzy_candidates(lowered, n=6)
        # also provide top_n from spellchecker
        candidates = list(spell.candidates(lowered))[:6]
        suggestions.append({
            "original": lowered,
            "spell_correction": corrected,
            "spell_candidates": candidates,
            "fuzzy_candidates": fuzzy
        })
    return suggestions

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/correct", methods=["POST"])
def api_correct():
    payload = request.json or {}
    text = payload.get("text", "")
    # Basic normalization
    text = text.strip()

    # Per-word spelling suggestions
    per_word = per_word_spell_suggestions(text)

    # A "quick grammar-corrected" version using textblob's correct (note: basic)
    try:
        blob = TextBlob(text)
        grammar_corrected = str(blob.correct())
    except Exception:
        grammar_corrected = text

    # A best-effort "best single suggestion" using combining techniques:
    # If text is one token, show top fuzzy/spell suggestion
    best_suggestion = None
    confidence = 0
    if len(text.split()) == 1:
        token = text
        # choose best candidate among spell and fuzzy by simple scoring
        spell_corr = spell.correction(token)
        fuzzy_list = fuzzy_candidates(token, n=6)
        # pick best fuzzy if its score is high
        if fuzzy_list:
            top_fuzzy = fuzzy_list[0]
            if top_fuzzy["score"] > 85:
                best_suggestion = top_fuzzy["word"]
                confidence = top_fuzzy["score"]
            else:
                # fallback to spell correction
                best_suggestion = spell_corr
                confidence = 70 if spell_corr != token else 40
        else:
            best_suggestion = spell_corr
            confidence = 70 if spell_corr != token else 40
    else:
        # for multiword, return grammar corrected as "best"
        best_suggestion = grammar_corrected
        confidence = 60

    return jsonify({
        "original": text,
        "per_word": per_word,
        "grammar_corrected": grammar_corrected,
        "best_suggestion": best_suggestion,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
