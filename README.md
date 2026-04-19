# ✨ Wordinator

A powerful NLP-powered web tool that analyzes and processes text in real time — combining spell checking, sentiment analysis, fuzzy word matching, and word frequency scoring in one clean interface.

---

## 🚀 Features

- 🔤 **Spell Checker** — Detects and suggests corrections for misspelled words
- 💬 **Sentiment Analysis** — Determines if text is positive, negative, or neutral
- 🔍 **Fuzzy Word Matching** — Finds near-match words even with typos or partial input
- 📊 **Word Frequency Scoring** — Ranks words by real-world usage using Zipf scores

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask |
| Language | Python |
| Spell Checking | pyspellchecker |
| Sentiment Analysis | TextBlob |
| Fuzzy Matching | RapidFuzz |
| Word Frequency | wordfreq |
| Tokenization | NLTK |

---

## 📁 Project Structure

```
Wordinator/
│
├── static/               # CSS / JS assets
├── templates/
│   └── index.html        # Web UI
├── nltk_data/            # Local NLTK data
├── app.py                # Flask app
└── requirements.txt
```

---

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/Gargie-ui/Wordinator
cd Wordinator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Then open `http://localhost:5000` in your browser.

---

## 📦 Requirements

```
flask
nltk
textblob
pyspellchecker
rapidfuzz
wordfreq
```

---

## 💡 How It Works

1. User enters text in the web interface
2. **SpellChecker** scans for misspelled words and suggests corrections
3. **TextBlob** analyzes the overall sentiment of the input
4. **RapidFuzz** finds the closest matching words using fuzzy string comparison
5. **wordfreq** scores each word by how commonly it appears in real-world language (Zipf scale)

---

## 👩‍💻 Author

**Gargi Channe**
- 🔗 [LinkedIn](https://www.linkedin.com/in/gargi-channe)
- 🐙 [GitHub](https://github.com/Gargie-ui)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
