const input = document.getElementById("inputText");
const btn = document.getElementById("btnCheck");
const results = document.getElementById("results");

btn.addEventListener("click", doCheck);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter") doCheck();
});

async function doCheck() {
  const text = input.value.trim();
  if (!text) {
    results.innerHTML = `<div class="text-sm text-red-500">Please type something.</div>`;
    return;
  }
  results.innerHTML = `<div class="text-sm text-slate-500">Checking…</div>`;

  try {
    const res = await fetch("/api/correct", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    renderResults(data);
  } catch (err) {
    results.innerHTML = `<div class="text-red-500">Error contacting server.</div>`;
    console.error(err);
  }
}

function renderResults(data) {
  const { original, best_suggestion, confidence, grammar_corrected, per_word } = data;

  let html = `
    <div class="p-3 border rounded bg-white/50">
      <div class="text-sm text-slate-600">Original:</div>
      <div class="font-medium mt-1">${escapeHtml(original)}</div>
    </div>

    <div class="p-3 border rounded bg-white/50 mt-2">
      <div class="text-sm text-slate-600">Best suggestion:</div>
      <div class="font-medium mt-1">${escapeHtml(best_suggestion)} 
        <span class="text-xs px-2 ml-2 bg-slate-100 rounded">${confidence}%</span>
      </div>
    </div>

    <div class="p-3 border rounded bg-white/50 mt-2">
      <div class="text-sm text-slate-600">Grammar-corrected (quick):</div>
      <div class="mt-1">${escapeHtml(grammar_corrected)}</div>
    </div>

    <div class="mt-3">
      <div class="text-sm text-slate-600 mb-2">Per-word suggestions:</div>
      <div class="space-y-2">`;

  per_word.forEach(pw => {
    html += `
      <div class="p-3 border rounded bg-slate-50">
        <div><strong>${escapeHtml(pw.original)}</strong></div>
        <div class="text-xs text-slate-600 mt-1">Spell correction: ${escapeHtml(pw.spell_correction)}</div>
        <div class="text-xs text-slate-600 mt-1">Spell candidates: ${escapeHtml(pw.spell_candidates.join(", "))}</div>
        <div class="text-xs text-slate-600 mt-1">Fuzzy suggestions: ${pw.fuzzy_candidates.map(f=>escapeHtml(f.word + " ("+f.score+"%)")).join(", ")}</div>
      </div>`;
  });

  html += `</div></div>`;

  results.innerHTML = html;
}

function escapeHtml(text) {
  if (!text && text !== 0) return "";
  return String(text)
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;");
}
