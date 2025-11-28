async function doCheck() {
    const inputEl = document.getElementById("userInput");
    const resultEl = document.getElementById("result");
    const text = inputEl.value.trim();

    if (!text) {
        resultEl.textContent = "Please enter a word or sentence.";
        return;
    }

    try {
        const response = await fetch("/api/correct", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });

        if (!response.ok) {
            throw new Error("Server Error: " + response.status);
        }

        const data = await response.json();

        console.log("API Response:", data);

        // Show result in page
        resultEl.innerHTML = `
            <p><strong>Original:</strong> ${data.original}</p>
            <p><strong>Best Suggestion:</strong> ${data.best_suggestion}</p>
            <p><strong>Confidence:</strong> ${data.confidence}%</p>
            <p><strong>Grammar Corrected:</strong> ${data.grammar_corrected}</p>
        `;
    } catch (err) {
        console.error(err);
        resultEl.textContent = "⚠️ Error contacting server. Please try again.";
    }
}
