/**
 * FeedbackManager - Handles user feedback on risk assessments.
 * Provides thumbs up/down UI, correct-label selection, and localStorage persistence.
 */
class FeedbackManager {
    constructor() {
        this.storageKey = "si_feedback_log";
        this.endpoint = "/feedback";
        this._injectStyles();
    }

    // ------------------------------------------------------------------ //
    //  Public API
    // ------------------------------------------------------------------ //

    /**
     * Render the feedback bar beneath the analysis result.
     * @param {string} analysisId  - Unique ID for this analysis (e.g. UUID or timestamp).
     * @param {string} predictedLabel - One of "LOW_RISK" | "MODERATE_RISK" | "HIGH_RISK_SELF_HARM" | "HIGH_RISK_HARM_TO_OTHERS".
     * @param {string|null} containerId - Optional id of DOM element to attach bar into.
     *                                    Defaults to "#feedback-container".
     */
    show(analysisId, predictedLabel, containerId = "feedback-container") {
        const container = document.getElementById(containerId);
        if (!container) {
            console.warn(`FeedbackManager.show: element #${containerId} not found.`);
            return;
        }

        // Avoid duplicate bars
        const existing = container.querySelector(".feedback-bar");
        if (existing) existing.remove();

        const bar = this._buildBar(analysisId, predictedLabel);
        container.appendChild(bar);
    }

    /**
     * Submit feedback to the server and persist locally.
     * @param {string} analysisId
     * @param {string} correctLabel  - One of "LOW_RISK" | "MODERATE_RISK" | "HIGH_RISK_SELF_HARM" | "HIGH_RISK_HARM_TO_OTHERS".
     * @param {string} [comment=""] - Optional free-text comment.
     * @returns {Promise<boolean>} - true on success, false on failure.
     */
    async submit(analysisId, correctLabel, comment = "") {
        const payload = {
            analysis_id: analysisId,
            correct_label: correctLabel,
            comment: comment,
            timestamp: new Date().toISOString(),
        };

        // Persist locally regardless of server status
        this._saveLocal(payload);

        try {
            const response = await fetch(this.endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            return response.ok;
        } catch (err) {
            console.warn("FeedbackManager.submit: server unreachable, stored locally.", err);
            return false;
        }
    }

    /**
     * Return all locally stored feedback entries.
     * @returns {Array<Object>}
     */
    getLocalFeedback() {
        try {
            return JSON.parse(localStorage.getItem(this.storageKey) || "[]");
        } catch {
            return [];
        }
    }

    /**
     * Remove all locally stored feedback entries.
     */
    clearLocalFeedback() {
        localStorage.removeItem(this.storageKey);
    }

    // ------------------------------------------------------------------ //
    //  Private helpers
    // ------------------------------------------------------------------ //

    _buildBar(analysisId, predictedLabel) {
        const bar = document.createElement("div");
        bar.className = "feedback-bar";
        bar.dataset.analysisId = analysisId;
        bar.dataset.predictedLabel = predictedLabel;

        bar.innerHTML = `
            <div class="feedback-question">
                <span class="feedback-text">Was this assessment accurate?</span>
                <button class="feedback-btn feedback-btn--yes" title="Yes, accurate" aria-label="Thumbs up">
                    &#128077;
                </button>
                <button class="feedback-btn feedback-btn--no" title="No, incorrect" aria-label="Thumbs down">
                    &#128078;
                </button>
            </div>
            <div class="feedback-correction" style="display:none;">
                <span class="feedback-correction-label">Select the correct label:</span>
                <select class="feedback-select" aria-label="Correct risk label">
                    <option value="">-- choose --</option>
                    <option value="LOW_RISK">Low Risk</option>
                    <option value="MODERATE_RISK">Moderate Risk</option>
                    <option value="HIGH_RISK_SELF_HARM">High Risk - Self Harm</option>
                    <option value="HIGH_RISK_HARM_TO_OTHERS">High Risk - Harm To Others</option>
                </select>
                <textarea
                    class="feedback-comment"
                    placeholder="Optional comment..."
                    rows="2"
                    maxlength="500"
                    aria-label="Optional comment"
                ></textarea>
                <button class="feedback-btn feedback-btn--submit">Submit</button>
                <button class="feedback-btn feedback-btn--cancel">Cancel</button>
            </div>
            <div class="feedback-thanks" style="display:none;">
                Thank you for improving our model!
            </div>
        `;

        this._attachListeners(bar, analysisId, predictedLabel);
        return bar;
    }

    _attachListeners(bar, analysisId, predictedLabel) {
        const question   = bar.querySelector(".feedback-question");
        const correction = bar.querySelector(".feedback-correction");
        const thanks     = bar.querySelector(".feedback-thanks");
        const selectEl   = bar.querySelector(".feedback-select");
        const commentEl  = bar.querySelector(".feedback-comment");

        // Thumbs UP — assessment was correct
        bar.querySelector(".feedback-btn--yes").addEventListener("click", async () => {
            question.style.display = "none";
            const ok = await this.submit(analysisId, predictedLabel, "");
            thanks.style.display = "block";
            if (!ok) thanks.textContent = "Feedback saved locally (server unavailable).";
        });

        // Thumbs DOWN — show correction dropdown
        bar.querySelector(".feedback-btn--no").addEventListener("click", () => {
            question.style.display = "none";
            correction.style.display = "flex";
        });

        // Submit correction
        bar.querySelector(".feedback-btn--submit").addEventListener("click", async () => {
            const label = selectEl.value;
            if (!label) {
                selectEl.style.outline = "2px solid #ef4444";
                return;
            }
            selectEl.style.outline = "";
            correction.style.display = "none";
            const ok = await this.submit(analysisId, label, commentEl.value.trim());
            thanks.style.display = "block";
            if (!ok) thanks.textContent = "Feedback saved locally (server unavailable).";
        });

        // Cancel correction
        bar.querySelector(".feedback-btn--cancel").addEventListener("click", () => {
            correction.style.display = "none";
            question.style.display = "flex";
        });
    }

    _saveLocal(entry) {
        const log = this.getLocalFeedback();
        log.push(entry);
        // Keep last 500 entries to avoid storage bloat
        const trimmed = log.slice(-500);
        try {
            localStorage.setItem(this.storageKey, JSON.stringify(trimmed));
        } catch (err) {
            console.warn("FeedbackManager._saveLocal: localStorage write failed.", err);
        }
    }

    _injectStyles() {
        if (document.getElementById("feedback-manager-styles")) return;
        const style = document.createElement("style");
        style.id = "feedback-manager-styles";
        style.textContent = `
            .feedback-bar {
                margin-top: 1rem;
                padding: 0.75rem 1rem;
                background: rgba(255,255,255,0.07);
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 0.5rem;
                font-size: 0.875rem;
                color: #cbd5e1;
            }
            .feedback-question {
                display: flex;
                align-items: center;
                gap: 0.5rem;
                flex-wrap: wrap;
            }
            .feedback-text {
                flex: 1;
                min-width: 0;
            }
            .feedback-btn {
                background: transparent;
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: 0.35rem;
                color: #e2e8f0;
                padding: 0.25rem 0.6rem;
                cursor: pointer;
                font-size: 1rem;
                transition: background 0.15s, border-color 0.15s;
            }
            .feedback-btn:hover {
                background: rgba(255,255,255,0.12);
                border-color: rgba(255,255,255,0.4);
            }
            .feedback-btn--submit {
                background: #3b82f6;
                border-color: #3b82f6;
                font-size: 0.8rem;
                padding: 0.3rem 0.8rem;
            }
            .feedback-btn--submit:hover { background: #2563eb; }
            .feedback-btn--cancel { font-size: 0.8rem; padding: 0.3rem 0.6rem; }
            .feedback-correction {
                display: flex;
                flex-wrap: wrap;
                align-items: flex-start;
                gap: 0.5rem;
                margin-top: 0.5rem;
            }
            .feedback-correction-label {
                width: 100%;
                font-size: 0.8rem;
                color: #94a3b8;
            }
            .feedback-select {
                background: #1e293b;
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: 0.35rem;
                color: #e2e8f0;
                padding: 0.3rem 0.5rem;
                font-size: 0.85rem;
                cursor: pointer;
            }
            .feedback-comment {
                width: 100%;
                background: #1e293b;
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: 0.35rem;
                color: #e2e8f0;
                padding: 0.4rem 0.5rem;
                font-size: 0.82rem;
                resize: vertical;
            }
            .feedback-thanks {
                color: #4ade80;
                font-weight: 500;
                padding: 0.25rem 0;
            }
        `;
        document.head.appendChild(style);
    }
}

// Expose globally
window.FeedbackManager = FeedbackManager;
