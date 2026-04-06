/**
 * app.js - Main Application Module
 * Suicidal Ideation Detection - Social Media Risk Assessment Platform
 *
 * Architecture:
 *   ApiClient      - All HTTP calls to the FastAPI backend
 *   HistoryManager - localStorage CRUD for analysis history
 *   UIController   - All DOM manipulation and rendering
 *   App            - Top-level orchestrator, wires everything together
 *
 * Usage (classic script, no bundler required):
 *   <script src="/static/js/charts.js"></script>
 *   <script src="/static/js/app.js"></script>
 *
 * The inline script in index.html duplicates some functions for
 * standalone operation; this file is the "organized" version.
 * Both can co-exist. This file guards against re-declaring globals.
 */

'use strict';

/* ============================================================
   CONSTANTS
============================================================= */

const SI_CONFIG = {
  API_BASE:       '',           // Same origin; override for dev: 'http://localhost:8000'
  ANALYZE_PATH:   '/analyze',
  STATS_PATH:     '/stats',
  FEEDBACK_PATH:  '/feedback',
  HISTORY_KEY:    'si_history',
  DRAFT_KEY:      'si_draft',
  STATS_INTERVAL: 30_000,       // ms between /stats polls
  HISTORY_MAX:    10,
  MAX_TEXT_LEN:   2000,
  SOFT_CHAR_WARN: 500,
  REQUEST_TIMEOUT:45_000,       // ms
};

/* ============================================================
   CLASS: ApiClient
============================================================= */

class ApiClient {
  /**
   * @param {string} baseUrl  API base URL (default same-origin)
   */
  constructor(baseUrl = SI_CONFIG.API_BASE) {
    this.baseUrl = baseUrl;
    this._pendingAnalyze = null;
  }

  /**
   * Analyze text synchronously (POST /analyze).
   * @param {string}  text
   * @param {boolean} useFallback  send use_ml_fallback: true
   * @returns {Promise<object>}  API response JSON
   */
  async analyze(text, useFallback = false) {
    // Cancel any pending analysis
    this.cancelAnalyze();

    const controller = new AbortController();
    this._pendingAnalyze = controller;

    const timeoutId = setTimeout(() => controller.abort(), SI_CONFIG.REQUEST_TIMEOUT);

    try {
      const resp = await fetch(`${this.baseUrl}${SI_CONFIG.ANALYZE_PATH}`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ text, use_ml_fallback: useFallback }),
        signal:  controller.signal,
      });

      clearTimeout(timeoutId);

      if (!resp.ok) {
        let detail = `HTTP ${resp.status}`;
        try {
          const j = await resp.json();
          detail = j.detail || j.message || detail;
        } catch(_) {}
        throw new ApiError(detail, resp.status);
      }

      return await resp.json();

    } catch (err) {
      clearTimeout(timeoutId);
      if (err.name === 'AbortError') throw new ApiError('Request timed out or was cancelled.', 408);
      if (err instanceof ApiError)  throw err;
      throw new ApiError(err.message || 'Network error', 0);
    } finally {
      this._pendingAnalyze = null;
    }
  }

  /**
   * Async (streaming / webhook) version. Falls back to regular analyze.
   * Extend this method to handle SSE or WebSocket streaming if the backend
   * supports it in the future.
   * @param {string} text
   * @returns {Promise<object>}
   */
  async analyzeAsync(text) {
    // Future: EventSource('/analyze-stream?text=...')
    return this.analyze(text, false);
  }

  /**
   * Fetch aggregate stats (GET /stats).
   * @returns {Promise<object>}
   */
  async getStats() {
    const resp = await fetch(`${this.baseUrl}${SI_CONFIG.STATS_PATH}`, {
      signal: AbortSignal.timeout(8000),
    });
    if (!resp.ok) throw new ApiError(`Stats fetch failed: ${resp.status}`, resp.status);
    return resp.json();
  }

  /**
   * Submit analyst feedback for a result (POST /feedback).
   * @param {object} data  { analysis_id, text, risk_level, correct_label, notes }
   * @returns {Promise<object>}
   */
  async postFeedback(data) {
    const resp = await fetch(`${this.baseUrl}${SI_CONFIG.FEEDBACK_PATH}`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(data),
      signal:  AbortSignal.timeout(10000),
    });
    if (!resp.ok) throw new ApiError(`Feedback submission failed: ${resp.status}`, resp.status);
    return resp.json();
  }

  /** Cancel any in-flight analyze request. */
  cancelAnalyze() {
    if (this._pendingAnalyze) {
      this._pendingAnalyze.abort();
      this._pendingAnalyze = null;
    }
  }
}

/* Custom error class for API failures */
class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.name   = 'ApiError';
    this.status = status;
  }
}

/* ============================================================
   CLASS: HistoryManager
============================================================= */

class HistoryManager {
  /**
   * @param {string} storageKey  localStorage key
   * @param {number} maxItems    rolling window size
   */
  constructor(storageKey = SI_CONFIG.HISTORY_KEY, maxItems = SI_CONFIG.HISTORY_MAX) {
    this.storageKey = storageKey;
    this.maxItems   = maxItems;
  }

  /**
   * Read all history entries (newest first).
   * @returns {HistoryEntry[]}
   */
  getAll() {
    try {
      const raw = localStorage.getItem(this.storageKey);
      return raw ? JSON.parse(raw) : [];
    } catch(_) {
      return [];
    }
  }

  /**
   * Find a single entry by id.
   * @param {number} id
   * @returns {HistoryEntry|null}
   */
  getById(id) {
    return this.getAll().find(e => e.id === id) || null;
  }

  /**
   * Prepend a new entry, enforce maxItems, persist.
   * @param {HistoryEntry} entry
   * @returns {HistoryEntry}  the stored entry (with generated id)
   */
  add(entry) {
    const history = this.getAll();
    const stored  = { id: Date.now(), created_at: new Date().toISOString(), ...entry };
    history.unshift(stored);
    if (history.length > this.maxItems) history.splice(this.maxItems);
    this._save(history);
    return stored;
  }

  /**
   * Remove entry by id.
   * @param {number} id
   * @returns {boolean} true if removed
   */
  remove(id) {
    const history = this.getAll();
    const idx     = history.findIndex(e => e.id === id);
    if (idx === -1) return false;
    history.splice(idx, 1);
    this._save(history);
    return true;
  }

  /** Delete all entries. */
  clear() {
    try { localStorage.removeItem(this.storageKey); } catch(_) {}
  }

  /**
   * Replace an existing entry (for updating with feedback, etc.).
   * @param {number} id
   * @param {object} patch  partial update
   * @returns {boolean}
   */
  update(id, patch) {
    const history = this.getAll();
    const item    = history.find(e => e.id === id);
    if (!item) return false;
    Object.assign(item, patch);
    this._save(history);
    return true;
  }

  /** Serialize to JSON-formatted string for export. */
  export() {
    return JSON.stringify(this.getAll(), null, 2);
  }

  /* ----- private ----- */
  _save(data) {
    try { localStorage.setItem(this.storageKey, JSON.stringify(data)); } catch(_) {}
  }
}

/**
 * @typedef {object} HistoryEntry
 * @property {number}  id
 * @property {string}  created_at
 * @property {string}  text         truncated input
 * @property {string}  risk_level
 * @property {number}  risk_score
 * @property {number}  confidence
 * @property {string}  model_used
 * @property {number}  elapsed_ms
 * @property {object}  [raw]        full API response
 */

/* ============================================================
   CLASS: UIController
============================================================= */

class UIController {
  /**
   * @param {object} elements  Map of id → DOM element (populated in constructor)
   */
  constructor() {
    this.els = {};
    this._resolveElements();
    this._gaugeInstance  = null;
    this._donutInstance  = null;
  }

  /* ----- DOM resolution ----- */

  _resolveElements() {
    const ids = [
      'post-input', 'char-counter', 'analyze-btn', 'ml-fallback-toggle',
      'results-panel', 'history-list',
      'stat-today', 'stat-high', 'stat-time', 'stat-model',
    ];
    ids.forEach(id => {
      const el = document.getElementById(id);
      if (el) this.els[id] = el;
    });
  }

  /* ----- Input / Textarea ----- */

  getInputText() {
    return this.els['post-input']?.value?.trim() || '';
  }

  setInputText(text) {
    if (this.els['post-input']) this.els['post-input'].value = text;
  }

  clearInput() {
    this.setInputText('');
    this.updateCharCounter(0);
  }

  updateCharCounter(len) {
    const el = this.els['char-counter'];
    if (!el) return;
    el.textContent = `${len} / ${SI_CONFIG.SOFT_CHAR_WARN} characters`;
    el.className   = 'char-counter';
    if (len > SI_CONFIG.MAX_TEXT_LEN * 0.95) el.classList.add('over');
    else if (len > SI_CONFIG.SOFT_CHAR_WARN)  el.classList.add('warn');
  }

  isFallbackEnabled() {
    return this.els['ml-fallback-toggle']?.checked || false;
  }

  /* ----- Button / Loading state ----- */

  setLoading(on) {
    const btn   = this.els['analyze-btn'];
    const input = this.els['post-input'];
    if (!btn || !input) return;

    if (on) {
      btn.classList.add('loading');
      btn.disabled      = true;
      const span        = btn.querySelector('.btn-text');
      if (span) span.textContent = 'Analyzing…';
      input.classList.add('loading-input');
      input.disabled    = true;
    } else {
      btn.classList.remove('loading');
      btn.disabled      = false;
      const span        = btn.querySelector('.btn-text');
      if (span) span.textContent = 'Analyze';
      input.classList.remove('loading-input');
      input.disabled    = false;
    }
  }

  shakeInput() {
    const el = this.els['post-input'];
    if (!el) return;
    el.style.animation = 'none';
    void el.offsetHeight;
    el.style.animation = 'borderGlow 0.3s ease 3';
    el.focus();
  }

  /* ----- Results panel ----- */

  showResultsLoading() {
    const panel = this.els['results-panel'];
    if (!panel) return;
    panel.classList.add('visible');
    panel.innerHTML = `
      <div class="processing-state">
        <div class="processing-ring"></div>
        <div class="processing-text">Running AI Agent Pipeline</div>
        <div class="processing-steps">
          <div class="processing-step active" id="ps-1">🤖 Classifier Agent: detecting risk signals...</div>
          <div class="processing-step"         id="ps-2">🔍 Explainer Agent: extracting linguistic patterns...</div>
          <div class="processing-step"         id="ps-3">💡 Recommender Agent: generating resources...</div>
        </div>
      </div>`;

    setTimeout(() => this._advanceProcessingStep(1, 2), 900);
    setTimeout(() => this._advanceProcessingStep(2, 3), 1800);
  }

  _advanceProcessingStep(done, active) {
    const doneEl   = document.getElementById(`ps-${done}`);
    const activeEl = document.getElementById(`ps-${active}`);
    if (doneEl)   { doneEl.classList.add('done');   doneEl.classList.remove('active'); }
    if (activeEl)   activeEl.classList.add('active');
  }

  renderResults(data, elapsedMs) {
    const panel = this.els['results-panel'];
    if (!panel) return;
    panel.classList.add('visible');

    const clf       = data.classification || {};
    const riskLevel = (clf.risk_level || data.risk_level || 'UNKNOWN').toUpperCase();
    const score     = typeof clf.risk_score === 'number' ? clf.risk_score : (typeof data.risk_score === 'number' ? data.risk_score : 0);
    const conf      = typeof clf.confidence === 'number' ? clf.confidence : (typeof data.confidence === 'number' ? data.confidence : 0);
    const model     = data.model || data.model_used || data.tier_used || 'AI Agent';
    const expl      = data.explanation || {};
    const recs      = data.recommendations || {};

    const cls    = UIController.riskClass(riskLevel);
    const emoji  = UIController.riskEmoji(riskLevel);
    const label  = UIController.riskLabel(riskLevel);

    panel.innerHTML = `
      <div class="risk-badge-wrap">
        <div class="risk-badge ${cls}">
          <div class="risk-dot"></div>
          ${emoji} ${label}
        </div>
        <div class="risk-meta">
          <span style="font-size:1.4rem;font-weight:800;color:var(--text-primary)">
            ${Math.round(score)}<span style="font-size:.7rem;color:var(--text-muted);font-weight:400">/100</span>
          </span>
          <span class="risk-model-tag">via ${this._esc(model)}</span>
          ${elapsedMs ? `<span class="risk-model-tag">${elapsedMs}ms</span>` : ''}
        </div>
      </div>

      <div class="gauge-row">
        <div class="gauge-wrap">
          <div class="gauge-canvas-wrap">
            <canvas id="risk-gauge" width="160" height="90" aria-label="Risk score gauge"></canvas>
            <div class="gauge-score" id="gauge-score-text">0</div>
          </div>
          <div class="gauge-label">Risk Score</div>
        </div>
        <div class="metrics-col">
          ${this._renderMetric('Risk Level', Math.round(score), '%', cls)}
          ${this._renderMetric('Confidence', Math.round(conf * 100), '%', 'conf')}
          ${this._renderScoreBreakdown(data)}
        </div>
      </div>

      <div class="sections-grid">
        ${this._renderClassifierSection(expl, data)}
        ${this._renderNormalizationSection(data.normalization || {})}
        ${this._renderSystemSignalsSection(data.system_signals || {})}
        ${this._renderLinguisticSection(expl)}
        ${this._renderRecsSection(recs, riskLevel)}
      </div>`;

    // Animate gauge
    if (window.SICharts) {
      try {
        this._gaugeInstance = new window.SICharts.RiskGauge('risk-gauge');
        this._gaugeInstance.draw(score);
      } catch(_) {
        this._fallbackGauge(score);
      }
    } else {
      this._fallbackGauge(score);
    }

    this._animCounter('gauge-score-text', 0, Math.round(score), 1000);

    // Update model indicator in stats bar
    const mEl = this.els['stat-model'];
    if (mEl) mEl.textContent = model.toLowerCase().includes('ml') ? 'ML' : 'Agent';

    setTimeout(() => panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 100);
  }

  _fallbackGauge(score) {
    // Inline canvas gauge if SICharts not loaded
    const canvas = document.getElementById('risk-gauge');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    canvas.width  = 160 * dpr;
    canvas.height = 90  * dpr;
    canvas.style.width  = '160px';
    canvas.style.height = '90px';
    ctx.scale(dpr, dpr);
    const cx = 80, cy = 80, r = 65;
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.lineWidth = 12; ctx.lineCap = 'round'; ctx.stroke();
    if (score > 0) {
      ctx.beginPath();
      ctx.arc(cx, cy, r, Math.PI, Math.PI + (score / 100) * Math.PI);
      ctx.strokeStyle = score > 60 ? '#ef4444' : score > 30 ? '#f59e0b' : '#22c55e';
      ctx.lineWidth = 12; ctx.stroke();
    }
  }

  renderError(message) {
    const panel = this.els['results-panel'];
    if (!panel) return;
    panel.classList.add('visible');
    panel.innerHTML = `
      <div class="error-card">
        <div class="error-icon">⚠️</div>
        <div>
          <div class="error-title">Analysis Failed</div>
          <div class="error-msg">${this._esc(message)}</div>
          <div style="margin-top:10px;">
            <button
              onclick="window.SIApp && window.SIApp.analyze()"
              style="background:none;border:1px solid rgba(239,68,68,0.3);color:#fca5a5;font-size:0.8rem;padding:6px 14px;border-radius:6px;cursor:pointer;font-family:inherit;"
            >Retry</button>
          </div>
        </div>
      </div>`;
  }

  /* ----- History panel ----- */

  renderHistory(entries, onItemClick, onClear) {
    const list = this.els['history-list'];
    if (!list) return;

    if (!entries.length) {
      list.innerHTML = '<div class="history-empty">No analyses yet. Run your first analysis above.</div>';
      return;
    }

    list.innerHTML = entries.map(entry => {
      const cls   = UIController.riskClass(entry.risk_level);
      const label = UIController.riskLabel(entry.risk_level);
    const color = cls === 'high' ? 'var(--risk-high)' : cls === 'moderate' ? 'var(--risk-mod)' : 'var(--risk-low)';
      const time  = UIController.formatRelativeTime(entry.created_at);
      const text  = entry.text || '';
      return `
        <div class="history-item"
             role="button" tabindex="0"
             data-history-id="${entry.id}"
             onkeydown="if(event.key==='Enter')this.click()">
          <div class="history-risk-dot" style="background:${color}"></div>
          <div class="history-text">${this._esc(text)}${text.length >= 120 ? '…' : ''}</div>
          <span class="history-badge ${cls}">${label}</span>
          <span class="history-time">${time}</span>
        </div>`;
    }).join('');

    // Bind click events
    list.querySelectorAll('.history-item').forEach(el => {
      el.addEventListener('click', () => {
        const id = parseInt(el.dataset.historyId, 10);
        if (onItemClick) onItemClick(id);
      });
    });
  }

  /* ----- Stats bar ----- */

  updateStats(data) {
    const { 'stat-today': todayEl, 'stat-high': highEl, 'stat-time': timeEl } = this.els;

    if (todayEl && data.total_today != null) {
      const cur = parseInt(todayEl.textContent, 10) || 0;
      this._animCounter('stat-today', cur, data.total_today, 500);
    }

    if (highEl && data.high_risk != null) {
      const cur = parseInt(highEl.textContent, 10) || 0;
      this._animCounter('stat-high', cur, data.high_risk, 500);
    }

    if (timeEl) {
      if      (data.avg_time_ms        != null) timeEl.textContent = data.avg_time_ms + 'ms';
      else if (data.avg_processing_time != null) timeEl.textContent = Math.round(data.avg_processing_time * 1000) + 'ms';
    }
  }

  /* ----- Render sub-helpers ----- */

  _renderMetric(name, value, unit, cls) {
    return `
      <div class="metric-item">
        <div class="metric-header">
          <span class="metric-name">${name}</span>
          <span class="metric-val">${value}${unit}</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill ${cls}" style="width:${value}%"></div>
        </div>
      </div>`;
  }

  _renderScoreBreakdown(data) {
    const sb = data.score_breakdown || data.scores || {};
    return Object.entries(sb).slice(0, 2).map(([k, v]) => {
      const pct = typeof v === 'number' ? Math.round(v * 100) : 0;
      return this._renderMetric(k, pct, '%', 'conf');
    }).join('');
  }

  _renderClassifierSection(expl, data) {
    const reasoning = data.classification?.reasoning || expl.reasoning || expl.explanation || data.reasoning || 'No reasoning provided by the classifier.';
    const signalSummary = expl.signal_summary || data.system_signals?.summary || '';
    return `
      <div class="collapsible open">
        <div class="collapsible-header" onclick="this.closest('.collapsible').classList.toggle('open')">
          <span class="collapsible-icon">🧠</span>
          <span class="collapsible-title">AI Classification Reasoning</span>
          <svg class="collapsible-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
        </div>
        <div class="collapsible-body">
          <p class="reasoning-text">${this._esc(reasoning)}</p>
          ${signalSummary ? `<p class="reasoning-text" style="margin-top:10px;color:var(--text-secondary)">${this._esc(signalSummary)}</p>` : ''}
        </div>
      </div>`;
  }

  _renderNormalizationSection(normalization) {
    const originalText = normalization.original_text || '';
    const normalizedText = normalization.normalized_text || originalText;
    const fixes = Array.isArray(normalization.fixes) ? normalization.fixes : [];
    const changed = Boolean(normalization.changed) || (originalText && normalizedText && originalText !== normalizedText);

    const fixTags = fixes.length
      ? fixes.map((fix) => {
          const from = fix.original || fix.pattern || fix.kind || 'pattern';
          const to = fix.replacement || '';
          const count = fix.count ? ` x${fix.count}` : '';
          return `<span class="tag linguistic">${this._esc(String(from))} -> ${this._esc(String(to))}${count}</span>`;
        }).join('')
      : '<span class="tag empty">No merged-word or slang fixes were needed.</span>';

    return `
      <div class="collapsible">
        <div class="collapsible-header" onclick="this.closest('.collapsible').classList.toggle('open')">
          <span class="collapsible-icon">🧩</span>
          <span class="collapsible-title">Normalization Agent</span>
          <svg class="collapsible-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
        </div>
        <div class="collapsible-body">
          <div class="platform-action-box"><strong>Agent View</strong>${changed ? 'Text was normalized before risk detection.' : 'Original text was already clear enough for direct analysis.'}</div>
          <div class="pattern-subsection" style="margin-top:12px">
            <div class="pattern-subsection-title">Original Text</div>
            <div class="reasoning-text">${this._esc(originalText || 'N/A')}</div>
          </div>
          <div class="pattern-subsection" style="margin-top:12px">
            <div class="pattern-subsection-title">Normalized Text</div>
            <div class="reasoning-text">${this._esc(normalizedText || 'N/A')}</div>
          </div>
          <div class="pattern-subsection" style="margin-top:12px">
            <div class="pattern-subsection-title">Detected Fixes</div>
            <div class="tag-list">${fixTags}</div>
          </div>
        </div>
      </div>`;
  }

  _renderSystemSignalsSection(systemSignals) {
    const angles = systemSignals.angles || {};
    const score = systemSignals.system_score || 0;
    const level = systemSignals.system_risk_level || 'LOW_RISK';
    const detected = Array.isArray(systemSignals.signals_detected) ? systemSignals.signals_detected : [];
    const items = [
      ['Harm To Others', angles.harm_to_others_intent],
      ['Explicit Intent', angles.explicit_intent],
      ['Planning / Preparation', angles.planning_preparation],
      ['Finality / Farewell', angles.finality_farewell],
      ['Self-Harm', angles.self_harm],
      ['Hopelessness', angles.hopelessness],
      ['Burden / Worthlessness', angles.burden_worthlessness],
      ['Isolation / Disconnection', angles.isolation_disconnection],
      ['Emotional Dysregulation', angles.emotional_dysregulation],
      ['Help Seeking', angles.help_seeking],
      ['Future Orientation', angles.future_orientation],
    ];

    return `
      <div class="collapsible">
        <div class="collapsible-header" onclick="this.closest('.collapsible').classList.toggle('open')">
          <span class="collapsible-icon">🧭</span>
          <span class="collapsible-title">System Signal Analysis</span>
          <svg class="collapsible-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
        </div>
        <div class="collapsible-body">
          <div class="platform-action-box"><strong>System View</strong>${this._esc(level.replaceAll('_', ' '))} · score ${score}/100</div>
          <div class="tag-list" style="margin-top:10px">
            ${items.map(([name, val]) => `<span class="tag ${val ? 'risk-indicator' : ''}">${this._esc(name)}: ${val ? 'present' : 'absent'}</span>`).join('')}
          </div>
          ${detected.length ? `
            <div class="pattern-subsection" style="margin-top:12px">
              <div class="pattern-subsection-title">Detected Signals</div>
              <div class="tag-list">
                ${detected.flatMap(s => (s.matches || []).slice(0,2)).slice(0,12).map(p => `<span class="tag linguistic">${this._esc(String(p))}</span>`).join('')}
              </div>
            </div>` : ''}
        </div>
      </div>`;
  }

  _renderLinguisticSection(expl) {
    const indicators = this._toArray(expl.risk_indicators);
    const protective = this._toArray(expl.protective_factors);
    const patterns   = this._toArray(expl.linguistic_patterns);

    const tagList = (items, cls) => items.length
      ? items.map(p => `<span class="tag ${cls}">${this._esc(String(p))}</span>`).join('')
      : '<span class="tag empty">None identified</span>';

    return `
      <div class="collapsible">
        <div class="collapsible-header" onclick="this.closest('.collapsible').classList.toggle('open')">
          <span class="collapsible-icon">🔍</span>
          <span class="collapsible-title">Linguistic Analysis</span>
          <svg class="collapsible-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
        </div>
        <div class="collapsible-body">
          <div class="pattern-subsection">
            <div class="pattern-subsection-title">⚠ Risk Indicators</div>
            <div class="tag-list">${tagList(indicators, 'risk-indicator')}</div>
          </div>
          <div class="pattern-subsection">
            <div class="pattern-subsection-title">🛡 Protective Factors</div>
            <div class="tag-list">${tagList(protective, 'protective')}</div>
          </div>
          <div class="pattern-subsection">
            <div class="pattern-subsection-title">📝 Linguistic Patterns</div>
            <div class="tag-list">${tagList(patterns, 'linguistic')}</div>
          </div>
        </div>
      </div>`;
  }

  _renderRecsSection(recs, riskLevel) {
    const resources        = this._toArray(recs.resources);
    const immediateAction  = recs.immediate_action || '';
    const platformAction   = recs.platform_action  || '';
    const isHigh           = riskLevel.startsWith('HIGH_RISK') || riskLevel === 'HIGH';
    const isOtherHarm      = riskLevel === 'HIGH_RISK_HARM_TO_OTHERS';
    const icons            = ['📞','💬','🌐','📚','🏥','💚','🤝','🆘'];

    const immediateHtml = immediateAction ? (isHigh
      ? `<div class="immediate-action-banner">
             <div class="icon-wrap">🚨</div>
             <div class="action-content">
               <h4>${isOtherHarm ? 'Violence Threat - Immediate Action Required' : 'Immediate Action Required'}</h4>
               <p>${this._esc(immediateAction)}</p>
             </div>
           </div>`
      : `<div style="background:var(--risk-mod-bg);border:1px solid rgba(245,158,11,.25);border-radius:var(--radius-md);padding:14px 16px;margin-bottom:12px;">
           <p style="font-size:.85rem;color:#fcd34d;">⚡ ${this._esc(immediateAction)}</p>
         </div>`) : '';

    const resourcesHtml = resources.length
      ? resources.map((r, i) => {
          const name = typeof r === 'string' ? r : (r.name || r.title || `Resource ${i+1}`);
          const desc = typeof r === 'object' ? (r.description || r.contact || r.url || '') : '';
          return `
            <div class="resource-card">
              <div class="resource-card-icon">${icons[i % icons.length]}</div>
              <div class="resource-card-body">
                <div class="resource-card-name">${this._esc(name)}</div>
                ${desc ? `<div class="resource-card-desc">${this._esc(desc)}</div>` : ''}
              </div>
            </div>`;
        }).join('')
      : `<div class="resource-card">
           <div class="resource-card-icon">📋</div>
           <div class="resource-card-body">
             <div class="resource-card-name">Monitor &amp; Support</div>
             <div class="resource-card-desc">Continue to monitor the user's wellbeing and offer support.</div>
           </div>
         </div>`;

    const platformHtml = platformAction
      ? `<div class="platform-action-box">
           <strong>Platform Action</strong>
           ${this._esc(platformAction)}
         </div>`
      : '';

    return `
      <div class="collapsible">
        <div class="collapsible-header" onclick="this.closest('.collapsible').classList.toggle('open')">
          <span class="collapsible-icon">💡</span>
          <span class="collapsible-title">Recommendations &amp; Resources</span>
          <svg class="collapsible-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
        </div>
        <div class="collapsible-body">
          <div class="recommendations-list">
            ${immediateHtml}
            ${resourcesHtml}
            ${platformHtml}
          </div>
        </div>
      </div>`;
  }

  /* ----- General utilities ----- */

  _esc(str) {
    const d = document.createElement('div');
    d.textContent = String(str);
    return d.innerHTML;
  }

  _toArray(val) {
    if (!val) return [];
    if (Array.isArray(val)) return val;
    if (typeof val === 'string') return val.split(',').map(s => s.trim()).filter(Boolean);
    return [String(val)];
  }

  _animCounter(elId, from, to, dur) {
    const el = document.getElementById(elId);
    if (!el || from === to) return;
    let start = null;
    const step = (ts) => {
      if (!start) start = ts;
      const p = Math.min((ts - start) / dur, 1);
      const ease = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(from + (to - from) * ease);
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /* ----- Static helpers (used by both UIController and inline script) ----- */

  static riskClass(level) {
    const l = (level || '').toUpperCase().replace(/[_\s]/g, '');
    if (l.includes('HIGH')) return 'high';
    if (l.includes('MOD'))  return 'moderate';
    return 'low';
  }

  static riskLabel(level) {
    const raw = (level || '').toUpperCase();
    const l = raw.replace(/[_\s]/g, '');
    if (raw === 'HIGH_RISK_HARM_TO_OTHERS') return 'HIGH RISK - HARM TO OTHERS';
    if (raw === 'HIGH_RISK_SELF_HARM') return 'HIGH RISK - SELF HARM';
    if (l.includes('HIGH')) return 'HIGH RISK';
    if (l.includes('MOD'))  return 'MODERATE RISK';
    if (l.includes('LOW'))  return 'LOW RISK';
    return level || 'UNKNOWN';
  }

  static riskEmoji(level) {
    const raw = (level || '').toUpperCase();
    const l = raw.replace(/[_\s]/g, '');
    if (raw === 'HIGH_RISK_HARM_TO_OTHERS') return '🛑';
    if (raw === 'HIGH_RISK_SELF_HARM') return '🚨';
    if (l.includes('HIGH')) return '🔴';
    if (l.includes('MOD'))  return '🟡';
    if (l.includes('LOW'))  return '🟢';
    return '⚪';
  }

  static formatRelativeTime(iso) {
    try {
      const diff = Date.now() - new Date(iso).getTime();
      if (diff < 60_000)    return 'just now';
      if (diff < 3_600_000) return Math.floor(diff / 60_000)    + 'm ago';
      if (diff < 86_400_000)return Math.floor(diff / 3_600_000) + 'h ago';
      return new Date(iso).toLocaleDateString();
    } catch(_) { return ''; }
  }
}

/* ============================================================
   CLASS: App - Top-level orchestrator
============================================================= */

class App {
  constructor() {
    this.api      = new ApiClient();
    this.history  = new HistoryManager();
    this.ui       = new UIController();

    this._isLoading     = false;
    this._currentResult = null;
    this._statsTimer    = null;
    this._analyzeStart  = null;

    // Session counters (supplement server stats)
    this._sessionToday    = 0;
    this._sessionHighRisk = 0;
    this._sessionTimes    = [];
  }

  /** Initialize event listeners, restore state, start polling. */
  init() {
    this._bindEvents();
    this._restoreDraft();
    this.ui.renderHistory(this.history.getAll(), id => this._loadHistoryItem(id));
    this._fetchStats();
    this._statsTimer = setInterval(() => this._fetchStats(), SI_CONFIG.STATS_INTERVAL);
  }

  /* ----- Core analyze flow ----- */

  async analyze() {
    const text = this.ui.getInputText();
    if (!text) { this.ui.shakeInput(); return; }
    if (this._isLoading) return;

    const useFallback = this.ui.isFallbackEnabled();
    this._isLoading   = true;
    this._analyzeStart = Date.now();

    this.ui.setLoading(true);
    this.ui.showResultsLoading();

    try {
      const data    = await this.api.analyze(text, useFallback);
      const elapsed = Date.now() - this._analyzeStart;
      const clf     = data.classification || {};

      this._currentResult = data;
      this.ui.renderResults(data, elapsed);

      // Persist to history
      const entry = this.history.add({
        text:       text.slice(0, 120),
        risk_level: clf.risk_level || data.risk_level || 'UNKNOWN',
        risk_score: clf.risk_score || data.risk_score || 0,
        confidence: clf.confidence || data.confidence || 0,
        model_used: data.model_used || '',
        elapsed_ms: elapsed,
        raw:        data,
      });

      this.ui.renderHistory(this.history.getAll(), id => this._loadHistoryItem(id));

      // Update session stats
      this._sessionToday++;
      this._sessionTimes.push(elapsed);
      const rl = ((data.classification && data.classification.risk_level) || data.risk_level || '').toUpperCase();
      if (rl.includes('HIGH')) this._sessionHighRisk++;
      this._updateSessionStats();

      // Clear draft
      try { localStorage.removeItem(SI_CONFIG.DRAFT_KEY); } catch(_) {}

    } catch (err) {
      this.ui.renderError(err.message || 'An unexpected error occurred.');
    } finally {
      this._isLoading = false;
      this.ui.setLoading(false);
    }
  }

  /* ----- History ----- */

  _loadHistoryItem(id) {
    const entry = this.history.getById(id);
    if (!entry) return;
    this.ui.setInputText(entry.text);
    this.ui.updateCharCounter(entry.text.length);
    this.ui.els['post-input']?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    this.ui.els['post-input']?.focus();
  }

  clearHistory() {
    if (!confirm('Clear all analysis history?')) return;
    this.history.clear();
    this.ui.renderHistory([], null);
  }

  /* ----- Stats ----- */

  async _fetchStats() {
    try {
      const data = await this.api.getStats();
      this.ui.updateStats(data);
    } catch(_) { /* non-critical */ }
  }

  _updateSessionStats() {
    const avgTime = this._sessionTimes.length
      ? Math.round(this._sessionTimes.reduce((a, b) => a + b, 0) / this._sessionTimes.length)
      : null;
    this.ui.updateStats({
      total_today: this._sessionToday,
      high_risk:   this._sessionHighRisk,
      avg_time_ms: avgTime,
    });
  }

  /* ----- Event binding ----- */

  _bindEvents() {
    // Textarea: character counter + draft auto-save
    const input = this.ui.els['post-input'];
    if (input) {
      input.addEventListener('input', () => {
        this.ui.updateCharCounter(input.value.length);
        try { localStorage.setItem(SI_CONFIG.DRAFT_KEY, input.value); } catch(_) {}
      });
    }

    // Keyboard shortcut: Ctrl/Cmd+Enter
    document.addEventListener('keydown', e => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        this.analyze();
      }
    });

    // Analyze button (the button in index.html calls analyzeText() inline;
    // this binding supports the external app.js flow as well)
    const btn = document.getElementById('analyze-btn');
    if (btn) {
      // Only add our listener if the onclick attribute is not already bound
      if (!btn.hasAttribute('data-app-bound')) {
        btn.setAttribute('data-app-bound', '1');
        btn.addEventListener('click', () => this.analyze());
      }
    }

    // Window resize: redraw gauge
    window.addEventListener('resize', () => {
      const currentScore = this._currentResult?.classification?.risk_score ?? this._currentResult?.risk_score;
      if (currentScore != null && window.SICharts) {
        try {
          const g = new window.SICharts.RiskGauge('risk-gauge');
          g.draw(currentScore, false);
        } catch(_) {}
      }
    });
  }

  /* ----- Draft ----- */

  _restoreDraft() {
    try {
      const draft = localStorage.getItem(SI_CONFIG.DRAFT_KEY);
      if (draft) {
        this.ui.setInputText(draft);
        this.ui.updateCharCounter(draft.length);
      }
    } catch(_) {}
  }

  /* ----- Feedback (public API) ----- */

  /**
   * Submit analyst feedback for the current result.
   * @param {string} correctLabel  'HIGH_RISK' | 'MODERATE_RISK' | 'LOW_RISK'
   * @param {string} [notes]
   */
  async submitFeedback(correctLabel, notes = '') {
    if (!this._currentResult) return;
    try {
      await this.api.postFeedback({
        text:          this.ui.getInputText(),
        risk_level:    this._currentResult.classification?.risk_level || this._currentResult.risk_level,
        correct_label: correctLabel,
        notes,
      });
    } catch(err) {
      console.warn('Feedback submission failed:', err.message);
    }
  }

  /* ----- Export history ----- */
  exportHistory() {
    const json = this.history.export();
    const blob = new Blob([json], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `si_history_${new Date().toISOString().slice(0,10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
}

/* ============================================================
   BOOTSTRAP
   Auto-initialize when DOM is ready. Expose as window.SIApp
   so the inline index.html script and other scripts can call
   window.SIApp.analyze() etc.
============================================================= */

(function bootstrap() {
  function init() {
    if (window.SIApp) return; // Already initialized by inline script
    const app      = new App();
    window.SIApp   = app;
    window.SIApp.ApiClient     = ApiClient;
    window.SIApp.HistoryManager = HistoryManager;
    window.SIApp.UIController  = UIController;
    app.init();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

/* ============================================================
   NAMED EXPORTS (for testing environments / ES module consumers)
============================================================= */
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ApiClient, ApiError, HistoryManager, UIController, App };
}
