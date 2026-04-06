/**
 * charts.js — Pure Vanilla JS Canvas Chart Utilities
 * Suicidal Ideation Detection — Social Media Risk Assessment
 *
 * Implements from scratch (no Chart.js):
 *   - RiskGauge     : Arc gauge 0-180°, animated, color-interpolated
 *   - DonutChart    : Three-segment donut, animated, with legend
 *   - BarChart      : Horizontal bar chart for confidence breakdown
 *   - LineChart     : Time-series risk score history
 *
 * All charts:
 *   - Smooth requestAnimationFrame animation
 *   - DPI-aware (devicePixelRatio)
 *   - Responsive (ResizeObserver)
 *   - Dark theme colors
 */

'use strict';

/* ============================================================
   SHARED UTILITIES
============================================================= */

/**
 * Linearly interpolate between two values.
 * @param {number} a
 * @param {number} b
 * @param {number} t  0–1
 */
function lerp(a, b, t) {
  return a + (b - a) * t;
}

/**
 * Ease-out cubic: fast start, slow stop.
 * @param {number} t  0–1
 */
function easeOutCubic(t) {
  return 1 - Math.pow(1 - t, 3);
}

/**
 * Ease-in-out sine.
 */
function easeInOutSine(t) {
  return -(Math.cos(Math.PI * t) - 1) / 2;
}

/**
 * Interpolate between two hex colors.
 * @param {string} hex1   e.g. "#22c55e"
 * @param {string} hex2   e.g. "#ef4444"
 * @param {number} t      0–1
 * @returns {string} interpolated hex
 */
function lerpColor(hex1, hex2, t) {
  const parse = h => [
    parseInt(h.slice(1,3), 16),
    parseInt(h.slice(3,5), 16),
    parseInt(h.slice(5,7), 16),
  ];
  const [r1,g1,b1] = parse(hex1);
  const [r2,g2,b2] = parse(hex2);
  const r = Math.round(lerp(r1, r2, t));
  const g = Math.round(lerp(g1, g2, t));
  const b = Math.round(lerp(b1, b2, t));
  return `rgb(${r},${g},${b})`;
}

/**
 * Three-stop color scale: green(0) → amber(0.5) → red(1).
 * @param {number} t  0–1
 * @returns {string} CSS color
 */
function riskColor(t) {
  const GREEN = '#22c55e';
  const AMBER = '#f59e0b';
  const RED   = '#ef4444';
  if (t <= 0.5) return lerpColor(GREEN, AMBER, t * 2);
  return lerpColor(AMBER, RED, (t - 0.5) * 2);
}

/**
 * Set canvas physical vs CSS size for sharp rendering on HiDPI.
 * @param {HTMLCanvasElement} canvas
 * @param {number} cssW
 * @param {number} cssH
 */
function setupCanvas(canvas, cssW, cssH) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width  = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  canvas.style.width  = cssW + 'px';
  canvas.style.height = cssH + 'px';
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return ctx;
}

/** Cancel a pending animation frame stored on an object. */
function cancelAnim(obj) {
  if (obj._raf) {
    cancelAnimationFrame(obj._raf);
    obj._raf = null;
  }
}

/* ============================================================
   CLASS: RiskGauge
   Arc gauge from 0 to 180°. Animated fill, color gradient.
============================================================= */

class RiskGauge {
  /**
   * @param {string}  canvasId  ID of <canvas> element
   * @param {object}  options
   * @param {number}  options.lineWidth       arc stroke width (default 14)
   * @param {number}  options.animDuration    ms (default 1200)
   * @param {boolean} options.showNeedle      show pointer needle (default true)
   * @param {boolean} options.showLabels      show 0/50/100 labels (default true)
   */
  constructor(canvasId, options = {}) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) throw new Error(`RiskGauge: canvas #${canvasId} not found`);

    this.opts = {
      lineWidth:    options.lineWidth    || 14,
      animDuration: options.animDuration || 1200,
      showNeedle:   options.showNeedle   !== false,
      showLabels:   options.showLabels   !== false,
    };

    this._score   = 0;    // current displayed score
    this._target  = 0;    // target score
    this._raf     = null;

    this._cssW = this.canvas.parentElement
      ? this.canvas.parentElement.clientWidth || 160
      : 160;
    this._cssH = Math.round(this._cssW * 0.56);

    this._ctx = setupCanvas(this.canvas, this._cssW, this._cssH);

    // Responsive redraw
    this._resizeObserver = new ResizeObserver(() => this._onResize());
    this._resizeObserver.observe(this.canvas.parentElement || document.body);

    this._drawStatic(0);
  }

  /**
   * Animate gauge from current value to `score`.
   * @param {number}  score      0–100
   * @param {boolean} animated   default true
   */
  draw(score, animated = true) {
    this._target = Math.max(0, Math.min(100, score));
    if (!animated) {
      this._score = this._target;
      this._drawStatic(this._score);
      return;
    }
    cancelAnim(this);
    const start     = this._score;
    const startTime = performance.now();
    const duration  = this.opts.animDuration;

    const tick = (now) => {
      const t = Math.min((now - startTime) / duration, 1);
      this._score = lerp(start, this._target, easeOutCubic(t));
      this._drawStatic(this._score);
      if (t < 1) this._raf = requestAnimationFrame(tick);
      else        this._raf = null;
    };
    this._raf = requestAnimationFrame(tick);
  }

  /**
   * Update to new score (alias for draw with animation).
   * @param {number} score  0–100
   */
  update(score) {
    this.draw(score, true);
  }

  /** Destroy — clean up ResizeObserver and pending animation. */
  destroy() {
    cancelAnim(this);
    if (this._resizeObserver) this._resizeObserver.disconnect();
  }

  /* ----- private ----- */

  _onResize() {
    if (!this.canvas.parentElement) return;
    const w = this.canvas.parentElement.clientWidth;
    if (!w || Math.abs(w - this._cssW) < 2) return;
    this._cssW = w;
    this._cssH = Math.round(w * 0.56);
    this._ctx  = setupCanvas(this.canvas, this._cssW, this._cssH);
    this._drawStatic(this._score);
  }

  _drawStatic(score) {
    const ctx    = this._ctx;
    const W      = this._cssW;
    const H      = this._cssH;
    const cx     = W / 2;
    const cy     = H - 14;
    const radius = Math.min(W * 0.44, H * 0.82);
    const lw     = this.opts.lineWidth;
    const t      = score / 100;

    ctx.clearRect(0, 0, W, H);

    // --- Track ---
    ctx.beginPath();
    ctx.arc(cx, cy, radius, Math.PI, 2 * Math.PI);
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.lineWidth   = lw;
    ctx.lineCap     = 'round';
    ctx.stroke();

    // --- Colored arc fill ---
    if (score > 0) {
      // Draw as gradient by segmenting the arc
      const segments = 60;
      const totalArc = Math.PI * t;
      for (let i = 0; i < segments; i++) {
        const pct  = i / segments;
        const pct2 = (i + 1) / segments;
        if (pct2 * Math.PI > totalArc + 0.001) break;

        const a1 = Math.PI + pct  * totalArc;
        const a2 = Math.PI + pct2 * totalArc;
        const mid = (a1 + a2) / 2;
        // color based on position along track, not fill progress
        const colorT = (mid - Math.PI) / Math.PI;

        ctx.beginPath();
        ctx.arc(cx, cy, radius, a1, a2);
        ctx.strokeStyle = riskColor(colorT);
        ctx.lineWidth   = lw;
        ctx.lineCap     = 'butt';
        ctx.stroke();
      }
    }

    // --- Glow on tip ---
    if (score > 2) {
      const tipA = Math.PI + t * Math.PI;
      const tx   = cx + radius * Math.cos(tipA);
      const ty   = cy + radius * Math.sin(tipA);
      const grd  = ctx.createRadialGradient(tx, ty, 0, tx, ty, lw * 1.5);
      grd.addColorStop(0,   riskColor(t) + 'cc');
      grd.addColorStop(1,   riskColor(t) + '00');
      ctx.beginPath();
      ctx.arc(tx, ty, lw * 1.5, 0, 2 * Math.PI);
      ctx.fillStyle = grd;
      ctx.fill();
    }

    // --- Needle ---
    if (this.opts.showNeedle) {
      const needleA = Math.PI + t * Math.PI;
      const nx = cx + radius * Math.cos(needleA);
      const ny = cy + radius * Math.sin(needleA);

      // Hub
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, 2 * Math.PI);
      ctx.fillStyle = '#f1f5f9';
      ctx.fill();

      // Stick
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(nx, ny);
      ctx.strokeStyle = '#f1f5f9';
      ctx.lineWidth   = 2.5;
      ctx.lineCap     = 'round';
      ctx.stroke();

      // Inner hub dot
      ctx.beginPath();
      ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
      ctx.fillStyle = '#0f172a';
      ctx.fill();
    }

    // --- Labels ---
    if (this.opts.showLabels) {
      ctx.font         = `bold ${Math.max(9, W * 0.058)}px system-ui,sans-serif`;
      ctx.fillStyle    = 'rgba(255,255,255,0.35)';
      ctx.textBaseline = 'middle';

      ctx.textAlign = 'right';
      ctx.fillText('0',   cx - radius - 4, cy);

      ctx.textAlign = 'left';
      ctx.fillText('100', cx + radius + 4, cy);

      ctx.textAlign = 'center';
      ctx.fillText('50',  cx, cy - radius - 6);
    }
  }
}

/* ============================================================
   CLASS: DonutChart
   Three-segment donut for low/moderate/high distribution.
============================================================= */

class DonutChart {
  /**
   * @param {string} canvasId
   * @param {object} data   { low: number, moderate: number, high: number }
   * @param {object} options
   * @param {number} options.size          canvas CSS size (default 180)
   * @param {number} options.holeRatio     inner radius ratio (default 0.58)
   * @param {number} options.animDuration  ms (default 900)
   * @param {number} options.gap           gap between segments in radians (default 0.04)
   */
  constructor(canvasId, data = {}, options = {}) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) throw new Error(`DonutChart: canvas #${canvasId} not found`);

    this.opts = {
      size:         options.size         || 180,
      holeRatio:    options.holeRatio    || 0.58,
      animDuration: options.animDuration || 900,
      gap:          options.gap          || 0.04,
    };

    this._data   = this._normalizeData(data);
    this._raf    = null;
    this._prog   = 0;   // animation progress 0–1

    const s  = this.opts.size;
    this._ctx = setupCanvas(this.canvas, s, s);

    this._resizeObserver = new ResizeObserver(() => this._onResize());
    if (this.canvas.parentElement) {
      this._resizeObserver.observe(this.canvas.parentElement);
    }
  }

  /** Re-draw with new data (animated). */
  update(data) {
    this._data = this._normalizeData(data);
    this.draw();
  }

  /** Animate from empty to full. */
  draw() {
    cancelAnim(this);
    const startTime = performance.now();
    const duration  = this.opts.animDuration;

    const tick = (now) => {
      this._prog = Math.min((now - startTime) / duration, 1);
      this._render(easeInOutSine(this._prog));
      if (this._prog < 1) this._raf = requestAnimationFrame(tick);
      else                 this._raf = null;
    };
    this._raf = requestAnimationFrame(tick);
  }

  destroy() {
    cancelAnim(this);
    if (this._resizeObserver) this._resizeObserver.disconnect();
  }

  /* ----- private ----- */

  _normalizeData(raw) {
    const segments = [
      { key: 'low',      label: 'Low Risk',      color: '#22c55e', pct: raw.low      || 0 },
      { key: 'moderate', label: 'Moderate Risk',  color: '#f59e0b', pct: raw.moderate || 0 },
      { key: 'high',     label: 'High Risk',      color: '#ef4444', pct: raw.high     || 0 },
    ];
    const total = segments.reduce((s, x) => s + x.pct, 0) || 1;
    segments.forEach(s => s.pct = s.pct / total);
    return segments;
  }

  _onResize() {
    const parent = this.canvas.parentElement;
    if (!parent) return;
    const w = parent.clientWidth;
    if (!w) return;
    const s = Math.min(w, this.opts.size);
    this._ctx = setupCanvas(this.canvas, s, s);
    this._render(easeInOutSine(this._prog));
  }

  _render(progress) {
    const ctx  = this._ctx;
    const w    = this.canvas.clientWidth  || this.opts.size;
    const h    = this.canvas.clientHeight || this.opts.size;
    const cx   = w / 2;
    const cy   = h / 2;
    const R    = Math.min(cx, cy) * 0.92;
    const r    = R * this.opts.holeRatio;
    const gap  = this.opts.gap;

    ctx.clearRect(0, 0, w, h);

    const totalArc = 2 * Math.PI * progress;
    let   angle    = -Math.PI / 2;

    // Background circle
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(255,255,255,0.03)';
    ctx.fill();

    for (const seg of this._data) {
      if (seg.pct <= 0) continue;
      const arc = seg.pct * totalArc;
      if (arc < 0.002) { angle += arc; continue; }

      const a1 = angle + gap / 2;
      const a2 = angle + arc - gap / 2;

      ctx.beginPath();
      ctx.moveTo(cx + r * Math.cos(a1), cy + r * Math.sin(a1));
      ctx.arc(cx, cy, R, a1, a2);
      ctx.arc(cx, cy, r, a2, a1, true);
      ctx.closePath();
      ctx.fillStyle = seg.color;
      ctx.fill();

      // Subtle inner shadow
      ctx.strokeStyle = 'rgba(0,0,0,0.3)';
      ctx.lineWidth   = 1;
      ctx.stroke();

      angle += arc;
    }

    // Center hole overlay
    ctx.beginPath();
    ctx.arc(cx, cy, r * 0.98, 0, 2 * Math.PI);
    ctx.fillStyle = '#1e293b';
    ctx.fill();
  }
}

/* ============================================================
   CLASS: BarChart
   Horizontal bar chart for confidence/score breakdown.
============================================================= */

class BarChart {
  /**
   * @param {string} canvasId
   * @param {Array}  data     Array of { label, value (0–1), color? }
   * @param {object} options
   * @param {number} options.barHeight     px (default 22)
   * @param {number} options.gap           px between bars (default 14)
   * @param {number} options.paddingX      horizontal padding (default 16)
   * @param {number} options.paddingY      vertical padding   (default 14)
   * @param {number} options.animDuration  ms (default 800)
   * @param {string} options.defaultColor  fallback bar color
   */
  constructor(canvasId, data = [], options = {}) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) throw new Error(`BarChart: canvas #${canvasId} not found`);

    this.opts = {
      barHeight:    options.barHeight    || 22,
      gap:          options.gap          || 16,
      paddingX:     options.paddingX     || 16,
      paddingY:     options.paddingY     || 14,
      animDuration: options.animDuration || 800,
      defaultColor: options.defaultColor || '#6366f1',
      labelWidth:   options.labelWidth   || 120,
    };

    this._data  = data;
    this._raf   = null;
    this._cssW  = 0;
    this._cssH  = 0;

    this._init();
  }

  /** Re-render with updated data. */
  update(data) {
    this._data = data;
    this._init();
  }

  draw() {
    this._init();
  }

  destroy() {
    cancelAnim(this);
    if (this._resizeObserver) this._resizeObserver.disconnect();
  }

  /* ----- private ----- */

  _init() {
    if (!this.canvas.parentElement) return;
    const pW = this.canvas.parentElement.clientWidth || 300;
    const { barHeight, gap, paddingY } = this.opts;
    const totalH = paddingY * 2 + this._data.length * (barHeight + gap) - gap;

    this._cssW = pW;
    this._cssH = Math.max(totalH, 60);
    this._ctx  = setupCanvas(this.canvas, this._cssW, this._cssH);
    this._animate();
  }

  _animate() {
    cancelAnim(this);
    const startTime = performance.now();
    const dur       = this.opts.animDuration;

    const tick = (now) => {
      const t = Math.min((now - startTime) / dur, 1);
      this._render(easeOutCubic(t));
      if (t < 1) this._raf = requestAnimationFrame(tick);
      else        this._raf = null;
    };
    this._raf = requestAnimationFrame(tick);
  }

  _render(progress) {
    const ctx = this._ctx;
    const W   = this._cssW;
    const H   = this._cssH;
    const { barHeight, gap, paddingX, paddingY, labelWidth, defaultColor } = this.opts;

    ctx.clearRect(0, 0, W, H);

    const trackX  = paddingX + labelWidth;
    const trackW  = W - trackX - paddingX;

    this._data.forEach((item, i) => {
      const y    = paddingY + i * (barHeight + gap);
      const val  = Math.max(0, Math.min(1, item.value || 0));
      const fillW = trackW * val * progress;
      const color = item.color || defaultColor;

      // Label
      ctx.font         = '12px system-ui,sans-serif';
      ctx.fillStyle    = 'rgba(148,163,184,0.9)';
      ctx.textAlign    = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(
        item.label.length > 16 ? item.label.slice(0, 15) + '…' : item.label,
        trackX - 10,
        y + barHeight / 2
      );

      // Track
      ctx.beginPath();
      ctx.roundRect(trackX, y, trackW, barHeight, barHeight / 2);
      ctx.fillStyle = 'rgba(255,255,255,0.05)';
      ctx.fill();

      // Fill
      if (fillW > 0) {
        ctx.beginPath();
        ctx.roundRect(trackX, y, fillW, barHeight, barHeight / 2);
        ctx.fillStyle = color;
        ctx.fill();
      }

      // Value label
      ctx.font         = 'bold 11px system-ui,sans-serif';
      ctx.fillStyle    = 'rgba(241,245,249,0.8)';
      ctx.textAlign    = 'left';
      ctx.fillText(
        Math.round(val * 100) + '%',
        trackX + fillW + 6,
        y + barHeight / 2
      );
    });
  }
}

/* ============================================================
   CLASS: LineChart
   Time-series risk score history with smooth curve.
============================================================= */

class LineChart {
  /**
   * @param {string} canvasId
   * @param {Array}  data      Array of { timestamp: ISO string | Date, score: 0–100 }
   * @param {object} options
   * @param {number} options.paddingX      horizontal padding (default 48)
   * @param {number} options.paddingY      vertical padding   (default 24)
   * @param {number} options.animDuration  ms (default 1000)
   * @param {number} options.lineWidth     (default 2.5)
   * @param {boolean} options.fillArea     gradient fill under curve (default true)
   * @param {boolean} options.showDots     dots at data points (default true)
   * @param {boolean} options.showGrid     horizontal grid lines (default true)
   * @param {number}  options.maxPoints    rolling window (default 50)
   */
  constructor(canvasId, data = [], options = {}) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) throw new Error(`LineChart: canvas #${canvasId} not found`);

    this.opts = {
      paddingX:     options.paddingX     || 48,
      paddingY:     options.paddingY     || 28,
      animDuration: options.animDuration || 1000,
      lineWidth:    options.lineWidth    || 2.5,
      fillArea:     options.fillArea     !== false,
      showDots:     options.showDots     !== false,
      showGrid:     options.showGrid     !== false,
      maxPoints:    options.maxPoints    || 50,
    };

    this._data  = [...data].slice(-this.opts.maxPoints);
    this._raf   = null;
    this._cssW  = 0;
    this._cssH  = 0;

    this._resizeObserver = new ResizeObserver(() => this._onResize());
    if (this.canvas.parentElement) {
      this._resizeObserver.observe(this.canvas.parentElement);
    }

    this._init();
  }

  /** Full re-render with new dataset. */
  draw() {
    this._init();
  }

  /** Add a single new data point and re-render. */
  addPoint(timestamp, score) {
    this._data.push({ timestamp, score });
    if (this._data.length > this.opts.maxPoints) {
      this._data.shift();
    }
    this._init();
  }

  /** Update entire dataset. */
  update(data) {
    this._data = [...data].slice(-this.opts.maxPoints);
    this._init();
  }

  destroy() {
    cancelAnim(this);
    if (this._resizeObserver) this._resizeObserver.disconnect();
  }

  /* ----- private ----- */

  _onResize() {
    this._init();
  }

  _init() {
    if (!this.canvas.parentElement) return;
    const pW = this.canvas.parentElement.clientWidth  || 400;
    const pH = this.canvas.parentElement.clientHeight || 220;

    this._cssW = pW;
    this._cssH = pH;
    this._ctx  = setupCanvas(this.canvas, pW, pH);
    this._animate();
  }

  _animate() {
    cancelAnim(this);
    const startTime = performance.now();
    const dur       = this.opts.animDuration;

    const tick = (now) => {
      const t = Math.min((now - startTime) / dur, 1);
      this._render(easeInOutSine(t));
      if (t < 1) this._raf = requestAnimationFrame(tick);
      else        this._raf = null;
    };
    this._raf = requestAnimationFrame(tick);
  }

  _render(progress) {
    const ctx = this._ctx;
    const W   = this._cssW;
    const H   = this._cssH;
    const { paddingX, paddingY, lineWidth, fillArea, showDots, showGrid } = this.opts;

    ctx.clearRect(0, 0, W, H);

    const data = this._data;
    if (!data.length) {
      ctx.font      = '13px system-ui,sans-serif';
      ctx.fillStyle = 'rgba(71,85,105,0.7)';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('No data available', W / 2, H / 2);
      return;
    }

    const plotW = W - paddingX * 2;
    const plotH = H - paddingY * 2;

    // Map data points to pixel coordinates
    const pts = data.map((d, i) => ({
      x: paddingX + (i / Math.max(data.length - 1, 1)) * plotW,
      y: paddingY + (1 - d.score / 100) * plotH,
      score: d.score,
      timestamp: d.timestamp,
    }));

    // --- Grid lines ---
    if (showGrid) {
      const gridLevels = [0, 25, 50, 75, 100];
      gridLevels.forEach(lvl => {
        const y = paddingY + (1 - lvl / 100) * plotH;
        ctx.beginPath();
        ctx.moveTo(paddingX, y);
        ctx.lineTo(W - paddingX, y);
        ctx.strokeStyle = 'rgba(255,255,255,0.05)';
        ctx.lineWidth   = 1;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Y-axis label
        ctx.font         = '10px system-ui,sans-serif';
        ctx.fillStyle    = 'rgba(71,85,105,0.8)';
        ctx.textAlign    = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(lvl, paddingX - 6, y);
      });
    }

    // Danger zone (score > 60) subtle fill
    const dangerY = paddingY + (1 - 60 / 100) * plotH;
    ctx.fillStyle = 'rgba(239,68,68,0.04)';
    ctx.fillRect(paddingX, paddingY, plotW, dangerY - paddingY);

    // --- Clip to animate from left ---
    ctx.save();
    ctx.beginPath();
    ctx.rect(paddingX, 0, plotW * progress, H);
    ctx.clip();

    // --- Gradient fill under curve ---
    if (fillArea && pts.length > 1) {
      const grad = ctx.createLinearGradient(0, paddingY, 0, H - paddingY);
      grad.addColorStop(0,   'rgba(99,102,241,0.25)');
      grad.addColorStop(0.6, 'rgba(99,102,241,0.08)');
      grad.addColorStop(1,   'rgba(99,102,241,0)');

      ctx.beginPath();
      ctx.moveTo(pts[0].x, H - paddingY);
      ctx.lineTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) {
        const cpx = (pts[i-1].x + pts[i].x) / 2;
        ctx.bezierCurveTo(cpx, pts[i-1].y, cpx, pts[i].y, pts[i].x, pts[i].y);
      }
      ctx.lineTo(pts[pts.length-1].x, H - paddingY);
      ctx.closePath();
      ctx.fillStyle = grad;
      ctx.fill();
    }

    // --- Smooth line ---
    if (pts.length > 1) {
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i++) {
        const cpx = (pts[i-1].x + pts[i].x) / 2;
        ctx.bezierCurveTo(cpx, pts[i-1].y, cpx, pts[i].y, pts[i].x, pts[i].y);
      }
      ctx.strokeStyle = '#6366f1';
      ctx.lineWidth   = lineWidth;
      ctx.lineCap     = 'round';
      ctx.lineJoin    = 'round';
      ctx.stroke();
    } else if (pts.length === 1) {
      ctx.beginPath();
      ctx.arc(pts[0].x, pts[0].y, 4, 0, 2 * Math.PI);
      ctx.fillStyle = '#6366f1';
      ctx.fill();
    }

    // --- Data point dots with risk colors ---
    if (showDots) {
      pts.forEach(pt => {
        const color = riskColor(pt.score / 100);
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 4.5, 0, 2 * Math.PI);
        ctx.fillStyle   = color;
        ctx.strokeStyle = '#0f172a';
        ctx.lineWidth   = 2;
        ctx.fill();
        ctx.stroke();
      });
    }

    ctx.restore();

    // --- X-axis timestamps ---
    if (data.length > 1) {
      const step = Math.max(1, Math.floor(data.length / 5));
      ctx.font         = '10px system-ui,sans-serif';
      ctx.fillStyle    = 'rgba(71,85,105,0.7)';
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'top';

      pts.forEach((pt, i) => {
        if (i % step !== 0 && i !== pts.length - 1) return;
        const label = formatAxisTime(data[i].timestamp);
        ctx.fillText(label, pt.x, H - paddingY + 5);
      });
    }
  }
}

/* ============================================================
   INTERNAL HELPERS
============================================================= */

/** Format a timestamp for axis label. */
function formatAxisTime(ts) {
  try {
    const d = ts instanceof Date ? ts : new Date(ts);
    if (isNaN(d.getTime())) return String(ts).slice(0, 5);
    const now  = Date.now();
    const diff = now - d.getTime();
    if (diff < 3600000) return Math.round(diff / 60000) + 'm';
    if (diff < 86400000) return d.getHours() + ':' + String(d.getMinutes()).padStart(2,'0');
    return (d.getMonth()+1) + '/' + d.getDate();
  } catch(_) {
    return '';
  }
}

/* ============================================================
   EXPORTS
   Compatible with both ES modules and classic <script> tags.
============================================================= */
(function exportCharts() {
  const exports = { RiskGauge, DonutChart, BarChart, LineChart };

  // ES module environment
  if (typeof window !== 'undefined') {
    window.SICharts = exports;
  }

  // CommonJS / Node (for testing)
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exports;
  }
})();
