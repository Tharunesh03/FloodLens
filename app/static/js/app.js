/* FloodLens v2 — main JS (theme, nav, charts, API demo) */
(function(){
  const root = document.documentElement;

  // ----- Theme toggle -----
  const themeBtn = document.getElementById('themeToggle');
  const iconSun = document.getElementById('iconSun');
  const iconMoon = document.getElementById('iconMoon');
  function themeColors(){
    const cs = getComputedStyle(document.body);
    return {
      bg: cs.backgroundColor,
      surface: document.querySelector('.card, .kpi, .chart-card') ? getComputedStyle(document.querySelector('.card')).backgroundColor : '#ffffff',
      fg: cs.color,
      border: cs.borderColor,
      muted: (function(){
        var el = document.createElement('span'); el.style.color='var(--muted)'; document.body.appendChild(el);
        var c = getComputedStyle(el).color; el.remove(); return c;
      })(),
      grid: (function(){
        var el = document.createElement('span'); el.style.color='var(--border)'; document.body.appendChild(el);
        var c = getComputedStyle(el).color; el.remove(); return c;
      })(),
    };
  }
  function applyTheme(t){
    root.setAttribute('data-theme', t);
    localStorage.setItem('floodlens-theme', t);
    if (iconSun && iconMoon){
      iconSun.style.display = t === 'dark' ? 'block' : 'none';
      iconMoon.style.display = t === 'dark' ? 'none' : 'block';
    }
    // Animate plots
    if (window.Plotly && window.__CHARTS__){
      var c = themeColors();
      Object.keys(window.__CHARTS__).forEach(id => {
        var el = document.getElementById('chart-' + id);
        if (!el) return;
        try {
          Plotly.relayout(el, {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: {color: c.fg},
            xaxis: {gridcolor: c.grid, linecolor: c.grid, zerolinecolor: c.grid},
            yaxis: {gridcolor: c.grid, linecolor: c.grid, zerolinecolor: c.grid},
            legend: {font: {color: c.fg}},
          });
        } catch(e){}
      });
    }
  }
  applyTheme(root.getAttribute('data-theme') || 'light');
  if (themeBtn){
    themeBtn.addEventListener('click', () => {
      const cur = root.getAttribute('data-theme');
      applyTheme(cur === 'dark' ? 'light' : 'dark');
    });
  }

  // ----- Mobile nav -----
  const toggle = document.getElementById('navToggle');
  const nav = document.getElementById('siteNav');
  if (toggle && nav){
    toggle.addEventListener('click', () => {
      const open = nav.classList.toggle('open');
      toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    const path = window.location.pathname;
    nav.querySelectorAll('a[data-route]').forEach(a => {
      const route = a.getAttribute('data-route');
      const match =
        (route === 'home' && path === '/') ||
        (route !== 'home' && path.startsWith('/' + route)) ||
        (route === 'dataset' && path.startsWith('/dataset'));
      if (match) a.classList.add('active');
    });
  }

  // ----- Flash auto-hide -----
  document.querySelectorAll('.flash').forEach(el => {
    setTimeout(() => { el.style.transition = 'opacity .5s, transform .5s'; el.style.opacity = '0'; el.style.transform='translateY(-6px)'; }, 6000);
    setTimeout(() => el.remove(), 6600);
  });

  // ----- Plotly charts -----
  function renderCharts(){
    if (!window.Plotly || !window.__CHARTS__) return;
    var c = themeColors();
    Object.entries(window.__CHARTS__).forEach(([id, chart]) => {
      if (!chart) return;
      const el = document.getElementById('chart-' + id);
      if (!el) return;
      try{
        const spec = JSON.parse(chart);
        const layout = Object.assign({}, spec.layout, {
          paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
          font:{color: c.fg, family: 'Inter, -apple-system, Segoe UI, Roboto, sans-serif', size: 12},
          margin: {l:50, r:20, t:50, b:50},
          xaxis: Object.assign({gridcolor: c.grid, linecolor: c.grid, zerolinecolor: c.grid}, spec.layout.xaxis || {}),
          yaxis: Object.assign({gridcolor: c.grid, linecolor: c.grid, zerolinecolor: c.grid}, spec.layout.yaxis || {}),
          legend: Object.assign({font: {color: c.fg}}, spec.layout.legend || {}),
        });
        Plotly.newPlot(el, spec.data, layout, {responsive:true, displaylogo:false, modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d']});
      }catch(e){
        console.error('Chart render failed for', id, e);
        const fb = el.parentNode.querySelector('.chart-fallback');
        if (fb) fb.style.display = 'block';
      }
    });
  }
  if (document.querySelectorAll('[id^="chart-"]').length){
    function loadPlotly(cb){
      const s = document.createElement('script');
      s.src = 'https://cdn.plot.ly/plotly-2.35.2.min.js';
      s.onload = cb;
      s.onerror = () => {
        // Fallback without integrity check if CDN fails with hash
        const s2 = document.createElement('script');
        s2.src = 'https://cdn.plot.ly/plotly-2.35.2.min.js';
        s2.onload = cb;
        s2.onerror = () => {
          document.querySelectorAll('.chart-fallback').forEach(el => el.style.display='block');
        };
        document.head.appendChild(s2);
      };
      document.head.appendChild(s);
    }
    loadPlotly(renderCharts);
  }

  // ----- Form submit: lock button briefly; re-enable after navigation or timeout -----
  const form = document.getElementById('predictForm');
  const btn = document.getElementById('btnSubmit');
  if (form && btn){
    form.addEventListener('submit', () => {
      btn.disabled = true;
      const origHTML = btn.innerHTML;
      btn.innerHTML = '<span class="spinner"></span> Estimating…';
      setTimeout(() => { btn.disabled = false; btn.innerHTML = origHTML; }, 12000);
    });
  }

  // ----- API demo -----
  function wire(id, outId, fn){
    const b = document.getElementById(id); if (!b) return;
    b.addEventListener('click', async () => {
      const out = document.getElementById(outId);
      out.textContent = 'Calling…';
      try{ fn(await fetch); } catch(e){ out.textContent = 'Error: ' + e; }
    });
  }
  wire('btnHealth','healthOut', async function(fetch){
    const r = await fetch('/api/health');
    document.getElementById('healthOut').textContent = JSON.stringify(await r.json(), null, 2);
  });
  wire('btnPredict','predictOut', async function(fetch){
    const payload = {
      Rainfall_mm: 120, Temperature_C: 28, Humidity_pct: 88,
      River_Discharge_m3_s: 3500, Water_Level_m: 6.5, Elevation_m: 80,
      Population_Density: 800, Land_Cover: "Agricultural", Soil_Type: "Clay",
      Infrastructure: 0, Historical_Floods: 1
    };
    const r = await fetch('/api/predict', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    document.getElementById('predictOut').textContent = JSON.stringify(await r.json(), null, 2);
  });

  // ----- Number animations for KPI values -----
  document.querySelectorAll('.kpi-value[data-target]').forEach(el => {
    const target = parseFloat(el.getAttribute('data-target'));
    const suffix = el.getAttribute('data-suffix') || '';
    const isInt = el.getAttribute('data-int') === '1';
    const dur = 900; const start = performance.now();
    function step(now){
      const t = Math.min(1, (now-start)/dur);
      const v = target * (1 - Math.pow(1-t, 3));
      el.textContent = (isInt ? Math.round(v).toLocaleString() : v.toFixed(2)) + suffix;
      if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  });
})();
