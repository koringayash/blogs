'use strict';

const Book = (() => {
  const $ = (sel) => document.querySelector(sel);
  const esc = (s) => String(s).replace(/[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  /* ---------- Theme (light by default, remembered in localStorage) ---------- */
  function initTheme() {
    const btn = $('#theme-btn');
    if (!btn) return;
    const label = () => {
      btn.textContent = document.documentElement.dataset.theme === 'dark' ? '☀️ Light' : '🌙 Dark';
    };
    label();
    btn.addEventListener('click', () => {
      const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('theme', next); } catch (e) { /* storage blocked: ignore */ }
      label();
      window.dispatchEvent(new Event('themechange')); // widgets and the concept map redraw
    });
  }

  /* ---------- Data ---------- */
  async function loadChapters() {
    const r = await fetch('chapters.json');
    if (!r.ok) throw new Error('chapters.json not found');
    return r.json();
  }

  function showError(el, err) {
    console.error(err);
    el.innerHTML = '<div class="err"><b>Could not load this page.</b><br>' +
      'If you opened the file by double-clicking it, that cannot work: the book loads its chapters ' +
      'with <code>fetch()</code>, which browsers block on <code>file://</code>. ' +
      'Run <code>python3 -m http.server</code> in the project folder and open ' +
      '<code>http://localhost:8000</code>, or use GitHub Pages.</div>';
  }

  /* ---------- Markdown + math pipeline ----------
     1. Pull every $$...$$ and $...$ out of the text (marked would mangle _ and \).
     2. Number display equations in reading order and resolve [[eq-id]] references.
     3. Expand :::box / :::tabs / :::figure / :::widget blocks.
     4. Run marked, then put the KaTeX-rendered math back. */
  function extractMath(src) {
    const stash = { display: [], inline: [] };
    const eqNum = {};
    let n = 0;

    src = src.replace(/\$\$([\s\S]+?)\$\$(?:[ \t]*\{#([\w-]+)\})?/g, (m, tex, id) => {
      n += 1;
      if (id) {
        if (eqNum[id]) console.warn('Duplicate equation id: ' + id);
        eqNum[id] = n;
      }
      stash.display.push({ tex: tex.trim(), id, num: n });
      return '\n\n@@D' + (stash.display.length - 1) + '@@\n\n';
    });

    src = src.replace(/\$([^$\n]+?)\$/g, (m, tex) => {
      stash.inline.push(tex);
      return '@@I' + (stash.inline.length - 1) + '@@';
    });

    src = src.replace(/\[\[([\w-]+)\]\]/g, (m, id) => {
      if (eqNum[id]) return '<a class="eqref" href="#' + id + '">Equation (' + eqNum[id] + ')</a>';
      console.warn('Unknown equation reference: ' + id);
      return '<span class="bad">[unknown equation: ' + esc(id) + ']</span>';
    });

    return { src, stash };
  }

  const renderTex = (tex, display) => {
    try { return katex.renderToString(tex, { displayMode: display, throwOnError: false }); }
    catch (e) { return esc(tex); }
  };

  const equationBox = ({ tex, id, num }) =>
    '<div class="math-box"' + (id ? ' id="' + id + '"' : '') + '><span></span>' +
    '<div class="eq">' + renderTex(tex, true) + '</div>' +
    '<span class="eqn">(' + num + ')</span></div>';

  function restoreMath(html, stash) {
    html = html.replace(/<p>\s*@@D(\d+)@@\s*<\/p>/g, (m, i) => equationBox(stash.display[i]));
    html = html.replace(/@@D(\d+)@@/g, (m, i) => equationBox(stash.display[i]));
    return html.replace(/@@I(\d+)@@/g, (m, i) => renderTex(stash.inline[i], false));
  }

  function expandBlocks(src) {
    // Tabs:  :::tabs Title  /  --- Tab name  /  text ...  /  :::
    src = src.replace(/^:::tabs[ \t]*(.*)\n([\s\S]*?)\n:::[ \t]*$/gm, (m, title, body) => {
      const parts = body.split(/^--- (.+)$/gm); // ['', name1, text1, name2, text2, ...]
      let bar = '', panels = '';
      for (let k = 1; k < parts.length; k += 2) {
        const idx = (k - 1) / 2, on = idx === 0 ? ' on' : '';
        bar += '<button class="tab' + on + '" data-tab="' + idx + '">' + esc(parts[k]) + '</button>';
        panels += '<div class="panel' + on + '" data-panel="' + idx + '">\n\n' + parts[k + 1].trim() + '\n\n</div>';
      }
      return '<div class="tabs"><div class="ct">' + esc(title) + '</div><div class="tabbar">' + bar + '</div>' + panels + '</div>';
    });

    // Boxes:  :::box step | Title  /  markdown ...  /  :::
    src = src.replace(/^:::box[ \t]+(\w+)[ \t]*\|?[ \t]*(.*)\n([\s\S]*?)\n:::[ \t]*$/gm,
      (m, kind, title, body) =>
        '<div class="callout ' + esc(kind) + '"><div class="ct">' + esc(title) + '</div>\n\n' + body + '\n\n</div>');

    // Placeholder figure:  :::figure F1.3 | Title | description
    src = src.replace(/^:::figure (.+)$/gm, (m, line) => {
      const [id, title, desc] = line.split('|').map((s) => s.trim());
      return '<div class="fig todo"><span class="tag">' + esc(id) + ' · DIAGRAM TO BE ADDED</span><br><b>' +
        esc(title || '') + '</b><br>' + esc(desc || '') + '</div>';
    });

    // Live widget:  :::widget name | Title
    src = src.replace(/^:::widget (\w+)(?:[ \t]*\|[ \t]*(.+))?$/gm, (m, name, title) =>
      '<div class="fig" data-widget="' + esc(name) + '"><span class="tag">INTERACTIVE</span> <b>' +
      esc(title || '') + '</b></div>');

    return src;
  }

  function bindTabs(root) {
    root.querySelectorAll('.tabs').forEach((box) => {
      box.querySelectorAll('.tab').forEach((btn) => btn.addEventListener('click', () => {
        box.querySelectorAll('.tab, .panel').forEach((el) => el.classList.remove('on'));
        btn.classList.add('on');
        box.querySelector('.panel[data-panel="' + btn.dataset.tab + '"]').classList.add('on');
      }));
    });
  }

  /* ---------- Reading enhancements (presentation only) ---------- */
  function initProgress() {
    const bar = $('#progress');
    if (!bar) return;
    const upd = () => {
      const h = document.documentElement.scrollHeight - innerHeight;
      bar.style.width = (h > 0 ? Math.min(100, (scrollY / h) * 100) : 0) + '%';
    };
    addEventListener('scroll', upd, { passive: true });
    addEventListener('resize', upd);
    upd();
  }

  const slug = (s) => s.toLowerCase().replace(/<[^>]+>/g, '').replace(/[^\w]+/g, '-').replace(/^-|-$/g, '');

  function enhance(main) {
    // Code blocks: syntax highlight, language label, copy button
    main.querySelectorAll('pre > code').forEach((code) => {
      const pre = code.parentElement;
      const m = (code.className || '').match(/language-(\w+)/);
      if (window.hljs) { try { hljs.highlightElement(code); } catch (e) { /* ignore */ } }
      if (m) { const l = document.createElement('span'); l.className = 'lang'; l.textContent = m[1]; pre.prepend(l); }
      const b = document.createElement('button'); b.className = 'copy'; b.type = 'button'; b.textContent = 'Copy';
      b.onclick = () => navigator.clipboard.writeText(code.innerText).then(() => {
        b.textContent = 'Copied'; setTimeout(() => { b.textContent = 'Copy'; }, 1400);
      });
      pre.append(b);
    });

    // Meta line under the title: reading time
    const h1 = main.querySelector('h1');
    const words = (main.innerText.match(/\S+/g) || []).length;
    if (h1) {
      const meta = document.createElement('div'); meta.className = 'meta';
      meta.textContent = Math.max(1, Math.round(words / 200)) + ' min read';
      const sub = h1.nextElementSibling;
      (sub && sub.matches('p') && sub.children.length === 1 && sub.firstElementChild.tagName === 'EM' ? sub : h1).after(meta);
    }

    // Heading anchors + side table of contents with scroll-spy
    const side = $('aside.toc-side');
    const heads = [...main.querySelectorAll('h2, h3')];
    const seen = {};
    heads.forEach((h) => {
      if (!h.id) { let id = slug(h.textContent) || 'section'; if (seen[id]) id += '-' + (++seen[id]); else seen[id] = 1; h.id = id; }
      const a = document.createElement('a'); a.className = 'anchor'; a.href = '#' + h.id; a.textContent = '#'; a.setAttribute('aria-label', 'Link to section');
      h.append(a);
    });
    if (!side || !heads.length) return;
    side.innerHTML = '<div class="lbl">On this page</div>' + heads.map((h) =>
      '<a href="#' + h.id + '"' + (h.tagName === 'H3' ? ' class="sub"' : '') + '>' +
      esc(h.textContent.replace(/#$/, '')) + '</a>').join('');
    const links = [...side.querySelectorAll('a')];
    const spy = () => {
      let cur = 0;
      heads.forEach((h, k) => { if (h.getBoundingClientRect().top < 120) cur = k; });
      links.forEach((l, k) => l.classList.toggle('on', k === cur));
    };
    addEventListener('scroll', spy, { passive: true });
    spy();
  }

  /* ---------- Pages ---------- */
  async function home() {
    initTheme();
    initProgress();
    const toc = $('#toc'), map = $('#map');
    try {
      const chapters = await loadChapters();

      let part = '', html = '';
      chapters.forEach((c) => {
        if (c.part !== part) { part = c.part; html += '<h3>' + esc(part) + '</h3>'; }
        html += '<a href="chapter.html?c=' + encodeURIComponent(c.id) + '"><b>' + esc(c.title) + '</b>' +
          '<span class="badge ' + esc(c.status) + '">' + esc(c.status) + '</span><br><small>' + esc(c.summary) + '</small></a>';
      });
      toc.innerHTML = html;

      const node = (id) => 'c' + id.slice(0, 2);
      let graph = 'graph LR\n';
      chapters.forEach((c) => {
        graph += ' ' + node(c.id) + '["Ch ' + parseInt(c.id, 10) + ' · ' + c.title.split(/[&,]/)[0].trim() + '"]\n';
        c.connects.forEach((t) => { graph += ' ' + node(c.id) + ' --> ' + node(t) + '\n'; });
        graph += ' click ' + node(c.id) + ' "chapter.html?c=' + c.id + '"\n';
      });

      const draw = async () => {
        const dark = document.documentElement.dataset.theme === 'dark';
        mermaid.initialize({ startOnLoad: false, securityLevel: 'loose', theme: 'base', themeVariables: {
          fontFamily: 'IBM Plex Sans, sans-serif', fontSize: '15px',
          primaryColor: dark ? '#1c1d22' : '#ffffff', primaryTextColor: dark ? '#e9e6df' : '#1d1c1a',
          primaryBorderColor: dark ? '#f08a5d' : '#b4441f', lineColor: dark ? '#9a958b' : '#6f6a62' } });
        map.removeAttribute('data-processed');
        map.textContent = graph;
        await mermaid.run({ nodes: [map] });
      };
      window.addEventListener('themechange', draw);
      await draw();
    } catch (err) { showError(toc, err); }
  }

  async function chapter() {
    initTheme();
    initProgress();
    const main = $('#main');
    try {
      const id = new URLSearchParams(location.search).get('c');
      const chapters = await loadChapters();
      const i = chapters.findIndex((c) => c.id === id);
      if (i < 0) { main.innerHTML = '<p>Unknown chapter. <a href="index.html">Back to contents</a></p>'; return; }
      const ch = chapters[i];
      document.title = ch.title + ' · ML Notes Book';

      const res = await fetch('chapters/' + ch.id + '.md');
      let body;
      if (res.ok) {
        const { src, stash } = extractMath(await res.text());
        body = restoreMath(marked.parse(expandBlocks(src)), stash);
      } else {
        // Outline page for chapters whose full text is not written yet
        body = '<h1>' + esc(ch.title) + '</h1>' +
          '<div class="callout warn"><div class="ct">Draft outline</div>Full text coming soon. ' + esc(ch.summary) + '</div>' +
          '<h2>Prerequisites</h2><ul>' + ch.prereq.map((p) => '<li>' + esc(p) + '</li>').join('') + '</ul>' +
          '<h2>Planned diagrams</h2>' + ch.figs.map((f) => {
            const [fid, title, desc] = f.split('|').map((s) => s.trim());
            return '<div class="fig todo"><span class="tag">' + esc(fid) + ' · DIAGRAM TO BE ADDED</span><br><b>' +
              esc(title) + '</b><br>' + esc(desc) + '</div>';
          }).join('');
      }

      const link = (c) => 'chapter.html?c=' + encodeURIComponent(c.id);
      const leadsTo = ch.connects.map((k) => chapters.find((c) => c.id === k)).filter(Boolean)
        .map((c) => '<a href="' + link(c) + '">' + esc(c.title) + '</a>').join(' · ');
      const prev = chapters[i - 1], next = chapters[i + 1];

      main.innerHTML = body +
        (leadsTo ? '<div class="callout step"><div class="ct">Leads to</div>' + leadsTo + '</div>' : '') +
        '<div class="nav"><span>' + (prev ? '<a href="' + link(prev) + '">← ' + esc(prev.title) + '</a>' : '') +
        '</span><span>' + (next ? '<a href="' + link(next) + '">' + esc(next.title) + ' →</a>' : '') + '</span></div>';

      bindTabs(main);
      enhance(main);
      main.querySelectorAll('[data-widget]').forEach((el) => {
        const w = Widgets[el.dataset.widget];
        if (w) w(el); else console.warn('Missing widget: ' + el.dataset.widget);
      });
      if (location.hash) { const t = document.getElementById(location.hash.slice(1)); if (t) t.scrollIntoView(); }
    } catch (err) { showError(main, err); }
  }

  return { home, chapter };
})();