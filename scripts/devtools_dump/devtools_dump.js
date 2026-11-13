/* ============================================================================
 * devtools_dump.js  -  colar no Console do DevTools da página.
 *
 * O que faz:
 *  1. Instala ganchos AGORA: console.*, erros, fetch, XHR, WebSocket, EventSource,
 *     postMessage. Tudo que acontecer depois de colar fica gravado com corpo.
 *  2. Ao rodar  await __dump.save()  coleta:
 *     - HTML original (refetch) + DOM vivo
 *     - todos os recursos vistos pela aba Network (performance entries) com timing
 *     - conteúdo de todo JS, CSS, JSON, HTML e source maps (//# sourceMappingURL)
 *     - scripts e estilos inline
 *     - cookies, localStorage, sessionStorage, nomes de IndexedDB, service workers
 *     - globais que a página criou em window (config, web3, wallet, store...)
 *     - log do console e do tráfego capturados desde que o script foi colado
 *  3. Baixa um único arquivo JSON:  dump_<host>_<data>.json
 *
 * Uso:
 *   - DevTools > Console > colar tudo > Enter
 *   - navegue/interaja na página por um tempo (o tráfego ao vivo é gravado)
 *   - await __dump.save()            // baixa o JSON
 *   - await __dump.save({binary:true})  // inclui imagens/fontes em base64 (pesado)
 *   - __dump.stats()                 // contagem do que já foi capturado
 *
 * Limites (do navegador, não do script):
 *   - console e respostas de rede ANTERIORES ao paste não são recuperáveis.
 *     Para isso: Network > "Preserve log" + recarregar > botão direito >
 *     "Save all as HAR with content". Use os dois juntos.
 *   - recursos de outro domínio sem CORS não podem ser relidos (fica o erro).
 *   - WebSockets abertos antes do paste: só as mensagens ENVIADAS são vistas.
 * ==========================================================================*/
(() => {
  if (window.__dump) { console.warn('[dump] já instalado. use await __dump.save()'); return; }

  const T0 = Date.now();
  const now = () => new Date().toISOString();
  const MAX_BODY = 5 * 1024 * 1024; // corta corpos maiores que 5 MB
  const clip = (s) => (typeof s === 'string' && s.length > MAX_BODY) ? s.slice(0, MAX_BODY) + '\n/*[dump: cortado]*/' : s;
  const tag = (x) => Object.prototype.toString.call(x);

  const state = { console: [], errors: [], http: [], ws: [], sse: [], messages: [] };

  // --- serialização segura ---------------------------------------------------
  const ser = (v) => {
    try {
      if (v === null || typeof v !== 'object') {
        if (typeof v === 'function') return `[Function ${v.name || 'anon'}]`;
        if (typeof v === 'bigint') return v.toString() + 'n';
        if (typeof v === 'symbol') return v.toString();
        return v;
      }
      if (v instanceof Error) return { error: v.name, message: v.message, stack: v.stack };
      if (v instanceof Node) return `[${v.nodeName}] ${(v.outerHTML || v.textContent || '').slice(0, 500)}`;
      if (v === window) return '[Window]';
      const seen = new WeakSet();
      return JSON.parse(JSON.stringify(v, (k, x) => {
        if (typeof x === 'bigint') return x.toString() + 'n';
        if (typeof x === 'function') return `[Function ${x.name || 'anon'}]`;
        if (x && typeof x === 'object') {
          if (x === window) return '[Window]';
          if (x instanceof Node) return `[${x.nodeName}]`;
          if (seen.has(x)) return '[Circular]';
          seen.add(x);
        }
        return x;
      }));
    } catch (e) { try { return String(v); } catch { return '[unserializable]'; } }
  };

  // --- console -----------------------------------------------------------------
  ['log', 'info', 'warn', 'error', 'debug', 'trace', 'table', 'dir', 'group', 'groupCollapsed'].forEach((lvl) => {
    const orig = console[lvl];
    if (typeof orig !== 'function') return;
    console[lvl] = function (...args) {
      try { state.console.push({ t: now(), level: lvl, args: args.map(ser) }); } catch {}
      return orig.apply(this, args);
    };
  });
  window.addEventListener('error', (e) => state.errors.push({
    t: now(), type: 'error', message: e.message, src: e.filename, line: e.lineno, col: e.colno,
    stack: e.error && e.error.stack,
    target: e.target && e.target !== window ? (e.target.src || e.target.href || e.target.nodeName) : undefined,
  }), true);
  window.addEventListener('unhandledrejection', (e) => state.errors.push({ t: now(), type: 'unhandledrejection', reason: ser(e.reason) }));

  // --- fetch -------------------------------------------------------------------
  const origFetch = window.fetch.bind(window);
  window.fetch = async function (input, init = {}) {
    const rec = {
      t: now(), kind: 'fetch',
      method: String(init.method || (input && input.method) || 'GET').toUpperCase(),
      url: String(input && input.url ? input.url : input), reqHeaders: {}, reqBody: null,
    };
    try {
      new Headers(init.headers || (input && input.headers) || {}).forEach((v, k) => (rec.reqHeaders[k] = v));
      if (init.body != null) rec.reqBody = typeof init.body === 'string' ? clip(init.body) : `[${tag(init.body)}]`;
    } catch {}
    state.http.push(rec);
    const t1 = performance.now();
    try {
      const res = await origFetch(input, init);
      rec.status = res.status; rec.ms = Math.round(performance.now() - t1); rec.resHeaders = {};
      res.headers.forEach((v, k) => (rec.resHeaders[k] = v));
      res.clone().text().then((b) => (rec.resBody = clip(b))).catch((e) => (rec.resBody = `[erro lendo corpo: ${e}]`));
      return res;
    } catch (e) { rec.error = String(e); throw e; }
  };

  // --- XHR -----------------------------------------------------------------------
  const XO = XMLHttpRequest.prototype.open, XS = XMLHttpRequest.prototype.send, XH = XMLHttpRequest.prototype.setRequestHeader;
  XMLHttpRequest.prototype.open = function (m, u, ...r) {
    this.__d = { t: now(), kind: 'xhr', method: String(m).toUpperCase(), url: String(u), reqHeaders: {} };
    return XO.call(this, m, u, ...r);
  };
  XMLHttpRequest.prototype.setRequestHeader = function (k, v) { if (this.__d) this.__d.reqHeaders[k] = v; return XH.call(this, k, v); };
  XMLHttpRequest.prototype.send = function (body) {
    const rec = this.__d || { t: now(), kind: 'xhr' };
    rec.reqBody = typeof body === 'string' ? clip(body) : body == null ? null : `[${tag(body)}]`;
    state.http.push(rec);
    const t1 = performance.now();
    this.addEventListener('loadend', () => {
      rec.status = this.status; rec.ms = Math.round(performance.now() - t1);
      rec.resHeaders = this.getAllResponseHeaders();
      try {
        rec.resBody = (this.responseType === '' || this.responseType === 'text') ? clip(this.responseText)
          : this.responseType === 'json' ? ser(this.response) : `[${this.responseType}]`;
      } catch (e) { rec.resBody = `[${e}]`; }
    });
    return XS.call(this, body);
  };

  // --- WebSocket ------------------------------------------------------------------
  const wsData = (d) => typeof d === 'string' ? clip(d) : `[${tag(d)} ${d.byteLength || d.size || ''}]`;
  const WSsend = WebSocket.prototype.send;
  WebSocket.prototype.send = function (d) { state.ws.push({ t: now(), dir: 'out', url: this.url, data: wsData(d) }); return WSsend.call(this, d); };
  const OrigWS = window.WebSocket;
  const HookWS = function (url, protocols) {
    const ws = protocols === undefined ? new OrigWS(url) : new OrigWS(url, protocols);
    state.ws.push({ t: now(), dir: 'open', url: String(url) });
    ws.addEventListener('message', (e) => state.ws.push({ t: now(), dir: 'in', url: ws.url, data: wsData(e.data) }));
    ws.addEventListener('close', (e) => state.ws.push({ t: now(), dir: 'close', url: ws.url, code: e.code, reason: e.reason }));
    return ws;
  };
  HookWS.prototype = OrigWS.prototype;
  Object.assign(HookWS, { CONNECTING: 0, OPEN: 1, CLOSING: 2, CLOSED: 3 });
  window.WebSocket = HookWS;

  // --- EventSource (SSE) e postMessage ---------------------------------------------
  if (window.EventSource) {
    const OrigES = window.EventSource;
    const HookES = function (url, cfg) {
      const es = new OrigES(url, cfg);
      es.addEventListener('message', (e) => state.sse.push({ t: now(), url: String(url), data: clip(e.data) }));
      return es;
    };
    HookES.prototype = OrigES.prototype;
    window.EventSource = HookES;
  }
  window.addEventListener('message', (e) => state.messages.push({ t: now(), origin: e.origin, data: ser(e.data) }));

  // --- coleta estática ----------------------------------------------------------------
  const TEXT_CT = /(javascript|ecmascript|json|css|html|xml|text\/|svg)/i;
  const TEXT_EXT = /\.(m?js|cjs|css|json|map|html?|txt|svg|xml|webmanifest)(\?|#|$)/i;
  const abs = (u, base = location.href) => { try { return new URL(u, base).href; } catch { return null; } };

  async function grab(url, binary) {
    const out = { url };
    try {
      const r = await origFetch(url, { credentials: 'include', cache: 'force-cache' });
      out.status = r.status; out.type = r.headers.get('content-type') || '';
      out.headers = {}; r.headers.forEach((v, k) => (out.headers[k] = v));
      if (TEXT_CT.test(out.type) || TEXT_EXT.test(url)) out.text = clip(await r.text());
      else if (binary) {
        const b = await r.blob(); out.size = b.size;
        if (b.size <= MAX_BODY) out.base64 = await new Promise((ok) => { const fr = new FileReader(); fr.onload = () => ok(String(fr.result).split(',')[1]); fr.readAsDataURL(b); });
      } else out.skipped = 'binário (use {binary:true})';
    } catch (e) { out.error = String(e); }
    return out;
  }

  function pageGlobals() {
    try {
      const ifr = document.createElement('iframe'); ifr.style.display = 'none'; document.documentElement.appendChild(ifr);
      const base = new Set(Object.getOwnPropertyNames(ifr.contentWindow)); ifr.remove();
      const g = {};
      for (const k of Object.getOwnPropertyNames(window)) {
        if (base.has(k) || k === '__dump') continue;
        try { const v = window[k]; g[k] = typeof v === 'function' ? `[Function ${v.name || k}]` : ser(v); } catch (e) { g[k] = `[${e}]`; }
      }
      return g;
    } catch (e) { return { error: String(e) }; }
  }

  const dumpStore = (s) => { const o = {}; try { for (let i = 0; i < s.length; i++) { const k = s.key(i); o[k] = s.getItem(k); } } catch (e) { o.__error = String(e); } return o; };

  async function save(opts = {}) {
    const binary = !!opts.binary;
    console.info('[dump] coletando...');
    const entries = performance.getEntriesByType('resource').map((e) => ({
      name: e.name, initiator: e.initiatorType, start: Math.round(e.startTime), ms: Math.round(e.duration),
      transfer: e.transferSize, encoded: e.encodedBodySize, decoded: e.decodedBodySize,
      protocol: e.nextHopProtocol, status: e.responseStatus,
    }));

    const urls = new Set(entries.map((e) => e.name));
    document.querySelectorAll('script[src],link[href],img[src],source[src],iframe[src],video[src],audio[src],embed[src]').forEach((el) => {
      const u = abs(el.getAttribute('src') || el.getAttribute('href')); if (u) urls.add(u);
    });
    urls.add(location.href);
    const queue = [...urls].filter((u) => /^https?:/.test(u));
    const total = queue.length;

    const resources = {};
    let done = 0;
    const worker = async () => {
      while (queue.length) {
        const u = queue.shift();
        resources[u] = await grab(u, binary);
        if (++done % 25 === 0) console.info(`[dump] ${done}/${total}`);
      }
    };
    await Promise.all(Array.from({ length: 6 }, worker));

    // source maps referenciados no JS/CSS (código-fonte original, se publicado)
    const maps = new Set();
    for (const r of Object.values(resources)) {
      if (!r.text) continue;
      const m = r.text.match(/[#@]\s*sourceMappingURL=([^\s'"*]+)\s*(\*\/)?\s*$/m);
      const ref = (r.headers && (r.headers['sourcemap'] || r.headers['x-sourcemap'])) || (m && m[1]);
      if (ref && !ref.startsWith('data:')) { const u = abs(ref, r.url); if (u && !resources[u]) maps.add(u); }
    }
    for (const u of maps) resources[u] = await grab(u, false);

    let idb = []; try { idb = indexedDB.databases ? await indexedDB.databases() : []; } catch {}
    let sw = []; try { sw = ((await navigator.serviceWorker?.getRegistrations?.()) || []).map((r) => ({ scope: r.scope, script: r.active && r.active.scriptURL })); } catch {}

    const dump = {
      meta: {
        url: location.href, title: document.title, savedAt: now(), installedAt: new Date(T0).toISOString(),
        userAgent: navigator.userAgent, referrer: document.referrer, viewport: [innerWidth, innerHeight],
      },
      navigation: performance.getEntriesByType('navigation').map((e) => e.toJSON()),
      domLive: document.documentElement.outerHTML,
      inlineScripts: [...document.querySelectorAll('script:not([src])')].map((s, i) => ({ i, type: s.type, id: s.id, text: s.textContent })),
      inlineStyles: [...document.querySelectorAll('style')].map((s, i) => ({ i, text: s.textContent })),
      metaTags: [...document.querySelectorAll('meta')].map((m) => ({ name: m.name || m.getAttribute('property') || m.httpEquiv, content: m.content })),
      links: [...new Set([...document.querySelectorAll('a[href]')].map((a) => a.href))],
      network: entries,
      resources,
      storage: { cookies: document.cookie, local: dumpStore(localStorage), session: dumpStore(sessionStorage), indexedDB: idb, serviceWorkers: sw },
      globals: pageGlobals(),
      live: state,
    };

    const blob = new Blob([JSON.stringify(dump)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `dump_${location.hostname}_${now().replace(/[:.]/g, '-')}.json`;
    document.documentElement.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 30000);
    console.info(`[dump] ok: ${Object.keys(resources).length} recursos (${maps.size} source maps), ${state.http.length} req ao vivo, ${state.ws.length} ws, ${state.console.length} logs, ${(blob.size / 1e6).toFixed(1)} MB`);
    return dump.meta;
  }

  window.__dump = {
    save, state,
    stats: () => Object.fromEntries(Object.entries(state).map(([k, v]) => [k, v.length])),
  };
  console.info('[dump] ganchos instalados. Interaja com a página e depois rode:  await __dump.save()');
})();
