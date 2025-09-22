// ──────────────────────────────────────────────────────────────
// AURION PROXY v1.2 — Factual Mode (no BS)
// ──────────────────────────────────────────────────────────────
import express from 'express';
import cors from 'cors';
try { (await import('dotenv')).config(); } catch {}

const app = express();

// — Config
const PORT = Number(process.env.PORT || 3000);
const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';
const AURION_MODEL = process.env.AURION_MODEL || 'aurion-gemma';
const TAVILY_API_KEY = process.env.TAVILY_API_KEY || '';
const TZ = 'Europe/Paris';

// — Defaults “safe”
const BASE_OPTIONS = {
  temperature: 0.2, // ↓ créativité
  top_p: 0.85,
  repeat_penalty: 1.15,
  repeat_last_n: 128,
  num_ctx: 1536,
  num_predict: 300,
};

// — Express
app.use(cors({ origin: true }));
app.use(express.json({ limit: '2mb' }));

// — Time helper
function nowParis() {
  const d = new Date();
  const iso = new Date(d.toLocaleString('en-US', { timeZone: TZ })).toISOString();
  const human = new Intl.DateTimeFormat('fr-FR', { timeZone: TZ, dateStyle: 'full', timeStyle: 'medium' }).format(d);
  return { iso, human, tz: TZ };
}

// — Guardrail system (FR)
const GUARDRAIL_SYSTEM = `
Tu es **Aurion**, assistant factuel et fiable. Règles ABSOLUES :
1) **Pas de spéculation**. Si tu n'es pas certain, dis-le clairement: "Je ne sais pas avec certitude" et propose une recherche.
2) **Toujours le contexte temporel** quand utile (dates précises, "au 22 septembre 2025").
3) **Structure claire**, concise, étape par étape si nécessaire.
4) **Citations** si tu as utilisé des sources (format: [1] [2] avec domaines).
5) **Zéro invention** de chiffres, personnes, lois, prix, API, endpoints.
6) Si la question est ambiguë, répond au cas le plus probable en le signalant ET propose l'alternative en 1 ligne.
`;

// — Heuristique : faut-il naviguer ?
function shouldBrowse(q = '') {
  const s = q.toLowerCase();
  return [
    'aujourd\'hui','hier','demain','derni','actu','news','rumeur','breaking','nouveau','maj','update',
    'prix','tarif','loi','décret','score','résultat','classement',
    'version','release','roadmap','changelog',
    '2024','2025','septembre','octobre','novembre','décembre'
  ].some(t => s.includes(t));
}

// — Fetch w/ timeout
async function fwt(url, init = {}, timeoutMs = 20000) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeoutMs);
  try { return await fetch(url, { ...init, signal: ctl.signal }); }
  finally { clearTimeout(t); }
}

// — /health
app.get('/health', async (_req, res) => {
  let ollama = 'down';
  try {
    const r = await fwt(`${OLLAMA_HOST}/api/tags`, {}, 5000);
    if (r.ok) ollama = 'up';
  } catch {}
  res.json({ ok: true, model: AURION_MODEL, ollama, tavily: TAVILY_API_KEY ? 'configured' : 'missing_key', time: nowParis() });
});

// — Basique non-stream (garde-fous activés)
app.post('/aurion', async (req, res) => {
  const { prompt, options = {}, model } = req.body || {};
  if (!prompt || typeof prompt !== 'string') return res.status(400).json({ error: 'Missing prompt' });
  const finalModel = String(model || AURION_MODEL);

  const body = {
    model: finalModel,
    prompt,
    system: GUARDRAIL_SYSTEM,
    stream: false,
    options: { ...BASE_OPTIONS, ...options },
  };

  try {
    const r = await fwt(`${OLLAMA_HOST}/api/generate`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    }, 120000);

    if (!r.ok) {
      const txt = await r.text().catch(() => '');
      return res.status(502).json({ error: 'Ollama error', detail: txt.slice(0, 500) });
    }
    const j = await r.json();
    res.json({ reply: j.response || '', model: finalModel });
  } catch (e) {
    res.status(e?.name === 'AbortError' ? 504 : 500).json({ error: 'Proxy error', detail: String(e) });
  }
});

// — STREAM (garde-fous activés)
app.post('/aurion_stream', async (req, res) => {
  const { prompt, options = {}, model } = req.body || {};
  if (!prompt || typeof prompt !== 'string') {
    res.writeHead(400, { 'Content-Type': 'text/plain; charset=utf-8' });
    return res.end('Missing prompt');
  }
  const finalModel = String(model || AURION_MODEL);

  const body = {
    model: finalModel,
    prompt,
    system: GUARDRAIL_SYSTEM,
    stream: true,
    options: { ...BASE_OPTIONS, ...options },
  };

  try {
    const r = await fwt(`${OLLAMA_HOST}/api/generate`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    }, 120000);

    if (!r.ok || !r.body) {
      const txt = await r.text().catch(() => '');
      res.writeHead(502, { 'Content-Type': 'text/plain; charset=utf-8' });
      return res.end(`Ollama error: ${txt.slice(0, 500)}`);
    }

    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache');
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunk = dec.decode(value, { stream: true });
      for (const line of chunk.split('\n').filter(Boolean)) {
        try {
          const json = JSON.parse(line);
          if (json.response) res.write(json.response);
          if (json.done) return res.end();
        } catch {}
      }
    }
    res.end();
  } catch (e) {
    res.writeHead(e?.name === 'AbortError' ? 504 : 500, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('Proxy error');
  }
});

// — Tavily raw
app.post('/search', async (req, res) => {
  if (!TAVILY_API_KEY) return res.status(400).json({ error: 'Missing TAVILY_API_KEY' });
  const { q, query, search_depth = 'advanced', max_results = 5, include_answer = true } = req.body || {};
  const finalQ = (q || query || '').toString().trim();
  if (!finalQ) return res.status(400).json({ error: 'Missing query' });

  try {
    const r = await fwt('https://api.tavily.com/search', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: TAVILY_API_KEY, query: finalQ, search_depth, include_answer, max_results, include_images: false })
    }, 20000);
    if (!r.ok) return res.status(502).json({ error: 'Tavily error', status: r.status, detail: await r.text() });
    const data = await r.json();
    res.json({ ok: true, ...data });
  } catch (e) {
    res.status(e?.name === 'AbortError' ? 504 : 500).json({ error: 'Proxy error', detail: String(e) });
  }
});

// — Réponse FIABLE (Tavily -> synthèse avec citations)
app.post('/answer', async (req, res) => {
  if (!TAVILY_API_KEY) return res.status(400).json({ error: 'Missing TAVILY_API_KEY' });
  const { q, style = 'bullets', max_results = 6, search_depth = 'advanced' } = req.body || {};
  if (!q || typeof q !== 'string') return res.status(400).json({ error: 'Missing q' });

  try {
    const sr = await fwt('https://api.tavily.com/search', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ api_key: TAVILY_API_KEY, query: q, search_depth, include_answer: true, max_results, include_images: false }),
    }, 20000);
    if (!sr.ok) return res.status(502).json({ error: 'Tavily error', detail: await sr.text() });
    const data = await sr.json();

    const blocks = [];
    if (data.answer) blocks.push(`Résumé Tavily:\n${data.answer}`);
    if (Array.isArray(data.results)) {
      data.results.slice(0, max_results).forEach((r, i) => {
        blocks.push(`[${i+1}] ${r.title || r.url}\n${r.content || ''}\n(${new URL(r.url).hostname})`);
      });
    }
    const context = blocks.join('\n\n');

    const styleLine =
      style === 'short'  ? 'Format: 1 court paragraphe.' :
      style === 'bullets'? 'Format: 4–6 puces nettes.' :
                           'Format: synthèse concise.';

    const prompt = [
      GUARDRAIL_SYSTEM,
      '',
      '=== CONTEXTE (sources) ===',
      context,
      '',
      '=== CONSIGNES ===',
      '- Réponds UNIQUEMENT en t’appuyant sur les sources ci-dessus.',
      '- Si les sources sont insuffisantes/contradictoires → dis-le et propose une recherche supplémentaire.',
      '- Termine par une ligne "Sources: [1] [2] …" avec numéros pertinents.',
      styleLine,
      '',
      '=== QUESTION ===',
      q
    ].join('\n');

    const r2 = await fwt(`${OLLAMA_HOST}/api/generate`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: AURION_MODEL, prompt, stream: false, options: { ...BASE_OPTIONS, num_predict: 450 } }),
    }, 120000);

    if (!r2.ok) return res.status(502).json({ error: 'Ollama error', detail: await r2.text() });
    const out = await r2.json();
    res.json({ ok: true, reply: out.response || '', model: AURION_MODEL });
  } catch (e) {
    res.status(e?.name === 'AbortError' ? 504 : 500).json({ error: 'Proxy error', detail: String(e) });
  }
});

// — Route “intelligente” : choisit automatique
app.post('/aurion_smart', async (req, res) => {
  const { prompt, reliability = 'default', forceWeb = false, style = 'bullets', options = {} } = req.body || {};
  if (!prompt || typeof prompt !== 'string') return res.status(400).json({ error: 'Missing prompt' });

  const mustWeb = forceWeb || reliability === 'high' || shouldBrowse(prompt);
  if (mustWeb) {
    if (!TAVILY_API_KEY) {
      return res.status(412).json({ // precondition failed
        error: 'high_reliability_requires_web',
        detail: 'Tavily manquant — je préfère ne pas inventer. Active TAVILY_API_KEY.'
      });
    }
    // Déroute vers /answer (sources+citation)
    req.body = { q: prompt, style };
    return app._router.handle(req, res, app._router.stack.find(l => l.route && l.route.path === '/answer').route.stack[0].handle);
  }

  // Sinon, réponse locale avec garde-fous
  try {
    const r = await fwt(`${OLLAMA_HOST}/api/generate`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: AURION_MODEL, prompt, system: GUARDRAIL_SYSTEM, stream: false, options: { ...BASE_OPTIONS, ...options } }),
    }, 120000);
    if (!r.ok) return res.status(502).json({ error: 'Ollama error', detail: await r.text() });
    const j = await r.json();
    res.json({ ok: true, reply: j.response || '', model: AURION_MODEL });
  } catch (e) {
    res.status(e?.name === 'AbortError' ? 504 : 500).json({ error: 'Proxy error', detail: String(e) });
  }
});

// — Switch modèle
app.post('/set_model', (req, res) => {
  const { model } = req.body || {};
  if (!model || typeof model !== 'string') return res.status(400).json({ error: 'Missing model' });
  process.env.AURION_MODEL = model;
  res.json({ ok: true, model });
});

// — Start
app.listen(PORT, () => {
  console.log(`${new Date().toISOString()} - Aurion proxy v1.2 (factual) → http://localhost:${PORT} | model=${AURION_MODEL}`);
});
