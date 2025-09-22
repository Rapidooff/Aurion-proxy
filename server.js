// ─────────────────────────────────────────────────────────────────────────────
// AURION PROXY — Express + Ollama (Gemma/Phi) + Tavily + APNS (iOS commands)
// Rapide, robuste, switch de modèle in-app, recherche web, smart routing,
// push iOS pour exécuter des commandes, et horloge Europe/Paris.
// ─────────────────────────────────────────────────────────────────────────────

/*
ENV attendus (.env) — exemples :
PORT=3000
OLLAMA_HOST=http://localhost:11434
AURION_MODEL=aurion-gemma
TAVILY_API_KEY=tvly_xxx

# APNS pour push vers ton app iOS (pour exécuter des commandes côté téléphone)
APNS_TEAM_ID=XXXXXXXXXX
APNS_KEY_ID=YYYYYYYYYY
APNS_BUNDLE_ID=com.rapido.aurion
APNS_PRIVATE_KEY_BASE64=...    (contenu base64 d'un .p8 ENTIER, sans retour ligne)
APNS_SANDBOX=true              (true = environnement de dev)
*/

const express = require('express');
const cors = require('cors');
const http2 = require('http2');
const jwt = require('jsonwebtoken');

// charge .env si présent
try { require('dotenv').config(); } catch {}

// ── App & middlewares
const app = express();
app.use(cors({ origin: true }));
app.use(express.json({ limit: '2mb' }));

// ── Config (env > défauts)
const PORT = Number(process.env.PORT || 3000);
const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';
const DEFAULT_MODEL = process.env.AURION_MODEL || 'aurion-gemma';
const REQUEST_TIMEOUT_MS = Number(process.env.REQUEST_TIMEOUT_MS || 120000);
const TAVILY_API_KEY = process.env.TAVILY_API_KEY || '';
const TZ = 'Europe/Paris';

// APNS (optionnels)
const APNS_TEAM_ID = process.env.APNS_TEAM_ID || '';
const APNS_KEY_ID = process.env.APNS_KEY_ID || '';
const APNS_BUNDLE_ID = process.env.APNS_BUNDLE_ID || '';
const APNS_PRIVATE_KEY_BASE64 = process.env.APNS_PRIVATE_KEY_BASE64 || '';
const APNS_SANDBOX = String(process.env.APNS_SANDBOX || 'true').toLowerCase() === 'true';

// ── Utils
const log = (...args) => console.log(new Date().toISOString(), '-', ...args);

function nowParis() {
  const d = new Date();
  const fmt = new Intl.DateTimeFormat('fr-FR', {
    timeZone: TZ, dateStyle: 'full', timeStyle: 'medium'
  }).format(d);
  return {
    iso: new Date(d.toLocaleString('en-US', { timeZone: TZ })).toISOString(),
    human: fmt,
    tz: TZ,
  };
}

// Timeouts pour fetch
async function fetchWithTimeout(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(id);
  }
}

// Modèle courant (réagit à /set_model)
function getCurrentModel() {
  return process.env.AURION_MODEL || DEFAULT_MODEL;
}

// Heuristique : faut-il “naviguer” (Tavily) ?
// Si la requête semble récente/sensible (news, prix, lois, “aujourd’hui”, “dernier”, “2025”, etc.)
function shouldBrowse(q = '') {
  const s = q.toLowerCase();
  const triggers = [
    'aujourd\'hui','hier','demain','derni', // dernier/dernière
    'actu','actualité','news','nouveau','mise à jour','changelog',
    'prix','tarif','loi','règlement','horaire','match','score',
    'programme','planning','disponible','sortie','release',
    'version','ios','android','macos','windows','kernel',
    '2023','2024','2025','2026','septembre','octobre','novembre','décembre'
  ];
  return triggers.some(t => s.includes(t));
}

// Réponse courte par défaut (safe) si l’app ne force rien
const DEFAULT_OPTIONS = {
  temperature: 0.3,
  top_p: 0.9,
  num_ctx: 1536,
  num_predict: 200,
  repeat_penalty: 1.1,
  repeat_last_n: 96,
};

// ─────────────────────────────────────────────────────────────────────────────
// Root & Health
// ─────────────────────────────────────────────────────────────────────────────

app.get('/', (_req, res) => res.send('Aurion proxy is running'));

app.get('/health', async (_req, res) => {
  let ollama = 'down';
  try {
    const r = await fetchWithTimeout(`${OLLAMA_HOST}/api/tags`, { method: 'GET' }, 5000);
    if (r.ok) ollama = 'up';
  } catch {}
  const tavily = TAVILY_API_KEY ? 'configured' : 'missing_key';
  const time = nowParis();
  res.json({ ok: true, model: getCurrentModel(), ollama, tavily, time });
});

// ─────────────────────────────────────────────────────────────────────────────
// OLLAMA — NON-STREAM
// body: { prompt, system?, options?, model? }
// ─────────────────────────────────────────────────────────────────────────────
app.post('/aurion', async (req, res) => {
  const { prompt, system, options, model } = req.body || {};
  if (!prompt || typeof prompt !== 'string') {
    return res.status(400).json({ error: 'Missing prompt (string expected)' });
  }
  const finalModel = (model && String(model)) || getCurrentModel();
  const body = {
    model: finalModel,
    prompt,
    stream: false,
    ...(system ? { system } : {}),
    options: { ...DEFAULT_OPTIONS, ...(options || {}) },
  };

  try {
    log('> /aurion', { model: finalModel, len: prompt.length });
    const r = await fetchWithTimeout(`${OLLAMA_HOST}/api/generate`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    if (!r.ok) {
      const text = await r.text().catch(() => '');
      log('! Ollama error', r.status, text.slice(0, 300));
      return res.status(502).json({ error: 'Ollama error', status: r.status, detail: text });
    }

    const data = await r.json();
    return res.json({
      reply: data.response || '',
      model: finalModel,
      metrics: {
        total_duration: data.total_duration,
        eval_count: data.eval_count,
        eval_duration: data.eval_duration,
        load_duration: data.load_duration,
        prompt_eval_count: data.prompt_eval_count,
      },
    });
  } catch (e) {
    const code = e?.name === 'AbortError' ? 504 : 500;
    log('! Proxy error /aurion', e?.name || '', e?.message || e);
    return res.status(code).json({ error: 'Proxy error', detail: String(e) });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// OLLAMA — STREAM (texte progressif)
// ─────────────────────────────────────────────────────────────────────────────
app.post('/aurion_stream', async (req, res) => {
  const { prompt, system, options, model } = req.body || {};
  if (!prompt || typeof prompt !== 'string') {
    res.writeHead(400, { 'Content-Type': 'text/plain; charset=utf-8' });
    return res.end('Missing prompt (string expected)');
  }
  const finalModel = (model && String(model)) || getCurrentModel();

  try {
    log('> /aurion_stream', { model: finalModel, len: prompt.length });

    const r = await fetchWithTimeout(`${OLLAMA_HOST}/api/generate`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: finalModel, prompt, stream: true,
        ...(system ? { system } : {}),
        options: { ...DEFAULT_OPTIONS, ...(options || {}) },
      }),
    });

    if (!r.ok || !r.body) {
      const text = await r.text().catch(() => '');
      res.writeHead(502, { 'Content-Type': 'text/plain; charset=utf-8' });
      return res.end(`Ollama error: ${text}`);
    }

    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let totalChars = 0;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      for (const line of chunk.split('\n').filter(Boolean)) {
        try {
          const json = JSON.parse(line);
          if (json.response) { totalChars += json.response.length; res.write(json.response); }
          if (json.done) { log('< /aurion_stream done', { totalChars }); return res.end(); }
        } catch { /* ignore fragments */ }
      }
    }
    res.end();
  } catch (e) {
    const msg = e?.name === 'AbortError' ? 'Proxy timeout' : `Proxy error: ${String(e)}`;
    res.writeHead(e?.name === 'AbortError' ? 504 : 500, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end(msg);
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// MODEL SWITCH — /set_model { model: "aurion-gemma" | "aurion-phi" }
// ─────────────────────────────────────────────────────────────────────────────
app.post('/set_model', (req, res) => {
  const { model } = req.body || {};
  if (!model || typeof model !== 'string') {
    return res.status(400).json({ error: 'Missing model (string expected)' });
  }
  process.env.AURION_MODEL = model;
  log('> Model switched to', model);
  res.json({ ok: true, model });
});

// ─────────────────────────────────────────────────────────────────────────────
// TAVILY — /search (brut) et /answer (search → résumé par LLM)
// ─────────────────────────────────────────────────────────────────────────────
app.post('/search', async (req, res) => {
  try {
    if (!TAVILY_API_KEY) return res.status(400).json({ error: 'Missing TAVILY_API_KEY' });
    const {
      q, query, search_depth = 'basic', include_answer = true,
      max_results = 5, include_images = false, include_domains, exclude_domains
    } = req.body || {};
    const finalQuery = (q || query || '').toString().trim();
    if (!finalQuery) return res.status(400).json({ error: 'Missing query (q or query)' });

    const body = {
      api_key: TAVILY_API_KEY,
      query: finalQuery, search_depth, include_answer, max_results, include_images,
      ...(Array.isArray(include_domains) ? { include_domains } : {}),
      ...(Array.isArray(exclude_domains) ? { exclude_domains } : {}),
    };

    log('> /search tavily', { q: finalQuery.slice(0, 80), max_results, search_depth });
    const r = await fetchWithTimeout('https://api.tavily.com/search', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }, 20000);

    if (!r.ok) {
      const text = await r.text().catch(() => '');
      log('! Tavily error', r.status, text.slice(0, 300));
      return res.status(502).json({ error: 'Tavily error', status: r.status, detail: text });
    }

    const data = await r.json();
    return res.json({ ok: true, ...data });
  } catch (e) {
    const code = e?.name === 'AbortError' ? 504 : 500;
    log('! Proxy error /search', e?.name || '', e?.message || e);
    return res.status(code).json({ error: 'Proxy error', detail: String(e) });
  }
});

// /answer : { q: "...", style?: "bullets"|"short"|"long", max_results?, search_depth? }
app.post('/answer', async (req, res) => {
  const { q, style = 'bullets', max_results = 5, search_depth = 'advanced' } = req.body || {};
  if (!q || typeof q !== 'string') return res.status(400).json({ error: 'Missing q (string)' });
  if (!TAVILY_API_KEY) return res.status(400).json({ error: 'Missing TAVILY_API_KEY' });

  try {
    // 1) Recherche Tavily
    const sr = await fetchWithTimeout('https://api.tavily.com/search', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        api_key: TAVILY_API_KEY, query: q, search_depth, include_answer: true,
        max_results, include_images: false
      }),
    }, 20000);
    if (!sr.ok) {
      const text = await sr.text().catch(() => '');
      return res.status(502).json({ error: 'Tavily error', detail: text });
    }
    const data = await sr.json();

    // 2) Construit un contexte compact pour le LLM
    const context = [
      data.answer ? `Réponse Tavily: ${data.answer}` : '',
      ...(Array.isArray(data.results) ? data.results.slice(0, max_results).map(r =>
        `• ${r.title || ''}\n${r.content || ''}`) : [])
    ].filter(Boolean).join('\n\n');

    // 3) Demande au LLM de synthétiser
    const prompt = [
      'Tu es Aurion. Utilise UNIQUEMENT les informations ci-dessous (contexte).',
      'Réponds en français, précis, et cite si nécessaire les sources en fin de réponse (titres courts).',
      style === 'bullets' ? 'Format: 4–6 puces concises.' :
      style === 'short' ? 'Format: 1 paragraphe court.' :
      'Format: synthèse développée, claire.',
      '',
      '=== CONTEXTE ===',
      context,
      '',
      '=== QUESTION ===',
      q
    ].join('\n');

    const finalModel = getCurrentModel();
    const r2 = await fetchWithTimeout(`${OLLAMA_HOST}/api/generate`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: finalModel, prompt,
        stream: false, options: { ...DEFAULT_OPTIONS, num_predict: 400 }
      }),
    });

    if (!r2.ok) {
      const text = await r2.text().catch(() => '');
      return res.status(502).json({ error: 'Ollama error', detail: text });
    }

    const out = await r2.json();
    return res.json({ ok: true, reply: out.response || '', model: finalModel });
  } catch (e) {
    const code = e?.name === 'AbortError' ? 504 : 500;
    log('! Proxy error /answer', e?.name || '', e?.message || e);
    return res.status(code).json({ error: 'Proxy error', detail: String(e) });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// /aurion_smart : route qui choisit toute seule (LLM direct vs Tavily)
// body: { prompt, forceWeb?: boolean, reliability?: "high"|"default", style? }
// ─────────────────────────────────────────────────────────────────────────────
app.post('/aurion_smart', async (req, res) => {
  const { prompt, forceWeb = false, reliability = 'default', style = 'bullets', options, model } = req.body || {};
  if (!prompt || typeof prompt !== 'string') {
    return res.status(400).json({ error: 'Missing prompt (string expected)' });
  }

  const mustWeb = forceWeb || shouldBrowse(prompt) || reliability === 'high';
  if (mustWeb && TAVILY_API_KEY) {
    // délègue à /answer
    req.body = { q: prompt, style };
    return app._router.handle(req, res, app._router.stack.find(l => l.route && l.route.path === '/answer').route.stack[0].handle);
  }

  // Sinon: direct LLM (non-stream) avec options safe
  const finalModel = (model && String(model)) || getCurrentModel();
  try {
    const r = await fetchWithTimeout(`${OLLAMA_HOST}/api/generate`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: finalModel, prompt, stream: false,
        options: { ...DEFAULT_OPTIONS, ...(options || {}) }
      }),
    });
    if (!r.ok) {
      const text = await r.text().catch(() => '');
      return res.status(502).json({ error: 'Ollama error', detail: text });
    }
    const out = await r.json();
    return res.json({ ok: true, reply: out.response || '', model: finalModel });
  } catch (e) {
    const code = e?.name === 'AbortError' ? 504 : 500;
    log('! Proxy error /aurion_smart', e?.name || '', e?.message || e);
    return res.status(code).json({ error: 'Proxy error', detail: String(e) });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// APNS — envoyer des commandes à l'app iOS (ton app exécute côté device)
// /notify : push simple
// /phone_command : push avec payload { type: "...", args: {...} }
// L'app iOS doit interpréter le payload et exécuter (ouvrir app, lancer nav,
// domotique, rappeler quelqu’un, activer un mode, etc.).
// ─────────────────────────────────────────────────────────────────────────────

function apnsIsConfigured() {
  return Boolean(APNS_TEAM_ID && APNS_KEY_ID && APNS_BUNDLE_ID && APNS_PRIVATE_KEY_BASE64);
}

function apnsToken() {
  // JWT pour APNS
  const privateKey = Buffer.from(APNS_PRIVATE_KEY_BASE64, 'base64').toString('utf8');
  return jwt.sign(
    { iss: APNS_TEAM_ID, iat: Math.floor(Date.now() / 1000) },
    privateKey,
    { algorithm: 'ES256', header: { alg: 'ES256', kid: APNS_KEY_ID } }
  );
}

// deviceToken: string (sans espaces)
// payload: JSON object (alert, command, etc.)
async function apnsSend(deviceToken, payload) {
  if (!apnsIsConfigured()) throw new Error('APNS not configured');
  const host = APNS_SANDBOX ? 'api.sandbox.push.apple.com' : 'api.push.apple.com';
  const client = http2.connect(`https://${host}`);
  const headers = {
    ':method': 'POST',
    ':path': `/3/device/${deviceToken}`,
    'authorization': `bearer ${apnsToken()}`,
    'apns-topic': APNS_BUNDLE_ID,
    'apns-push-type': payload.command ? 'background' : 'alert',
    'content-type': 'application/json',
  };

  return new Promise((resolve, reject) => {
    const req = client.request(headers);
    req.setEncoding('utf8');
    let data = '';
    req.on('response', (headers) => {
      // collect if needed
    });
    req.on('data', (chunk) => { data += chunk; });
    req.on('end', () => {
      client.close();
      if (data && data.includes('BadDeviceToken')) return reject(new Error('BadDeviceToken'));
      resolve(data || '{}');
    });
    req.on('error', (err) => { client.close(); reject(err); });
    req.end(JSON.stringify({ aps: { alert: payload.alert, sound: payload.sound ? 'default' : undefined, contentAvailable: payload.command ? 1 : 0 }, ...payload }));
  });
}

// Push simple (notification)
app.post('/notify', async (req, res) => {
  const { deviceToken, title = 'Aurion', body = '', sound = true, extra = {} } = req.body || {};
  if (!deviceToken) return res.status(400).json({ error: 'Missing deviceToken' });
  if (!apnsIsConfigured()) return res.status(400).json({ error: 'APNS not configured' });

  try {
    const result = await apnsSend(deviceToken, { alert: { title, body }, sound, ...extra });
    return res.json({ ok: true, result });
  } catch (e) {
    return res.status(502).json({ error: 'APNS error', detail: String(e) });
  }
});

// Commande téléphone (ton app iOS exécute)
// body: { deviceToken, type, args? }
// Exemples de type: "open_app", "call_contact", "reminder", "navigate", "home_action", "say"
// Ta logique côté app doit gérer ces types et exécuter localement.
app.post('/phone_command', async (req, res) => {
  const { deviceToken, type, args = {} } = req.body || {};
  if (!deviceToken || !type) return res.status(400).json({ error: 'Missing deviceToken or type' });
  if (!apnsIsConfigured()) return res.status(400).json({ error: 'APNS not configured' });

  const payload = {
    command: { type, args, ts: nowParis() },
    // Optionnel: message visible
    alert: { title: 'Aurion', body: `Commande: ${type}` },
    sound: false
  };

  try {
    const result = await apnsSend(deviceToken, payload);
    return res.json({ ok: true, result });
  } catch (e) {
    return res.status(502).json({ error: 'APNS error', detail: String(e) });
  }
});

// ─────────────────────────────────────────────────────────────────────────────
// Start
// ─────────────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  log(`Aurion proxy running → http://localhost:${PORT} | model=${getCurrentModel()}`);
});
