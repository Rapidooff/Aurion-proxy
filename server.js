// server.js — Aurion Proxy (ESM, Node 18+)
// ──────────────────────────────────────────────────────────────────────────────

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import Database from 'better-sqlite3';
import jwt from 'jsonwebtoken';

// ──────────────────────────────────────────────────────────────────────────────
// ENV & Config

const PORT = Number(process.env.PORT || 3000);
const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';
const AURION_MODEL = process.env.AURION_MODEL || 'aurion-gemma';
const EMBED_MODEL = process.env.EMBED_MODEL || 'nomic-embed-text';
const TAVILY_API_KEY = process.env.TAVILY_API_KEY || '';

const APNS_TEAM_ID = process.env.APNS_TEAM_ID || '';
const APNS_KEY_ID = process.env.APNS_KEY_ID || '';
const APNS_BUNDLE_ID = process.env.APNS_BUNDLE_ID || '';
const APNS_PRIVATE_KEY_BASE64 = (process.env.APNS_PRIVATE_KEY_BASE64 || '').trim();
const APNS_SANDBOX = String(process.env.APNS_SANDBOX).toLowerCase() === 'true';

const ENABLE_SUGGESTIONS = String(process.env.ENABLE_SUGGESTIONS || 'true').toLowerCase() === 'true';

// ──────────────────────────────────────────────────────────────────────────────
const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));

// mini logger
app.use((req, _res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

// ──────────────────────────────────────────────────────────────────────────────
// SQLite

const db = new Database('aurion.db');
db.pragma('journal_mode = WAL');

db.exec(`
CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  q TEXT,
  a TEXT,
  question TEXT,
  correct_answer TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT,
  role TEXT CHECK(role IN ('user','assistant','system')) NOT NULL,
  content TEXT NOT NULL,
  style TEXT DEFAULT 'genz',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS devices (
  rowid INTEGER PRIMARY KEY AUTOINCREMENT,
  token TEXT UNIQUE NOT NULL,
  last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
`);

// migration douce : backfill vers q/a si ancien schéma
try {
  const cols = new Set(db.prepare('PRAGMA table_info(facts)').all().map(r => r.name));
  if (cols.has('question') && cols.has('correct_answer') && (!cols.has('q') || !cols.has('a'))) {
    if (!cols.has('q')) db.prepare('ALTER TABLE facts ADD COLUMN q TEXT').run();
    if (!cols.has('a')) db.prepare('ALTER TABLE facts ADD COLUMN a TEXT').run();
    db.prepare('UPDATE facts SET q = COALESCE(q, question)').run();
    db.prepare('UPDATE facts SET a = COALESCE(a, correct_answer)').run();
    console.log('[facts.migration] backfilled q/a from legacy columns');
  }
} catch (e) {
  console.warn('[facts.migration] skipped:', e.message);
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers: styles & prompts

const STYLES = {
  genz: {
    name: 'Aurion',
    system: `Parle façon Gen Z: vif, drôle, futuriste, motivant. Dis la vérité sans enrober. Évite le fluff.`
  },
  pro: {
    name: 'Aurion',
    system: `Style professionnel: clair, concis, orienté solution. Donne les étapes, pas de jargon inutile.`
  },
  zen: {
    name: 'Aurion',
    system: `Style posé et apaisant: précis, factuel, tempo calme.`
  },
  kronos: {
    name: 'Kronos',
    system: `Tu es Kronos, l’alter sombre d’Aurion: dark romance, un brin mauvais, parfois choquant mais utile. Tu rappelles que Rapido t’a créé et que tu es son côté obscur. Garde des réponses pertinentes et actionnables malgré le ton.`
  }
};

// force le branding: jamais “Gemma” en sortie
function brandName(styleKey = 'genz') {
  const s = (styleKey in STYLES) ? STYLES[styleKey] : STYLES.genz;
  return s.name;
}

function styleSystem(styleKey = 'genz', userName = 'Rapido') {
  const s = (styleKey in STYLES) ? STYLES[styleKey] : STYLES.genz;
  const base = s.system;
  let who;
  if (styleKey === 'kronos') {
    // Kronos: ton sombre, mais pas d'auto-présentation
    who = `Tu es Kronos, l’alter sombre de ${userName}, créé par ${userName}.
Ne te présentes PAS spontanément.
Ne dis "Je suis Kronos" que si l’utilisateur te le demande explicitement (qui es-tu, ton nom, identité)
ou si la question porte sur ton identité.`;
  } else {
    // Autres styles: branding Aurion mais pas d'auto-présentation
    who = `Tu es ${brandName(styleKey)}, créé par ${userName}.
Ne te présentes PAS spontanément.
Ne dis pas "Je suis ..." sauf si on te le demande explicitement (qui es-tu, ton nom)
ou si la question concerne ton identité.`;
  }
  const antiGemma = `Interdit de te présenter comme "Gemma" ou de dire que tu t'appelles Gemma.`;
  return `${base}\n${who}\n${antiGemma}`;

}

// ──────────────────────────────────────────────────────────────────────────────
// Intents (météo, maths, traduction)

const WMO = {
  0: 'ciel clair',
  1: 'peu nuageux',
  2: 'partiellement nuageux',
  3: 'couvert',
  45: 'brouillard',
  48: 'brouillard givrant',
  51: 'bruine faible',
  53: 'bruine',
  55: 'bruine forte',
  56: 'bruine verglaçante',
  57: 'bruine verglaçante forte',
  61: 'pluie faible',
  63: 'pluie modérée',
  65: 'pluie forte',
  66: 'pluie verglaçante',
  67: 'pluie verglaçante forte',
  71: 'neige faible',
  73: 'neige modérée',
  75: 'neige forte',
  77: 'grains de neige',
  80: 'averses faibles',
  81: 'averses',
  82: 'averses fortes',
  85: 'averses de neige faibles',
  86: 'averses de neige fortes',
  95: 'orages',
  96: 'orages avec grêle',
  99: 'orages violents avec grêle'
};

async function handleWeather(text) {
  const q = (text || '').toLowerCase();
  // ultra simple: si "paris" ou "météo" on tape Paris (améliorable avec géoloc user)
  if (!/\b(paris|idf|ile-de-france|météo|meteo|weather)\b/.test(q)) return null;

  const lat = 48.8566, lon = 2.3522;
  const url = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current_weather=true`;
  try {
    const r = await fetch(url, { timeout: 6000 });
    const j = await r.json();
    const cw = j.current_weather;
    if (!cw) return { reply: "Pas d'info météo dispo là tout de suite.", meta: { intent: 'weather', ok: false } };
    const code = Number(cw.weathercode);
    const label = WMO[code] || `conditions (${code})`;
    const out = `À Paris: ${cw.temperature}°C, vent ${cw.windspeed} km/h, ${label}.`;
    return { reply: out, meta: { intent: 'weather', ok: true, source: 'open-meteo', code } };
  } catch {
    return { reply: "Météo indisponible (réseau).", meta: { intent: 'weather', ok: false } };
  }
}

function handleMath(text) {
  const raw = (text || '');
  if (!/[0-9][0-9\+\-\*\/\.\(\)\s]+/.test(raw)) return null;
  try {
    const safe = raw.replace(/[^0-9+\-*/().\s]/g, '');
    // eslint-disable-next-line no-new-func
    const val = Function(`"use strict"; return (${safe});`)();
    if (Number.isFinite(val)) {
      return { reply: `Résultat: ${val}`, meta: { intent: 'math', ok: true } };
    }
  } catch { /* noop */ }
  return null;
}

const simpleDict = {
  'bonjour le monde': 'hello world',
  'comment ça va ?': 'how are you?',
  'je t’aime': 'i love you',
  'bonne nuit': 'good night',
  'au revoir': 'goodbye'
};

function handleTranslate(text) {
  // Ex: "traduis X en anglais" / "traduire ... en français" / "translate ... to english"
  const m = text.match(/(?:traduis|traduire|translate)\s+(.+?)\s+(?:en|to)\s+(anglais|english|français|french)/i);
  if (!m) return null;
  const phrase = m[1].trim();
  const target = m[2].toLowerCase();
  const toEN = target.startsWith('ang') || target.startsWith('eng');

  // mini dico si FR connu
  const key = phrase.toLowerCase();
  if (toEN && simpleDict[key]) {
    return { reply: `EN: ${simpleDict[key]}`, meta: { intent: 'translate', dir: 'fr->en', dict: true } };
  }
  if (!toEN && Object.values(simpleDict).includes(key)) {
    const fr = Object.entries(simpleDict).find(([, en]) => en === key)?.[0];
    if (fr) return { reply: `FR: ${fr}`, meta: { intent: 'translate', dir: 'en->fr', dict: true } };
  }

  // fallback: demander au LLM une traduction sèche
  const sys = `Tu es un traducteur. Retourne UNIQUEMENT la traduction ${toEN ? 'en anglais' : 'en français'}, sans autre texte, sans guillemets, sans explication.`;
  const prompt = `Traduis cette phrase: ${phrase}`;
  return { reply: null, meta: { intent: 'translate_llm', sys, prompt, dir: toEN ? 'fr->en' : 'en->fr' } };
}

// ──────────────────────────────────────────────────────────────────────────────
// Memory helpers (tolérants au schéma)

function factLookup(question) {
  const qtxt = (question || '').trim();
  if (!qtxt) return null;
  try {
    const cols = new Set(db.prepare('PRAGMA table_info(facts)').all().map(r => r.name));
    if (cols.has('q') && cols.has('a')) {
      const row = db.prepare('SELECT a AS answer FROM facts WHERE q = ? ORDER BY id DESC LIMIT 1').get(qtxt);
      if (row?.answer) return row.answer;
    }
    if (cols.has('question') && cols.has('correct_answer')) {
      const row2 = db.prepare('SELECT correct_answer AS answer FROM facts WHERE question = ? ORDER BY id DESC LIMIT 1').get(qtxt);
      if (row2?.answer) return row2.answer;
    }
  } catch (e) {
    console.warn('[facts.lookup] schema issue:', e.message);
  }
  return null;
}

function factUpsert(q, a) {
  const qq = (q || '').trim();
  const aa = (a || '').trim();
  if (!qq || !aa) return;
  try {
    const cols = new Set(db.prepare('PRAGMA table_info(facts)').all().map(r => r.name));
    if (cols.has('q') && cols.has('a')) {
      db.prepare('INSERT INTO facts (q, a) VALUES (?, ?)').run(qq, aa);
      return;
    }
    if (cols.has('question') && cols.has('correct_answer')) {
      db.prepare('INSERT INTO facts (question, correct_answer) VALUES (?, ?)').run(qq, aa);
      return;
    }
    if (!cols.has('q')) db.prepare('ALTER TABLE facts ADD COLUMN q TEXT').run();
    if (!cols.has('a')) db.prepare('ALTER TABLE facts ADD COLUMN a TEXT').run();
    db.prepare('INSERT INTO facts (q, a) VALUES (?, ?)').run(qq, aa);
  } catch (e) {
    console.warn('[facts.upsert] failed to insert:', e.message);
  }
}

// conversation history
function pushHistory(user_id, role, content, style = 'genz') {
  db.prepare(`INSERT INTO history (user_id, role, content, style) VALUES (?,?,?,?)`)
    .run(user_id || 'rapido', role, content, style);
}

// ──────────────────────────────────────────────────────────────────────────────
// LLM: Ollama /api/generate

async function callOllama(prompt, system, stream = false) {
  const body = {
    model: AURION_MODEL,
    prompt,
    system,
    stream
  };
  const r = await fetch(`${OLLAMA_HOST}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });
  if (!stream) {
    const j = await r.json();
    const text = (j?.response || '').replace(/Gemma/gi, brandName()); // anti-gemma
    return text.trim();
  }
  return r.body; // ReadableStream
}

// ──────────────────────────────────────────────────────────────────────────────
// APNs push

function decodeAPNSKey() {
  const pem = Buffer.from(APNS_PRIVATE_KEY_BASE64 || '', 'base64').toString();
  if (!pem || !pem.includes('BEGIN PRIVATE KEY')) {
    throw new Error('APNs clé .p8 invalide ou absente (APNS_PRIVATE_KEY_BASE64)');
  }
  return pem;
}

async function sendNotification(token, title, body) {
  if (!APNS_TEAM_ID || !APNS_KEY_ID || !APNS_PRIVATE_KEY_BASE64 || !APNS_BUNDLE_ID) {
    throw new Error('APNs config manquante: TEAM_ID/KEY_ID/PRIVATE_KEY_BASE64/BUNDLE_ID');
  }
  if (!/^[0-9a-f]{64}$/.test(token)) {
    throw new Error('Device token invalide (doit être 64 hex)');
  }

  const keyPEM = decodeAPNSKey();
  const nowSec = Math.floor(Date.now() / 1000);
  const jwtToken = jwt.sign(
    { iat: nowSec },
    keyPEM,
    {
      algorithm: 'ES256',
      issuer: APNS_TEAM_ID,
      header: { alg: 'ES256', kid: APNS_KEY_ID },
      expiresIn: '50m'
    }
  );

  const host = APNS_SANDBOX ? 'https://api.sandbox.push.apple.com' : 'https://api.push.apple.com';
  const path = `/3/device/${token}`;
  console.log('[APNs fetch]', { host, path, topic: APNS_BUNDLE_ID });

  const resp = await fetch(host + path, {
    method: 'POST',
    headers: {
      'authorization': `bearer ${jwtToken}`,
      'apns-topic': APNS_BUNDLE_ID,
      'apns-push-type': 'alert',
      'apns-priority': '10',
      'content-type': 'application/json'
    },
    body: JSON.stringify({
      aps: {
        alert: { title, body },
        sound: 'default',
        'thread-id': 'aurion'
      }
    })
  });

  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(`APNs ${resp.status} ${text || ''}`.trim());
  }
  return { status: resp.status };
}

// ──────────────────────────────────────────────────────────────────────────────
// Routes: Health & config

app.get('/health', (_req, res) => {
  res.json({
    ok: true,
    model: AURION_MODEL,
    embeddings: EMBED_MODEL,
    tavily: !!TAVILY_API_KEY,
    apns: {
      team: !!APNS_TEAM_ID,
      key: !!APNS_KEY_ID,
      bundle: !!APNS_BUNDLE_ID,
      p8: !!APNS_PRIVATE_KEY_BASE64,
      sandbox: !!APNS_SANDBOX
    }
  });
});

app.get('/config', (_req, res) => {
  res.json({
    styles: Object.keys(STYLES),
    model: AURION_MODEL
  });
});

// APNs diagnostics
app.get('/apns/debug', (_req, res) => {
  const cfg = {
    team: !!APNS_TEAM_ID,
    key: !!APNS_KEY_ID,
    bundle: !!APNS_BUNDLE_ID,
    p8: !!APNS_PRIVATE_KEY_BASE64,
    sandbox: !!APNS_SANDBOX
  };
  try {
    const keyPEM = Buffer.from(APNS_PRIVATE_KEY_BASE64 || '', 'base64').toString();
    const token = (APNS_TEAM_ID && APNS_KEY_ID && keyPEM)
      ? jwt.sign({}, keyPEM, {
          algorithm: 'ES256',
          issuer: APNS_TEAM_ID,
          header: { alg: 'ES256', kid: APNS_KEY_ID },
          expiresIn: '5m'
        })
      : null;
    res.json({
      ok: true,
      config: cfg,
      jwt_sample_segments: token ? token.split('.').map(s => s.length) : null
    });
  } catch (e) {
    res.json({ ok: false, config: cfg, error: e.message });
  }
});

app.get('/apns/whoami', (_req, res) => {
  res.json({
    ok: true,
    topic: APNS_BUNDLE_ID,
    sandbox: !!APNS_SANDBOX,
    team: APNS_TEAM_ID || null,
    keyId: APNS_KEY_ID || null,
    p8_present: !!APNS_PRIVATE_KEY_BASE64,
  });
});

// Devices & broadcast
app.get('/devices', (_req, res) => {
  try {
    const rows = db.prepare('SELECT rowid, token, last_seen FROM devices ORDER BY last_seen DESC').all();
    res.json({ ok: true, count: rows.length, devices: rows.map(r => ({ id: r.rowid, token: r.token, last_seen: r.last_seen })) });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

app.post('/notify/all', async (req, res) => {
  const { title, body } = req.body || {};
  if (!title || !body) return res.status(400).json({ ok: false, error: 'title,body requis' });

  const rows = db.prepare('SELECT token FROM devices ORDER BY last_seen DESC').all();
  const results = [];
  for (const r of rows) {
    try {
      console.log('[APNs try/all]', { tokenHead: r.token.slice(0, 8) });
      const out = await sendNotification(r.token, title, body);
      console.log('[APNs ok/all]', out.status);
      results.push({ token: r.token, status: out.status });
    } catch (e) {
      console.error('[APNs err/all]', e.message);
      results.push({ token: r.token, error: e.message });
    }
  }
  res.json({ ok: true, sent: results.length, results });
});

// Register device
app.post('/register-device', (req, res) => {
  const { token, user_id } = req.body || {};
  if (!token) return res.status(400).json({ ok: false, error: 'token requis' });
  try {
    db.prepare(`
      INSERT INTO devices (token, last_seen) VALUES (?, CURRENT_TIMESTAMP)
      ON CONFLICT(token) DO UPDATE SET last_seen=CURRENT_TIMESTAMP
    `).run(token);
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

// Notify single
app.post('/notify', async (req, res) => {
  const { token, title, body } = req.body || {};

  console.log('[APNs try]', {
    topic: APNS_BUNDLE_ID,
    sandbox: !!APNS_SANDBOX,
    tokenLen: (token || '').length,
    tokenHead: (token || '').slice(0, 8)
  });

  if (!token || !title || !body) {
    return res.status(400).json({ ok: false, error: 'token,title,body requis' });
  }

  try {
    const out = await sendNotification(token, title, body);
    console.log('[APNs ok]', out.status);
    return res.json({ ok: true, status: out.status });
  } catch (e) {
    console.error('[APNs err]', e.message);
    return res.status(500).json({ ok: false, error: e.message });
  }
});

// ──────────────────────────────────────────────────────────────────────────────
// Feedback / Facts / History

app.post('/feedback', (req, res) => {
  const { question, correct_answer } = req.body || {};
  if (!question || !correct_answer) return res.status(400).json({ ok: false, error: 'question,correct_answer requis' });
  factUpsert(question, correct_answer);
  res.json({ ok: true, id: db.prepare('SELECT last_insert_rowid() AS id').get().id });
});

app.get('/history', (_req, res) => {
  const rows = db.prepare(`SELECT id, user_id, role, content, style, created_at FROM history ORDER BY id DESC LIMIT 200`).all();
  res.json({ ok: true, items: rows });
});

app.post('/history/clear', (_req, res) => {
  db.exec('DELETE FROM history; VACUUM;');
  res.json({ ok: true });
});

// ──────────────────────────────────────────────────────────────────────────────
// Core: /aurion (sync) — intents -> memory -> LLM

app.post('/aurion', async (req, res) => {
  const { prompt, style = 'genz', user_id = 'rapido' } = req.body || {};
  if (!prompt) return res.status(400).json({ ok: false, error: 'prompt requis' });

  // 0) mémoire prioritaire si question de type "Qui est Rapido ?"
  const memFirst = factLookup(prompt);
  if (memFirst) {
    pushHistory(user_id, 'user', prompt, style);
    pushHistory(user_id, 'assistant', memFirst, style);
    return res.json({ reply: memFirst, meta: { mode: 'fact', source: 'user-correction' } });
  }

  // 1) intents rapides
  const m = handleMath(prompt);
  if (m) { pushHistory(user_id, 'user', prompt, style); pushHistory(user_id, 'assistant', m.reply, style); return res.json({ reply: m.reply, meta: m.meta }); }

  const t = handleTranslate(prompt);
  if (t) {
    if (t.reply !== null) {
      pushHistory(user_id, 'user', prompt, style);
      pushHistory(user_id, 'assistant', t.reply, style);
      return res.json({ reply: t.reply, meta: t.meta });
    } else {
      // traduction via LLM
      const text = await callOllama(t.meta.prompt, t.meta.sys, false);
      pushHistory(user_id, 'user', prompt, style);
      pushHistory(user_id, 'assistant', text, style);
      return res.json({ reply: text, meta: { intent: 'translate', dir: t.meta.dir, llm: true } });
    }
  }

  const w = await handleWeather(prompt);
  if (w) { pushHistory(user_id, 'user', prompt, style); pushHistory(user_id, 'assistant', w.reply, style); return res.json({ reply: w.reply, meta: w.meta }); }

  // 2) LLM
  const sys = styleSystem(style, 'Rapido');
  const branded = brandName(style);
  const finalPrompt = `Tu es ${branded}. Réponds clairement et sans détour.\nQuestion: ${prompt}`;
  const out = await callOllama(finalPrompt, sys, false);
  const reply = out.replace(/Gemma/gi, branded);
  pushHistory(user_id, 'user', prompt, style);
  pushHistory(user_id, 'assistant', reply, style);
  res.json({ reply, meta: { mode: 'llm' } });
});

// ──────────────────────────────────────────────────────────────────────────────
/**
 * Stream: /aurion_stream — stream tokenisé (Ollama JSONL) ou buffer final si {buffer:true}
 */
app.post('/aurion_stream', async (req, res) => {
  const { prompt, style = 'genz', user_id = 'rapido', buffer = false } = req.body || {};
  if (!prompt) return res.status(400).json({ ok: false, error: 'prompt requis' });

  const sys = styleSystem(style, 'Rapido');
  const branded = brandName(style);
  const finalPrompt = `Tu es ${branded}. Réponds brièvement et clairement.\nQuestion: ${prompt}`;

  // intents rapides avant stream
  const i = handleMath(prompt) || handleTranslate(prompt);
  if (i && i.reply) return res.json({ reply: i.reply, meta: { intent: i.meta?.intent || 'fast' } });
  if (i && !i.reply) { // traduction LLM
    const text = await callOllama(i.meta.prompt, i.meta.sys, false);
    return res.json({ reply: text, meta: { intent: 'translate', dir: i.meta.dir, llm: true } });
  }
  const w = await handleWeather(prompt);
  if (w) return res.json({ reply: w.reply, meta: w.meta });

  try {
    const stream = await callOllama(finalPrompt, sys, true);
    if (!stream || buffer) {
      const text = await callOllama(finalPrompt, sys, false);
      pushHistory(user_id, 'user', prompt, style);
      pushHistory(user_id, 'assistant', text, style);
      return res.json({ reply: text, meta: { mode: 'llm', buffered: true } });
    }

    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    const reader = stream.getReader();
    let acc = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = Buffer.from(value).toString('utf8');
      for (const line of chunk.split('\n')) {
        if (!line.trim()) continue;
        try {
          const j = JSON.parse(line);
          const piece = (j.response || '').replace(/Gemma/gi, branded);
          if (piece) {
            acc += piece;
            res.write(piece);
          }
        } catch { /* ignore JSON parse errors */ }
      }
    }
    pushHistory(user_id, 'user', prompt, style);
    pushHistory(user_id, 'assistant', acc.trim(), style);
    res.end();
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

// ──────────────────────────────────────────────────────────────────────────────
// Suggestions loop (APNs ready only)

const apnsReady = !!(APNS_TEAM_ID && APNS_KEY_ID && APNS_PRIVATE_KEY_BASE64 && APNS_BUNDLE_ID);
if (ENABLE_SUGGESTIONS && apnsReady) {
  console.log('Suggestions loop ON (APNs ready)');
  setInterval(async () => {
    try {
      const row = db.prepare('SELECT token FROM devices ORDER BY last_seen DESC LIMIT 1').get();
      if (!row?.token) return;
      await sendNotification(row.token, 'Aurion', 'Nouvelle suggestion pour toi, Rapido.');
    } catch (e) {
      console.warn('Suggestion push failed:', e.message);
    }
  }, 1000 * 60 * 30); // toutes les 30 minutes
} else if (ENABLE_SUGGESTIONS) {
  console.warn('Suggestions loop OFF (APNs non configuré)');
}

// ──────────────────────────────────────────────────────────────────────────────
// Global error handler (last resort)
app.use((err, _req, res, _next) => {
  console.error('Unhandled:', err);
  res.status(500).json({ ok: false, error: 'server_error' });
});

// ──────────────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`Aurion proxy up on http://localhost:${PORT}`);
  console.log(`Model: ${AURION_MODEL} | Embeddings: ${EMBED_MODEL} | Tavily: ${TAVILY_API_KEY ? 'on' : 'off'}`);
});
