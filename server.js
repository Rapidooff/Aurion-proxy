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

const AURION_MODEL_PRIMARY   = process.env.AURION_MODEL_PRIMARY || process.env.AURION_MODEL || 'aurion-gemma';
const AURION_MODEL_SECONDARY = process.env.AURION_MODEL_SECONDARY || 'aurion-phi';
const MODELS = {
  primary:   AURION_MODEL_PRIMARY,
  secondary: AURION_MODEL_SECONDARY,
  gemma:     AURION_MODEL_PRIMARY,
  phi:       AURION_MODEL_SECONDARY
};
function chooseModel(hint, style) {
  if (hint && MODELS[hint]) return MODELS[hint];
  if (style === 'kronos') return MODELS.secondary; // Kronos → modèle “rugueux”
  return MODELS.primary;
}

const EMBED_MODEL     = process.env.EMBED_MODEL || 'nomic-embed-text';
const TAVILY_API_KEY  = (process.env.TAVILY_API_KEY || '').trim();

const APNS_TEAM_ID = process.env.APNS_TEAM_ID || '';
const APNS_KEY_ID  = process.env.APNS_KEY_ID || '';
const APNS_BUNDLE_ID = process.env.APNS_BUNDLE_ID || '';
const APNS_PRIVATE_KEY_BASE64 = (process.env.APNS_PRIVATE_KEY_BASE64 || '').trim();
const APNS_SANDBOX = String(process.env.APNS_SANDBOX).toLowerCase() === 'true';

const ENABLE_SUGGESTIONS = String(process.env.ENABLE_SUGGESTIONS || 'true').toLowerCase() === 'true';

// LLM options (anti-coupure + cohérence)
const LLM_NUM_CTX      = Number(process.env.LLM_NUM_CTX || 8192);
const LLM_NUM_PREDICT  = Number(process.env.LLM_NUM_PREDICT || 1024);
const LLM_TEMPERATURE  = Number(process.env.LLM_TEMPERATURE || 0.5);

// ──────────────────────────────────────────────────────────────────────────────
const app = express();
app.use(cors());
app.use(express.json({ limit: '1mb' }));
app.use((req, _res, next) => { console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`); next(); });

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
  session_id TEXT,
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

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  title TEXT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cache (
  key TEXT PRIMARY KEY,
  answer TEXT NOT NULL,
  created_at INTEGER NOT NULL
);
`);
// ── Auto-migrations de colonnes manquantes ───────────────────────────────────
try {
  // history.session_id
  const histCols = new Set(db.prepare('PRAGMA table_info(history)').all().map(r => r.name));
  if (!histCols.has('session_id')) {
    db.prepare('ALTER TABLE history ADD COLUMN session_id TEXT').run();
    console.log('[migrate] history.session_id -> ADDED');
  }

  // facts.q / facts.a déjà géré plus bas, on laisse.
} catch (e) {
  console.warn('[migrate] skipped:', e.message);
}
// migration douce q/a
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
// Styles & branding

const STYLES = {
  genz:  { name: 'Aurion',  system: `Gen Z: vif, drôle, futuriste, motivant. Dis la vérité sans enrober.` },
  pro:   { name: 'Aurion',  system: `Pro: clair, concis, orienté solution avec étapes actionnables.` },
  zen:   { name: 'Aurion',  system: `Zen: posé, précis, factuel, tempo calme.` },
  kronos:{ name: 'Kronos',  system: `Kronos: alter sombre d’Aurion (dark romance), un brin mauvais mais utile. Rappelle que Rapido t’a créé. Reste pertinent malgré le ton.` }
};

function brandName(styleKey = 'genz') {
  const s = STYLES[styleKey] || STYLES.genz;
  return s.name;
}

function styleSystem(styleKey = 'genz', userName = 'Rapido') {
  const s = STYLES[styleKey] || STYLES.genz;
  const base = s.system;
  let who;
  if (styleKey === 'kronos') {
    who = `Tu es Kronos, l’alter sombre de ${userName}, créé par ${userName}.
Ne te présentes PAS spontanément. Ne dis "Je suis Kronos" que si on te le demande ou si la question porte sur ton identité.`;
  } else {
    who = `Tu es ${brandName(styleKey)}, créé par ${userName}.
Ne te présentes PAS spontanément. Ne dis pas "Je suis ..." sauf si on te le demande explicitement ou si la question concerne ton identité.`;
  }
  const antiGemma = `Interdit de te présenter comme "Gemma" ou de dire que tu t'appelles Gemma.`;
  const quality = `Structure: commence par un résumé clair, puis détaille si utile. Termine tes explications, pas de coupure.`;
  return `${base}\n${who}\n${antiGemma}\n${quality}`;
}

// ──────────────────────────────────────────────────────────────────────────────
// Intents

// — Météo stricte
const WMO = {0:'ciel clair',1:'peu nuageux',2:'partiellement nuageux',3:'couvert',45:'brouillard',48:'brouillard givrant',
51:'bruine faible',53:'bruine',55:'bruine forte',56:'bruine verglaçante',57:'bruine verglaçante forte',
61:'pluie faible',63:'pluie modérée',65:'pluie forte',66:'pluie verglaçante',67:'pluie verglaçante forte',
71:'neige faible',73:'neige modérée',75:'neige forte',77:'grains de neige',80:'averses faibles',81:'averses',82:'averses fortes',
85:'averses de neige faibles',86:'averses de neige fortes',95:'orages',96:'orages avec grêle',99:'orages violents avec grêle'};

async function handleWeather(text) {
  const q = (text || '').toLowerCase();
  if (!/(météo|meteo|quel\s+temps|pluie|ensoleillé|ensoleille|neige|vent|prévisions|previsions|meteo à|meteo de)/i.test(q)) return null;
  // simple: Paris par défaut
  const lat = 48.8566, lon = 2.3522;
  try {
    const r = await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current_weather=true`, { timeout: 6000 });
    const j = await r.json();
    const cw = j.current_weather;
    if (!cw) return { reply: "Pas d'info météo dispo.", meta: { intent: 'weather', ok: false } };
    const label = WMO[Number(cw.weathercode)] || `conditions (${cw.weathercode})`;
    return { reply: `À Paris: ${cw.temperature}°C, vent ${cw.windspeed} km/h, ${label}.`, meta: { intent: 'weather', ok: true, source: 'open-meteo' } };
  } catch {
    return { reply: "Météo indisponible (réseau).", meta: { intent: 'weather', ok: false } };
  }
}

// — Fuseaux & heure locale
const CITY_TZ = {
  'paris': 'Europe/Paris',
  'new york': 'America/New_York',
  'nyc': 'America/New_York',
  'tokyo': 'Asia/Tokyo',
  'londres': 'Europe/London',
  'dubai': 'Asia/Dubai',
  'los angeles': 'America/Los_Angeles',
  'la': 'America/Los_Angeles'
};
function extractCity(text) {
  const q = (text || '').toLowerCase();
  for (const k of Object.keys(CITY_TZ)) {
    if (q.includes(k)) return { city: k, tz: CITY_TZ[k] };
  }
  return { city: 'paris', tz: 'Europe/Paris' };
}
function handleTime(text) {
  const q = (text || '').toLowerCase();
  if (!/(quelle\s+heure|quelle\s+date|aujourd'hui|maintenant|time|heure)/i.test(q)) return null;
  const { city, tz } = extractCity(q);
  try {
    const d = new Date();
    const fmtDate = new Intl.DateTimeFormat('fr-FR', { timeZone: tz, dateStyle: 'full' }).format(d);
    const fmtTime = new Intl.DateTimeFormat('fr-FR', { timeZone: tz, timeStyle: 'medium' }).format(d);
    return { reply: `Nous sommes le ${fmtDate}, il est ${fmtTime} (${tz}).`, meta: { intent: 'time', ok: true, city, tz } };
  } catch {
    return { reply: `Heure locale indisponible.`, meta: { intent: 'time', ok: false } };
  }
}

// — Maths strictes (éviter années/dates)
const MATH_WORD_GUARDS = /(siècle|siecle|année|annee|date|vers\s+\d{3,4}|en\s+\d{3,4}|révolution|histoire|qui|quelle|quand|où|ou|prix|combien)/i;
function handleMath(text) {
  const raw = (text || '');
  const hasOp = /[\+\-\*\/]/.test(raw) && /[0-9]/.test(raw);
  const looksLikeYearOnly = /^\s*\d{3,4}\s*$/.test(raw);
  if (!hasOp || looksLikeYearOnly || MATH_WORD_GUARDS.test(raw)) return null;
  try {
    const safe = raw.replace(/[^0-9+\-*/().\s]/g, '');
    // eslint-disable-next-line no-new-func
    const val = Function(`"use strict"; return (${safe});`)();
    if (Number.isFinite(val)) return { reply: `Résultat: ${val}`, meta: { intent: 'math', ok: true } };
  } catch {}
  return null;
}

// — Conversion d’unités
const UNIT_FACTORS = {
  km: 1000, m: 1, cm: 0.01,
  mi: 1609.344, miles: 1609.344,
  ft: 0.3048, 'pied': 0.3048, 'pieds': 0.3048,
  in: 0.0254, pouce: 0.0254, pouces: 0.0254,
  kg: 1, g: 0.001, lb: 0.45359237, lbs: 0.45359237, oz: 0.028349523125
};
function handleConvert(text) {
  const q = (text || '').toLowerCase();
  // températures
  const temp = q.match(/([-+]?\d+(?:[.,]\d+)?)\s*°?\s*(c|celsius|f|fahrenheit)\s*(?:en|to|vers)\s*(c|celsius|f|fahrenheit)/i);
  if (temp) {
    let val = parseFloat(temp[1].replace(',', '.'));
    const from = temp[2][0].toLowerCase(), to = temp[3][0].toLowerCase();
    if (from === to) return { reply: `${val.toFixed(2)} °${from.toUpperCase()}`, meta: { intent: 'convert', kind: 'temp' } };
    const out = (from === 'c') ? (val * 9/5 + 32) : ((val - 32) * 5/9);
    const lab = (to === 'c') ? '°C' : '°F';
    return { reply: `${out.toFixed(2)} ${lab}`, meta: { intent: 'convert', kind: 'temp' } };
  }
  // longueurs/poids
  const m = q.match(/([-+]?\d+(?:[.,]\d+)?)\s*([a-zéû]+)\s*(?:en|to|vers)\s*([a-zéû]+)/i);
  if (!m) return null;
  const val = parseFloat(m[1].replace(',', '.'));
  const from = m[2], to = m[3];
  const fm = UNIT_FACTORS[from]; const tm = UNIT_FACTORS[to];
  if (!fm || !tm) return null;
  // distance via m, masse via kg, sinon assume distance/masse unique
  let base, out;
  if (['kg','g','lb','lbs','oz'].includes(from) || ['kg','g','lb','lbs','oz'].includes(to)) {
    // masse
    const K = { kg:1, g:0.001, lb:0.45359237, lbs:0.45359237, oz:0.028349523125 };
    if (!K[from] || !K[to]) return null;
    base = val * K[from];
    out = base / K[to];
  } else {
    // longueur
    const L = { km:1000, m:1, cm:0.01, mi:1609.344, miles:1609.344, ft:0.3048, 'pied':0.3048,'pieds':0.3048, in:0.0254, pouce:0.0254, pouces:0.0254 };
    if (!L[from] || !L[to]) return null;
    base = val * L[from];
    out = base / L[to];
  }
  return { reply: `${val} ${from} ≈ ${out.toFixed(4)} ${to}`, meta: { intent: 'convert', ok: true } };
}

// — Traduction
const simpleDict = {'bonjour le monde':'hello world','comment ça va ?':'how are you?','je t’aime':'i love you','bonne nuit':'good night','au revoir':'goodbye'};
function handleTranslate(text) {
  const m = text.match(/(?:traduis|traduire|translate)\s+(.+?)\s+(?:en|to)\s+(anglais|english|français|french)/i);
  if (!m) return null;
  const phrase = m[1].trim(); const target = m[2].toLowerCase();
  const toEN = target.startsWith('ang') || target.startsWith('eng');
  const key = phrase.toLowerCase();
  if (toEN && simpleDict[key]) return { reply: `EN: ${simpleDict[key]}`, meta: { intent: 'translate', dir: 'fr->en', dict: true } };
  if (!toEN && Object.values(simpleDict).includes(key)) {
    const fr = Object.entries(simpleDict).find(([, en]) => en === key)?.[0];
    if (fr) return { reply: `FR: ${fr}`, meta: { intent: 'translate', dir: 'en->fr', dict: true } };
  }
  const sys = `Traducteur: retourne UNIQUEMENT la traduction ${toEN ? 'en anglais' : 'en français'}, sans guillemets ni explications.`;
  const prompt = `Traduis cette phrase: ${phrase}`;
  return { reply: null, meta: { intent: 'translate_llm', sys, prompt, dir: toEN ? 'fr->en' : 'en->fr' } };
}

// — Recherche web (Tavily) avec cache (TTL 6h)
const CACHE_TTL_MS = 6 * 60 * 60 * 1000;
function looksLikeResearch(text) {
  return /(aujourd'hui|derni(ers|ères)|actualité|news|prix|coût|tarif|programme|calendrier|horaire|score|mercato|bourse|loi|décret|2024|2025)/i.test(text || '');
}
function cacheGet(key) {
  const row = db.prepare('SELECT answer, created_at FROM cache WHERE key = ?').get(key);
  if (!row) return null;
  if (Date.now() - Number(row.created_at) > CACHE_TTL_MS) { db.prepare('DELETE FROM cache WHERE key=?').run(key); return null; }
  return row.answer;
}
function cacheSet(key, answer) {
  db.prepare('INSERT OR REPLACE INTO cache (key, answer, created_at) VALUES (?,?,?)').run(key, answer, Date.now());
}
async function handleResearch(text) {
  if (!TAVILY_API_KEY) return null;
  if (!looksLikeResearch(text)) return null;
  const key = `tavily:${text.trim().toLowerCase()}`;
  const cached = cacheGet(key);
  if (cached) return { reply: cached, meta: { intent: 'research', ok: true, provider: 'tavily', cached: true } };
  try {
    const resp = await fetch('https://api.tavily.com/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${TAVILY_API_KEY}` },
      body: JSON.stringify({ query: text, search_depth: 'advanced', max_results: 5 })
    });
    const j = await resp.json();
    const items = (j?.results || []).slice(0, 5);
    if (!items.length) return { reply: "Pas de résultats fiables trouvés.", meta: { intent: 'research', ok: false } };
    const bullets = items.map((r, i) => `- ${i+1}. ${r.title || 'Source'} — ${r.url}`).join('\n');
    const summary = j?.answer || "Voici les points clés trouvés :";
    const out = `${summary}\n\nSources:\n${bullets}`;
    cacheSet(key, out);
    return { reply: out, meta: { intent: 'research', ok: true, provider: 'tavily', cached: false } };
  } catch {
    return { reply: "Recherche indisponible (réseau).", meta: { intent: 'research', ok: false } };
  }
}

// — Identité
function handleIdentity(text, style = 'genz') {
  if (!/^\s*(qui\s+es[-\s]?tu|tu\s+es\s+qui|qui\s+êtes[-\s]?vous)\s*\??$/i.test(text || '')) return null;
  if (style === 'kronos') {
    return {
      reply: "Je suis Kronos, l’alter sombre façonné par Rapido. Je coupe le bruit, j’éclaire l’ombre. Pose ta question.",
      meta: { intent: 'identity', style }
    };
  }
  return {
    reply: "Je suis Aurion, un assistant conçu par Rapido pour répondre vite, clair et juste. Dis-moi ce dont tu as besoin.",
    meta: { intent: 'identity', style }
  };
}

// — Classif debug
function classifyIntent(prompt) {
  if (handleMath(prompt)) return 'math';
  if (/(traduis|traduire|translate)\b/i.test(prompt)) return 'translate';
  if (/(quelle\s+heure|quelle\s+date|aujourd'hui|maintenant|time|heure)/i.test(prompt)) return 'time';
  if (/\b(plan|go-to-market|lancement|roadmap|business|produit)\b/i.test(prompt)) return 'business';
  if (/(météo|meteo|quel\s+temps|pluie|ensoleillé|ensoleille|neige|vent|prévisions|previsions)/i.test(prompt)) return 'weather';
  if (looksLikeResearch(prompt)) return 'research';
  if (/\b(qui|quelle|quand|où|ou|combien|pourquoi)\b/i.test(prompt)) return 'factual';
  return 'general';
}

// — Contraintes
function mustBeTwoSentences(prompt) {
  return /(\b2\s*phrases?\b|\bdeux\s*phrases?\b)/i.test(prompt || '');
}
function businessScaffold(prompt) {
  const ok = /\b(plan|go-to-market|lancement|roadmap|business|produit)\b/i.test(prompt || '');
  if (!ok) return '';
  return `
Format demandé (liste concise):
1) Cible & valeur
2) Proposition & différenciation
3) Messages clés
4) Canaux & calendrier (J-30 → J+30)
5) KPI & boucle d'apprentissage
Réponds sans généralités ni phrases creuses.`;
}

// — Réglage longueur
function lengthConstraint(len = 'medium') {
  const L = String(len || 'medium').toLowerCase();
  if (L === 'short') return { txt: '\nRéponds en 1–3 phrases maximum.', mult: 0.6 };
  if (L === 'long')  return { txt: '\nRéponds de façon détaillée mais structurée (≈8–12 phrases).', mult: 1.4 };
  return { txt: '', mult: 1.0 };
}

// ──────────────────────────────────────────────────────────────────────────────
// Mémoire & Sessions

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
  const qq = (q || '').trim(); const aa = (a || '').trim();
  if (!qq || !aa) return;
  try {
    const cols = new Set(db.prepare('PRAGMA table_info(facts)').all().map(r => r.name));
    if (cols.has('q') && cols.has('a')) { db.prepare('INSERT INTO facts (q, a) VALUES (?, ?)').run(qq, aa); return; }
    if (cols.has('question') && cols.has('correct_answer')) { db.prepare('INSERT INTO facts (question, correct_answer) VALUES (?, ?)').run(qq, aa); return; }
    if (!cols.has('q')) db.prepare('ALTER TABLE facts ADD COLUMN q TEXT').run();
    if (!cols.has('a')) db.prepare('ALTER TABLE facts ADD COLUMN a TEXT').run();
    db.prepare('INSERT INTO facts (q, a) VALUES (?, ?)').run(qq, aa);
  } catch (e) { console.warn('[facts.upsert] failed:', e.message); }
}

function pushHistory(session_id, user_id, role, content, style = 'genz') {
  db.prepare(`INSERT INTO history (session_id, user_id, role, content, style) VALUES (?,?,?,?,?)`)
    .run(session_id || null, user_id || 'rapido', role, content, style);
}
function pullRecentHistory(session_id, limit = 6) {
  if (session_id) {
    return db.prepare(`SELECT role, content FROM history WHERE session_id = ? ORDER BY id DESC LIMIT ?`).all(session_id, limit).reverse();
  }
  return db.prepare(`SELECT role, content FROM history ORDER BY id DESC LIMIT ?`).all(limit).reverse();
}
function renderContext(history = []) {
  if (!history.length) return '';
  const lines = history.map(h => `${h.role}: ${h.content}`.trim());
  return `Contexte récent:\n${lines.join('\n')}\n---\n`;
}

// Sessions routes
app.post('/session/start', (req, res) => {
  const { session_id, title } = req.body || {};
  const id = session_id || `s_${Date.now().toString(36)}_${Math.random().toString(36).slice(2,8)}`;
  db.prepare('INSERT OR IGNORE INTO sessions (id, title) VALUES (?,?)').run(id, title || 'Conversation');
  res.json({ ok: true, session_id: id });
});
app.get('/sessions', (_req, res) => {
  const rows = db.prepare('SELECT id, title, created_at FROM sessions ORDER BY created_at DESC').all();
  res.json({ ok: true, items: rows });
});
app.get('/session/:id/history', (req, res) => {
  const id = req.params.id;
  const rows = db.prepare('SELECT id, role, content, style, created_at FROM history WHERE session_id = ? ORDER BY id ASC').all(id);
  res.json({ ok: true, items: rows });
});
app.post('/session/:id/rename', (req, res) => {
  const id = req.params.id; const { title } = req.body || {};
  db.prepare('UPDATE sessions SET title = ? WHERE id = ?').run(title || 'Conversation', id);
  res.json({ ok: true });
});
app.post('/session/:id/clear', (req, res) => {
  const id = req.params.id; db.prepare('DELETE FROM history WHERE session_id = ?').run(id); res.json({ ok: true });
});

// ──────────────────────────────────────────────────────────────────────────────
// LLM : Ollama generate + post-traitements

function chooseOptions(model, intent='general', lengthMult = 1.0) {
  const basePredict = Math.round(LLM_NUM_PREDICT * Math.max(0.5, Math.min(2.0, lengthMult)));
  let opt = {
    num_ctx: LLM_NUM_CTX,
    num_predict: basePredict,
    temperature: LLM_TEMPERATURE,
    top_p: 0.9, top_k: 50,
    repeat_penalty: 1.1, repeat_last_n: 256
  };
  // Phi: plus sec par défaut
  if (/phi/i.test(model)) opt.temperature = Math.min(opt.temperature, 0.35);
  // Business / factual / science → cadrer
  if (intent === 'business') opt.temperature = Math.min(opt.temperature, (/phi/i.test(model) ? 0.25 : 0.35));
  if (intent === 'science' || intent === 'factual') opt.temperature = Math.min(opt.temperature, 0.4);
  return opt;
}

async function callOllama(prompt, system, stream = false, modelName, options = {}) {
  const model = modelName || MODELS.primary;
  const body = { model, prompt, system, stream, options: { ...options } };
  const r = await fetch(`${OLLAMA_HOST}/api/generate`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body)
  });
  if (!stream) {
    const j = await r.json();
    return (j?.response || '');
  }
  return r.body;
}

function stripAutoIntro(text) {
  return (text || '').replace(/(^|\n)\s*je suis\s+(aurion|kronos).*?créé par rapido\.?/gi, '').trim();
}
function denoise(text) {
  return (text || '')
    .replace(/\b(nous sommes|je suis content|j'espère que|avez-vous des questions\??)\b.*$/gmi, '')
    .replace(/\n{3,}/g, '\n\n')
    .replace(/[ \t]+$/gm, '')
    .trim();
}
function tidy(text) {
  return (text || '').replace(/\n{3,}/g, '\n\n').replace(/[ \t]+$/gm, '').trim();
}
function seemsCut(text) {
  const t = (text || '').trim();
  return !!t && !/[.!?…]$/.test(t);
}
async function ensureComplete(base, sys, model) {
  let reply = tidy(stripAutoIntro(base));
  if (!seemsCut(reply)) return reply;
  const cont = await callOllama(`Termine la dernière réponse sans répéter le début. Conclus proprement en 1–2 phrases.`, sys, false, model, { num_predict: 200, temperature: 0.3 });
  reply = tidy(stripAutoIntro(`${reply} ${cont}`));
  return reply;
}

// ──────────────────────────────────────────────────────────────────────────────
// APNs push (inchangé, mais robuste)

function decodeAPNSKey() {
  const pem = Buffer.from(APNS_PRIVATE_KEY_BASE64 || '', 'base64').toString();
  if (!pem || !pem.includes('BEGIN PRIVATE KEY')) throw new Error('APNs clé .p8 invalide ou absente (APNS_PRIVATE_KEY_BASE64)');
  return pem;
}
async function sendNotification(token, title, body) {
  if (!APNS_TEAM_ID || !APNS_KEY_ID || !APNS_PRIVATE_KEY_BASE64 || !APNS_BUNDLE_ID) throw new Error('APNs config manquante');
  if (!/^[0-9a-f]{64}$/.test(token)) throw new Error('Device token invalide (64 hex)');
  const keyPEM = decodeAPNSKey();
  const nowSec = Math.floor(Date.now() / 1000);
  const jwtToken = jwt.sign({ iat: nowSec }, keyPEM, { algorithm: 'ES256', issuer: APNS_TEAM_ID, header: { alg: 'ES256', kid: APNS_KEY_ID }, expiresIn: '50m' });
  const host = APNS_SANDBOX ? 'https://api.sandbox.push.apple.com' : 'https://api.push.apple.com';
  const resp = await fetch(`${host}/3/device/${token}`, {
    method: 'POST',
    headers: { authorization: `bearer ${jwtToken}`, 'apns-topic': APNS_BUNDLE_ID, 'apns-push-type': 'alert', 'apns-priority': '10', 'content-type': 'application/json' },
    body: JSON.stringify({ aps: { alert: { title, body }, sound: 'default', 'thread-id': 'aurion' } })
  });
  if (!resp.ok) throw new Error(`APNs ${resp.status} ${await resp.text().catch(()=>'')}`.trim());
  return { status: resp.status };
}

// ──────────────────────────────────────────────────────────────────────────────
// Health / Models / Intents debug

app.get('/health', (_req, res) => {
  const sessionsCount = db.prepare('SELECT COUNT(*) AS c FROM sessions').get().c;
  res.json({
    ok: true,
    models: MODELS,
    ctx: LLM_NUM_CTX, predict: LLM_NUM_PREDICT,
    tavily: !!TAVILY_API_KEY,
    sessions: sessionsCount,
    apns: { team: !!APNS_TEAM_ID, key: !!APNS_KEY_ID, bundle: !!APNS_BUNDLE_ID, p8: !!APNS_PRIVATE_KEY_BASE64, sandbox: !!APNS_SANDBOX }
  });
});
app.get('/models', (_req, res) => res.json({ ok: true, models: MODELS }));
app.get('/intent_debug', (req, res) => {
  const q = String(req.query.q || '');
  res.json({ ok: true, input: q,
    math: !!handleMath(q),
    translate: /(traduis|traduire|translate)\b/i.test(q),
    time: /(quelle\s+heure|quelle\s+date|aujourd'hui|maintenant|time|heure)/i.test(q),
    business: /\b(plan|go-to-market|lancement|roadmap|business|produit)\b/i.test(q),
    weather: /(météo|meteo|quel\s+temps|pluie|ensoleillé|ensoleille|neige|vent|prévisions|previsions)/i.test(q),
    research: looksLikeResearch(q),
    class: classifyIntent(q)
  });
});

// APNs & devices
app.get('/apns/debug', (_req, res) => {
  const cfg = { team: !!APNS_TEAM_ID, key: !!APNS_KEY_ID, bundle: !!APNS_BUNDLE_ID, p8: !!APNS_PRIVATE_KEY_BASE64, sandbox: !!APNS_SANDBOX };
  try {
    const keyPEM = Buffer.from(APNS_PRIVATE_KEY_BASE64 || '', 'base64').toString();
    const token = (APNS_TEAM_ID && APNS_KEY_ID && keyPEM) ? jwt.sign({}, keyPEM, { algorithm: 'ES256', issuer: APNS_TEAM_ID, header: { alg: 'ES256', kid: APNS_KEY_ID }, expiresIn: '5m' }) : null;
    res.json({ ok: true, config: cfg, jwt_sample_segments: token ? token.split('.').map(s => s.length) : null });
  } catch (e) { res.json({ ok: false, config: cfg, error: e.message }); }
});
app.get('/apns/whoami', (_req, res) => res.json({ ok: true, topic: APNS_BUNDLE_ID, sandbox: !!APNS_SANDBOX, team: APNS_TEAM_ID || null, keyId: APNS_KEY_ID || null, p8_present: !!APNS_PRIVATE_KEY_BASE64 }));
app.get('/devices', (_req, res) => {
  try {
    const rows = db.prepare('SELECT rowid, token, last_seen FROM devices ORDER BY last_seen DESC').all();
    res.json({ ok: true, count: rows.length, devices: rows.map(r => ({ id: r.rowid, token: r.token, last_seen: r.last_seen })) });
  } catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});
app.post('/register-device', (req, res) => {
  const { token } = req.body || {};
  if (!token) return res.status(400).json({ ok: false, error: 'token requis' });
  try {
    db.prepare(`INSERT INTO devices (token, last_seen) VALUES (?, CURRENT_TIMESTAMP)
      ON CONFLICT(token) DO UPDATE SET last_seen=CURRENT_TIMESTAMP`).run(token);
    res.json({ ok: true });
  } catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});
app.post('/notify', async (req, res) => {
  const { token, title, body } = req.body || {};
  console.log('[APNs try]', { topic: APNS_BUNDLE_ID, sandbox: !!APNS_SANDBOX, tokenLen: (token || '').length, tokenHead: (token || '').slice(0,8) });
  if (!token || !title || !body) return res.status(400).json({ ok: false, error: 'token,title,body requis' });
  try { const out = await sendNotification(token, title, body); console.log('[APNs ok]', out.status); res.json({ ok: true, status: out.status }); }
  catch (e) { console.error('[APNs err]', e.message); res.status(500).json({ ok: false, error: e.message }); }
});
app.post('/notify/all', async (req, res) => {
  const { title, body } = req.body || {};
  if (!title || !body) return res.status(400).json({ ok: false, error: 'title,body requis' });
  const rows = db.prepare('SELECT token FROM devices ORDER BY last_seen DESC').all();
  const results = [];
  for (const r of rows) {
    try { const out = await sendNotification(r.token, title, body); results.push({ token: r.token, status: out.status }); }
    catch (e) { results.push({ token: r.token, error: e.message }); }
  }
  res.json({ ok: true, sent: results.length, results });
});

// Feedback / history
app.post('/feedback', (req, res) => {
  const { question, correct_answer } = req.body || {};
  if (!question || !correct_answer) return res.status(400).json({ ok: false, error: 'question,correct_answer requis' });
  factUpsert(question, correct_answer);
  res.json({ ok: true, id: db.prepare('SELECT last_insert_rowid() AS id').get().id });
});
app.get('/history', (_req, res) => {
  const rows = db.prepare(`SELECT id, session_id, user_id, role, content, style, created_at FROM history ORDER BY id DESC LIMIT 200`).all();
  res.json({ ok: true, items: rows });
});
app.post('/history/clear', (_req, res) => { db.exec('DELETE FROM history; VACUUM;'); res.json({ ok: true }); });

// ──────────────────────────────────────────────────────────────────────────────
// Core: /aurion (sync) — intents -> mémoire -> LLM (+ contexte + finisher)

app.post('/aurion', async (req, res) => {
  const { prompt, style = 'genz', user_id = 'rapido', model, session_id, response_length = 'medium' } = req.body || {};
  if (!prompt) return res.status(400).json({ ok: false, error: 'prompt requis' });
  const chosen = chooseModel(model, style);

  const lenCtl = lengthConstraint(response_length);

  // Mémoire / identité
  const mem = factLookup(prompt);
  if (mem) { pushHistory(session_id, user_id, 'user', prompt, style); pushHistory(session_id, user_id, 'assistant', mem, style); return res.json({ reply: mem, meta: { mode: 'fact', source: 'user-correction', model: chosen } }); }
  const idt = handleIdentity(prompt, style);
  if (idt) { pushHistory(session_id, user_id, 'user', prompt, style); pushHistory(session_id, user_id, 'assistant', idt.reply, style); return res.json({ reply: idt.reply, meta: { ...idt.meta, model: chosen } }); }

  // Intents rapides (ordre optimisé)
  const t = handleTranslate(prompt); if (t) {
    if (t.reply !== null) { pushHistory(session_id, user_id, 'user', prompt, style); pushHistory(session_id, user_id, 'assistant', t.reply, style); return res.json({ reply: t.reply, meta: { ...t.meta, model: chosen } }); }
    const text = await callOllama(t.meta.prompt, t.meta.sys, false, chosen, { temperature: 0.2, num_predict: 160 });
    const clean = tidy(stripAutoIntro(text)); pushHistory(session_id, user_id, 'user', prompt, style); pushHistory(session_id, user_id, 'assistant', clean, style);
    return res.json({ reply: clean, meta: { intent: 'translate', dir: t.meta.dir, llm: true, model: chosen } });
  }
  const tm = handleTime(prompt); if (tm) { pushHistory(session_id, user_id, 'user', prompt, style); pushHistory(session_id, user_id, 'assistant', tm.reply, style); return res.json({ reply: tm.reply, meta: { ...tm.meta, model: chosen } }); }
  const w = await handleWeather(prompt); if (w) { pushHistory(session_id, user_id, 'user', prompt, style); pushHistory(session_id, user_id, 'assistant', w.reply, style); return res.json({ reply: w.reply, meta: { ...w.meta, model: chosen } }); }
  const conv = handleConvert(prompt); if (conv) { pushHistory(session_id, user_id, 'user', prompt, style); pushHistory(session_id, user_id, 'assistant', conv.reply, style); return res.json({ reply: conv.reply, meta: { ...conv.meta, model: chosen } }); }
  const m = handleMath(prompt); if (m) { pushHistory(session_id, user_id, 'user', prompt, style); pushHistory(session_id, user_id, 'assistant', m.reply, style); return res.json({ reply: m.reply, meta: { ...m.meta, model: chosen } }); }
  const rsch = await handleResearch(prompt); if (rsch) { pushHistory(session_id, user_id, 'user', prompt, style); pushHistory(session_id, user_id, 'assistant', rsch.reply, style); return res.json({ reply: rsch.reply, meta: { ...rsch.meta, model: chosen } }); }

  // LLM + contraintes
  const sys = styleSystem(style, 'Rapido');
  const context = renderContext(pullRecentHistory(session_id, 6));
  const enforce = mustBeTwoSentences(prompt) ? `\nContraintes: réponds en exactement 2 phrases, sans politesses finales.` : '';
  const biz = businessScaffold(prompt);
  const finalPrompt = `${context}Réponds clairement et sans détour.${lenCtl.txt}${enforce}${biz}\nQuestion: ${prompt}`;

  const intentGuess = classifyIntent(prompt);
  const opts = chooseOptions(chosen, intentGuess, lenCtl.mult);
  const raw = await callOllama(finalPrompt, sys, false, chosen, opts);
  const branded = raw.replace(/Gemma/gi, brandName(style)).replace(/\bPhi\b/gi, brandName(style));
  const completed = await ensureComplete(branded, sys, chosen);
  const reply = denoise(tidy(stripAutoIntro(completed)));

  pushHistory(session_id, user_id, 'user', prompt, style);
  pushHistory(session_id, user_id, 'assistant', reply, style);
  res.json({ reply, meta: { mode: 'llm', model: chosen, intent: intentGuess, length: response_length } });
});

// ──────────────────────────────────────────────────────────────────────────────
// Stream: /aurion_stream — buffer optionnel + finisher

app.post('/aurion_stream', async (req, res) => {
  const { prompt, style = 'genz', user_id = 'rapido', buffer = false, model, session_id, response_length = 'medium' } = req.body || {};
  if (!prompt) return res.status(400).json({ ok: false, error: 'prompt requis' });
  const chosen = chooseModel(model, style);
  const lenCtl = lengthConstraint(response_length);
  const sys = styleSystem(style, 'Rapido');
  const context = renderContext(pullRecentHistory(session_id, 6));
  const enforce = mustBeTwoSentences(prompt) ? `\nContraintes: réponds en exactement 2 phrases, sans politesses finales.` : '';
  const biz = businessScaffold(prompt);
  const finalPrompt = `${context}Réponds brièvement et clairement.${lenCtl.txt}${enforce}${biz}\nQuestion: ${prompt}`;

  // intents avant stream (ordre optimisé)
  const t = handleTranslate(prompt); if (t) {
    if (t.reply !== null) return res.json({ reply: t.reply, meta: { ...t.meta, model: chosen } });
    const text = await callOllama(t.meta.prompt, t.meta.sys, false, chosen, { temperature: 0.2, num_predict: 160 });
    return res.json({ reply: tidy(stripAutoIntro(text)), meta: { intent: 'translate', dir: t.meta.dir, llm: true, model: chosen } });
  }
  const tm = handleTime(prompt); if (tm) return res.json({ reply: tm.reply, meta: { ...tm.meta, model: chosen } });
  const w = await handleWeather(prompt); if (w) return res.json({ reply: w.reply, meta: { ...w.meta, model: chosen } });
  const conv = handleConvert(prompt); if (conv) return res.json({ reply: conv.reply, meta: { ...conv.meta, model: chosen } });
  const m = handleMath(prompt); if (m) return res.json({ reply: m.reply, meta: { ...m.meta, model: chosen } });
  const rsch = await handleResearch(prompt); if (rsch) return res.json({ reply: rsch.reply, meta: { ...rsch.meta, model: chosen } });

  try {
    const stream = await callOllama(finalPrompt, sys, true, chosen, chooseOptions(chosen, classifyIntent(prompt), lenCtl.mult));
    if (!stream || buffer) {
      const text = await callOllama(finalPrompt, sys, false, chosen, chooseOptions(chosen, classifyIntent(prompt), lenCtl.mult));
      const completed = await ensureComplete(text, sys, chosen);
      const clean = denoise(tidy(stripAutoIntro(completed)));
      pushHistory(session_id, user_id, 'user', prompt, style); pushHistory(session_id, user_id, 'assistant', clean, style);
      return res.json({ reply: clean, meta: { mode: 'llm', buffered: true, model: chosen } });
    }
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    const reader = stream.getReader(); let acc = '';
    while (true) {
      const { done, value } = await reader.read(); if (done) break;
      const chunk = Buffer.from(value).toString('utf8');
      for (const line of chunk.split('\n')) {
        if (!line.trim()) continue;
        try {
          const j = JSON.parse(line);
          const piece = (j.response || '');
          if (piece) { acc += piece; res.write(piece.replace(/Gemma/gi, brandName(style)).replace(/\bPhi\b/gi, brandName(style))); }
        } catch {}
      }
    }
    const completed = await ensureComplete(acc, sys, chosen);
    const clean = denoise(tidy(stripAutoIntro(completed)));
    pushHistory(session_id, user_id, 'user', prompt, style);
    pushHistory(session_id, user_id, 'assistant', clean, style);
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
    } catch (e) { console.warn('Suggestion push failed:', e.message); }
  }, 1000 * 60 * 30);
} else if (ENABLE_SUGGESTIONS) {
  console.warn('Suggestions loop OFF (APNs non configuré)');
}

// ──────────────────────────────────────────────────────────────────────────────
// Global error handler + listen

app.use((err, _req, res, _next) => { console.error('Unhandled:', err); res.status(500).json({ ok: false, error: 'server_error' }); });

app.listen(PORT, () => {
  console.log(`Aurion proxy up on http://localhost:${PORT}`);
  console.log(`Models: primary=${MODELS.primary} secondary=${MODELS.secondary} | ctx=${LLM_NUM_CTX} predict=${LLM_NUM_PREDICT}`);
  console.log(`Embeddings: ${EMBED_MODEL} | Tavily: ${TAVILY_API_KEY ? 'on' : 'off'}`);
});
