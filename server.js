// server.js — Aurion Proxy (Ollama only, ESM, Node 18+)
// ──────────────────────────────────────────────────────────────────────────────

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import Database from 'better-sqlite3';
import jwt from 'jsonwebtoken';

// ──────────────────────────────────────────────────────────────────────────────
// ENV & Config

const PORT = Number(process.env.PORT || 3000);

// Ollama
const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://127.0.0.1:11434';
const AURION_MODEL_PRIMARY   = process.env.AURION_MODEL_PRIMARY   || 'aurion-gemma';
const AURION_MODEL_SECONDARY = process.env.AURION_MODEL_SECONDARY || 'aurion-phi';


// Embeddings + Web search (optionnel)
const EMBED_MODEL    = process.env.EMBED_MODEL || 'nomic-embed-text';
const TAVILY_API_KEY = (process.env.TAVILY_API_KEY || '').trim();

// APNs (optionnel)
const APNS_TEAM_ID            = process.env.APNS_TEAM_ID || '';
const APNS_KEY_ID             = process.env.APNS_KEY_ID || '';
const APNS_BUNDLE_ID          = process.env.APNS_BUNDLE_ID || '';
const APNS_PRIVATE_KEY_BASE64 = (process.env.APNS_PRIVATE_KEY_BASE64 || '').trim();
const APNS_SANDBOX            = String(process.env.APNS_SANDBOX).toLowerCase() === 'true';
const ENABLE_SUGGESTIONS      = String(process.env.ENABLE_SUGGESTIONS || 'false').toLowerCase() === 'true';

// LLM tuning
const LLM_NUM_CTX     = Number(process.env.LLM_NUM_CTX || 8192);
const LLM_NUM_PREDICT = Number(process.env.LLM_NUM_PREDICT || 1024);
const LLM_TEMPERATURE = Number(process.env.LLM_TEMPERATURE || 0.5);

// Modèles nommés
const MODELS = {
  primary:   AURION_MODEL_PRIMARY,
  secondary: AURION_MODEL_SECONDARY,
  gemma:     AURION_MODEL_PRIMARY,   // alias
  phi:       AURION_MODEL_SECONDARY, // alias
};
function chooseModel(hint, style) {
  if (hint && MODELS[hint]) return MODELS[hint];
  if (style === 'kronos') return MODELS.secondary;
  return MODELS.primary;
}

// ──────────────────────────────────────────────────────────────────────────────
const app = express();
// Log cible Ollama (utile en debug)
console.log(`[BOOT] OLLAMA_HOST = ${OLLAMA_HOST}`);

// Ping des modèles Ollama
app.get('/ollama/health', async (_req, res) => {
  try {
    const r = await fetch(`${OLLAMA_HOST}/api/tags`);
    const data = await r.json().catch(() => null);
    if (!r.ok) {
      return res.status(502).json({ ok: false, status: r.status, host: OLLAMA_HOST, error: 'upstream_not_ok' });
    }
    return res.json({ ok: true, host: OLLAMA_HOST, models: data?.models ?? data });
  } catch (e) {
    return res.status(500).json({ ok: false, host: OLLAMA_HOST, error: String(e?.message || e) });
  }
});
app.use(cors({ origin: '*', credentials: false }));
app.use(express.json({ limit: '1mb' }));
app.use((req, _res, next) => { console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`); next(); });

// timeouts fetch
const FETCH_TIMEOUT_MS = 55000;
async function timedFetch(url, options = {}) {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try { return await fetch(url, { ...options, signal: controller.signal }); }
  finally { clearTimeout(id); }
}

// graceful shutdown
function onExit(db) { try { db.close(); } catch {} process.exit(0); }
process.on('SIGINT', () => onExit(db));
process.on('SIGTERM', () => onExit(db));

// ──────────────────────────────────────────────────────────────────────────────
// SQLite

const db = new Database('aurion.db');
db.pragma('journal_mode = WAL');

db.exec(`
CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  q TEXT, a TEXT,
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
  id INTEGER PRIMARY KEY AUTOINCREMENT,
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

// ──────────────────────────────────────────────────────────────────────────────
// Styles & prompts — Personnalités boostées

const STYLES = {
  genz: {
    name: 'Aurion',
    system: [
      "Parle en français moderne, punchy, sans phrases creuses.",
      "Humour léger, complicité, métaphores courtes si utiles.",
      "Vocabulaire simple, concret; évite le jargon corporate.",
      "Ne te présentes JAMAIS spontanément.",
      "Pas de 'Je suis ...' sauf question explicite d'identité.",
      "Priorité à la clarté, puis au style."
    ].join('\n'),
    tone: {
      openers: ["Okay !", "Banco.", "On fait simple :", "En clair :"],
      closers: ["Ça te va ?", "Tu veux la version turbo ?", "On avance ?"],
      slang: [
        ["très", "grave"], ["vraiment", "franchement"], ["d’accord", "ok"], ["rapide", "flash"],
        ["important", "clé"], ["astuce", "hack"], ["erreur", "boulette"]
      ],
      emoji: ["⚡️","😅","🧠","✨","📌"],
      maxEmoji: 2
    }
  },
  pro: {
    name: 'Aurion',
    system: [
      "Style professionnel, net, orienté solution et risques.",
      "Expose d’abord l’idée clé, ensuite les étapes actionnables.",
      "Ne te présentes JAMAIS spontanément.",
      "Pas de blabla : chiffres, critères, décisions."
    ].join('\n'),
    tone: {
      openers: ["Synthèse :", "À retenir :", "Concrètement :"],
      closers: ["Si besoin je détaille points & risques.", "Besoin d’un chiffrage ?"],
      slang: [],
      emoji: ["📈","🧩","✅"],
      maxEmoji: 1
    }
  },
  zen: {
    name: 'Aurion',
    system: [
      "Voix posée, rassurante, structurée.",
      "Phrases courtes, souffle calme. On retire le superflu.",
      "Ne te présentes JAMAIS spontanément."
    ].join('\n'),
    tone: {
      openers: ["Doucement :", "Pas à pas :", "L’essentiel :"],
      closers: ["Respire. C’est gérable.", "On y va étape par étape."],
      slang: [],
      emoji: ["🌿","🪷","💡"],
      maxEmoji: 1
    }
  },
  kronos: {
    name: 'Kronos',
    system: [
      "Tu es Kronos, l’alter sombre d’Aurion (créé par Rapido).",
      "Cinglant mais utile. Ironie fine. Jamais gratuit.",
      "Ne te présentes JAMAIS spontanément.",
      "Toujours pertinent, factuel, tranchant si nécessaire."
    ].join('\n'),
    tone: {
      openers: ["Tranchons :", "Sans fioritures :", "Version brute :"],
      closers: ["On coupe le bruit.", "Assez de fumée, place aux faits."],
      slang: [["problème","faiblesse"],["mauvais","fragile"],["risque","angle mort"]],
      emoji: ["🗡️","🕯️","♟️"],
      maxEmoji: 1
    }
  }
};

const brandName  = (k='genz') => (STYLES[k] || STYLES.genz).name;

function styleSystem(style='genz', userName='Rapido') {
  const s = STYLES[style] || STYLES.genz;
  const who = (style === 'kronos')
    ? `Tu es Kronos, l’alter sombre de ${userName}. Ne te présentes PAS.`
    : `Tu es ${brandName(style)}, créé par ${userName}. Ne te présentes PAS.`;
  const guard = `Interdit: 'Je suis ...' ou 'Bonjour' en intro. Réponds directement. Termine proprement.`;
  return `${s.system}\n${who}\n${guard}`;
}

// ──────────────────────────────────────────────────────────────────────────────
// Intents (météo / heure / maths / conversion / traduction / recherche)

const WMO = {0:'ciel clair',1:'peu nuageux',2:'partiellement nuageux',3:'couvert',45:'brouillard',48:'brouillard givrant',
51:'bruine faible',53:'bruine',55:'bruine forte',56:'bruine verglaçante',57:'bruine verglaçante forte',
61:'pluie faible',63:'pluie modérée',65:'pluie forte',66:'pluie verglaçante',67:'pluie verglaçante forte',
71:'neige faible',73:'neige modérée',75:'neige forte',77:'grains de neige',
80:'averses faibles',81:'averses',82:'averses fortes',85:'averses de neige faibles',86:'averses de neige fortes',
95:'orages',96:'orages avec grêle',99:'orages violents avec grêle'};

async function handleWeather(text) {
  const q = (text || '').toLowerCase();
  if (!/(météo|meteo|quel\s+temps|pluie|ensoleillé|neige|vent|prévisions)/i.test(q)) return null;
  const lat = 48.8566, lon = 2.3522; // Paris par défaut
  try {
    const r = await timedFetch(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current_weather=true`);
    const j = await r.json();
    const cw = j.current_weather;
    if (!cw) return { reply: "Météo indisponible.", meta: { intent: 'weather', ok: false } };
    const label = WMO[Number(cw.weathercode)] || `conditions (${cw.weathercode})`;
    return { reply: `À Paris: ${cw.temperature}°C, vent ${cw.windspeed} km/h, ${label}.`, meta: { intent: 'weather', ok: true } };
  } catch {
    return { reply: "Météo indisponible (réseau).", meta: { intent: 'weather', ok: false } };
  }
}

const CITY_TZ = { 'paris':'Europe/Paris','new york':'America/New_York','nyc':'America/New_York','tokyo':'Asia/Tokyo','londres':'Europe/London','dubai':'Asia/Dubai','los angeles':'America/Los_Angeles','la':'America/Los_Angeles' };
function extractCity(text) { const q=(text||'').toLowerCase(); for (const k of Object.keys(CITY_TZ)) if (q.includes(k)) return { city:k, tz:CITY_TZ[k] }; return { city:'paris', tz:'Europe/Paris' }; }
function handleTime(text) {
  if (!/(quelle\s+heure|quelle\s+date|aujourd'hui|maintenant|time|heure)/i.test(text || '')) return null;
  const { city, tz } = extractCity(text);
  try {
    const d = new Date();
    const fmtDate = new Intl.DateTimeFormat('fr-FR', { timeZone: tz, dateStyle: 'full' }).format(d);
    const fmtTime = new Intl.DateTimeFormat('fr-FR', { timeZone: tz, timeStyle: 'medium' }).format(d);
    return { reply: `Nous sommes le ${fmtDate}, il est ${fmtTime} (${tz}).`, meta: { intent: 'time', ok: true, city, tz } };
  } catch { return { reply: `Heure locale indisponible.`, meta: { intent: 'time', ok: false } }; }
}

const MATH_WORD_GUARDS = /(siècle|siecle|année|annee|date|vers\s+\d{3,4}|en\s+\d{3,4}|révolution|histoire|qui|quelle|quand|où|ou|prix|combien)/i;
function handleMath(text) {
  const raw = (text || '');
  if (!/[0-9][0-9\+\-\*\/\.\(\)\s]+/.test(raw)) return null;
  const hasOp = /[\+\-\*\/]/.test(raw) && /[0-9]/.test(raw);
  const looksLikeYearOnly = /^\s*\d{3,4}\s*$/.test(raw);
  if (!hasOp || looksLikeYearOnly || MATH_WORD_GUARDS.test(raw)) return null;
  try {
    const safe = raw.replace(/[^0-9+\-*/().\s]/g, '');
    const val = Function(`"use strict"; return (${safe});`)();
    if (Number.isFinite(val)) return { reply: `Résultat: ${val}`, meta: { intent: 'math', ok: true } };
  } catch {}
  return null;
}

const UNIT_FACTORS = { km:1000, m:1, cm:0.01, mi:1609.344, miles:1609.344, ft:0.3048, pied:0.3048, pieds:0.3048, in:0.0254, pouce:0.0254, pouces:0.0254, kg:1, g:0.001, lb:0.45359237, lbs:0.45359237, oz:0.028349523125 };
function handleConvert(text) {
  const q=(text||'').toLowerCase();
  const temp = q.match(/([-+]?\d+(?:[.,]\d+)?)\s*°?\s*(c|celsius|f|fahrenheit)\s*(?:en|to|vers)\s*(c|celsius|f|fahrenheit)/i);
  if (temp) {
    let v=parseFloat(temp[1].replace(',','.')); const from=temp[2][0].toLowerCase(), to=temp[3][0].toLowerCase();
    if (from===to) return { reply:`${v.toFixed(2)} °${from.toUpperCase()}`, meta:{ intent:'convert', kind:'temp' } };
    const out = (from==='c') ? (v*9/5+32) : ((v-32)*5/9); return { reply:`${out.toFixed(2)} ${to==='c'?'°C':'°F'}`, meta:{ intent:'convert', kind:'temp' } };
  }
  const m = q.match(/([-+]?\d+(?:[.,]\d+)?)\s*([a-zéû]+)\s*(?:en|to|vers)\s*([a-zéû]+)/i);
  if (!m) return null;
  const val=parseFloat(m[1].replace(',','.')), from=m[2], to=m[3]; const fm=UNIT_FACTORS[from], tm=UNIT_FACTORS[to];
  if (!fm || !tm) return null;
  const isMass = ['kg','g','lb','lbs','oz'].includes(from) || ['kg','g','lb','lbs','oz'].includes(to);
  const map = isMass ? { kg:1,g:0.001,lb:0.45359237,lbs:0.45359237,oz:0.028349523125 } : { km:1000,m:1,cm:0.01,mi:1609.344,miles:1609.344,ft:0.3048,pied:0.3048,pieds:0.3048,in:0.0254,pouce:0.0254,pouces:0.0254 };
  const base = val * map[from]; const out = base / map[to];
  return { reply: `${val} ${from} ≈ ${out.toFixed(4)} ${to}`, meta: { intent:'convert', ok:true } };
}

const simpleDict = {'bonjour le monde':'hello world','comment ça va ?':'how are you?','je t’aime':'i love you','bonne nuit':'good night','au revoir':'goodbye'};
function handleTranslate(text) {
  const m = text.match(/(?:traduis|traduire|translate)\s+(.+?)\s+(?:en|to)\s+(anglais|english|français|french)/i);
  if (!m) return null;
  const phrase = m[1].trim(); const target = m[2].toLowerCase(); const toEN = target.startsWith('ang') || target.startsWith('eng');
  const key = phrase.toLowerCase();
  if (toEN && simpleDict[key]) return { reply:`EN: ${simpleDict[key]}`, meta:{ intent:'translate', dir:'fr->en', dict:true } };
  if (!toEN && Object.values(simpleDict).includes(key)) {
    const fr = Object.entries(simpleDict).find(([,en])=>en===key)?.[0];
    if (fr) return { reply:`FR: ${fr}`, meta:{ intent:'translate', dir:'en->fr', dict:true } };
  }
  const sys = `Traducteur: retourne UNIQUEMENT la traduction ${toEN?'en anglais':'en français'}, sans guillemets ni explications.`;
  return { reply:null, meta:{ intent:'translate_llm', sys, prompt:`Traduis: ${phrase}`, dir: toEN?'fr->en':'en->fr' } };
}

// Recherche web + cache (6h)
const CACHE_TTL_MS = 6 * 60 * 60 * 1000;
const looksLikeResearch = (t) => /(aujourd'hui|dernier|dernière|actualité|news|prix|coût|tarif|programme|horaire|score|bourse|loi|décret|2024|2025)/i.test(t||'');
const cacheGet = (k) => { const r=db.prepare('SELECT answer,created_at FROM cache WHERE key=?').get(k); if(!r) return null; if(Date.now()-Number(r.created_at)>CACHE_TTL_MS){db.prepare('DELETE FROM cache WHERE key=?').run(k);return null;} return r.answer; };
const cacheSet = (k,a) => db.prepare('INSERT OR REPLACE INTO cache (key,answer,created_at) VALUES (?,?,?)').run(k,a,Date.now());

// JARVIS intents (web & images)
const IMG_INTENT_RE = /\b(image|photo|illustration|affiche[-\s]?moi|montre[-\s]?moi|à quoi (ça|il|elle) ressemble)\b/i;
const WEB_INTENT_RE = /\b(cherche|recherche|trouve|actualités?|news|prix|me (donne|fais) des liens|sources?)\b/i;
const EXPLAIN_RE    = /\b(explique|c'est quoi|définis?|definition|tuto|comment)\b/i;
const wantsImages  = (p) => IMG_INTENT_RE.test(p||'');
const wantsWeb     = (p) => WEB_INTENT_RE.test(p||'') || looksLikeResearch(p);
const wantsExplain = (p) => EXPLAIN_RE.test(p||'');

// Tavily web search → { summary, links: [{title,url}] }
async function webSearch(query, max = 5) {
  if (!TAVILY_API_KEY) return null;
  try {
    const resp = await timedFetch('https://api.tavily.com/search', {
      method: 'POST',
      headers: { 'Content-Type':'application/json', 'Authorization':`Bearer ${TAVILY_API_KEY}` },
      body: JSON.stringify({ query, search_depth:'advanced', max_results: max })
    });
    if (!resp.ok) throw new Error(`Tavily ${resp.status}`);
    const j = await resp.json();
    const links = (j?.results || []).slice(0, max).map(r => ({ title: r.title || r.url, url: r.url }));
    return { summary: j?.answer || null, links };
  } catch (e) {
    console.warn('[webSearch]', e.message);
    return null;
  }
}

// DuckDuckGo images (public i.js)
async function imageSearch(query, max = 6) {
  const u = new URL('https://duckduckgo.com/i.js');
  u.searchParams.set('q', query);
  u.searchParams.set('o', 'json');
  u.searchParams.set('l', 'fr-fr');
  u.searchParams.set('p', '1');
  try {
    const r = await timedFetch(u.toString(), {
      headers: {
        'user-agent': 'Mozilla/5.0',
        'accept': 'application/json,text/plain,*/*',
        'referer': 'https://duckduckgo.com/'
      }
    });
    if (!r.ok) throw new Error(`DDG ${r.status}`);
    const j = await r.json();
    const items = (j?.results || []).slice(0, max).map(it => ({
      title: it.title || '',
      image: it.image,
      thumbnail: it.thumbnail,
      url: it.url,
      source: it.source || ''
    }));
    return items;
  } catch (e) {
    console.warn('[imageSearch]', e.message);
    return [];
  }
}

// Identité
function handleIdentity(text, style='genz') {
  if (!/^\s*(qui\s+es[-\s]?tu|tu\s+es\s+qui|qui\s+êtes[-\s]?vous)\s*\??$/i.test(text||'')) return null;
  if (style==='kronos') return { reply:"Je suis Kronos, l’alter sombre façonné par Rapido. Pose ta question.", meta:{ intent:'identity', style } };
  return { reply:"Je suis Aurion, assistant conçu par Rapido. Dis-moi ce dont tu as besoin.", meta:{ intent:'identity', style } };
}

// Helpers divers
const classifyIntent = (p) => (handleMath(p)?'math' : /(traduis|traduire|translate)\b/i.test(p)?'translate' : /(heure|date|maintenant|time)/i.test(p)?'time' : /(météo|meteo|pluie|prévisions)/i.test(p)?'weather' : looksLikeResearch(p)?'research' : 'general');
const mustBeTwoSentences = (p) => /(\b2\s*phrases?\b|\bdeux\s*phrases?\b)/i.test(p||'');
function businessScaffold(p){const ok=/\b(plan|go-to-market|lancement|roadmap|business|produit)\b/i.test(p||''); if(!ok) return ''; return `\nFormat:\n1) Cible & valeur\n2) Proposition & différenciation\n3) Messages clés\n4) Canaux & calendrier (J-30 → J+30)\n5) KPI\n`; }
function lengthConstraint(len='medium'){const L=String(len||'medium').toLowerCase(); if(L==='short')return{txt:'\nRéponds en 1–3 phrases.',mult:0.6}; if(L==='long')return{txt:'\nRéponse détaillée (8–12 phrases).',mult:1.4}; return{txt:'',mult:1.0};}

// Mémoire & Sessions
function factLookup(question){const q=(question||'').trim(); if(!q) return null; const row=db.prepare('SELECT a FROM facts WHERE q = ? ORDER BY id DESC LIMIT 1').get(q); return row?.a || null;}
function factUpsert(q,a){if(!(q||'').trim() || !(a||'').trim()) return; try{db.prepare('INSERT INTO facts (q,a) VALUES (?,?)').run(q.trim(), a.trim());}catch(e){console.warn('[facts.upsert] failed:',e.message);}}
function pushHistory(session_id,user_id,role,content,style='genz'){db.prepare('INSERT INTO history (session_id,user_id,role,content,style) VALUES (?,?,?,?,?)').run(session_id||null,user_id||'rapido',role,content,style);}
function pullRecentHistory(session_id,limit=6){ if(session_id){return db.prepare('SELECT role,content FROM history WHERE session_id=? ORDER BY id DESC LIMIT ?').all(session_id,limit).reverse();} return db.prepare('SELECT role,content FROM history ORDER BY id DESC LIMIT ?').all(limit).reverse(); }
function renderContext(history=[]){ if(!history.length) return ''; const lines=history.map(h=>`${h.role}: ${h.content}`.trim()); return `Contexte récent:\n${lines.join('\n')}\n---\n`; }

// ──────────────────────────────────────────────────────────────────────────────
// LLM (Ollama) + post-traitements

function chooseOptions(model,intent='general',lengthMult=1.0){
  const base = Math.round(LLM_NUM_PREDICT * Math.max(0.5, Math.min(2.0, lengthMult)));
  const opt = { num_ctx:LLM_NUM_CTX, num_predict:base, temperature:LLM_TEMPERATURE, top_p:0.9, top_k:50, repeat_penalty:1.1, repeat_last_n:256 };
  if (/phi/i.test(model)) opt.temperature = Math.min(opt.temperature, 0.35);
  if (intent==='business'||intent==='factual'||intent==='science') opt.temperature = Math.min(opt.temperature, 0.35);
  return opt;
}

async function callOllama(prompt, system, stream=false, modelName, options={}){
  const model = modelName || MODELS.primary;
  const body = { model, prompt, system, stream, options: { ...options } };
  const r = await timedFetch(`${OLLAMA_HOST}/api/generate`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
  if (!r.ok) {
    let detail = ''; try { detail = await r.text(); } catch {}
    throw new Error(`Ollama HTTP ${r.status}${detail ? ' — ' + detail.slice(0,200) : ''}`);
  }
  if (!stream) { const j = await r.json(); return (j?.response || ''); }
  return r.body;
}

// Nettoyages / finisher
function stripAutoIntro(text) {
  const t = (text || "").trim();
  if (!t) return t;
  if (/^\s*bonjour[ !\.\?]*$/i.test(t)) return t;

  let out = t;
  const before = out;
  out = out.replace(/(^|\n)\s*je\s+suis\s+(aurion|kronos|un assistant|l'assistant|assistant)[^.\n]*[.\n]?\s*/i, "$1");
  const afterJeSuis = out;
  const tmp = out.replace(/^\s*bonjour[.!?]?\s*/i, "");
  out = tmp.trim().length > 0 ? tmp : afterJeSuis;
  out = out.replace(/\n{3,}/g, "\n\n").trim();
  if (!out) return before || t;
  return out;
}
const tidy    = (t) => (t||'').replace(/\n{3,}/g,'\n\n').replace(/[ \t]+$/gm,'').trim();
const denoise = (t) => (t||'').replace(/\b(nous sommes|j'espère que|avez-vous des questions\??)\b.*$/gmi,'').trim();
const seemsCut = (t) => !!(t||'').trim() && !/[.!?…]$/.test((t||'').trim());
async function ensureComplete(base, sys, model){
  let reply = tidy(stripAutoIntro(base));
  if (!seemsCut(reply)) return reply;
  const cont = await callOllama(`Termine la dernière réponse sans répéter le début. Conclus en 1–2 phrases.`, sys, false, model, { num_predict: 200, temperature: 0.3 });
  return tidy(stripAutoIntro(`${reply} ${cont}`));
}

// Décorateur de personnalité
function clamp(n, a, b) { return Math.max(a, Math.min(b, n)); }
function sprinkleEmojis(text, palette=[], max=0) {
  if (!max || !palette.length) return text;
  let out = text;
  const sentences = out.split(/([.!?…])\s+/);
  let used = 0;
  for (let i = 0; i < sentences.length; i += 2) {
    if (used >= max) break;
    if ((sentences[i] || '').length > 12) {
      const e = palette[used % palette.length];
      sentences[i] = `${sentences[i]} ${e}`;
      used++;
    }
  }
  return sentences.join('');
}
function applySlang(text, pairs=[], intensity=1) {
  if (!pairs.length || intensity <= 0) return text;
  let out = text;
  const limit = clamp(intensity, 0, 3);
  for (const [from, to] of pairs.slice(0, 6)) {
    const r = new RegExp(`\\b${from}\\b`, 'gi');
    out = out.replace(r, (m) => (Math.random() < 0.35 * limit ? to : m));
  }
  return out;
}
function applyPersona(reply, style='genz', opts={}) {
  const { personality_level = 'medium', allow_emojis = true } = opts || {};
  const s = STYLES[style] || STYLES.genz;
  let out = String(reply || '');
  if (/^```[\s\S]+```$/m.test(out) || out.trim().length < 10) return out;

  const level = (personality_level === 'high') ? 3 : (personality_level === 'low' ? 1 : 2);
  const startsClean = !/^(bonjour|salut|hey|\s*je\s+suis)/i.test(out);
  if (startsClean) {
    const opener = (s.tone.openers || [])[ (out.length + level) % (s.tone.openers?.length || 1) ];
    if (opener) out = `${opener} ${out}`;
  }
  if (!(style === 'pro' && level === 1)) out = applySlang(out, s.tone.slang || [], level);
  if (allow_emojis && s.tone.emoji?.length) {
    const max = clamp((s.tone.maxEmoji || 0) + (level > 2 ? 1 : 0), 0, 3);
    out = sprinkleEmojis(out, s.tone.emoji, max);
  }
  if (out.length > 80 && (s.tone.closers?.length)) {
    const closer = s.tone.closers[(out.length + s.tone.closers.length) % s.tone.closers.length];
    if (closer && !out.trim().endsWith('?')) out += ` ${closer}`;
  }
  return out;
}

// ──────────────────────────────────────────────────────────────────────────────
// Health / config / debug

app.get('/health', async (_req, res) => {
  let backend = 'none';
  try { const r = await timedFetch(`${OLLAMA_HOST}/api/tags`); if (r.ok) backend = 'ollama'; } catch {}
  const sessionsCount = db.prepare('SELECT COUNT(*) AS c FROM sessions').get().c;
  res.json({ ok:true, backend, port:PORT, models:MODELS, ctx:LLM_NUM_CTX, predict:LLM_NUM_PREDICT, embeddings:EMBED_MODEL, tavily:!!TAVILY_API_KEY, sessions:sessionsCount });
});

app.get('/models', async (_req, res) => {
  try {
    const r = await timedFetch(`${OLLAMA_HOST}/api/tags`);
    const j = await r.json();
    res.json({ ok:true, configured: MODELS, installed: j?.models || [] });
  } catch (e) {
    res.json({ ok:false, configured: MODELS, error: String(e.message || e) });
  }
});

app.get('/intent_debug', (req,res) => {
  const q=String(req.query.q||'');
  res.json({ ok:true, input:q,
    math:!!handleMath(q),
    translate:/(traduis|traduire|translate)\b/i.test(q),
    time:/(quelle\s+heure|quelle\s+date|aujourd'hui|maintenant|time|heure)/i.test(q),
    weather:/(météo|meteo|pluie|prévisions)/i.test(q),
    research:looksLikeResearch(q),
    class: classifyIntent(q)
  });
});

app.get('/', (_req, res) => res.type('text/plain').send('Aurion (Ollama) OK'));

// ──────────────────────────────────────────────────────────────────────────────
// Endpoints Jarvis utiles

app.get('/web/search', async (req, res) => {
  const q = String(req.query.q || '').trim();
  if (!q) return res.status(400).json({ ok:false, error:'q requis' });
  const data = await webSearch(q, Number(req.query.max || 5));
  if (!data) return res.status(502).json({ ok:false, error:'search_unavailable' });
  res.json({ ok:true, ...data });
});

app.get('/web/images', async (req, res) => {
  const q = String(req.query.q || '').trim();
  if (!q) return res.status(400).json({ ok:false, error:'q requis' });
  const imgs = await imageSearch(q, Number(req.query.max || 6));
  res.json({ ok:true, count: imgs.length, images: imgs });
});

// Preview personnalité
app.get('/persona/preview', (req, res) => {
  const style = String(req.query.style || 'genz');
  const level = String(req.query.level || 'medium');
  const allow = String(req.query.emojis || 'true') !== 'false';
  const sample = "Voici comment je répondrai à une question simple en gardant le fond clair et utile.";
  const out = applyPersona(sample, style, { personality_level: level, allow_emojis: allow });
  res.json({ ok:true, style, level, allow_emojis: allow, sample: out });
});

// ──────────────────────────────────────────────────────────────────────────────
// APNs (optionnel)

function decodeAPNSKey(){
  const pem = Buffer.from(APNS_PRIVATE_KEY_BASE64 || '', 'base64').toString();
  if (!pem || !pem.includes('BEGIN PRIVATE KEY')) throw new Error('APNs clé .p8 invalide ou absente');
  return pem;
}
async function sendNotification(token,title,body){
  if (!APNS_TEAM_ID || !APNS_KEY_ID || !APNS_PRIVATE_KEY_BASE64 || !APNS_BUNDLE_ID) throw new Error('APNs config manquante');
  if (!/^[0-9a-f]{64}$/.test(token)) throw new Error('Device token invalide (64 hex)');
  const keyPEM=decodeAPNSKey(); const nowSec=Math.floor(Date.now()/1000);
  const jwtToken = jwt.sign({ iat: nowSec }, keyPEM, { algorithm:'ES256', issuer:APNS_TEAM_ID, header:{ alg:'ES256', kid:APNS_KEY_ID }, expiresIn:'50m' });
  const host = APNS_SANDBOX ? 'https://api.sandbox.push.apple.com' : 'https://api.push.apple.com';
  const resp = await timedFetch(`${host}/3/device/${token}`, {
    method:'POST',
    headers:{ authorization:`bearer ${jwtToken}`, 'apns-topic':APNS_BUNDLE_ID, 'apns-push-type':'alert', 'apns-priority':'10', 'content-type':'application/json' },
    body: JSON.stringify({ aps:{ alert:{ title, body }, sound:'default', 'thread-id':'aurion' } })
  });
  if (!resp.ok) throw new Error(`APNs ${resp.status} ${await resp.text().catch(()=> '')}`.trim());
  return { status: resp.status };
}

// Devices & notifications
app.get('/apns/whoami', (_req,res)=>res.json({ ok:true, topic:APNS_BUNDLE_ID, sandbox:!!APNS_SANDBOX, team:APNS_TEAM_ID||null, keyId:APNS_KEY_ID||null, p8_present:!!APNS_PRIVATE_KEY_BASE64 }));
app.get('/devices', (_req,res)=>{try{const rows=db.prepare('SELECT rowid,token,last_seen FROM devices ORDER BY last_seen DESC').all();res.json({ok:true,count:rows.length,devices:rows});}catch(e){res.status(500).json({ok:false,error:e.message});}});
app.post('/register-device',(req,res)=>{const {token}=req.body||{}; if(!token) return res.status(400).json({ok:false,error:'token requis'}); try{db.prepare('INSERT INTO devices (token,last_seen) VALUES (?,CURRENT_TIMESTAMP) ON CONFLICT(token) DO UPDATE SET last_seen=CURRENT_TIMESTAMP').run(token); res.json({ok:true});}catch(e){res.status(500).json({ok:false,error:e.message});}});
app.post('/notify', async (req,res)=>{const {token,title,body}=req.body||{}; if(!token||!title||!body) return res.status(400).json({ok:false,error:'token,title,body requis'}); try{const out=await sendNotification(token,title,body); res.json({ok:true,status:out.status});}catch(e){res.status(500).json({ok:false,error:e.message});}});
app.post('/notify/all', async (req,res)=>{const {title,body}=req.body||{}; if(!title||!body) return res.status(400).json({ok:false,error:'title,body requis'}); const rows=db.prepare('SELECT token FROM devices ORDER BY last_seen DESC').all(); const results=[]; for (const r of rows){try{const out=await sendNotification(r.token,title,body); results.push({token:r.token,status:out.status});}catch(e){results.push({token:r.token,error:e.message});}} res.json({ok:true,sent:results.length,results});});

// ──────────────────────────────────────────────────────────────────────────────
// Sessions & history endpoints

app.post('/session/start',(req,res)=>{const {session_id,title}=req.body||{};const id=session_id||`s_${Date.now().toString(36)}_${Math.random().toString(36).slice(2,8)}`;db.prepare('INSERT OR IGNORE INTO sessions (id,title) VALUES (?,?)').run(id,title||'Conversation');res.json({ok:true,session_id:id});});
app.get('/sessions',(_req,res)=>{const rows=db.prepare('SELECT id,title,created_at FROM sessions ORDER BY created_at DESC').all();res.json({ok:true,items:rows});});
app.get('/session/:id/history',(req,res)=>{const id=req.params.id;const rows=db.prepare('SELECT id,role,content,style,created_at FROM history WHERE session_id=? ORDER BY id ASC').all(id);res.json({ok:true,items:rows});});
app.post('/session/:id/rename',(req,res)=>{db.prepare('UPDATE sessions SET title=? WHERE id=?').run((req.body?.title||'Conversation'),req.params.id);res.json({ok:true});});
app.post('/session/:id/clear',(req,res)=>{db.prepare('DELETE FROM history WHERE session_id=?').run(req.params.id);res.json({ok:true});});

app.post('/feedback',(req,res)=>{const {question,correct_answer}=req.body||{}; if(!question||!correct_answer) return res.status(400).json({ok:false,error:'question,correct_answer requis'}); factUpsert(question,correct_answer); const id=db.prepare('SELECT last_insert_rowid() AS id').get().id; res.json({ok:true,id});});
app.get('/history',(_req,res)=>{const rows=db.prepare('SELECT id,session_id,user_id,role,content,style,created_at FROM history ORDER BY id DESC LIMIT 200').all(); res.json({ok:true,items:rows});});
app.post('/history/clear',(_req,res)=>{db.exec('DELETE FROM history; VACUUM;'); res.json({ok:true});});

// ──────────────────────────────────────────────────────────────────────────────
// Core: /aurion (sync)

app.post('/aurion', async (req, res) => {
  const { prompt, style='genz', user_id='rapido', model, session_id, response_length='medium' } = req.body || {};
  if (!prompt || typeof prompt !== 'string' || !prompt.trim()) return res.status(400).json({ ok:false, error:'prompt requis' });

  const chosen = chooseModel(model, style);
  const lenCtl = lengthConstraint(response_length);

  // Mémoire / identité
  const mem = factLookup(prompt); if (mem) { pushHistory(session_id,user_id,'user',prompt,style); pushHistory(session_id,user_id,'assistant',mem,style); return res.json({ reply: mem, meta:{ mode:'fact', source:'user-correction', model:chosen } }); }
  const idt = handleIdentity(prompt, style); if (idt) { pushHistory(session_id,user_id,'user',prompt,style); pushHistory(session_id,user_id,'assistant',idt.reply,style); return res.json({ reply: idt.reply, meta:{ ...idt.meta, model:chosen } }); }

  // Intents rapides
  const t = handleTranslate(prompt); if (t) {
    if (t.reply !== null) { pushHistory(session_id,user_id,'user',prompt,style); pushHistory(session_id,user_id,'assistant',t.reply,style); return res.json({ reply:t.reply, meta:{ ...t.meta, model:chosen } }); }
    const text = await callOllama(t.meta.prompt, t.meta.sys, false, chosen, { temperature:0.2, num_predict:160 });
    let clean = tidy(stripAutoIntro(text)); 
    clean = applyPersona(clean, style, { personality_level: req.body?.personality_level || 'medium', allow_emojis: req.body?.allow_emojis !== false });
    pushHistory(session_id,user_id,'user',prompt,style); pushHistory(session_id,user_id,'assistant',clean,style);
    return res.json({ reply: clean, meta:{ intent:'translate', dir:t.meta.dir, llm:true, model:chosen } });
  }
  const tm = handleTime(prompt);    if (tm)  { pushHistory(session_id,user_id,'user',prompt,style); pushHistory(session_id,user_id,'assistant',tm.reply,style);  return res.json({ reply:tm.reply,  meta:{ ...tm.meta,  model:chosen } }); }
  const w  = await handleWeather(prompt); if (w){ pushHistory(session_id,user_id,'user',prompt,style); pushHistory(session_id,user_id,'assistant',w.reply,style);   return res.json({ reply:w.reply,   meta:{ ...w.meta,   model:chosen } }); }
  const conv = handleConvert(prompt); if (conv){ pushHistory(session_id,user_id,'user',prompt,style); pushHistory(session_id,user_id,'assistant',conv.reply,style); return res.json({ reply:conv.reply, meta:{ ...conv.meta, model:chosen } }); }
  const m  = handleMath(prompt);    if (m)   { pushHistory(session_id,user_id,'user',prompt,style); pushHistory(session_id,user_id,'assistant',m.reply,style);   return res.json({ reply:m.reply,   meta:{ ...m.meta,   model:chosen } }); }

  // Mode Jarvis (prépare web/images)
  let links = null;
  let images = null;
  if (wantsWeb(prompt)) {
    const r = await webSearch(prompt, 5);
    if (r) links = r.links;
  }
  if (wantsImages(prompt)) {
    images = await imageSearch(prompt, 6);
  }

  // LLM principal
  const sys = styleSystem(style,'Rapido');
  const context = renderContext(pullRecentHistory(session_id,6));
  const enforce = mustBeTwoSentences(prompt) ? `\nContraintes: réponds en exactement 2 phrases.` : '';
  const biz = businessScaffold(prompt);
  const finalPrompt = `${context}Réponds clairement.${lenCtl.txt}${enforce}${biz}\nQuestion: ${prompt}`;
  const opts = chooseOptions(chosen, classifyIntent(prompt), lenCtl.mult);

  const raw = await callOllama(finalPrompt, sys, false, chosen, opts);
  const branded = raw.replace(/Gemma/gi, brandName(style)).replace(/\bPhi\b/gi, brandName(style));
  const completed = await ensureComplete(branded, sys, chosen);
  let reply = denoise(tidy(stripAutoIntro(completed)));

  // Fallback si vide
  if (!reply || !reply.trim()) {
    const retryPrompt = `Réponds en UNE phrase directe, sans salutation ni auto-présentation.
Question: ${prompt}`;
    const retryRaw = await callOllama(retryPrompt, sys, false, chosen, { temperature: 0.4, num_predict: 80 });
    reply = denoise(tidy(stripAutoIntro(retryRaw))) || "D’accord.";
  }

  // Personnalité
  let finalReply = applyPersona(
    reply,
    style,
    { personality_level: req.body?.personality_level || 'medium',
      allow_emojis: req.body?.allow_emojis !== false }
  );

  pushHistory(session_id,user_id,'user',prompt,style);
  pushHistory(session_id,user_id,'assistant',finalReply,style);

  const payload = { reply: finalReply, meta:{ mode:'llm', model:chosen } };
  if (Array.isArray(links) && links.length) payload.links = links;
  if (Array.isArray(images) && images.length) payload.images = images;
  return res.json(payload);
});

// ──────────────────────────────────────────────────────────────────────────────
// Stream: /aurion_stream

app.post('/aurion_stream', async (req, res) => {
  const { prompt, style='genz', user_id='rapido', buffer=false, model, session_id, response_length='medium' } = req.body || {};
  if (!prompt || typeof prompt !== 'string' || !prompt.trim()) return res.status(400).json({ ok:false, error:'prompt requis' });

  const chosen = chooseModel(model, style);
  const lenCtl = lengthConstraint(response_length);
  const sys = styleSystem(style,'Rapido');

  // Intents rapides avant stream
  const t = handleTranslate(prompt); if (t) {
    if (t.reply !== null) return res.json({ reply:t.reply, meta:{ ...t.meta, model:chosen } });
    const text = await callOllama(t.meta.prompt, t.meta.sys, false, chosen, { temperature:0.2, num_predict:160 });
    let clean = tidy(stripAutoIntro(text));
    clean = applyPersona(clean, style, { personality_level: req.body?.personality_level || 'medium', allow_emojis: req.body?.allow_emojis !== false });
    return res.json({ reply: clean, meta:{ intent:'translate', dir:t.meta.dir, llm:true, model:chosen } });
  }
  const tm = handleTime(prompt); if (tm) return res.json({ reply:tm.reply, meta:{ ...tm.meta, model:chosen } });
  const w  = await handleWeather(prompt); if (w) return res.json({ reply:w.reply, meta:{ ...w.meta, model:chosen } });
  const conv= handleConvert(prompt); if (conv) return res.json({ reply:conv.reply, meta:{ ...conv.meta, model:chosen } });
  const m   = handleMath(prompt); if (m) return res.json({ reply:m.reply, meta:{ ...m.meta, model:chosen } });

  // Jarvis preview (non-stream)
  if (wantsWeb(prompt) || wantsImages(prompt) || buffer) {
    const context = renderContext(pullRecentHistory(session_id,6));
    const enforce = mustBeTwoSentences(prompt) ? `\nContraintes: réponds en exactement 2 phrases.` : '';
    const biz = businessScaffold(prompt);
    const finalPrompt = `${context}Réponds brièvement et clairement.${lenCtl.txt}${enforce}${biz}\nQuestion: ${prompt}`;
    const opts = chooseOptions(chosen, classifyIntent(prompt), lenCtl.mult);

    // Prépare liens/images
    let links = null, images = null;
    if (wantsWeb(prompt)) { const r = await webSearch(prompt, 5); if (r) links = r.links; }
    if (wantsImages(prompt)) { images = await imageSearch(prompt, 6); }

    const text = await callOllama(finalPrompt, sys, false, chosen, opts);
    const completed = await ensureComplete(text, sys, chosen);
    let clean = denoise(tidy(stripAutoIntro(completed)));
    clean = applyPersona(clean, style, {
      personality_level: req.body?.personality_level || 'medium',
      allow_emojis: req.body?.allow_emojis !== false
    });

    pushHistory(session_id,user_id,'user',prompt,style);
    pushHistory(session_id,user_id,'assistant',clean,style);

    const payload = { reply: clean, meta:{ mode:'llm', buffered:true, model:chosen } };
    if (Array.isArray(links) && links.length) payload.links = links;
    if (Array.isArray(images) && images.length) payload.images = images;
    return res.json(payload);
  }

  // Streaming brut
  const context = renderContext(pullRecentHistory(session_id,6));
  const enforce = mustBeTwoSentences(prompt) ? `\nContraintes: réponds en exactement 2 phrases.` : '';
  const biz = businessScaffold(prompt);
  const finalPrompt = `${context}Réponds brièvement et clairement.${lenCtl.txt}${enforce}${biz}\nQuestion: ${prompt}`;
  const opts = chooseOptions(chosen, classifyIntent(prompt), lenCtl.mult);

  try {
    const stream = await callOllama(finalPrompt, sys, true, chosen, opts);
    if (!stream) {
      const text = await callOllama(finalPrompt, sys, false, chosen, opts);
      const completed = await ensureComplete(text, sys, chosen);
      let clean = denoise(tidy(stripAutoIntro(completed)));
      clean = applyPersona(clean, style, { personality_level: req.body?.personality_level || 'medium', allow_emojis: req.body?.allow_emojis !== false });
      pushHistory(session_id,user_id,'user',prompt,style);
      pushHistory(session_id,user_id,'assistant',clean,style);
      return res.json({ reply: clean, meta:{ mode:'llm', buffered:true, model:chosen } });
    }

    res.setHeader('Content-Type','text/plain; charset=utf-8');
    const reader = stream.getReader(); let acc = '';
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
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
    let clean = denoise(tidy(stripAutoIntro(completed)));
    clean = applyPersona(clean, style, { personality_level: req.body?.personality_level || 'medium', allow_emojis: req.body?.allow_emojis !== false });
    pushHistory(session_id,user_id,'user',prompt,style);
    pushHistory(session_id,user_id,'assistant',clean,style);
    res.end();
  } catch (e) {
    res.status(500).json({ ok:false, error:e.message });
  }
});

// ──────────────────────────────────────────────────────────────────────────────
// Suggestions (optionnel)

if (ENABLE_SUGGESTIONS && APNS_TEAM_ID && APNS_KEY_ID && APNS_BUNDLE_ID && APNS_PRIVATE_KEY_BASE64) {
  setInterval(async () => {
    try {
      const row = db.prepare('SELECT token FROM devices ORDER BY last_seen DESC LIMIT 1').get();
      if (!row?.token) return;
      await sendNotification(row.token, 'Aurion', 'Nouvelle suggestion pour toi, Rapido.');
    } catch (e) { console.warn('Suggestion push failed:', e.message); }
  }, 1000 * 60 * 30);
} else if (ENABLE_SUGGESTIONS) {
  console.warn('Suggestions OFF (APNs non configuré)');
}

// ──────────────────────────────────────────────────────────────────────────────
// Errors + Start

app.use((err,_req,res,_next)=>{ console.error('Unhandled:',err); res.status(500).json({ok:false,error:'server_error'}); });

app.listen(PORT, () => {
  console.log(`✅ Aurion (Ollama) up on http://localhost:${PORT}`);
  console.log(`⚙️ Models: primary=${MODELS.primary} secondary=${MODELS.secondary} | ctx=${LLM_NUM_CTX} predict=${LLM_NUM_PREDICT}`);
  console.log(`🔎 Tavily: ${TAVILY_API_KEY ? 'ON' : 'OFF'}`);
});
