// server.js — Aurion Proxy (ESM) vULTIMATE
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import Database from 'better-sqlite3';
import jwt from 'jsonwebtoken';
import { Buffer } from 'node:buffer';
import http2 from 'node:http2';

// ---------- CONFIG ----------
const BOOT_TS = Date.now();
const PORT = Number(process.env.PORT || 3000);
const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || process.env.AURION_MODEL || 'gemma3:1b';
const EMBED_MODEL = process.env.EMBED_MODEL || 'nomic-embed-text';
const TAVILY_API_KEY = process.env.TAVILY_API_KEY || '';
const FACT_SIM_THRESHOLD = Number(process.env.FACT_SIM_THRESHOLD || 0.85);

// History & memory
const HISTORY_MAX_RAW_CHARS = Number(process.env.HISTORY_MAX_RAW_CHARS || 4500);
const HISTORY_SUMMARY_TRIGGER = Number(process.env.HISTORY_SUMMARY_TRIGGER || 6500);
const HISTORY_KEEP_LAST = Number(process.env.HISTORY_KEEP_LAST || 16);

// Suggestions / APNs
const ENABLE_SUGGESTIONS = process.env.ENABLE_SUGGESTIONS === 'true';
const APNS_TEAM_ID = process.env.APNS_TEAM_ID || '';
const APNS_KEY_ID = process.env.APNS_KEY_ID || '';
const APNS_BUNDLE_ID = process.env.APNS_BUNDLE_ID || '';
const APNS_PRIVATE_KEY_BASE64 = process.env.APNS_PRIVATE_KEY_BASE64 || '';
const APNS_SANDBOX = process.env.APNS_SANDBOX === 'true';

// ---------- APP ----------
const app = express();
app.use(cors());
app.use(express.json({ limit: '4mb' }));
app.use((req, _res, next) => { console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`); next(); });

// ---------- DB ----------
const db = new Database('aurion.db');
db.exec(`
PRAGMA journal_mode=WAL;

/* Facts + embeddings */
CREATE TABLE IF NOT EXISTS facts (
  id INTEGER PRIMARY KEY,
  question_norm TEXT NOT NULL UNIQUE,
  answer TEXT NOT NULL,
  source TEXT DEFAULT 'user-correction',
  ttl_days INTEGER DEFAULT NULL,
  created_at TEXT DEFAULT (datetime('now')),
  updated_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS embeddings (
  fact_id INTEGER NOT NULL UNIQUE,
  vector BLOB NOT NULL,
  FOREIGN KEY(fact_id) REFERENCES facts(id) ON DELETE CASCADE
);

/* Conversations + messages */
CREATE TABLE IF NOT EXISTS conversations (
  id INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL UNIQUE,
  user_id TEXT DEFAULT NULL,
  summary TEXT DEFAULT '',
  title TEXT DEFAULT NULL,
  updated_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY,
  conversation_id INTEGER NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
  content TEXT NOT NULL,
  created_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_conv_created ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at);

/* Flags (archive) */
CREATE TABLE IF NOT EXISTS conversation_flags (
  id INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL UNIQUE,
  archived INTEGER DEFAULT 0 CHECK(archived IN (0,1))
);

/* User preferences */
CREATE TABLE IF NOT EXISTS user_prefs (
  id INTEGER PRIMARY KEY,
  user_id TEXT NOT NULL,
  pref_key TEXT NOT NULL,
  pref_value TEXT NOT NULL,
  updated_at TEXT DEFAULT (datetime('now')),
  UNIQUE(user_id, pref_key)
);

/* Devices (APNs) */
CREATE TABLE IF NOT EXISTS devices (
  id INTEGER PRIMARY KEY,
  token TEXT UNIQUE,
  user_id TEXT DEFAULT 'default',
  last_seen TEXT DEFAULT (datetime('now'))
);
`);

// ---------- UTILS: temps ----------
function parisNow() {
  const tz = 'Europe/Paris';
  const now = new Date();
  const date = new Intl.DateTimeFormat('fr-FR', { timeZone: tz, weekday:'long', year:'numeric', month:'long', day:'numeric' }).format(now);
  const time = new Intl.DateTimeFormat('fr-FR', { timeZone: tz, hour:'2-digit', minute:'2-digit', second:'2-digit' }).format(now);
  const iso = new Date(now.toLocaleString('en-US', { timeZone: tz })).toISOString();
  return { tz, date, time, iso, epoch: now.getTime() };
}
const timeSummary = () => { const n = parisNow(); return `${n.date} — ${n.time} (${n.tz})`; };

// ---------- UTILS: texte & identité ----------
const CREATOR = 'Rapido';
function normQ(q) {
  return (q || '').toLowerCase().trim().replace(/\s+/g,' ').replace(/[!?.,;:()"'’“”«»]/g,'');
}
function sanitizeIdentity(text, style='aurion') {
  if (!text) return text;
  let out = text.replace(/\bGemma\b/gi, style==='kronos' ? 'Kronos' : 'Aurion');
  out = out.replace(/\b(?:je suis|moi c.?est) (?:un )?(?:mod[eè]le|llm|ia)[^.\n]*[.\n]?/gi, () => style==='kronos' ? 'Je suis Kronos.' : 'Je suis Aurion.');
  return out;
}

// ---------- STYLES ----------
const replyStyles = {
  'aurion':   ({ dateLine }) => `Horloge synchronisée: ${dateLine}. Prêt à exécuter.`,
  'genz':     ({ dateLine }) => `Mise à jour IRL: ${dateLine} — synchro ok. ⏱️`,
  'pro':      ({ dateLine }) => `Contexte temporel: ${dateLine}.`,
  'friendly': ({ dateLine }) => `On est le ${dateLine} 😉`,
  'sobre':    ({ dateLine }) => `Nous sommes le ${dateLine}.`,
  'kronos':   ({ dateLine }) => `Les aiguilles griffent la nuit: ${dateLine}. Le velours de l'ombre serre nos promesses.`
};
function renderTimeAnswer(style='aurion'){ const n=parisNow(); const dateLine=`${n.date}, ${n.time} (${n.tz})`; return (replyStyles[style]||replyStyles.aurion)({dateLine,now:n}); }

const stylePrompts = {
  aurion:  null,
  genz:    `Reformule en style GenZ: bref, taquin, moderne. Max 2 emojis. Pas de vulgarité. Ne modifie pas les faits.`,
  pro:     `Reformule en style professionnel: concis, précis, sans emoji, ton neutre. Ne modifie pas les faits.`,
  friendly:`Reformule en style amical: chaleureux, accessible, rassurant. Max 1 emoji. Ne modifie pas les faits.`,
  sobre:   `Reformule sobre: phrases courtes, zéro emoji, aucune exagération. Ne modifie pas les faits.`,
  kronos:  `Reformule en style "dark romance" intensifié: sombre, baroque, envoûtant. Tension suggérée (jamais explicite). Évoque nuit, serments, murmures, velours, acier, cendre. Ton élégant, légèrement menaçant, sans vulgarité. Interdits: haine, discrimination, sexe explicite, mineurs, violence graphique, gore, automutilation, insultes, menaces. Zéro emoji. Ne modifie pas les faits ni les chiffres.`
};
function renderIdentity(style='aurion'){
  if (style==='kronos') return `Je suis Kronos — ton côté sombre, forgé par ${CREATOR}. J’avance dans la nuit, élégant et tranchant, pour dire ce que la lumière n’ose pas.`;
  const v={
    aurion:`Je suis Aurion, créé par ${CREATOR}. Assistant concis, fiable et connecté au temps réel.`,
    genz:`Yo — je suis Aurion, build par ${CREATOR}. Je réponds vite, propre, et je m’adapte. 🚀`,
    pro:`Je suis Aurion, créé par ${CREATOR}. Assistant robuste, factuel et orienté résultats.`,
    friendly:`Salut ! Moi c’est Aurion, créé par ${CREATOR}. Je t’aide simplement, sans prise de tête.`,
    sobre:`Aurion. Créé par ${CREATOR}.`
  }; return v[style]||v.aurion;
}
function identitySystemLine(style='aurion'){
  return (style==='kronos')
    ? `Identité: Tu es KRONOS, l’alter sombre d'Aurion, forgé par ${CREATOR}. Style dark romance élégant, jamais vulgaire. Toujours exact.`
    : `Identité: Tu es AURION, assistant créé par ${CREATOR}. Sois factuel, clair et fiable.`;
}

// ---------- INTENTS ----------
function isTimeQuery(text){const q=(text||'').toLowerCase();return [/quelle heure/,/il est quelle heure/,/donne l'heure/,/donne l’heure/,/c[’']?est quel jour/,/on est quel jour/,/la date/,/quel jour sommes-nous/,/\btime\b|\bdate\b/].some(r=>r.test(q));}
function isIdentityQuery(text){const q=(text||'').toLowerCase();return [/t[’' ]?es qui/,/\bqui es[- ]?tu\b/,/\btu es qui\b/,/\bqui (?:es|est) (?:aurion|kronos)\b/,/\bprésente[- ]?toi\b/,/\bqui (?:êtes|etes) vous\b/].some(r=>r.test(q));}
function isMathQuery(text){const q=(text||'').toLowerCase().trim();return /(\d+[\s]*[+\-*/^%][\s]*\d+)|\b(calcul|combien font|=)\b/.test(q);}
function isTranslateQuery(text){const q=(text||'').toLowerCase();return /\b(tradui[st]|translate|en anglais|en français|to english|to french|in french)\b/.test(q);}
function detectCityForWeather(text){const q=(text||'').toLowerCase();if(!/\b(meteo|météo|weather)\b/.test(q))return null;const m=text.match(/(?:m[ée]t[ée]o|weather)\s+(?:à|a|de|sur)?\s*([A-Za-zÀ-ÿ' -]{2,})/i);if(m&&m[1])return m[1].trim();const m2=text.match(/\b(?:à|a|sur|de)\s+([A-ZÀ-Ÿ][A-Za-zÀ-ÿ' -]+)/);if(m2&&m2[1])return m2[1].trim();return null;}

// ---------- Embeddings (Ollama) ----------
async function embed(text){
  const res=await fetch(`${OLLAMA_HOST}/api/embeddings`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:EMBED_MODEL,prompt:text})});
  if(!res.ok) throw new Error(`Embedding failed: ${res.status} ${await res.text()}`);
  const data=await res.json(); return Float32Array.from(data.embedding||[]);
}
function cosineSim(a,b){let dot=0,na=0,nb=0;const n=Math.min(a.length,b.length);for(let i=0;i<n;i++){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}return (na===0||nb===0)?0:dot/(Math.sqrt(na)*Math.sqrt(nb));}

// ---------- FactStore ----------
async function upsertFact(question,answer,source='user-correction',ttlDays=null){
  const qn=normQ(question);
  db.prepare(`INSERT INTO facts(question_norm,answer,source,ttl_days) VALUES(?,?,?,?)
              ON CONFLICT(question_norm) DO UPDATE SET answer=excluded.answer,source=excluded.source,ttl_days=excluded.ttl_days,updated_at=datetime('now')`)
    .run(qn,answer,source,ttlDays);
  const {id}=db.prepare(`SELECT id FROM facts WHERE question_norm=?`).get(qn);
  const vec=await embed(qn); const buf=Buffer.from(new Float32Array(vec).buffer);
  db.prepare(`INSERT INTO embeddings(fact_id,vector) VALUES(?,?)
              ON CONFLICT(fact_id) DO UPDATE SET vector=excluded.vector`).run(id,buf);
  return id;
}
async function lookupFact(question,threshold=FACT_SIM_THRESHOLD){
  db.prepare(`DELETE FROM facts WHERE ttl_days IS NOT NULL AND datetime(updated_at,'+'||ttl_days||' days') < datetime('now')`).run();
  const qn=normQ(question); const qv=await embed(qn);
  const rows=db.prepare(`SELECT f.id,f.question_norm,f.answer,f.source,f.updated_at,e.vector FROM facts f JOIN embeddings e ON f.id=e.fact_id`).all();
  let best=null; for(const r of rows){const vec=new Float32Array(r.vector.buffer,r.vector.byteOffset,r.vector.byteLength/4);const sim=cosineSim(qv,vec);if(!best||sim>best.sim)best={...r,sim};}
  return (best&&best.sim>=threshold)?best:null;
}
function forgetFact(question){const qn=normQ(question);const row=db.prepare(`SELECT id FROM facts WHERE question_norm=?`).get(qn);if(!row)return false;db.prepare(`DELETE FROM facts WHERE id=?`).run(row.id);return true;}

// ---------- Conversations & prefs ----------
function getOrCreateConversation(session_id,user_id=null){
  let conv=db.prepare(`SELECT * FROM conversations WHERE session_id=?`).get(session_id);
  if(!conv){db.prepare(`INSERT INTO conversations(session_id,user_id,summary) VALUES(?,?,?)`).run(session_id,user_id,'');conv=db.prepare(`SELECT * FROM conversations WHERE session_id=?`).get(session_id);}
  return conv;
}
function appendMessage(conversation_id,role,content){
  db.prepare(`INSERT INTO messages(conversation_id,role,content) VALUES(?,?,?)`).run(conversation_id,role,content);
  db.prepare(`UPDATE conversations SET updated_at=datetime('now') WHERE id=?`).run(conversation_id);
}
function fetchRecentMessages(conversation_id,limit=HISTORY_KEEP_LAST){
  return db.prepare(`SELECT role,content,created_at FROM messages WHERE conversation_id=? ORDER BY datetime(created_at) DESC LIMIT ?`).all(conversation_id,limit).reverse();
}
function estimateChars(summary,msgs){const s=(summary||'').length;const m=msgs.reduce((a,r)=>a+r.content.length,0);return s+m;}
async function summarizeConversation(summary,msgs){
  const context=[summary?`Résumé courant:\n${summary}\n`:'','Historique récent (rôle: texte):',...msgs.map(m=>`- ${m.role}: ${m.content}`)].filter(Boolean).join('\n');
  const sys='Tu résumes une conversation. Conserve décisions, faits utiles, préférences, ton. 8 lignes max, clair et actionnable.';
  const out=await askLLM(context+'\n\nRésumé condensé:',sys,0.2,220); return (out||'').trim();
}
async function getHistoryContext(session_id,user_id=null){
  const conv=getOrCreateConversation(session_id,user_id); let messages=fetchRecentMessages(conv.id,HISTORY_KEEP_LAST);
  if(estimateChars(conv.summary,messages)>HISTORY_SUMMARY_TRIGGER){
    const newSummary=await summarizeConversation(conv.summary,messages);
    db.prepare(`UPDATE conversations SET summary=?,updated_at=datetime('now') WHERE id=?`).run(newSummary,conv.id);
    const idsToKeep=db.prepare(`SELECT id FROM messages WHERE conversation_id=? ORDER BY datetime(created_at) DESC LIMIT ?`).all(conv.id,HISTORY_KEEP_LAST).map(r=>r.id);
    if(idsToKeep.length){db.prepare(`DELETE FROM messages WHERE conversation_id=? AND id NOT IN (${idsToKeep.map(()=>'?').join(',')})`).run(conv.id,...idsToKeep);}
    messages=fetchRecentMessages(conv.id,HISTORY_KEEP_LAST);
  }
  return { conv, messages };
}
function buildConversationPrefix(convSummary,messages,maxChars=HISTORY_MAX_RAW_CHARS){
  let lines=[]; if(convSummary) lines.push(`Résumé conversation (mémoire):\n${convSummary}\n`); lines.push('Derniers échanges (rôle: texte):');
  let used=0; for(const m of messages){const line=`- ${m.role}: ${m.content}`; if(used+line.length>maxChars)break; lines.push(line); used+=line.length;}
  return lines.join('\n');
}
function setUserPref(user_id,key,value){ if(!user_id||!key)return; db.prepare(`INSERT INTO user_prefs(user_id,pref_key,pref_value) VALUES(?,?,?)
  ON CONFLICT(user_id,pref_key) DO UPDATE SET pref_value=excluded.pref_value,updated_at=datetime('now')`).run(user_id,key,value); }
function getUserPref(user_id,key){ if(!user_id||!key)return null; const row=db.prepare(`SELECT pref_value FROM user_prefs WHERE user_id=? AND pref_key=?`).get(user_id,key); return row?row.pref_value:null; }
function maybeCapturePrefs(user_id,prompt){ const nick=prompt.match(/appelle[- ]?moi\s+([A-Za-zÀ-ÿ' -]{2,})/i); if(nick&&nick[1]) setUserPref(user_id,'nickname',nick[1].trim()); }

// ---------- Stylize & LLM ----------
async function askLLM(prompt,systemPrompt='',temperature=0.6,maxTokens=512){
  const res=await fetch(`${OLLAMA_HOST}/api/generate`,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({model:OLLAMA_MODEL,prompt:systemPrompt?`${systemPrompt}\n\n${prompt}`:prompt,options:{temperature,num_predict:maxTokens},stream:false})});
  if(!res.ok) throw new Error(`Ollama generate failed: ${res.status} ${await res.text()}`);
  const data=await res.json(); return data.response||'';
}
async function streamLLM(prompt,systemPrompt='',temperature=0.6){
  const res=await fetch(`${OLLAMA_HOST}/api/generate`,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({model:OLLAMA_MODEL,prompt:systemPrompt?`${systemPrompt}\n\n${prompt}`:prompt,options:{temperature},stream:true})});
  if(!res.ok) throw new Error(`Ollama stream failed: ${res.status} ${await res.text()}`); return res;
}
async function stylize(text,style){
  if(!style||style==='aurion'||!stylePrompts[style]) return (text||'').trim();
  const plain=(text||'').trim(); if(!plain) return plain;
  const sys='Tu reformules en respectant strictement les faits, chiffres et URLs. Pas d’ajouts factuels.';
  const prompt=`${stylePrompts[style]}\n\nTexte à reformuler (fr):\n"""${plain}"""\n\nRéécris sans altérer les faits:`;
  try { let out=await askLLM(prompt,sys,0.3,Math.min(800,plain.length+160)); out=sanitizeIdentity(out,style); return (out||'').trim(); } catch { return plain; }
}

// ---------- Web (Tavily) ----------
async function tavilySearch(query){
  if(!TAVILY_API_KEY) return null;
  const res=await fetch('https://api.tavily.com/search',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({api_key:TAVILY_API_KEY,query,search_depth:'advanced',include_answer:true,max_results:5})});
  if(!res.ok) throw new Error(`Tavily failed: ${res.status} ${await res.text()}`); return res.json();
}

// ---------- Intent handlers ----------
function mathSafeEval(expr){ const s=(expr||'').replace(/,/g,'.').trim(); if(!/^[0-9+\-*/().%^ \t]*$/.test(s)) throw new Error('expression invalide'); if(s.length>200) throw new Error('expression trop longue');
  const canon=s.replace(/\^/g,'**'); const f=new Function(`return (${canon});`); const val=f(); if(!isFinite(val)) throw new Error('résultat invalide'); return val; }
async function handleTranslate(prompt,style){
  const isToEN=/\b(en anglais|to english)\b/i.test(prompt); const isToFR=/\b(en français|in french|to french)\b/i.test(prompt);
  const sys='Tu es un traducteur fiable. Garde le sens exact, pas de notes ni de guillemets superflus.'; const core=prompt.replace(/\b(tradui[st]|translate)\b.*?:?/i,'').trim();
  const direction=isToEN?'FR->EN':(isToFR?'EN->FR':'AUTO'); const req=`Direction: ${direction}\nTexte:\n"""${core}"""\nTraduction:`; let out=await askLLM(req,sys,0.2,Math.min(800,core.length+160));
  out=sanitizeIdentity(out,style); return (await stylize(out,style)).trim();
}
async function handleWeather(city,style){
  try{
    const q=encodeURIComponent(city);
    const geo=await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${q}&count=1&language=fr&format=json`); const g=await geo.json();
    if(!g.results||!g.results.length) return (await stylize(`Je n'ai pas trouvé ${city}.`,style));
    const { latitude, longitude, name, country }=g.results[0]; const lat=Number(latitude).toFixed(3), lon=Number(longitude).toFixed(3);
    const wx=await fetch(`https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,weather_code&timezone=Europe%2FParis`); const w=await wx.json();
    const t=w?.current?.temperature_2m; const code=w?.current?.weather_code; let base=`Météo actuelle à ${name} (${country}) : ${t!==undefined?`${t}°C`:'température inconnue'}.`; if(code!==undefined) base+=` Code météo ${code}.`;
    base=sanitizeIdentity(base,style); return (await stylize(base,style)).trim();
  }catch{ return (await stylize(`Service météo indisponible pour "${city}".`,style)).trim(); }
}

// ---------- Sessions helpers ----------
function setArchived(session_id,archived=true){
  db.prepare(`INSERT INTO conversation_flags(session_id,archived) VALUES(?,?) ON CONFLICT(session_id) DO UPDATE SET archived=excluded.archived`).run(session_id,archived?1:0);
}
function isArchived(session_id){ const r=db.prepare(`SELECT archived FROM conversation_flags WHERE session_id=?`).get(session_id); return r?!!r.archived:false; }
function upsertConversationTitle(session_id,title){ db.prepare(`UPDATE conversations SET title=? WHERE session_id=?`).run(title,session_id); }
async function generateConversationTitle(session_id){
  const conv=db.prepare(`SELECT id,title FROM conversations WHERE session_id=?`).get(session_id); if(!conv) return null;
  const msgs=db.prepare(`SELECT role,content FROM messages WHERE conversation_id=? ORDER BY datetime(created_at) ASC LIMIT 30`).all(conv.id);
  const seed=msgs.map(m=>`- ${m.role}: ${m.content}`).join('\n'); const sys='Propose un titre ultra concis (3-7 mots), clair, sans emoji, en français.';
  const out=await askLLM(`Historique (extrait):\n${seed}\n\nTitre:`,sys,0.2,40); const title=(out||'Conversation').replace(/\n/g,' ').trim().slice(0,80); upsertConversationTitle(session_id,title); return title;
}
function searchSessions(query='',limit=20,offset=0){
  const q=`%${(query||'').trim()}%`;
  const rows=db.prepare(`
    SELECT c.session_id,c.user_id,c.title,c.summary,c.updated_at,IFNULL(f.archived,0) AS archived,
      (SELECT MIN(created_at) FROM messages m WHERE m.conversation_id=c.id) AS started_at,
      (SELECT COUNT(*) FROM messages m2 WHERE m2.conversation_id=c.id) AS msg_count
    FROM conversations c LEFT JOIN conversation_flags f ON f.session_id=c.session_id
    WHERE (c.title LIKE ? OR c.summary LIKE ? OR EXISTS (
      SELECT 1 FROM messages m3 WHERE m3.conversation_id=c.id AND m3.content LIKE ?
    ))
    ORDER BY datetime(c.updated_at) DESC
    LIMIT ? OFFSET ?
  `).all(q,q,q,limit,offset);
  return rows;
}
function cloneSession(from_session_id,new_session_id){
  const src=db.prepare(`SELECT * FROM conversations WHERE session_id=?`).get(from_session_id); if(!src) throw new Error('session source introuvable');
  db.prepare(`INSERT INTO conversations(session_id,user_id,summary,title,updated_at) VALUES(?,?,?,?,datetime('now'))`).run(new_session_id,src.user_id,src.summary||'',src.title||null);
  const dst=db.prepare(`SELECT id FROM conversations WHERE session_id=?`).get(new_session_id);
  const msgs=db.prepare(`SELECT role,content,created_at FROM messages WHERE conversation_id=? ORDER BY datetime(created_at) ASC`).all(src.id);
  const ins=db.prepare(`INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)`);
  const tx=db.transaction(items=>{ for(const m of items) ins.run(dst.id,m.role,m.content,m.created_at); }); tx(msgs); return { ok:true, count: msgs.length };
}
function exportSession(session_id,format='json'){
  const conv=db.prepare(`SELECT * FROM conversations WHERE session_id=?`).get(session_id); if(!conv) return null;
  const messages=db.prepare(`SELECT role,content,created_at FROM messages WHERE conversation_id=? ORDER BY datetime(created_at) ASC`).all(conv.id);
  if(format==='txt'){
    const head=`# ${conv.title||'Conversation'}\nSession: ${session_id}\nMàJ: ${conv.updated_at}\n\n`;
    const body=messages.map(m=>`[${m.created_at}] ${m.role.toUpperCase()}: ${m.content}`).join('\n');
    return { mime:'text/plain; charset=utf-8', body: head+body };
  }
  return { mime:'application/json; charset=utf-8', body: JSON.stringify({ session_id, title: conv.title, updated_at: conv.updated_at, summary: conv.summary, messages }, null, 2) };
}

// ---------- Core orchestration ----------
async function answerWithContext({ prompt, system, temperature, maxTokens, style, session_id, user_id }){
  if(user_id) maybeCapturePrefs(user_id,prompt);

  if(isTimeQuery(prompt)){ const ans=await stylize(renderTimeAnswer(style||'aurion'),style); return { reply: ans, meta:{mode:'time',style,now:parisNow()}, skipStore:true }; }
  if(isIdentityQuery(prompt)){ const ans=await stylize(renderIdentity(style||'aurion'),style); return { reply: ans, meta:{mode:'identity',style}, skipStore:true }; }
  if(isMathQuery(prompt)){ const expr=(prompt.match(/[-+*/().^%0-9\t ]+/g)||[]).join(' ').trim(); try{ const val=mathSafeEval(expr);
      const styled=await stylize(`Résultat: ${val}`,style); return { reply: styled, meta:{mode:'math',expr}, skipStore:false };
    }catch(e){ const styled=await stylize(`Expression invalide.`,style); return { reply: styled, meta:{mode:'math-error',error:e.message}, skipStore:false }; } }
  if(isTranslateQuery(prompt)){ const out=await handleTranslate(prompt,style); return { reply: out, meta:{mode:'translate'}, skipStore:false }; }
  const maybeCity=detectCityForWeather(prompt); if(maybeCity){ const out=await handleWeather(maybeCity,style); return { reply: out, meta:{mode:'weather',city:maybeCity}, skipStore:false }; }

  const fact=await lookupFact(prompt);
  if(fact){ let ans=fact.answer; ans=sanitizeIdentity(ans,style); ans=await stylize(ans,style);
    return { reply: ans, meta:{mode:'fact',confidence: fact.sim, source: fact.source, cache:'FactStore', updated_at: fact.updated_at}, skipStore:false }; }

  let conv=null, messages=[], historyPrefix='';
  if(session_id){ const ctx=await getHistoryContext(session_id,user_id||null); conv=ctx.conv; messages=ctx.messages;
    const nickname=user_id?getUserPref(user_id,'nickname'):null; const nickLine=nickname?`Préférence: L'utilisateur préfère être appelé "${nickname}".`:''; historyPrefix=[nickLine,buildConversationPrefix(ctx.conv.summary,messages)].filter(Boolean).join('\n'); }

  let web=null;
  if(TAVILY_API_KEY && /\b(qui est|qui sont|derni[eè]res?\s+(news|actu)|quelles? sont les|combien vaut|prix|sortie|release)\b/i.test(prompt)){
    try{ web=await tavilySearch(prompt); }catch(e){ console.warn('Tavily error:', e.message); }
  }

  const nowLine=`Contexte actuel: ${timeSummary()}. Adapte "aujourd'hui/demain/hier/ce soir" au fuseau Europe/Paris.`;
  const finalSystem=[identitySystemLine(style),'Tu es Aurion: concis, fiable, factuel.',nowLine,historyPrefix?`\n${historyPrefix}\n`:'',web&&web.answer?'Utilise les éléments fiables trouvés en ligne si fournis.':''].filter(Boolean).join('\n');

  let basePrompt=prompt;
  if(web&&web.answer){ const ctx=[`Contexte web (Tavily): ${web.answer}`,...(web.results||[]).slice(0,3).map((r,i)=>`Source ${i+1}: ${r.title} — ${r.url}`)].join('\n'); basePrompt=`${ctx}\n\nQuestion: ${prompt}\nRéponse:`; }

  let reply=await askLLM(basePrompt,finalSystem,Number(temperature??0.6),Number(maxTokens??600));
  reply=sanitizeIdentity((reply||'').trim(),style); reply=await stylize(reply,style);

  if(session_id&&conv){
    appendMessage(conv.id,'user',prompt);
    appendMessage(conv.id,'assistant',reply);
    try{
      const convRow=db.prepare(`SELECT title FROM conversations WHERE session_id=?`).get(session_id);
      if(!convRow?.title){ const msgCount=db.prepare(`SELECT COUNT(*) AS c FROM messages WHERE conversation_id=?`).get(conv.id).c; if(msgCount>=8){ generateConversationTitle(session_id).catch(()=>{}); } }
    }catch{}
  }
  return { reply, meta:{ mode: web?'web+llm':'llm', history: !!session_id, sources: web?.results?.slice(0,3)?.map(r=>({title:r.title,url:r.url})) || [] }, skipStore:false };
}

// ---------- APNs ----------
function buildAPNsJWT(){
  const key=Buffer.from(APNS_PRIVATE_KEY_BASE64,'base64').toString();
  return jwt.sign({}, key, { algorithm:'ES256', issuer: APNS_TEAM_ID, header:{alg:'ES256',kid:APNS_KEY_ID}, expiresIn:'50m' });
}
async function sendNotification(token, title, body) {
  if (!APNS_TEAM_ID || !APNS_KEY_ID || !APNS_BUNDLE_ID || !APNS_PRIVATE_KEY_BASE64) {
    throw new Error('APNs non configuré');
  }

  const audienceHost = APNS_SANDBOX ? 'api.sandbox.push.apple.com' : 'api.push.apple.com';
  const apnsPath = `/3/device/${token}`;

  // JWT signé ES256
  const keyPEM = Buffer.from(APNS_PRIVATE_KEY_BASE64, 'base64').toString();
  const jwtToken = jwt.sign({}, keyPEM, {
    algorithm: 'ES256',
    issuer: APNS_TEAM_ID,
    header: { alg: 'ES256', kid: APNS_KEY_ID },
    expiresIn: '50m'
  });

  // Session HTTP/2 (obligatoire pour APNs)
  const client = http2.connect(`https://${audienceHost}:443`, { ALPNProtocols: ['h2'] });

  const payload = JSON.stringify({
    aps: {
      alert: { title, body },
      sound: 'default'
    }
  });

  const headers = {
    ':method': 'POST',
    ':path': apnsPath,
    'authorization': `bearer ${jwtToken}`,
    'apns-topic': APNS_BUNDLE_ID,
    'apns-push-type': 'alert',
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(payload).toString()
  };

  const result = await new Promise((resolve, reject) => {
    const req = client.request(headers);
    let data = '';
    req.setEncoding('utf8');

    req.on('response', (h) => { req.statusCode = h[':status']; });
    req.on('data', (chunk) => { data += chunk; });
    req.on('end', () => {
      const status = req.statusCode || 0;
      if (status >= 200 && status < 300) {
        resolve({ ok: true, status, body: data });
      } else {
        reject(new Error(`APNs ${status} ${data || ''}`.trim()));
      }
    });
    req.on('error', (err) => reject(err));

    req.end(payload);
  });

  client.close();
  return result;
}
// ---------- Suggestions ----------
async function generateSuggestion(user_id='default'){
  const nickname=getUserPref(user_id,'nickname'); const who=nickname||'toi';
  const samples=[
    `Veux-tu que je résume ta dernière session pour ${who} ?`,
    `Je peux créer un rappel dans 2h pour relancer le projet — on y va ?`,
    `On cale un mini plan d’action en 3 étapes pour avancer maintenant ?`,
    `Je peux mémoriser cette info clé pour ${who}. Tu veux ?`
  ];
  return samples[Math.floor(Math.random()*samples.length)];
}
async function pushSuggestionsLoop(){
  if(!ENABLE_SUGGESTIONS) return;
  console.log('Suggestions loop ON (every 5min)');
  setInterval(async ()=>{
    try{
      const devices=db.prepare(`SELECT token,user_id FROM devices`).all(); if(!devices.length) return;
      for(const d of devices){ const suggestion=await generateSuggestion(d.user_id); await sendNotification(d.token,'Suggestion Aurion',suggestion); console.log('Pushed →',d.user_id,suggestion); }
    }catch(e){ console.error('Suggestion loop error:',e.message); }
  }, 5*60*1000);
}

// ---------- ROUTES: health/status/basic ----------
app.get('/health', async (_req,res)=>{ let ollama='down'; try{ const ping=await fetch(`${OLLAMA_HOST}/api/tags`); if(ping.ok) ollama='up'; }catch{ ollama='down'; }
  res.json({ ok:true, model:OLLAMA_MODEL, embed:EMBED_MODEL, ollama, tavily: !!TAVILY_API_KEY, time:{ iso:new Date().toISOString(), human: timeSummary(), tz:'Europe/Paris' } }); });
app.get('/status',(_req,res)=>{ const facts=db.prepare(`SELECT COUNT(*) AS c FROM facts`).get().c; const convs=db.prepare(`SELECT COUNT(*) AS c FROM conversations`).get().c; const prefs=db.prepare(`SELECT COUNT(*) AS c FROM user_prefs`).get().c;
  res.json({ ok:true, uptime_sec:Math.round((Date.now()-BOOT_TS)/1000), model:OLLAMA_MODEL, embed:EMBED_MODEL, tavily:!!TAVILY_API_KEY, suggestions:ENABLE_SUGGESTIONS, counts:{facts, sessions:convs, prefs} }); });
app.get('/now',(_req,res)=>{ const n=parisNow(); res.json({ ok:true, now:n, human:`${n.date} — ${n.time} (${n.tz})` }); });

// ---------- ROUTES: devices/APNs ----------
app.post('/register-device', (req, res) => {
  const { token, user_id } = req.body || {};
  if (!token) return res.status(400).json({ ok:false, error:'token requis' });
  db.prepare(`
    INSERT INTO devices(token, user_id) VALUES(?,?)
    ON CONFLICT(token) DO UPDATE SET last_seen=datetime('now'), user_id=excluded.user_id
  `).run(token, user_id || 'default');
  res.json({ ok:true });
});

app.post('/notify', async (req, res) => {
  const { token, title, body } = req.body || {};
  if (!token || !title || !body) {
    return res.status(400).json({ ok: false, error: 'token,title,body requis' });
  }
  try {
    const out = await sendNotification(token, title, body);
    res.json({ ok: true, status: out.status });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});
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
// ---------- ROUTES: facts ----------
app.post('/feedback', async (req,res)=>{ try{ const {question,correct_answer,source,ttl_days}=req.body||{}; if(!question||!correct_answer) return res.status(400).json({ok:false,error:'question et correct_answer requis'}); const id=await upsertFact(question,correct_answer,source||'user-correction',ttl_days??null); res.json({ok:true,id}); }
  catch(e){ console.error(e); res.status(500).json({ok:false,error:e.message}); } });
app.post('/forget',(req,res)=>{ const {question}=req.body||{}; if(!question) return res.status(400).json({ok:false,error:'question requise'}); const ok=forgetFact(question); res.json({ok}); });
app.get('/facts',(req,res)=>{ const q=req.query.q||''; if(!q){ const all=db.prepare(`SELECT id,question_norm,source,ttl_days,updated_at FROM facts ORDER BY updated_at DESC LIMIT 200`).all(); return res.json({ok:true,count:all.length,items:all}); }
  const qn=normQ(q); const row=db.prepare(`SELECT * FROM facts WHERE question_norm=?`).get(qn); if(!row) return res.json({ok:true,found:false}); res.json({ok:true,found:true,item:row}); });

// ---------- ROUTES: history ----------
app.get('/history',(req,res)=>{ const session_id=req.query.session_id; if(!session_id) return res.status(400).json({ok:false,error:'session_id requis'}); const conv=db.prepare(`SELECT * FROM conversations WHERE session_id=?`).get(session_id);
  if(!conv) return res.json({ok:true,found:false}); const messages=db.prepare(`SELECT role,content,created_at FROM messages WHERE conversation_id=? ORDER BY datetime(created_at) ASC`).all(conv.id);
  res.json({ ok:true, found:true, summary:conv.summary, title:conv.title, messages }); });
app.post('/history/clear',(req,res)=>{ const {session_id}=req.body||{}; if(!session_id) return res.status(400).json({ok:false,error:'session_id requis'}); const conv=db.prepare(`SELECT * FROM conversations WHERE session_id=?`).get(session_id);
  if(!conv) return res.json({ok:true,cleared:false}); db.prepare(`DELETE FROM messages WHERE conversation_id=?`).run(conv.id); db.prepare(`UPDATE conversations SET summary='',title=NULL,updated_at=datetime('now') WHERE id=?`).run(conv.id); res.json({ok:true,cleared:true}); });

// ---------- ROUTES: sessions (list/search/title/archive/clone/export/import) ----------
app.get('/sessions',(req,res)=>{ const query=(req.query.query||'').toString(); const limit=Math.min(Number(req.query.limit||20),100); const offset=Math.max(Number(req.query.offset||0),0);
  const archived=(req.query.archived||'all').toString(); let rows=searchSessions(query,limit,offset); if(archived==='0') rows=rows.filter(r=>!r.archived); if(archived==='1') rows=rows.filter(r=>!!r.archived);
  res.json({ ok:true, count: rows.length, items: rows }); });
app.get('/sessions/:session_id',(req,res)=>{ const session_id=req.params.session_id; const conv=db.prepare(`SELECT * FROM conversations WHERE session_id=?`).get(session_id);
  if(!conv) return res.status(404).json({ok:false,error:'not found'}); const messages=db.prepare(`SELECT role,content,created_at FROM messages WHERE conversation_id=? ORDER BY datetime(created_at) ASC`).all(conv.id);
  res.json({ ok:true, session:{ session_id, title:conv.title, summary:conv.summary, updated_at:conv.updated_at, archived:isArchived(session_id) }, messages }); });
app.post('/sessions/:session_id/title', async (req,res)=>{ const session_id=req.params.session_id; const conv=db.prepare(`SELECT * FROM conversations WHERE session_id=?`).get(session_id);
  if(!conv) return res.status(404).json({ok:false,error:'not found'}); const {title}=req.body||{}; if(title&&title.trim()){ upsertConversationTitle(session_id,title.trim().slice(0,80)); return res.json({ok:true,title:title.trim().slice(0,80)}); }
  const auto=await generateConversationTitle(session_id); res.json({ok:true,title:auto}); });
app.post('/sessions/:session_id/archive',(req,res)=>{ const session_id=req.params.session_id; const {archived}=req.body||{}; setArchived(session_id,!!archived); res.json({ok:true,session_id,archived:!!archived}); });
app.post('/sessions/clone',(req,res)=>{ const {from_session_id,new_session_id}=req.body||{}; if(!from_session_id||!new_session_id) return res.status(400).json({ok:false,error:'from_session_id et new_session_id requis'});
  try{ const out=cloneSession(from_session_id,new_session_id); res.json({ok:true,...out}); }catch(e){ res.status(400).json({ok:false,error:e.message}); } });
app.get('/sessions/:session_id/export',(req,res)=>{ const session_id=req.params.session_id; const format=(req.query.format||'json').toString(); const out=exportSession(session_id,format);
  if(!out) return res.status(404).json({ok:false,error:'not found'}); res.setHeader('Content-Type',out.mime); res.send(out.body); });
app.post('/sessions/import',(req,res)=>{ const payload=req.body; try{ const {session_id,title,summary,messages}=payload||{};
  if(!session_id||!Array.isArray(messages)) return res.status(400).json({ok:false,error:'session_id + messages requis'});
  let conv=db.prepare(`SELECT * FROM conversations WHERE session_id=?`).get(session_id);
  if(!conv){ db.prepare(`INSERT INTO conversations(session_id,summary,title,updated_at) VALUES(?,?,?,datetime('now'))`).run(session_id,summary||'',title||null); conv=db.prepare(`SELECT * FROM conversations WHERE session_id=?`).get(session_id); }
  const ins=db.prepare(`INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)`);
  const tx=db.transaction(items=>{ for(const m of items){ ins.run(conv.id,m.role||'user',m.content||'',m.created_at||null); } }); tx(messages);
  if(title) upsertConversationTitle(session_id,title); res.json({ok:true,imported:messages.length});
} catch(e){ res.status(400).json({ok:false,error:e.message}); } });

// ---------- ROUTES: core LLM ----------
app.post('/aurion', async (req,res)=>{ try{ const {prompt,system,temperature,maxTokens,style='aurion',session_id,user_id}=req.body||{};
  if(!prompt) return res.status(400).json({error:'prompt requis'}); const out=await answerWithContext({prompt,system,temperature,maxTokens,style,session_id,user_id}); res.json(out);
} catch(e){ console.error(e); res.status(500).json({error:e.message}); } });

app.post('/aurion_once', async (req,res)=>{ try{ const {prompt,system,temperature,maxTokens=280,style='aurion',session_id,user_id}=req.body||{};
  if(!prompt) return res.status(400).json({error:'prompt requis'}); const out=await answerWithContext({prompt,system,temperature,maxTokens,style,session_id,user_id}); res.json(out);
} catch(e){ console.error(e); res.status(500).json({error:e.message}); } });

app.post('/aurion_stream', async (req,res)=>{ try{
  const {prompt,system,temperature,style='aurion',buffer=false,session_id,user_id}=req.body||{};
  if(!prompt) return res.status(400).json({error:'prompt requis'});

  if(isTimeQuery(prompt)){ const out=await stylize(renderTimeAnswer(style),style);
    if(buffer){ res.setHeader('Content-Type','application/json; charset=utf-8'); return res.end(JSON.stringify({reply:out,meta:{mode:'time',style,now:parisNow(),buffered:true}})); }
    res.setHeader('Content-Type','text/plain; charset=utf-8'); return res.end(out); }
  if(isIdentityQuery(prompt)){ const out=await stylize(renderIdentity(style),style);
    if(buffer){ res.setHeader('Content-Type','application/json; charset=utf-8'); return res.end(JSON.stringify({reply:out,meta:{mode:'identity',style,buffered:true}})); }
    res.setHeader('Content-Type','text/plain; charset=utf-8'); return res.end(out); }

  const fact=await lookupFact(prompt);
  if(fact){ let ans=fact.answer; ans=sanitizeIdentity(ans,style); ans=await stylize(ans,style);
    if(buffer){ res.setHeader('Content-Type','application/json; charset=utf-8'); return res.end(JSON.stringify({reply:ans,meta:{mode:'fact',confidence:fact.sim,source:fact.source,cache:'FactStore',updated_at:fact.updated_at,buffered:true}})); }
    res.setHeader('Content-Type','text/plain; charset=utf-8'); return res.end(ans); }

  let conv=null,messages=[],historyPrefix='';
  if(session_id){ const ctx=await getHistoryContext(session_id,user_id||null); conv=ctx.conv; messages=ctx.messages;
    const nickname=user_id?getUserPref(user_id,'nickname'):null; const nickLine=nickname?`Préférence: L'utilisateur préfère être appelé "${nickname}".`:''; historyPrefix=[nickLine,buildConversationPrefix(ctx.conv.summary,messages)].filter(Boolean).join('\n'); }
  const nowLine=`Contexte actuel: ${timeSummary()}. Adapte "aujourd'hui/demain/hier/ce soir" au fuseau Europe/Paris.`;
  const finalSystem=[identitySystemLine(style),nowLine,historyPrefix?`\n${historyPrefix}\n`:''].filter(Boolean).join('\n');
  const r=await streamLLM(prompt,finalSystem,Number(temperature??0.6));

  if(buffer){
    const decoder=new TextDecoder(); let carry=''; let full='';
    for await (const chunk of r.body){ const text=decoder.decode(chunk,{stream:true}); carry+=text; let idx; while((idx=carry.indexOf('\n'))>=0){ const line=carry.slice(0,idx).trim(); carry=carry.slice(idx+1); if(!line)continue; try{ const obj=JSON.parse(line); if(obj.response) full+=obj.response; }catch{} } }
    if(carry.trim()){ try{ const last=JSON.parse(carry.trim()); if(last.response) full+=last.response; }catch{} }
    const clean=await stylize(sanitizeIdentity(full.trim(),style),style);
    if(session_id&&conv){ appendMessage(conv.id,'user',prompt); appendMessage(conv.id,'assistant',clean); }
    res.setHeader('Content-Type','application/json; charset=utf-8'); return res.end(JSON.stringify({reply:clean,meta:{mode:'llm',buffered:true,history:!!session_id}}));
  }

  // pass-through NDJSON; store only after
  res.setHeader('Content-Type','application/x-ndjson; charset=utf-8');
  const decoder=new TextDecoder(); let carry=''; let full='';
  for await (const chunk of r.body){ res.write(chunk); const text=decoder.decode(chunk,{stream:true}); carry+=text; let idx; while((idx=carry.indexOf('\n'))>=0){ const line=carry.slice(0,idx).trim(); carry=carry.slice(idx+1); if(!line)continue; try{ const obj=JSON.parse(line); if(obj.response) full+=obj.response; }catch{} } }
  if(session_id&&conv){ appendMessage(conv.id,'user',prompt); appendMessage(conv.id,'assistant',sanitizeIdentity(full.trim(),style)); }
  res.end();
} catch(e){ console.error(e); res.status(500).json({error:e.message}); } });

app.post('/aurion_smart', async (req,res)=>{ try{
  const {prompt,reliability='normal',style='aurion',system,session_id,user_id}=req.body||{};
  if(!prompt) return res.status(400).json({error:'prompt requis'});
  const out=await answerWithContext({prompt,system,temperature:0.6,maxTokens:700,style,session_id,user_id});
  res.json(out);
} catch(e){ console.error(e); res.status(500).json({error:e.message}); } });

// ---------- START ----------
app.listen(PORT,()=>{ const factsCount=db.prepare(`SELECT COUNT(*) AS c FROM facts`).get().c;
  console.log(`Aurion proxy up on http://localhost:${PORT}`);
  console.log(`Model: ${OLLAMA_MODEL} | Embeddings: ${EMBED_MODEL} | Tavily: ${TAVILY_API_KEY?'on':'off'} | Facts: ${factsCount} | Suggestions: ${ENABLE_SUGGESTIONS?'on':'off'}`);
  if(ENABLE_SUGGESTIONS) pushSuggestionsLoop();
});
