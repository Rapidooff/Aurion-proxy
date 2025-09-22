// server.js — Aurion Proxy (ESM) full
// Run: node server.js  (Node 18+)
// Features: FactStore (SQLite+embeddings), chat history+summary, time (Europe/Paris),
// styles (incl. KRONOS), stylized rephrasing, stream buffer NDJSON, Tavily optional.

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import Database from 'better-sqlite3';
import { Buffer } from 'node:buffer';

// ---------- CONFIG ----------
const PORT = Number(process.env.PORT || 3000);
const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || process.env.AURION_MODEL || 'gemma3:1b';
const EMBED_MODEL = process.env.EMBED_MODEL || 'nomic-embed-text';
const TAVILY_API_KEY = process.env.TAVILY_API_KEY || '';
const FACT_SIM_THRESHOLD = Number(process.env.FACT_SIM_THRESHOLD || 0.85);

// History limits
const HISTORY_MAX_RAW_CHARS = Number(process.env.HISTORY_MAX_RAW_CHARS || 4000);
const HISTORY_SUMMARY_TRIGGER = Number(process.env.HISTORY_SUMMARY_TRIGGER || 6000);
const HISTORY_KEEP_LAST = Number(process.env.HISTORY_KEEP_LAST || 14);

// ---------- APP ----------
const app = express();
app.use(cors());
app.use(express.json({ limit: '2mb' }));

// ---------- DB ----------
const db = new Database('fact_store.db');
db.exec(`
PRAGMA journal_mode=WAL;

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

CREATE TABLE IF NOT EXISTS conversations (
  id INTEGER PRIMARY KEY,
  session_id TEXT NOT NULL,
  user_id TEXT DEFAULT NULL,
  summary TEXT DEFAULT '',
  updated_at TEXT DEFAULT (datetime('now')),
  UNIQUE(session_id)
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
`);

// ---------- UTILS: text ----------
function normalizeQuestion(q) {
  return (q || '')
    .toLowerCase()
    .trim()
    .replace(/\s+/g, ' ')
    .replace(/[!?.,;:()"'’“”«»]/g, '');
}

// ---------- UTILS: time (Europe/Paris) ----------
function parisNow() {
  const tz = 'Europe/Paris';
  const now = new Date();
  const date = new Intl.DateTimeFormat('fr-FR', {
    timeZone: tz, weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
  }).format(now);
  const time = new Intl.DateTimeFormat('fr-FR', {
    timeZone: tz, hour: '2-digit', minute: '2-digit', second: '2-digit'
  }).format(now);
  const iso = new Date(now.toLocaleString('en-US', { timeZone: tz })).toISOString();
  return { tz, date, time, iso, epoch: now.getTime() };
}
function timeSummary() {
  const n = parisNow();
  return `${n.date} — ${n.time} (${n.tz})`;
}

// ---------- Time reply styles (incl. KRONOS) ----------
const replyStyles = {
  'sobre':    ({ dateLine }) => `Nous sommes le ${dateLine}.`,
  'friendly': ({ dateLine }) => `On est le ${dateLine} 😉`,
  'genz':     ({ dateLine }) => `Mise à jour IRL: ${dateLine} — synchro ok. ⏱️`,
  'pro':      ({ dateLine }) => `Contexte temporel: ${dateLine}.`,
  'aurion':   ({ dateLine }) => `Horloge synchronisée: ${dateLine}. Prêt à exécuter.`,
'kronos':   ({ dateLine }) => `Les aiguilles griffent la nuit: ${dateLine}. Le velours de l'ombre serre nos promesses.`,
};
function renderTimeAnswer(style = 'aurion') {
  const n = parisNow();
  const dateLine = `${n.date}, ${n.time} (${n.tz})`;
  const f = replyStyles[style] || replyStyles['aurion'];
  return f({ dateLine, now: n });
}
function isTimeQuery(text) {
  const q = (text || '').toLowerCase();
  return [
    /quelle heure/, /il est quelle heure/,
    /donne l'heure/, /donne l’heure/,
    /c[’']?est quel jour/, /on est quel jour/,
    /la date/, /quel jour sommes-nous/,
    /\btime\b|\bdate\b/
  ].some(r => r.test(q));
}

// ---------- Math ----------
function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ---------- Embeddings (Ollama) ----------
async function embed(text) {
  const res = await fetch(`${OLLAMA_HOST}/api/embeddings`, {
    method: 'POST',
    headers: { 'Content-Type':'application/json' },
    body: JSON.stringify({ model: EMBED_MODEL, prompt: text })
  });
  if (!res.ok) throw new Error(`Embedding failed: ${res.status} ${await res.text()}`);
  const data = await res.json();
  return Float32Array.from(data.embedding || []);
}

// ---------- FactStore ----------
async function upsertFact(question, answer, source = 'user-correction', ttlDays = null) {
  const qn = normalizeQuestion(question);
  db.prepare(`
    INSERT INTO facts(question_norm, answer, source, ttl_days)
    VALUES(?, ?, ?, ?)
    ON CONFLICT(question_norm) DO UPDATE SET
      answer=excluded.answer,
      source=excluded.source,
      ttl_days=excluded.ttl_days,
      updated_at=datetime('now')
  `).run(qn, answer, source, ttlDays);

  const row = db.prepare(`SELECT id FROM facts WHERE question_norm=?`).get(qn);
  const vec = await embed(qn);
  const buf = Buffer.from(new Float32Array(vec).buffer);
  db.prepare(`
    INSERT INTO embeddings(fact_id, vector) VALUES(?, ?)
    ON CONFLICT(fact_id) DO UPDATE SET vector=excluded.vector
  `).run(row.id, buf);
  return row.id;
}
async function lookupFact(question, threshold = FACT_SIM_THRESHOLD) {
  const qn = normalizeQuestion(question);
  const qv = await embed(qn);

  db.prepare(`
    DELETE FROM facts WHERE ttl_days IS NOT NULL
      AND datetime(updated_at, '+' || ttl_days || ' days') < datetime('now')
  `).run();

  const rows = db.prepare(`
    SELECT f.id, f.question_norm, f.answer, f.source, f.updated_at, e.vector
    FROM facts f JOIN embeddings e ON f.id = e.fact_id
  `).all();

  let best = null;
  for (const r of rows) {
    const buf = r.vector;
    const vec = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
    const sim = cosineSim(qv, vec);
    if (!best || sim > best.sim) best = { ...r, sim };
  }
  if (best && best.sim >= threshold) return best;
  return null;
}
function forgetFact(question) {
  const qn = normalizeQuestion(question);
  const row = db.prepare(`SELECT id FROM facts WHERE question_norm=?`).get(qn);
  if (!row) return false;
  db.prepare(`DELETE FROM facts WHERE id=?`).run(row.id);
  return true;
}

// ---------- LLM (Ollama) ----------
async function askLLM(prompt, systemPrompt = '', temperature = 0.6, maxTokens = 512) {
  const res = await fetch(`${OLLAMA_HOST}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type':'application/json' },
    body: JSON.stringify({
      model: OLLAMA_MODEL,
      prompt: systemPrompt ? `${systemPrompt}\n\n${prompt}` : prompt,
      options: { temperature, num_predict: maxTokens },
      stream: false
    })
  });
  if (!res.ok) throw new Error(`Ollama generate failed: ${res.status} ${await res.text()}`);
  const data = await res.json();
  return data.response || '';
}
async function streamLLM(prompt, systemPrompt = '', temperature = 0.6) {
  const res = await fetch(`${OLLAMA_HOST}/api/generate`, {
    method: 'POST',
    headers: { 'Content-Type':'application/json' },
    body: JSON.stringify({
      model: OLLAMA_MODEL,
      prompt: systemPrompt ? `${systemPrompt}\n\n${prompt}` : prompt,
      options: { temperature },
      stream: true
    })
  });
  if (!res.ok) throw new Error(`Ollama stream failed: ${res.status} ${await res.text()}`);
  return res; // ReadableStream (NDJSON)
}

// ---------- Tavily (optional) ----------
async function tavilySearch(query) {
  if (!TAVILY_API_KEY) return null;
  const res = await fetch('https://api.tavily.com/search', {
    method: 'POST',
    headers: { 'Content-Type':'application/json' },
    body: JSON.stringify({
      api_key: TAVILY_API_KEY,
      query,
      search_depth: 'advanced',
      include_answer: true,
      max_results: 5
    })
  });
  if (!res.ok) throw new Error(`Tavily failed: ${res.status} ${await res.text()}`);
  return res.json();
}

// ---------- Conversation history ----------
function getOrCreateConversation(session_id, user_id = null) {
  if (!session_id) throw new Error('session_id requis');
  let conv = db.prepare(`SELECT * FROM conversations WHERE session_id=?`).get(session_id);
  if (!conv) {
    db.prepare(`INSERT INTO conversations(session_id, user_id, summary) VALUES(?,?,?)`)
      .run(session_id, user_id, '');
    conv = db.prepare(`SELECT * FROM conversations WHERE session_id=?`).get(session_id);
  }
  return conv;
}
function appendMessage(conversation_id, role, content) {
  db.prepare(`INSERT INTO messages(conversation_id, role, content) VALUES(?,?,?)`)
    .run(conversation_id, role, content);
  db.prepare(`UPDATE conversations SET updated_at=datetime('now') WHERE id=?`)
    .run(conversation_id);
}
function fetchRecentMessages(conversation_id, limit = HISTORY_KEEP_LAST) {
  return db.prepare(`
    SELECT role, content, created_at
    FROM messages
    WHERE conversation_id=?
    ORDER BY datetime(created_at) DESC
    LIMIT ?
  `).all(conversation_id, limit).reverse();
}
function estimateChars(summary, msgs) {
  const s = (summary || '').length;
  const m = msgs.reduce((acc, r) => acc + r.content.length, 0);
  return s + m;
}
async function summarizeConversation(summary, msgs) {
  const context = [
    summary ? `Résumé courant:\n${summary}\n` : '',
    'Historique récent (rôle: texte):',
    ...msgs.map(m => `- ${m.role}: ${m.content}`)
  ].filter(Boolean).join('\n');
  const sys = 'Tu résumes une conversation utilisateur/assistant. Conserve décisions, faits utiles, préférences, et le ton. 8 lignes max, clair et actionnable.';
  const out = await askLLM(context + '\n\nRésumé condensé:', sys, 0.2, 220);
  return (out || '').trim();
}
async function getHistoryContext(session_id, user_id = null) {
  const conv = getOrCreateConversation(session_id, user_id);
  let messages = fetchRecentMessages(conv.id, HISTORY_KEEP_LAST);
  if (estimateChars(conv.summary, messages) > HISTORY_SUMMARY_TRIGGER) {
    const newSummary = await summarizeConversation(conv.summary, messages);
    db.prepare(`UPDATE conversations SET summary=?, updated_at=datetime('now') WHERE id=?`)
      .run(newSummary, conv.id);
    const idsToKeep = db.prepare(`
      SELECT id FROM messages WHERE conversation_id=?
      ORDER BY datetime(created_at) DESC LIMIT ?
    `).all(conv.id, HISTORY_KEEP_LAST).map(r => r.id);
    if (idsToKeep.length) {
      db.prepare(`
        DELETE FROM messages
        WHERE conversation_id=?
        AND id NOT IN (${idsToKeep.map(()=>'?').join(',')})
      `).run(conv.id, ...idsToKeep);
    }
    messages = fetchRecentMessages(conv.id, HISTORY_KEEP_LAST);
  }
  return { conv, messages };
}
function buildConversationPrefix(convSummary, messages, maxChars = HISTORY_MAX_RAW_CHARS) {
  let lines = [];
  if (convSummary) lines.push(`Résumé conversation (mémoire):\n${convSummary}\n`);
  lines.push('Derniers échanges (rôle: texte):');
  let used = 0;
  for (const m of messages) {
    const line = `- ${m.role}: ${m.content}`;
    if (used + line.length > maxChars) break;
    lines.push(line);
    used += line.length;
  }
  return lines.join('\n');
}

// ---------- Stylized rephrasing ----------
const stylePrompts = {
  aurion:  null,
  genz:    `Reformule en style GenZ: bref, taquin, moderne. Max 1–2 emojis. Pas de vulgarité. Ne change pas les faits.`,
  pro:     `Reformule en style professionnel: concis, précis, sans emoji, ton neutre. Ne change pas les faits.`,
  friendly:`Reformule en style amical: chaleureux, accessible, rassurant. Max 1 emoji. Ne change pas les faits.`,
  sobre:   `Reformule de manière sobre et minimale: phrases courtes, zéro emoji, aucune exagération. Ne change pas les faits.`,
kronos:  `Reformule en style "dark romance" intensifié: sombre, baroque, envoûtant, avec une tension sensuelle suggérée (jamais explicite). Évoque la nuit, les serments, les murmures, le velours, l'acier, la cendre. Ton élégant et dramatique, légèrement menaçant, mais sans vulgarité. Interdits absolus: haine, discrimination, sexualité explicite, sexualisation des mineurs, violence graphique, gore, incitation à l'automutilation, insultes, menaces directes. Zéro emoji. Ne change pas les faits ni les chiffres, ne supprime pas d'informations.`
};
async function stylize(text, style) {
  if (!style || style === 'aurion' || !stylePrompts[style]) return (text || '').trim();
  const plain = (text || '').trim();
  if (!plain) return plain;
  const sys = 'Tu reformules un texte en respectant strictement les faits, les chiffres et les URLs. Pas d’ajouts factuels.';
  const prompt = [
    `Consigne de style:\n${stylePrompts[style]}`,
    `\nTexte à reformuler (en français):\n"""`,
    plain,
    `"""\n\nRéécris le texte dans le style demandé, sans rajouter de faits:`
  ].join('');
  try {
    const out = await askLLM(prompt, sys, 0.3, Math.min(800, plain.length + 150));
    return (out || '').trim();
  } catch { return plain; }
}

// ---------- Core helper: answer with context ----------
async function answerWithContext({ prompt, system, temperature, maxTokens, style, session_id, user_id }) {
  if (isTimeQuery(prompt)) {
    return { reply: renderTimeAnswer(style || 'aurion'), meta: { mode:'time', style: style || 'aurion', now: parisNow() }, skipStore: false };
  }
  const hit = await lookupFact(prompt);
  if (hit) {
    return { reply: hit.answer, meta: { mode:'fact', confidence: hit.sim, source: hit.source, cache:'FactStore', updated_at: hit.updated_at }, skipStore: false };
  }
  let conv = null, messages = [], historyPrefix = '';
  if (session_id) {
    const result = await getHistoryContext(session_id, user_id || null);
    conv = result.conv;
    messages = result.messages;
    historyPrefix = buildConversationPrefix(result.conv.summary, messages);
  }
  const nowLine = `Contexte actuel: ${timeSummary()}. Adapte "aujourd'hui/demain/hier/ce soir" au fuseau Europe/Paris.`;
  const historyLine = historyPrefix ? `\n${historyPrefix}\n` : '';
  const finalSystem = [system || '', nowLine, historyLine].filter(Boolean).join('\n');

  let reply = await askLLM(prompt, finalSystem, Number(temperature ?? 0.6), Number(maxTokens ?? 512));
  reply = (reply || '').trim();
  if (style && style !== 'aurion') {
    reply = await stylize(reply, style);
  }
  if (session_id && conv) {
    appendMessage(conv.id, 'user', prompt);
    appendMessage(conv.id, 'assistant', reply);
  }
  return { reply, meta: { mode:'llm', history: !!session_id } };
}

// ---------- ROUTES ----------
app.get('/health', async (_req, res) => {
  let ollama = 'down';
  try { const ping = await fetch(`${OLLAMA_HOST}/api/tags`); if (ping.ok) ollama = 'up'; } catch { ollama = 'down'; }
  res.json({ ok: true, model: OLLAMA_MODEL, ollama, tavily: TAVILY_API_KEY ? 'configured' : 'off', time: { iso: new Date().toISOString(), human: timeSummary(), tz: 'Europe/Paris' } });
});

app.get('/now', (_req, res) => {
  const n = parisNow();
  res.json({ ok: true, now: n, human: `${n.date} — ${n.time} (${n.tz})` });
});

// History admin
app.get('/history', (req, res) => {
  const session_id = req.query.session_id;
  if (!session_id) return res.status(400).json({ ok:false, error:'session_id requis' });
  const conv = db.prepare(`SELECT * FROM conversations WHERE session_id=?`).get(session_id);
  if (!conv) return res.json({ ok:true, found:false });
  const messages = db.prepare(`SELECT role, content, created_at FROM messages WHERE conversation_id=? ORDER BY datetime(created_at) ASC`).all(conv.id);
  res.json({ ok:true, found:true, summary: conv.summary, messages });
});
app.post('/history/clear', (req, res) => {
  const { session_id } = req.body || {};
  if (!session_id) return res.status(400).json({ ok:false, error:'session_id requis' });
  const conv = db.prepare(`SELECT * FROM conversations WHERE session_id=?`).get(session_id);
  if (!conv) return res.json({ ok:true, cleared:false });
  db.prepare(`DELETE FROM messages WHERE conversation_id=?`).run(conv.id);
  db.prepare(`UPDATE conversations SET summary='', updated_at=datetime('now') WHERE id=?`).run(conv.id);
  res.json({ ok:true, cleared:true });
});
app.post('/history/append', (req, res) => {
  const { session_id, user_id, role, content } = req.body || {};
  if (!session_id || !role || !content) return res.status(400).json({ ok:false, error:'session_id, role, content requis' });
  const conv = getOrCreateConversation(session_id, user_id || null);
  appendMessage(conv.id, role, content);
  res.json({ ok:true });
});

// FactStore endpoints
app.post('/feedback', async (req, res) => {
  try {
    const { question, correct_answer, source, ttl_days } = req.body || {};
    if (!question || !correct_answer) return res.status(400).json({ ok:false, error:'question et correct_answer requis' });
    const id = await upsertFact(question, correct_answer, source || 'user-correction', ttl_days ?? null);
    res.json({ ok:true, id });
  } catch (e) { console.error(e); res.status(500).json({ ok:false, error: e.message }); }
});
app.post('/forget', (req, res) => {
  const { question } = req.body || {};
  if (!question) return res.status(400).json({ ok:false, error:'question requise' });
  const ok = forgetFact(question);
  res.json({ ok });
});
app.get('/facts', (req, res) => {
  const q = req.query.q || '';
  if (!q) {
    const all = db.prepare(`SELECT id, question_norm, source, ttl_days, updated_at FROM facts ORDER BY updated_at DESC LIMIT 200`).all();
    return res.json({ ok:true, count: all.length, items: all });
  }
  const qn = normalizeQuestion(q);
  const row = db.prepare(`SELECT * FROM facts WHERE question_norm=?`).get(qn);
  if (!row) return res.json({ ok:true, found:false });
  res.json({ ok:true, found:true, item: row });
});

// Core endpoints
app.post('/aurion', async (req, res) => {
  try {
    const { prompt, system, temperature, maxTokens, style = 'aurion', session_id, user_id } = req.body || {};
    if (!prompt) return res.status(400).json({ error:'prompt requis' });
    const out = await answerWithContext({ prompt, system, temperature, maxTokens, style, session_id, user_id });
    res.json(out);
  } catch (e) { console.error(e); res.status(500).json({ error: e.message }); }
});

app.post('/aurion_once', async (req, res) => {
  try {
    const { prompt, system, temperature, maxTokens = 256, style = 'aurion', session_id, user_id } = req.body || {};
    if (!prompt) return res.status(400).json({ error:'prompt requis' });
    const out = await answerWithContext({ prompt, system, temperature, maxTokens, style, session_id, user_id });
    res.json(out);
  } catch (e) { console.error(e); res.status(500).json({ error: e.message }); }
});

app.post('/aurion_stream', async (req, res) => {
  try {
    const { prompt, system, temperature, style = 'aurion', buffer = false, session_id, user_id } = req.body || {};
    if (!prompt) return res.status(400).json({ error:'prompt requis' });

    // Time intent
    if (isTimeQuery(prompt)) {
      const out = renderTimeAnswer(style);
      if (buffer) {
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        return res.end(JSON.stringify({ reply: out, meta: { mode:'time', style, now: parisNow(), buffered: true } }));
      }
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      return res.end(out);
    }

    // FactStore
    const hit = await lookupFact(prompt);
    if (hit) {
      const payload = { reply: hit.answer, meta: { mode:'fact', confidence: hit.sim, source: hit.source, cache:'FactStore', updated_at: hit.updated_at } };
      if (buffer) {
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        return res.end(JSON.stringify({ ...payload, buffered: true }));
      }
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      return res.end(hit.answer);
    }

    // History
    let conv = null, messages = [], historyPrefix = '';
    if (session_id) {
      const result = await getHistoryContext(session_id, user_id || null);
      conv = result.conv;
      messages = result.messages;
      historyPrefix = buildConversationPrefix(result.conv.summary, messages);
    }

    const nowLine = `Contexte actuel: ${timeSummary()}. Adapte "aujourd'hui/demain/hier/ce soir" au fuseau Europe/Paris.`;
    const historyLine = historyPrefix ? `\n${historyPrefix}\n` : '';
    const finalSystem = [system || '', nowLine, historyLine].filter(Boolean).join('\n');

    const r = await streamLLM(prompt, finalSystem, Number(temperature ?? 0.6));

    if (buffer) {
      const decoder = new TextDecoder();
      let carry = '';
      let full = '';
      for await (const chunk of r.body) {
        const text = decoder.decode(chunk, { stream: true });
        carry += text;
        let idx;
        while ((idx = carry.indexOf('\n')) >= 0) {
          const line = carry.slice(0, idx).trim();
          carry = carry.slice(idx + 1);
          if (!line) continue;
          try {
            const obj = JSON.parse(line);
            if (obj.response) full += obj.response;
          } catch {}
        }
      }
      if (carry.trim()) {
        try { const last = JSON.parse(carry.trim()); if (last.response) full += last.response; } catch {}
      }

      // persist history
      if (session_id && conv) {
        appendMessage(conv.id, 'user', prompt);
        appendMessage(conv.id, 'assistant', (full || '').trim());
      }

      let final = (full || '').trim();
      if (style && style !== 'aurion') {
        final = await stylize(final, style);
      }
      res.setHeader('Content-Type', 'application/json; charset=utf-8');
      return res.end(JSON.stringify({ reply: final, meta: { mode:'llm', buffered: true, history: !!session_id } }));
    }

    // Raw NDJSON pass-through; we also accumulate for history (not stylized live)
    res.setHeader('Content-Type', 'application/x-ndjson; charset=utf-8');
    const decoder = new TextDecoder();
    let carry = '';
    let full = '';
    for await (const chunk of r.body) {
      res.write(chunk);
      const text = decoder.decode(chunk, { stream: true });
      carry += text;
      let idx;
      while ((idx = carry.indexOf('\n')) >= 0) {
        const line = carry.slice(0, idx).trim();
        carry = carry.slice(idx + 1);
        if (!line) continue;
        try {
          const obj = JSON.parse(line);
          if (obj.response) full += obj.response;
        } catch {}
      }
    }
    if (session_id && conv) {
      appendMessage(conv.id, 'user', prompt);
      appendMessage(conv.id, 'assistant', (full || '').trim());
    }
    res.end();
  } catch (e) { console.error(e); res.status(500).json({ error: e.message }); }
});

app.post('/aurion_smart', async (req, res) => {
  try {
    const { prompt, reliability = 'normal', style = 'paragraph', system, session_id, user_id } = req.body || {};
    if (!prompt) return res.status(400).json({ error:'prompt requis' });

    const hit = await lookupFact(prompt);
    if (hit) {
      return res.json({ reply: hit.answer, meta: { mode:'fact', confidence: hit.sim, source: hit.source, cache:'FactStore', updated_at: hit.updated_at } });
    }

    let conv = null, messages = [], historyPrefix = '';
    if (session_id) {
      const result = await getHistoryContext(session_id, user_id || null);
      conv = result.conv;
      messages = result.messages;
      historyPrefix = buildConversationPrefix(result.conv.summary, messages);
    }

    const factual = reliability === 'high';
    let web = null;
    if (factual && TAVILY_API_KEY) {
      try { web = await tavilySearch(prompt); }
      catch (e) { console.warn('Tavily error:', e.message); }
    }

    const sysStyle = (style === 'bullets')
      ? 'Réponds en puces courtes, précises, avec dates chiffrées si pertinent.'
      : 'Réponds clairement, en français, sans blabla.';
    const nowLine = `Contexte actuel: ${timeSummary()}. Ajuste les références temporelles (Europe/Paris).`;
    const historyLine = historyPrefix ? `\n${historyPrefix}\n` : '';

    const finalSystem = [
      'Tu es Aurion: concis, fiable, factuel.',
      sysStyle,
      nowLine,
      historyLine,
      web && web.answer ? 'Utilise les éléments fiables trouvés en ligne si fournis.' : ''
    ].filter(Boolean).join('\n');

    let draft = '';
    if (web && web.answer) {
      const ctx = [
        `Contexte web (synthèse Tavily): ${web.answer}`,
        ...(web.results || []).slice(0, 3).map((r, i) => `Source ${i+1}: ${r.title} — ${r.url}`)
      ].join('\n');
      draft = await askLLM(`${ctx}\n\nQuestion: ${prompt}\nRéponse:`, finalSystem, 0.4, 700);
    } else {
      draft = await askLLM(prompt, [system || '', finalSystem].filter(Boolean).join('\n'), 0.5, 600);
    }

    // persist + style
    if (session_id && conv) {
      appendMessage(conv.id, 'user', prompt);
      appendMessage(conv.id, 'assistant', (draft || '').trim());
    }
    let finalDraft = (draft || '').trim();
    if (style && ['aurion','genz','pro','friendly','sobre','kronos'].includes(style)) {
      // ici "style" est utilisé pour stylize seulement si c'est une de nos clés
      finalDraft = await stylize(finalDraft, style === 'paragraph' ? 'aurion' : style);
    }

    res.json({
      reply: finalDraft,
      meta: { mode: web ? 'web+llm' : 'llm', sources: web?.results?.slice(0,3)?.map(r => ({ title: r.title, url: r.url })) || [], reliability, history: !!session_id }
    });
  } catch (e) { console.error(e); res.status(500).json({ error: e.message }); }
});

// ---------- START ----------
app.listen(PORT, () => {
  console.log(`Aurion proxy up on http://localhost:${PORT}`);
  console.log(`Model: ${OLLAMA_MODEL} | Embeddings: ${EMBED_MODEL} | Tavily: ${TAVILY_API_KEY ? 'on' : 'off'}`);
});
