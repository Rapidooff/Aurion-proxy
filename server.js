// server.js — Aurion Proxy (ESM) full power
// Run: node server.js  (Node 18+)
// Features: mémoire locale (SQLite + embeddings), horloge Europe/Paris, styles,
// intents heure/date, /aurion, /aurion_stream (buffer NDJSON), /aurion_once, /aurion_smart

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import Database from 'better-sqlite3';
import { Buffer } from 'node:buffer';

// ---------- CONFIG ----------
const PORT = Number(process.env.PORT || 3000);
const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';
// Accepte OLLAMA_MODEL ou AURION_MODEL (fallback)
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || process.env.AURION_MODEL || 'gemma3:1b';
const EMBED_MODEL = process.env.EMBED_MODEL || 'nomic-embed-text';
const TAVILY_API_KEY = process.env.TAVILY_API_KEY || ''; // facultatif
const FACT_SIM_THRESHOLD = Number(process.env.FACT_SIM_THRESHOLD || 0.85);

// ---------- APP ----------
const app = express();
app.use(cors());
app.use(express.json({ limit: '2mb' }));

// ---------- DB (FactStore) ----------
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
`);

// ---------- UTILS: texte ----------
function normalizeQuestion(q) {
  return (q || '')
    .toLowerCase()
    .trim()
    .replace(/\s+/g, ' ')
    .replace(/[!?.,;:()"'’“”«»]/g, '');
}

// ---------- UTILS: temps Europe/Paris ----------
function parisNow() {
  const tz = 'Europe/Paris';
  const now = new Date();
  const date = new Intl.DateTimeFormat('fr-FR', {
    timeZone: tz, weekday: 'long', year: 'numeric', month: 'long', day: 'numeric'
  }).format(now);
  const time = new Intl.DateTimeFormat('fr-FR', {
    timeZone: tz, hour: '2-digit', minute: '2-digit', second: '2-digit'
  }).format(now);
  // ISO alignée sur le TZ pour logs
  const iso = new Date(now.toLocaleString('en-US', { timeZone: tz })).toISOString();
  return { tz, date, time, iso, epoch: now.getTime() };
}
function timeSummary() {
  const n = parisNow();
  return `${n.date} — ${n.time} (${n.tz})`;
}

// ---------- Styles de réponses pour l'heure/date ----------
const replyStyles = {
  'sobre':   ({ dateLine }) => `Nous sommes le ${dateLine}.`,
  'friendly':({ dateLine }) => `On est le ${dateLine} 😉`,
  'genz':    ({ dateLine }) => `Mise à jour IRL: ${dateLine} — synchro ok. ⏱️`,
  'pro':     ({ dateLine }) => `Contexte temporel: ${dateLine}.`,
  'aurion':  ({ dateLine }) => `Horloge synchronisée: ${dateLine}. Prêt à exécuter.`,
};
function renderTimeAnswer(style = 'aurion') {
  const n = parisNow();
  const dateLine = `${n.date}, ${n.time} (${n.tz})`;
  const f = replyStyles[style] || replyStyles['aurion'];
  return f({ dateLine, now: n });
}

// ---------- Intent: requêtes d'heure/date ----------
function isTimeQuery(text) {
  const q = (text || '').toLowerCase();
  return [
    /quelle heure/,
    /il est quelle heure/,
    /donne l'heure/,
    /donne l’heure/,
    /c[’']?est quel jour/,
    /on est quel jour/,
    /la date/,
    /quel jour sommes-nous/,
    /\btime\b|\bdate\b/
  ].some(r => r.test(q));
}

// ---------- Math: cosinus ----------
function cosineSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
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

// ---------- FactStore ops ----------
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

  // purge TTL
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
    const buf = r.vector; // Buffer
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

// ---------- Tavily (optionnel) ----------
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

// ---------- ROUTES ----------
app.get('/health', async (_req, res) => {
  let ollama = 'down';
  try {
    const ping = await fetch(`${OLLAMA_HOST}/api/tags`);
    if (ping.ok) ollama = 'up';
  } catch { ollama = 'down'; }
  res.json({
    ok: true,
    model: OLLAMA_MODEL,
    ollama,
    tavily: TAVILY_API_KEY ? 'configured' : 'off',
    time: {
      iso: new Date().toISOString(),
      human: timeSummary(),
      tz: 'Europe/Paris'
    }
  });
});

app.get('/now', (_req, res) => {
  const n = parisNow();
  res.json({ ok: true, now: n, human: `${n.date} — ${n.time} (${n.tz})` });
});

app.post('/feedback', async (req, res) => {
  try {
    const { question, correct_answer, source, ttl_days } = req.body || {};
    if (!question || !correct_answer) {
      return res.status(400).json({ ok:false, error:'question et correct_answer requis' });
    }
    const id = await upsertFact(question, correct_answer, source || 'user-correction', ttl_days ?? null);
    res.json({ ok:true, id });
  } catch (e) {
    console.error(e);
    res.status(500).json({ ok:false, error: e.message });
  }
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
    const all = db.prepare(`
      SELECT id, question_norm, source, ttl_days, updated_at
      FROM facts ORDER BY updated_at DESC LIMIT 200
    `).all();
    return res.json({ ok:true, count: all.length, items: all });
  }
  const qn = normalizeQuestion(q);
  const row = db.prepare(`SELECT * FROM facts WHERE question_norm=?`).get(qn);
  if (!row) return res.json({ ok:true, found:false });
  res.json({ ok:true, found:true, item: row });
});

// ---- /aurion : time intent -> FactStore -> LLM (avec contexte temps) ----
app.post('/aurion', async (req, res) => {
  try {
    const { prompt, system, temperature, maxTokens, style = 'aurion' } = req.body || {};
    if (!prompt) return res.status(400).json({ error:'prompt requis' });

    if (isTimeQuery(prompt)) {
      return res.json({ reply: renderTimeAnswer(style), meta: { mode:'time', style, now: parisNow() } });
    }

    const hit = await lookupFact(prompt);
    if (hit) {
      return res.json({
        reply: hit.answer,
        meta: { mode:'fact', confidence: hit.sim, source: hit.source, cache:'FactStore', updated_at: hit.updated_at }
      });
    }

    const nowLine = `Contexte actuel: ${timeSummary()}. Réponds en cohérence avec cette date/heure (Europe/Paris).`;
    const finalSystem = [system || '', nowLine].filter(Boolean).join('\n');
    const reply = await askLLM(
      prompt,
      finalSystem,
      Number(temperature ?? 0.6),
      Number(maxTokens ?? 512)
    );
    res.json({ reply: (reply || '').trim(), meta:{ mode:'llm' } });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

// ---- /aurion_once : réponse non-stream, toujours “une phrase”/bloc complet ----
app.post('/aurion_once', async (req, res) => {
  try {
    const { prompt, system, temperature, maxTokens = 256, style = 'aurion' } = req.body || {};
    if (!prompt) return res.status(400).json({ error:'prompt requis' });

    if (isTimeQuery(prompt)) {
      return res.json({ reply: renderTimeAnswer(style), meta: { mode:'time', style, now: parisNow() } });
    }

    const hit = await lookupFact(prompt);
    if (hit) {
      return res.json({ reply: hit.answer, meta: { mode:'fact', confidence: hit.sim, source: hit.source } });
    }

    const nowLine = `Contexte actuel: ${timeSummary()}. Réponds en cohérence avec cette date/heure (Europe/Paris).`;
    const finalSystem = [system || '', nowLine].filter(Boolean).join('\n');
    const reply = await askLLM(prompt, finalSystem, Number(temperature ?? 0.6), Number(maxTokens));

    res.json({ reply: (reply || '').trim(), meta: { mode:'llm' } });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

// ---- /aurion_stream : stream NDJSON ou buffer complet fiable ----
app.post('/aurion_stream', async (req, res) => {
  try {
    const { prompt, system, temperature, style = 'aurion', buffer = false } = req.body || {};
    if (!prompt) return res.status(400).json({ error:'prompt requis' });

    // time intent -> direct
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
      const payload = {
        reply: hit.answer,
        meta: { mode:'fact', confidence: hit.sim, source: hit.source, cache:'FactStore', updated_at: hit.updated_at }
      };
      if (buffer) {
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        return res.end(JSON.stringify({ ...payload, buffered: true }));
      }
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      return res.end(hit.answer);
    }

    // LLM stream
    const nowLine = `Contexte actuel: ${timeSummary()}. Réponds en cohérence avec cette date/heure (Europe/Paris).`;
    const finalSystem = [system || '', nowLine].filter(Boolean).join('\n');
    const r = await streamLLM(prompt, finalSystem, Number(temperature ?? 0.6));

    if (buffer) {
      // Accumule le NDJSON proprement (Node 18+/20: chunks = Uint8Array)
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
          } catch { /* ignore malformed split; handled by carry buffer */ }
        }
      }
      if (carry.trim()) {
        try {
          const last = JSON.parse(carry.trim());
          if (last.response) full += last.response;
        } catch { /* ignore tail */ }
      }
      res.setHeader('Content-Type', 'application/json; charset=utf-8');
      return res.end(JSON.stringify({ reply: full.trim(), meta: { mode:'llm', buffered: true } }));
    }

    // Mode stream brut (NDJSON)
    res.setHeader('Content-Type', 'application/x-ndjson; charset=utf-8');
    for await (const chunk of r.body) {
      res.write(chunk); // NDJSON via Ollama
    }
    res.end();
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

// ---- /aurion_smart : FactStore -> (optionnel) Web -> LLM (contexte temps) ----
app.post('/aurion_smart', async (req, res) => {
  try {
    const { prompt, reliability = 'normal', style = 'paragraph', system } = req.body || {};
    if (!prompt) return res.status(400).json({ error:'prompt requis' });

    const hit = await lookupFact(prompt);
    if (hit) {
      return res.json({
        reply: hit.answer,
        meta: { mode:'fact', confidence: hit.sim, source: hit.source, cache:'FactStore', updated_at: hit.updated_at }
      });
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
    const nowLine = `Contexte actuel: ${timeSummary()}. Si la question implique aujourd'hui/demain/hier/ce soir, ajuste les dates et formulations en Europe/Paris.`;

    const finalSystem = [
      'Tu es Aurion: concis, fiable, factuel.',
      sysStyle,
      nowLine,
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

    res.json({
      reply: (draft || '').trim(),
      meta: {
        mode: web ? 'web+llm' : 'llm',
        sources: web?.results?.slice(0,3)?.map(r => ({ title: r.title, url: r.url })) || [],
        reliability
      }
    });
  } catch (e) {
    console.error(e);
    res.status(500).json({ error: e.message });
  }
});

// ---------- START ----------
app.listen(PORT, () => {
  console.log(`Aurion proxy up on http://localhost:${PORT}`);
  console.log(`Model: ${OLLAMA_MODEL} | Embeddings: ${EMBED_MODEL} | Tavily: ${TAVILY_API_KEY ? 'on' : 'off'}`);
});
