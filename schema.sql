CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc TEXT NOT NULL,
  symbol TEXT NOT NULL,
  bars_lookback_min INTEGER NOT NULL,
  headlines_count INTEGER NOT NULL,
  raw_headlines TEXT NOT NULL,     -- JSON list
  btc_filtered_headlines TEXT NOT NULL, -- JSON list
  finbert_score REAL NOT NULL,
  llm_score REAL NOT NULL,
  raw_narrative REAL NOT NULL,
  decayed_narrative REAL NOT NULL,
  price_score REAL NOT NULL,
  volume_z REAL NOT NULL,
  divergence REAL NOT NULL,
  divergence_threshold REAL NOT NULL,
  confidence REAL NOT NULL,
  action TEXT NOT NULL,
  reason TEXT NOT NULL,
  summary TEXT NOT NULL,           -- short natural-language summary
  detail TEXT NOT NULL             -- longer explanation
);
