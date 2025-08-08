# Project Roadmap: Tracer Bullet

This is a flexible, milestone-based roadmap. We will build at our own pace and check off each phase as it is completed.

### Phase 0: Foundation Setup
- [ ] Create project folder and initialize Git repository.
- [ ] Create and link a private GitHub remote repository.
- [ ] Create all initial documentation files (`README.md`, `roadmap.md`, `knowledge_wiki.md`, `dev_log.md`).
- [ ] Register for all necessary API keys (`Alpaca`, `Perplexity`).
- [ ] Set up the local SQLite database and its initial schema.

### Phase 1: The "Tracer Bullet" (Monolithic Proof of Concept)
- [ ] Build a single script (`tracer_bullet.py`) that proves the end-to-end concept.
- [ ] The script must:
    - [ ] Connect to the Alpaca API and fetch the price of one asset.
    - [ ] Connect to the Perplexity API and fetch a summary for that asset.
    - [ ] Perform a crude, combined analysis and print a result.

### Phase 2: The Professional Refactor & V1 Build
- [ ] Refactor the `tracer_bullet.py` logic into a clean, modular architecture.
- [ ] Create and test dedicated fetcher and analysis scripts.
- [ ] Implement the full "Narrative vs. Price Momentum" logic.
- [ ] Integrate the SQLite database for professional logging.
- [ ] The deliverable is a fully functional V1 agent that can autonomously run, analyze, and place paper trades.

### Phase 3: Backtesting & Optimization
- [ ] Build the `backtester.py` module.
- [ ] Use Alpaca's historical data to test and optimize the V1 agent's strategy.
- [ ] Refine the parameters in our `analysis_core.py` based on historical performance.

### Phase 4: Live Validation & V2 Planning
- [ ] Deploy the optimized agent to run continuously in paper trading mode.
- [ ] Monitor its live performance and gather data.
- [ ] Plan the roadmap for V2, which could include live trading with real capital.