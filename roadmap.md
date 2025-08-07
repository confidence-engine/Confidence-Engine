# Project Roadmap: Tracer Bullet

This is a flexible, milestone-based roadmap. We will check off each phase as it is completed, at our own pace.

### Phase 0: Foundation Setup
- [ ] Create project folder and initialize Git repository.
- [ ] Create and link a private GitHub remote repository.
- [ ] Create all initial documentation files (`README.md`, `roadmap.md`, `knowledge_wiki.md`).
- [ ] Register for all necessary API keys (`The Odds API`, `Dexsport`, `Perplexity`).
- [ ] Set up the local SQLite database and its initial schema.

### Phase 1: The Tracer Bullet (Monolithic Script)
- [ ] Build a single, monolithic script (`tracer_bullet.py`) that proves the end-to-end concept.
- [ ] The script must:
    - [ ] Fetch odds from `The Odds API` for a hardcoded event.
    - [ ] Fetch odds from `Dexsport` for the same event.
    - [ ] Send a query to the `Perplexity API` for qualitative context.
    - [ ] Parse the results from all three sources.
    - [ ] Perform a simple value comparison.
    - [ ] Print a final, reasoned result to the terminal.

### Phase 2: The Professional Refactor
- [ ] Refactor the `tracer_bullet.py` logic into a clean, modular architecture.
- [ ] Create and test a dedicated `odds_api_fetcher.py`.
- [ ] Create and test a dedicated `dexsport_fetcher.py`.
- [ ] Create and test a dedicated `perplexity_fetcher.py`.
- [ ] Create and test a robust `nlp_parser.py` module.
- [ ] Create and test the `analysis_core.py` with the refined scoring logic.
- [ ] Create the main `agent.py` orchestrator script.

### Phase 3: The Live Signal Service (V1 MVP)
- [ ] Build the professional `db_manager.py` with a robust SQLite logging schema.
- [ ] Integrate the SQLite logger into the `agent.py` script.
- [ ] Build the `signal_publisher.py` to send formatted messages to Discord/Telegram.
- [ ] Deploy the agent to run on a schedule and begin live, non-monetary signal generation and validation.