
Project Charter & Roadmap: Project Tracer Bullet


1. Project Overview

Mission:
To build a professional-grade Quantitative Alpha Engine. The V1 of this agent will be an autonomous system designed to find and validate trading opportunities in the stock and crypto markets. Its core function is to identify short-term, sentiment-driven inefficiencies.
The primary goal of this project is the deep acquisition of elite skills in AI development, data engineering, and quantitative finance. The agent itself is the verifiable proof of work.
Core Strategy: "Narrative vs. Price Momentum Divergence"
Our agent's "alpha," or competitive edge, is derived from a sophisticated, hybrid intelligence model.
1. It ingests two independent data streams from the Alpaca API: real-time price action and real-time news headlines.
2. It uses a powerful research agent (the Perplexity API) to synthesize the raw news into a high-level, qualitative narrative.
3. Its Analysis Core, powered by a finBERT model, quantifies this narrative into a "Narrative Momentum Score."
4. It simultaneously calculates a "Price Momentum Score" from the market data.
5. Its primary function is to identify and act upon divergences between these two momentum scores, exploiting the market's temporary under-reaction or over-reaction to new information.

2. The Optimized Technology Stack

This stack is selected for its professional-grade capabilities, reliability, and adherence to our "no cost" V1 principle.
Category	Tool	Rationale
Language	Python 3.9+	The industry standard for data science and algorithmic trading.
Core Library	Pandas	For all data manipulation and analysis.
Development Environment	Cursor	For AI-assisted coding, debugging, and project management.
Primary Data & Trading	Alpaca API	A stable, developer-first API for all price, news, and paper trading.
Qualitative Research	Perplexity API	A powerful AI engine for synthesizing narrative and context.
Sentiment Analysis	Hugging Face Transformers	Specifically, the ProsusAI/finbert model for financial context.
NLP	spaCy	For robust keyword extraction and language processing.
Database	SQLite	A simple, powerful, and file-based database for professional logging.
Version Control	Git / GitHub	For professional code management and automated data logging.
3. Project Architecture

The agent is a modular system with a clear separation of concerns.
* Data Pipeline (alpaca_fetcher.py, perplexity_fetcher.py): A simplified and robust pipeline with only two primary data fetchers.
* Analysis Core (analysis_core.py, sentiment_analyzer_v2.py, keyword_extractor.py): The "brain" of the agent. It contains the logic for NLP, sentiment scoring, and the final divergence analysis to generate a Confidence Score.
* Execution & Logging (agent.py, db_manager.py): The main orchestrator that runs the scan, executes paper trades via the Alpaca API, and logs all data and decisions to the SQLite database.

4. Risk Assessment & Mitigation

* Primary Technical Risk: Building a robust NLP parser to reliably extract structured data from the unstructured text of the Perplexity API.
    * Mitigation: We will use a strict Test-Driven Development (TDD) methodology for this specific component, with a comprehensive suite of tests to ensure its accuracy.
* Primary Strategic Risk: The "Narrative vs. Price" strategy, while sound, may not be profitable.
    * Mitigation: We will build a comprehensive Backtesting Engine in Phase 3 to rigorously test and optimize the strategy against historical data before ever deploying it with real capital.

5. The Project Roadmap (Milestone-Based)

This is our flexible, milestone-based roadmap. We will build at your pace and check off each phase as it is completed.
* [ ] Phase 0: Foundation Setup
    * Create the Project-Tracer-Bullet folder and initialize the Git repository.
    * Register for an Alpaca account and get paper trading API keys.
    * Securely store all API keys in the .env file.
    * Set up the local logs.db SQLite database file and its initial schema.
* [ ] Phase 1: The "Tracer Bullet" (Monolithic Proof of Concept)
    * Build a single, monolithic script (tracer_bullet.py) that proves the end-to-end concept works.
    * The script must:
        1. Connect to the Alpaca API and fetch the price of one asset.
        2. Connect to the Perplexity API and fetch a summary for that asset.
        3. Perform a crude, combined analysis and print a result.
* [ ] Phase 2: The Professional Refactor & V1 Build
    * Refactor the tracer_bullet.py logic into our clean, modular architecture (alpaca_fetcher, perplexity_fetcher, analysis_core, agent.py).
    * Implement the full "Narrative vs. Price Momentum" logic.
    * Integrate the SQLite database for professional logging.
    * The deliverable is a fully functional V1 agent that can autonomously run, analyze, and place paper trades.
* [ ] Phase 3: Backtesting & Optimization
    * Build the backtester.py module.
    * Use Alpaca's historical data to test and optimize the V1 agent's strategy.
    * Refine the parameters in our analysis_core.py based on historical performance.
* [ ] Phase 4: Live Validation & V2 Planning
    * Deploy the optimized agent to run continuously in paper trading mode.
    * Monitor its live performance and gather data.
    * Plan the roadmap for V2, which could include live trading with real capital.
This is the definitive blueprint for our project. The brainstorming is complete.
