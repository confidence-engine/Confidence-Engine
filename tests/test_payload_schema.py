def test_payload_keys_unchanged(monkeypatch):
    # Intercept DB save and exports to capture payload
    captured = {}

    def fake_save_run(payload):
        captured.update(payload)
        return 999

    def fake_export_run_json(run_id, payload):
        pass

    def fake_save_bars_csv(run_id, bars):
        pass

    def fake_save_accepted_txt(run_id, lines):
        pass

    # Avoid network/telegram
    monkeypatch.setenv("TB_NO_TELEGRAM", "1")

    # Stub heavy ML deps so importing finbert doesn't require torch/transformers
    import sys
    import types

    torch_stub = types.SimpleNamespace()

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    torch_stub.no_grad = lambda: _NoGrad()
    torch_stub.softmax = lambda x, dim=-1: x
    sys.modules.setdefault("torch", torch_stub)

    class _Tok:
        def __call__(self, *a, **k):
            return {}

    class _Model:
        def eval(self):
            return None

        def __call__(self, *a, **k):
            return types.SimpleNamespace(logits=None)

    class _TransformersModule(types.SimpleNamespace):
        class AutoTokenizer:
            @staticmethod
            def from_pretrained(_name):
                return _Tok()

        class AutoModelForSequenceClassification:
            @staticmethod
            def from_pretrained(_name):
                return _Model()

    # Provide a minimal package-like module structure for transformers to satisfy
    # sentence_transformers imports (which expect submodules like .configuration_utils)
    transformers_pkg = types.ModuleType("transformers")
    transformers_pkg.AutoTokenizer = _TransformersModule.AutoTokenizer
    transformers_pkg.AutoModelForSequenceClassification = _TransformersModule.AutoModelForSequenceClassification
    # Create dummy submodules needed by sentence_transformers.backend.load
    configuration_utils = types.ModuleType("transformers.configuration_utils")
    class _PretrainedConfig:  # minimal placeholder
        pass
    configuration_utils.PretrainedConfig = _PretrainedConfig
    sys.modules.setdefault("transformers", transformers_pkg)
    sys.modules.setdefault("transformers.configuration_utils", configuration_utils)

    # Stub sentence_transformers to avoid pulling heavy deps
    st_mod = types.ModuleType("sentence_transformers")

    class SentenceTransformer:  # type: ignore
        def __init__(self, _name: str):
            pass

        def encode(self, texts, normalize_embeddings: bool = False):
            import numpy as np

            n = len(texts)
            embs = np.zeros((n, 3), dtype=float)
            for i, t in enumerate(texts):
                h = sum(ord(c) for c in str(t)) % 997
                vec = np.array([((h % 7) + 1), ((h % 11) + 1), ((h % 13) + 1)], dtype=float)
                if normalize_embeddings:
                    norm = float(np.linalg.norm(vec) or 1.0)
                    vec = vec / norm
                embs[i] = vec
            return embs

    st_mod.SentenceTransformer = SentenceTransformer
    sys.modules.setdefault("sentence_transformers", st_mod)

    # Patch targets inside module namespace after import
    # Stub 'alpaca' module to avoid importing alpaca-trade-api
    alpaca_stub = types.ModuleType("alpaca")

    class DummyBars:
        def __init__(self):
            from datetime import datetime, timezone, timedelta

            # minimal index-like with to_pydatetime
            class _IdxItem:
                def __init__(self, dt):
                    self._dt = dt

                def to_pydatetime(self):
                    return self._dt

            now = datetime.now(timezone.utc)
            self._idx = [_IdxItem(now - timedelta(minutes=2)), _IdxItem(now - timedelta(minutes=1)), _IdxItem(now)]

        def __len__(self):
            return len(self._idx)

        @property
        def index(self):
            return self._idx

        def tail(self, n):
            return self

    def _recent_bars(symbol, lookback):
        return DummyBars()

    def _latest_headlines(symbol, limit):
        return ["Bitcoin rises as ETF inflows grow"]

    alpaca_stub.recent_bars = _recent_bars
    alpaca_stub.latest_headlines = _latest_headlines
    sys.modules.setdefault("alpaca", alpaca_stub)

    import tracer_bullet as tb

    monkeypatch.setattr(tb, "save_run", fake_save_run)
    monkeypatch.setattr(tb, "export_run_json", fake_export_run_json)
    monkeypatch.setattr(tb, "save_bars_csv", fake_save_bars_csv)
    monkeypatch.setattr(tb, "save_accepted_txt", fake_save_accepted_txt)

    # Also patch upstream data sources to return deterministic minimal data
    monkeypatch.setenv("PPLX_ENABLED", "false")
    monkeypatch.setenv("USE_COINDESK", "false")

    def fake_fetch_coindesk_titles():
        return []

    def fake_fetch_pplx_headlines_with_rotation(keys, hours=24):
        return [], [], None

    def fake_filter_relevant_weighted(items, threshold, source_lookup_fn, weight_overrides=None):
        # Accept the single item
        return [(items[0], 0.5, 0.5, "alpaca")], []

    class Nar:
        narrative_momentum_score = 0.7
        confidence = 0.8
        narrative_summary = "Positive catalysts expected."

    def fake_make_from_headlines(used, threshold):
        return Nar()

    def fake_sentiment_robust(used):
        return 0.2, [0.1, 0.3], []

    def fake_price_score(bars):
        return 0.1, 0.0

    def fake_adaptive_trigger(base, vz):
        return base

    # Patch in module namespace
    # recent_bars and latest_headlines provided by stub module already
    monkeypatch.setattr(tb, "fetch_coindesk_titles", fake_fetch_coindesk_titles)
    monkeypatch.setattr(tb, "fetch_pplx_headlines_with_rotation", fake_fetch_pplx_headlines_with_rotation)
    monkeypatch.setattr(tb, "filter_relevant_weighted", fake_filter_relevant_weighted)
    monkeypatch.setattr(tb, "make_from_headlines", fake_make_from_headlines)
    monkeypatch.setattr(tb, "sentiment_robust", fake_sentiment_robust)
    monkeypatch.setattr(tb, "price_score", fake_price_score)
    monkeypatch.setattr(tb, "adaptive_trigger", fake_adaptive_trigger)

    # Run
    tb.main()

    # Validate presence of key fields (don't remove existing keys)
    expected_keys = {
        "ts_utc",
        "symbol",
        "bars_lookback_min",
        "headlines_count",
        "raw_headlines",
        "btc_filtered_headlines",
        "relevance_details",
        "finbert_score",
        "finbert_kept_count",
        "finbert_dropped_count",
        "finbert_kept_scores",
        "finbert_dropped_scores",
        "llm_score",
        "raw_narrative",
        "decayed_narrative",
        "price_score",
        "volume_z",
        "divergence",
        "divergence_threshold",
        "confidence",
        "action",
        "reason",
        "alpha_summary",
        "alpha_next_steps",
        "story_excerpt",
        "summary",
        "detail",
        "pplx_provenance",
        "pplx_last_error",
        "source_diversity",
        "cascade_detector",
        "contrarian_viewport",
    }

    missing = expected_keys.difference(captured.keys())
    assert not missing, f"Missing keys: {missing}"


