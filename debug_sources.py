from config import settings
from alpaca import latest_headlines
from pplx_fetcher import fetch_pplx_headlines_with_rotation
try:
    from coindesk_rss import fetch_coindesk_titles
except Exception:
    fetch_coindesk_titles = None

def main():
    print(f"[DebugSources] symbol={settings.symbol}")
    alp = latest_headlines(settings.symbol, settings.headlines_limit)
    print(f"- Alpaca: {len(alp)}")
    for t in alp[:5]:
        print(f"  • {t}")

    pplx_titles, pplx_items, pplx_err = ([], [], None)
    if settings.pplx_enabled and settings.pplx_api_keys:
        pplx_titles, pplx_items, pplx_err = fetch_pplx_headlines_with_rotation(
            settings.pplx_api_keys, hours=settings.pplx_hours
        )
    print(f"- Perplexity: {len(pplx_titles)} (err={pplx_err})")
    for t in pplx_titles[:5]:
        print(f"  • {t}")

    if getattr(settings, "use_coindesk", False) and fetch_coindesk_titles:
        cd = fetch_coindesk_titles()
        print(f"- CoinDesk: {len(cd)}")
        for t in cd[:5]:
            print(f"  • {t}")
    else:
        print("- CoinDesk: disabled")

if __name__ == "__main__":
    main()
