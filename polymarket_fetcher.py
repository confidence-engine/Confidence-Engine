"""
DEPRECATED: Use `providers/polymarket_pplx.py` (Perplexity Pro provider) and
`scripts/polymarket_bridge.py` for mapping/filtering.
This native fetcher remains only for historical reference and may be removed after v4.3.
"""
import requests
import time
import pandas as pd

SPORTS_KEYWORDS = ['nba', 'nfl', 'soccer', 'mlb', 'tennis', 'hockey', 'golf', 'football', 'basketball']

def fetch_polymarket_sports_markets(limit=50, max_pages=10):
    base_url = "https://gamma-api.polymarket.com/markets"
    params = {
        "active": "true",
        "tagSlug": "sports",  # Can try multiple calls with different tagSlugs like 'nba', 'nfl'
        "offset": 0,
        "limit": limit
    }

    all_markets = []
    pages_fetched = 0

    print(f"--- Fetching ALL LIVE Sports Markets from Polymarket (Filtered by Keywords) ---")

    while pages_fetched < max_pages:
        print(f"  -> Fetching page {pages_fetched + 1}...")
        response = requests.get(base_url, params=params, timeout=20)
        if response.status_code != 200:
            print(f"Error: Received {response.status_code} status from API")
            break

        markets_page = response.json()
        if not markets_page or not isinstance(markets_page, list):
            print("  -> No more markets or unexpected response format. Ending fetch.")
            break

        # Filter markets by:
        # - Not closed, not resolved, not paused, not cancelled
        # - category includes 'sports'
        # - question or event title contains sports-related keywords
        filtered_markets = []
        for m in markets_page:
            if m.get('closed') or m.get('resolved') or m.get('paused') or m.get('cancelled'):
                continue

            category = m.get('category', '').lower()
            question = m.get('question', '').lower()
            event_title = (m.get('event_title') or '').lower()

            if 'sports' not in category:
                continue

            # Check if question or event title contains sports keywords
            if any(kw in question for kw in SPORTS_KEYWORDS) or \
               any(kw in event_title for kw in SPORTS_KEYWORDS):
                filtered_markets.append(m)

        all_markets.extend(filtered_markets)

        if len(markets_page) < limit:
            print("  -> Last page reached.")
            break

        pages_fetched += 1
        params['offset'] += limit
        time.sleep(0.5)  # Be kind to API

    if all_markets:
        df = pd.DataFrame(all_markets).drop_duplicates(subset=['id'])
        print(f"\nTotal live sports markets after filtering: {len(df)}")
        return df
    else:
        print("No live sports markets found with keyword filtering.")
        return None

# Example usage:
if __name__ == "__main__":
    sports_df = fetch_polymarket_sports_markets()
    if sports_df is not None:
        print(sports_df[['question', 'volume', 'category', 'closed', 'event_title']].head())
    else:
        print("No suitable sports markets found.")
