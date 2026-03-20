from StockSentiment import fetch_articles, parse_date, UNIQUE_FIELD_JSON
from getsentimentFinBERT import get_finbert_score
from getsentimentFlair import get_flair_score
import time
import pandas as pd

def fetch_cnbc_articles(ticker, max_pages=1):
    """Fetch real CNBC articles without scoring them."""
    articles = []
    seen_urls = set()
    endindex = 0

    for page in range(max_pages):
        data = fetch_articles(endindex, ticker)
        if not data or 'results' not in data:
            break

        for article in data['results']:
            url = article.get(UNIQUE_FIELD_JSON)
            if url in seen_urls:
                continue
            seen_urls.add(url)

            title = article.get('cn:title', '').strip()
            description = article.get('description', '').strip()
            pub_date = parse_date(article.get('datePublished'))

            if title:
                articles.append({
                    "title": title,
                    "description": description,
                    "published_date": pub_date,
                    "url": url,
                    "input_text": f"{title}: {description}",
                })

        metadata = data.get('metadata', {})
        page_size = metadata.get('pagesize', 100)
        if len(data.get('results', [])) < page_size:
            break
        endindex += page_size

    return articles


def run_comparison(ticker="SPY", max_pages=1):
    print(f"Fetching CNBC articles for '{ticker}'...")
    articles = fetch_cnbc_articles(ticker, max_pages)
    print(f"{len(articles)} articles loaded.\n")

    if not articles:
        print("No articles found.")
        return

    print("=" * 100)
    print(f"{'SENTIMENT MODEL COMPARISON — REAL CNBC NEWS':^100}")
    print(f"{'FinBERT vs. Flair | Ticker: ' + ticker:^100}")
    print("=" * 100)

    finbert_scores = []
    flair_scores = []
    finbert_total_time = 0
    flair_total_time = 0

    for i, art in enumerate(articles, 1):
        text = art["input_text"]

        start = time.perf_counter()
        fb_score = get_finbert_score(text)
        finbert_total_time += time.perf_counter() - start

        start = time.perf_counter()
        fl_score = get_flair_score(text)
        flair_total_time += time.perf_counter() - start

        finbert_scores.append(fb_score)
        flair_scores.append(fl_score)

        diff = abs(fb_score - fl_score)
        print(f"\n[{i}] {art['title'][:85]}{'...' if len(art['title']) > 85 else ''}")
        print(f"    FinBERT: {fb_score:+.4f}  |  Flair: {fl_score:+.4f}  |  Diff: {diff:.4f}")

    # Zusammenfassung
    n = len(articles)
    avg_fb = sum(finbert_scores) / n
    avg_fl = sum(flair_scores) / n
    avg_diff = sum(abs(f - l) for f, l in zip(finbert_scores, flair_scores)) / n
    agree = sum(1 for f, l in zip(finbert_scores, flair_scores) if (f > 0) == (l > 0) or f == 0 or l == 0)

    print("\n" + "=" * 100)
    print(f"{'ZUSAMMENFASSUNG':^100}")
    print("=" * 100)

    print(f"\n  {'Metrik':<40} {'FinBERT':>10} {'Flair':>10}")
    print(f"  {'-' * 60}")
    print(f"  {'Artikel analysiert':<40} {n:>10} {n:>10}")
    print(f"  {'Durchschnittlicher Score':<40} {avg_fb:>+10.4f} {avg_fl:>+10.4f}")
    print(f"  {'Gesamtzeit (s)':<40} {finbert_total_time:>10.2f} {flair_total_time:>10.2f}")
    print(f"  {'Zeit pro Artikel (ms)':<40} {finbert_total_time/n*1000:>10.1f} {flair_total_time/n*1000:>10.1f}")
    print(f"\n  Durchschnittliche Abweichung:  {avg_diff:.4f}")
    print(f"  Richtungsübereinstimmung:      {agree}/{n} ({agree/n*100:.0f}%)")

    # Größte Abweichungen
    diffs = [(abs(f - l), i) for i, (f, l) in enumerate(zip(finbert_scores, flair_scores))]
    diffs.sort(reverse=True)

    print(f"\n  Top 5 größte Abweichungen:")
    for diff, idx in diffs[:5]:
        print(f"    [{idx+1}] Diff: {diff:.4f} | {articles[idx]['title'][:70]}")

    print()


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    pages = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run_comparison(ticker, pages)
