# mixue indonesia sentiment

Social media analytics project: what do people actually say about Mixue Indonesia? Opinions are crawled from Instagram (#mixueindonesia) and from one leading news portal (suara.com), cleaned, filtered from bots, then modeled with sentiment analysis, aspect-based sentiment (ABSA) and topic modeling, following the bootcamp project brief.

## how to run

```
pip install -r requirements.txt
python scrape.py      # instagram, needs your own login (non-API scraping per the brief)
python news.py        # suara.com, no login needed
python clean.py       # preprocessing + bot detection
python sentiment.py   # indoBERT labeling + bert vs classical comparison
python absa.py        # sentiment per aspect (harga, rasa, pelayanan, tempat)
python topics.py      # lda topic modeling
python analytics.py   # account + media analysis
streamlit run dashboard.py   # dashboard + auto ppt generator (the brief's report automation)
```

each script can be rerun; scrapers append what they have not saved yet and dedupe on load.

## dashboard

`streamlit run dashboard.py` opens a plain app over the pipeline output (no model rerun, opens in seconds) with a ppt generator on the second page:

![dashboard](docs/screenshots/dashboard-overview.png)
![tren dan kata kunci](docs/screenshots/dashboard-trend.png)
![evaluasi model](docs/screenshots/dashboard-metrics.png)
![penjelajah data](docs/screenshots/dashboard-explorer.png)
![generator ppt](docs/screenshots/ppt-generator.png)

## scripts vs the brief

| brief item | script |
|---|---|
| source 1: instagram comments (username, text, timestamp) | scrape.py |
| source 2: one leading news portal, articles + reader comments if available | news.py |
| stage 01: case folding, noise removal, stopword removal + stemming (sastrawi) | clean.py |
| stage 02: bot vs non-bot detection (rule-based) | clean.py |
| sentiment analysis + visualization | sentiment.py |
| bert vs non-bert comparison (accuracy, precision, recall, f1) | sentiment.py |
| aspect-based sentiment analysis | absa.py |
| topic modeling (lda) | topics.py |
| account analysis (kol, engagement) + media analysis | analytics.py |
| report automation: streamlit ui + ai-generated ppt | dashboard.py |

## data files

- data/mixueindonesia_*.csv - instagram posts under the hashtag
- data/mixueindonesia_comments_*.csv - comments per post
- data/news_articles.csv, data/news_comments.csv - suara.com articles and reader comments
- data/clean.csv - merged comments, preprocessed, bot-flagged
- data/results.csv - every organic comment with label and model confidence
- data/metrics.csv - the evaluation table (svm + bert, per class and macro) shown on the dashboard
- data/aspect_summary.csv - sentiment per aspect
- data/smsa_test.tsv - public smssa benchmark split, downloaded once to score indoBERT itself
