# real-estate-buyer-segmentation_project
An AI-driven buyer segmentation pipeline and interactive Streamlit dashboard for real estate market intelligence, utilizing K-Means clustering and hierarchical validation.

# Parcl Buyer Segmentation & Investment Profiling — Streamlit App

## What this is
A self-contained Streamlit dashboard that cleans `clients.csv` + `properties.csv`,
engineers buyer-level features, clusters buyers with K-Means (validated against
Hierarchical clustering), and visualizes the results across four modules:
Segmentation Overview, Investor Behavior Dashboard, Geographic Buyer Analysis,
and a Segment Insights Panel — plus an optional Model Diagnostics tab.

## Folder contents
```
app.py              # the entire application (data cleaning -> clustering -> dashboard)
requirements.txt     # Python dependencies
data/clients.csv     # bundled default dataset (2,000 clients)
data/properties.csv  # bundled default dataset (10,000 listings)
```

## How to run
1. Install dependencies (Python 3.9+ recommended):
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the app from this folder:
   ```bash
   streamlit run app.py
   ```
3. The app opens in your browser at `http://localhost:8501` and loads the
   bundled `data/clients.csv` and `data/properties.csv` automatically.
4. To analyze your own data instead, open the **Data Source** panel in the
   sidebar, check "Upload my own files", and upload CSVs with the same column
   names described at the top of `app.py`.

## Using the dashboard
- **Sidebar → Clustering Model**: choose the number of segments (k, 2-8).
  Segment names are generated automatically from each cluster's behavior
  (spend, portfolio size, price point, financing/company mix).
- **Sidebar → Filters**: narrow the dashboard to a country, region,
  acquisition purpose, client type, or segment.
- **Segmentation Overview**: cluster sizes, PCA projection, snapshot table.
- **Investor Behavior Dashboard**: spend distributions, financing behavior,
  acquisition purpose split, price-vs-size scatter, sales volume over time.
- **Geographic Buyer Analysis**: choropleth map, top countries, country →
  region → segment breakdown.
- **Segment Insights Panel**: deep-dive on one segment at a time, with a
  CSV download of that segment's client list.
- **Model Diagnostics**: elbow method, silhouette score curve, and a
  Hierarchical-clustering cross-check (Adjusted Rand Index + dendrogram).

## Notes
- Nothing in this app calls external APIs — everything runs locally against
  the two CSV files.
- The clustering re-runs (cached) whenever you change k; filters only change
  what's *displayed*, not how clusters are computed, so segment definitions
  stay stable while you explore.

