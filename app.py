"""
================================================================================
 Buyer Segmentation & Investment Profiling for Real Estate Market Intelligence
 --------------------------------------------------------------------------
 Client   : Parcl Co. Limited
 Partner  : Unified Mentor
 Author   : Data Science Team
 Purpose  : End-to-end Streamlit application that cleans raw client and
            property-transaction data, engineers buyer-level features,
            clusters buyers with K-Means (validated against Hierarchical
            clustering), and exposes an interactive market-intelligence
            dashboard for marketing, sales and investment-strategy teams.

 Run with:
     streamlit run app.py

 Expected input files (bundled defaults in ./data/, or upload your own):
     clients.csv     -> client_id, client_type, first_name, last_name,
                         date_of_birth, gender, country, region,
                         acquisition_purpose, satisfaction_score,
                         loan_applied, referral_channel
     properties.csv  -> listing_id, tower_number, transaction_date,
                         unit_category, unit_number, floor_area_sqft,
                         sale_price, listing_status, client_ref
================================================================================
"""

import io
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# PAGE CONFIG & GLOBAL STYLE
# ==============================================================================
st.set_page_config(
    page_title="Parcl | Buyer Segmentation Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ["#2E5B88", "#C1440E", "#3E8E5A", "#8E5AA8", "#D4A017", "#4E7C8C", "#B0413E"]

# Maps our raw country labels to names Plotly's built-in choropleth understands
COUNTRY_TO_ISO_NAME = {
    "USA": "United States",
    "UK": "United Kingdom",
}

st.markdown(
    """
    <style>
    .metric-card {background-color:#f7f9fc; border-radius:10px; padding:14px;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

REFERENCE_DATE = pd.Timestamp("2026-01-01")  # analysis "as of" date


# ==============================================================================
# STEP 1-3: DATA LOADING, CLEANING & FEATURE ENGINEERING  (cached)
# ==============================================================================
@st.cache_data(show_spinner=False)
def load_raw(clients_bytes, properties_bytes):
    clients = pd.read_csv(io.BytesIO(clients_bytes))
    properties = pd.read_csv(io.BytesIO(properties_bytes))
    return clients, properties


@st.cache_data(show_spinner=False)
def clean_and_engineer(clients_bytes, properties_bytes):
    """Full Step 1-3 pipeline: cleaning, de-duplication, date/currency parsing,
    buyer-level feature engineering (RFM-style + demographic features)."""
    clients, properties = load_raw(clients_bytes, properties_bytes)

    # ---- Data Cleaning -------------------------------------------------
    clients = clients.drop_duplicates(subset="client_id").copy()
    properties = properties.drop_duplicates(subset="listing_id").copy()

    for col in ["client_type", "gender", "country", "region",
                "acquisition_purpose", "loan_applied", "referral_channel"]:
        clients[col] = clients[col].astype(str).str.strip()

    clients = clients.dropna(subset=["client_id"])

    # Money / date parsing on the property table
    properties["sale_price_clean"] = (
        properties["sale_price"].astype(str).replace(r"[\$,]", "", regex=True).astype(float)
    )
    properties["transaction_date_clean"] = pd.to_datetime(
        properties["transaction_date"], format="%d-%m-%Y", errors="coerce"
    )
    # mixed-format date-of-birth strings ("05-11-1968" / "11/26/1962")
    clients["date_of_birth_clean"] = pd.to_datetime(
        clients["date_of_birth"], format="mixed", dayfirst=False, errors="coerce"
    )
    clients["age"] = ((REFERENCE_DATE - clients["date_of_birth_clean"]).dt.days / 365.25).round().astype("Int64")

    sold = properties[properties["listing_status"] == "Sold"].copy()

    # ---- Buyer-level (RFM-style) aggregation from transactions ---------
    agg = sold.groupby("client_ref").agg(
        n_properties=("listing_id", "count"),
        total_spend=("sale_price_clean", "sum"),
        avg_price=("sale_price_clean", "mean"),
        max_price=("sale_price_clean", "max"),
        avg_floor_area=("floor_area_sqft", "mean"),
        n_towers=("tower_number", pd.Series.nunique),
        pct_apartment=("unit_category", lambda x: (x == "Apartment").mean()),
        first_purchase=("transaction_date_clean", "min"),
        last_purchase=("transaction_date_clean", "max"),
    ).reset_index().rename(columns={"client_ref": "client_id"})

    agg["tenure_months"] = ((REFERENCE_DATE - agg["first_purchase"]).dt.days / 30.44).round(1)
    agg["recency_months"] = ((REFERENCE_DATE - agg["last_purchase"]).dt.days / 30.44).round(1)

    df = clients.merge(agg, on="client_id", how="left")

    # Clients present but with no recorded sold transaction -> fill with 0 / neutral values
    fill_zero_cols = ["n_properties", "total_spend", "avg_price", "max_price",
                       "avg_floor_area", "n_towers", "pct_apartment",
                       "tenure_months", "recency_months"]
    for col in fill_zero_cols:
        df[col] = df[col].fillna(0)

    df["age"] = df["age"].fillna(df["age"].median())
    df["full_name"] = df["first_name"].astype(str) + " " + df["last_name"].astype(str)
    df["country_display"] = df["country"].replace(COUNTRY_TO_ISO_NAME)

    return df, properties


# ==============================================================================
# STEP 4-5: FEATURE MATRIX, CLUSTERING & EVALUATION (cached)
# ==============================================================================
NUMERIC_FEATURES = [
    "age", "satisfaction_score", "n_properties", "total_spend", "avg_price",
    "avg_floor_area", "n_towers", "pct_apartment", "tenure_months", "recency_months",
]
CATEGORICAL_FEATURES = ["client_type", "acquisition_purpose", "loan_applied"]


@st.cache_data(show_spinner=False)
def build_feature_matrix(df: pd.DataFrame):
    X_num = df[NUMERIC_FEATURES].copy()
    scaler = StandardScaler()
    X_num_scaled = pd.DataFrame(
        scaler.fit_transform(X_num), columns=NUMERIC_FEATURES, index=df.index
    )
    X_cat = pd.get_dummies(df[CATEGORICAL_FEATURES], drop_first=False)
    X = pd.concat([X_num_scaled, X_cat], axis=1)
    return X


@st.cache_data(show_spinner=False)
def elbow_and_silhouette(_X, k_min=2, k_max=8):
    ks, inertias, sils = [], [], []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(_X)
        ks.append(k)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(_X, labels))
    return pd.DataFrame({"k": ks, "inertia": inertias, "silhouette": sils})


@st.cache_data(show_spinner=False)
def fit_kmeans(_X, k):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(_X)
    sil = silhouette_score(_X, labels)
    return labels, sil, km.cluster_centers_


@st.cache_data(show_spinner=False)
def fit_hierarchical(_X, k, sample_n=800, seed=42):
    """Hierarchical clustering used to validate K-Means (Step 4). Sampled for
    tractability since agglomerative clustering is O(n^2) in memory/time."""
    n = _X.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=min(sample_n, n), replace=False)
    Xs = _X.iloc[idx]
    hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
    labels = hc.fit_predict(Xs)
    Z = linkage(Xs, method="ward")
    return idx, labels, Z


def auto_label_segments(df: pd.DataFrame, cluster_col="cluster") -> dict:
    """Data-driven descriptive naming so labels stay meaningful for any k chosen
    by the user (not hard-coded to exactly 4 clusters). Each cluster is scored
    against four archetype signatures (z-scored on cluster-mean features) and
    greedily assigned the best-fitting, still-unclaimed archetype; any extra
    clusters beyond the four archetypes fall back to a numbered generic label."""
    prof = df.groupby(cluster_col).agg(
        n_properties=("n_properties", "mean"),
        total_spend=("total_spend", "mean"),
        avg_price=("avg_price", "mean"),
        pct_loan=("loan_applied", lambda x: (x == "Yes").mean()),
        pct_company=("client_type", lambda x: (x == "Company").mean()),
    )
    z = (prof - prof.mean()) / prof.std(ddof=0).replace(0, 1)

    scores = pd.DataFrame({
        "Portfolio Investors": z["n_properties"] * 0.5 + z["total_spend"] * 0.5,
        "Luxury Single-Asset Buyers": z["avg_price"] - 0.4 * z["n_properties"],
        "Loan-Dependent / Corporate Growth Buyers": 0.5 * z["pct_loan"] + 0.5 * z["pct_company"],
        "Value-Focused Mainstream Buyers": -(z["avg_price"] + z["total_spend"]),
    })

    labels = {}
    remaining_clusters = list(scores.index)
    remaining_names = list(scores.columns)
    while remaining_clusters and remaining_names:
        sub = scores.loc[remaining_clusters, remaining_names]
        best = sub.stack().idxmax()  # (cluster, archetype)
        cl, name = best
        labels[cl] = name
        remaining_clusters.remove(cl)
        remaining_names.remove(name)

    for i, cl in enumerate(remaining_clusters, start=1):
        labels[cl] = f"Additional Segment {i}"

    return labels


# ==============================================================================
# SIDEBAR — DATA SOURCE, MODEL CONTROLS, FILTERS
# ==============================================================================
st.sidebar.title("🏙️ Parcl Buyer Intelligence")
st.sidebar.caption("Machine-learning buyer segmentation & investment profiling")

# Bundled default data lives in a "data" folder next to this script. Using an
# absolute path (anchored to this file, not the shell's current directory)
# means the app auto-loads correctly no matter where `streamlit run` is
# launched from.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CLIENTS_PATH = os.path.join(APP_DIR, "data", "clients.csv")
DEFAULT_PROPERTIES_PATH = os.path.join(APP_DIR, "data", "properties.csv")


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


clients_bytes, properties_bytes = None, None
default_data_found = os.path.exists(DEFAULT_CLIENTS_PATH) and os.path.exists(DEFAULT_PROPERTIES_PATH)

if default_data_found:
    clients_bytes = _read_bytes(DEFAULT_CLIENTS_PATH)
    properties_bytes = _read_bytes(DEFAULT_PROPERTIES_PATH)

with st.sidebar.expander("📁 Data Source", expanded=not default_data_found):
    if default_data_found:
        st.success("Using bundled dataset (data/clients.csv + data/properties.csv).")
    else:
        st.warning("Bundled dataset not found next to app.py — please upload both files.")

    use_uploaded = st.checkbox("Use my own files instead", value=not default_data_found)
    if use_uploaded:
        up_clients = st.file_uploader("clients.csv", type="csv")
        up_props = st.file_uploader("properties.csv", type="csv")
        if up_clients is not None:
            clients_bytes = up_clients.getvalue()
        if up_props is not None:
            properties_bytes = up_props.getvalue()

if not clients_bytes or not properties_bytes:
    st.info("Upload both `clients.csv` and `properties.csv` in the sidebar to continue.")
    st.stop()

df, properties_clean = clean_and_engineer(clients_bytes, properties_bytes)
X = build_feature_matrix(df)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Clustering Model")
k = st.sidebar.slider("Number of buyer segments (k)", min_value=2, max_value=8, value=4)
show_diagnostics = st.sidebar.checkbox("Show model diagnostics tab", value=True)

labels, sil_score, centers = fit_kmeans(X, k)
df = df.copy()
df["cluster"] = labels
segment_names = auto_label_segments(df)
df["segment"] = df["cluster"].map(segment_names)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Filters")
f_country = st.sidebar.multiselect("Country", sorted(df["country"].unique()))
f_region = st.sidebar.multiselect("Region", sorted(df["region"].unique()))
f_purpose = st.sidebar.multiselect("Acquisition Purpose", sorted(df["acquisition_purpose"].unique()))
f_ctype = st.sidebar.multiselect("Client Type", sorted(df["client_type"].unique()))
f_segment = st.sidebar.multiselect("Segment", sorted(df["segment"].unique()))

fdf = df.copy()
if f_country:
    fdf = fdf[fdf["country"].isin(f_country)]
if f_region:
    fdf = fdf[fdf["region"].isin(f_region)]
if f_purpose:
    fdf = fdf[fdf["acquisition_purpose"].isin(f_purpose)]
if f_ctype:
    fdf = fdf[fdf["client_type"].isin(f_ctype)]
if f_segment:
    fdf = fdf[fdf["segment"].isin(f_segment)]

if fdf.empty:
    st.warning("No clients match the current filter selection. Adjust filters in the sidebar.")
    st.stop()

# ==============================================================================
# HEADER + KPI ROW
# ==============================================================================
st.title("Machine-Learning Buyer Segmentation & Investment Profiling")
st.caption("Real Estate Market Intelligence — Parcl Co. Limited × Unified Mentor")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Clients (filtered)", f"{len(fdf):,}", f"of {len(df):,} total")
k2.metric("Segments", f"{fdf['segment'].nunique()}")
k3.metric("Total Portfolio Value", f"${fdf['total_spend'].sum()/1e6:,.1f}M")
k4.metric("Avg. Property Price", f"${fdf['avg_price'].mean():,.0f}")
k5.metric("Silhouette Score (k={})".format(k), f"{sil_score:.3f}")

tabs = st.tabs([
    "📊 Segmentation Overview",
    "💰 Investor Behavior Dashboard",
    "🌍 Geographic Buyer Analysis",
    "🧩 Segment Insights Panel",
] + (["🔬 Model Diagnostics"] if show_diagnostics else []))

# ------------------------------------------------------------------------------
# TAB 1 — SEGMENTATION OVERVIEW
# ------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Buyer Segmentation Overview")
    c1, c2 = st.columns([1, 1])

    with c1:
        seg_counts = fdf["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "count"]
        fig = px.pie(seg_counts, names="segment", values="count", hole=0.45,
                     color="segment", color_discrete_sequence=PALETTE,
                     title="Cluster Distribution")
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X.loc[fdf.index])
        plot_df = fdf[["segment"]].copy()
        plot_df["PC1"], plot_df["PC2"] = coords[:, 0], coords[:, 1]
        fig = px.scatter(plot_df, x="PC1", y="PC2", color="segment",
                          color_discrete_sequence=PALETTE, opacity=0.65,
                          title=f"PCA Projection of Segments (explains {pca.explained_variance_ratio_.sum()*100:.0f}% variance)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Segment Snapshot")
    snap = fdf.groupby("segment").agg(
        clients=("client_id", "count"),
        avg_age=("age", "mean"),
        avg_satisfaction=("satisfaction_score", "mean"),
        avg_properties=("n_properties", "mean"),
        avg_total_spend=("total_spend", "mean"),
        avg_price=("avg_price", "mean"),
        pct_investment=("acquisition_purpose", lambda x: (x == "Investment").mean() * 100),
        pct_loan=("loan_applied", lambda x: (x == "Yes").mean() * 100),
    ).round(1).sort_values("avg_total_spend", ascending=False)
    st.dataframe(snap.style.format({
        "avg_total_spend": "${:,.0f}", "avg_price": "${:,.0f}",
        "pct_investment": "{:.1f}%", "pct_loan": "{:.1f}%"
    }), use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2 — INVESTOR BEHAVIOR DASHBOARD
# ------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Investor Behavior Dashboard")
    c1, c2 = st.columns(2)

    with c1:
        fig = px.box(fdf, x="segment", y="total_spend", color="segment",
                     color_discrete_sequence=PALETTE, points=False,
                     title="Total Spend Distribution by Segment")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        loan_tab = (fdf.groupby(["segment", "loan_applied"]).size()
                    .reset_index(name="count"))
        fig = px.bar(loan_tab, x="segment", y="count", color="loan_applied",
                     barmode="stack", title="Financing (Loan) Behavior by Segment",
                     color_discrete_sequence=["#2E5B88", "#C1440E"])
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        purpose_tab = (fdf.groupby(["segment", "acquisition_purpose"]).size()
                       .reset_index(name="count"))
        fig = px.bar(purpose_tab, x="segment", y="count", color="acquisition_purpose",
                     barmode="group", title="Acquisition Purpose by Segment",
                     color_discrete_sequence=["#3E8E5A", "#8E5AA8"])
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.scatter(fdf, x="avg_floor_area", y="avg_price", color="segment",
                          size="n_properties", color_discrete_sequence=PALETTE,
                          opacity=0.6, hover_data=["full_name", "country"],
                          title="Price vs. Floor Area (bubble = # properties owned)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Sales Volume Over Time (Sold Transactions)")
    ts = properties_clean[properties_clean["listing_status"] == "Sold"].copy()
    ts = ts.merge(df[["client_id", "segment"]], left_on="client_ref", right_on="client_id", how="left")
    if f_segment:
        ts = ts[ts["segment"].isin(f_segment)]
    ts_month = ts.groupby([pd.Grouper(key="transaction_date_clean", freq="MS")]).agg(
        n_sales=("listing_id", "count"), revenue=("sale_price_clean", "sum")
    ).reset_index()
    fig = go.Figure()
    fig.add_bar(x=ts_month["transaction_date_clean"], y=ts_month["n_sales"], name="Units Sold", marker_color="#2E5B88")
    fig.add_trace(go.Scatter(x=ts_month["transaction_date_clean"], y=ts_month["revenue"] / ts_month["revenue"].max() * ts_month["n_sales"].max(),
                              name="Revenue (scaled)", yaxis="y", mode="lines+markers", line=dict(color="#C1440E")))
    fig.update_layout(title="Monthly Sales Volume (2024-2025)", xaxis_title="Month", yaxis_title="Units Sold")
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 3 — GEOGRAPHIC BUYER ANALYSIS
# ------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Geographic Buyer Analysis")
    c1, c2 = st.columns([1.3, 1])

    with c1:
        geo = fdf.groupby("country_display").agg(
            clients=("client_id", "count"), total_spend=("total_spend", "sum")
        ).reset_index()
        fig = px.choropleth(geo, locations="country_display", locationmode="country names",
                             color="clients", color_continuous_scale="Blues",
                             title="Client Concentration by Country")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        top_c = fdf["country"].value_counts().head(10).reset_index()
        top_c.columns = ["country", "clients"]
        fig = px.bar(top_c.sort_values("clients"), x="clients", y="country", orientation="h",
                     color_discrete_sequence=["#2E5B88"], title="Top 10 Countries by Client Count")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Region-Level Detail")
    region_tab = fdf.groupby(["country", "region"]).agg(
        clients=("client_id", "count"),
        avg_total_spend=("total_spend", "mean"),
        pct_investment=("acquisition_purpose", lambda x: (x == "Investment").mean() * 100),
        top_segment=("segment", lambda x: x.mode().iat[0] if not x.mode().empty else "-"),
    ).round(1).sort_values("clients", ascending=False).reset_index()
    st.dataframe(region_tab, use_container_width=True, height=350)

    fig = px.sunburst(fdf, path=["country", "region", "segment"], values=None,
                       color="country", title="Country → Region → Segment Breakdown",
                       color_discrete_sequence=PALETTE)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 4 — SEGMENT INSIGHTS PANEL
# ------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Segment Insights Panel")
    chosen = st.selectbox("Choose a segment to inspect in detail", sorted(fdf["segment"].unique()))
    seg_df = fdf[fdf["segment"] == chosen]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clients in Segment", f"{len(seg_df):,}", f"{len(seg_df)/len(fdf)*100:.1f}% of filtered")
    c2.metric("Avg. Total Spend", f"${seg_df['total_spend'].mean():,.0f}")
    c3.metric("Avg. Properties Owned", f"{seg_df['n_properties'].mean():.1f}")
    c4.metric("Avg. Satisfaction", f"{seg_df['satisfaction_score'].mean():.2f} / 5")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top Countries**")
        st.dataframe(seg_df["country"].value_counts().head(5).rename("clients"), use_container_width=True)
        st.markdown("**Referral Channel Mix**")
        fig = px.pie(seg_df, names="referral_channel", color_discrete_sequence=PALETTE, hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Client Type / Loan / Purpose Mix**")
        for col in ["client_type", "loan_applied", "acquisition_purpose"]:
            st.caption(col.replace("_", " ").title())
            st.bar_chart(seg_df[col].value_counts())

    st.markdown("#### Full Descriptive Statistics")
    st.dataframe(seg_df[NUMERIC_FEATURES].describe().T.round(2), use_container_width=True)

    csv_bytes = seg_df.drop(columns=["date_of_birth_clean"], errors="ignore").to_csv(index=False).encode("utf-8")
    st.download_button(f"⬇️ Download '{chosen}' client list (CSV)", data=csv_bytes,
                        file_name=f"segment_{chosen.replace(' ', '_')}.csv", mime="text/csv")

    st.markdown("#### All Clients (filtered view, with segment labels)")
    display_cols = ["client_id", "full_name", "client_type", "country", "region",
                     "age", "acquisition_purpose", "loan_applied", "n_properties",
                     "total_spend", "avg_price", "satisfaction_score", "segment"]
    st.dataframe(fdf[display_cols].sort_values("total_spend", ascending=False), use_container_width=True, height=350)

# ------------------------------------------------------------------------------
# TAB 5 — MODEL DIAGNOSTICS (elbow, silhouette, hierarchical validation)
# ------------------------------------------------------------------------------
if show_diagnostics:
    with tabs[4]:
        st.subheader("Clustering Model Diagnostics")
        st.caption("Evidence used to select the optimal number of segments (k) and validate K-Means against Hierarchical clustering.")

        diag_df = elbow_and_silhouette(X, 2, 8)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.line(diag_df, x="k", y="inertia", markers=True, title="Elbow Method — Inertia (WCSS) vs k")
            fig.add_vline(x=k, line_dash="dash", line_color="#C1440E")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.line(diag_df, x="k", y="silhouette", markers=True, title="Silhouette Score vs k")
            fig.add_vline(x=k, line_dash="dash", line_color="#C1440E")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"**Current selection:** k = {k} &nbsp;|&nbsp; Silhouette = {sil_score:.3f} "
            f"&nbsp;|&nbsp; Best silhouette in range 2-8 is at k = "
            f"{int(diag_df.loc[diag_df['silhouette'].idxmax(), 'k'])} "
            f"({diag_df['silhouette'].max():.3f})."
        )
        st.info(
            "Silhouette values in this dataset are modest (≈0.10-0.17), which is typical for "
            "buyer-segmentation problems that mix categorical demographic data with continuous "
            "transaction behavior — clusters reflect gradients of buyer behavior rather than "
            "fully separated groups. k is therefore chosen balancing statistical fit **and** "
            "business interpretability."
        )

        st.markdown("#### Hierarchical Clustering Validation")
        idx, hc_labels, Z = fit_hierarchical(X, k)
        km_labels_sample = df["cluster"].values[idx]
        ari = adjusted_rand_score(km_labels_sample, hc_labels)
        st.metric("Adjusted Rand Index (K-Means vs. Hierarchical, sampled n=800)", f"{ari:.3f}")

        fig = plt.figure(figsize=(10, 4))
        dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90.)
        plt.title("Hierarchical Clustering Dendrogram (truncated, sampled clients)")
        plt.xlabel("Sample clusters")
        plt.ylabel("Ward distance")
        st.pyplot(fig)

st.markdown("---")
st.caption(
    "Built for Parcl Co. Limited × Unified Mentor — Machine Learning based Buyer Segmentation "
    "and Investment Profiling for Real Estate Market Intelligence."
)
