"""
app.py
=======
Aplikasi Streamlit untuk mendemonstrasikan model klasifikasi Profit_Category
("Rugi" / "Untung Rendah" / "Untung Tinggi") pada dataset Global Skincare &
Beauty E-Store.

Cara menjalankan:
    1. Pastikan folder ./artifacts sudah berisi hasil training
       (jalankan `python train_model.py` jika belum ada).
    2. streamlit run app.py
"""

import json
import os

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Klasifikasi Profit | Skincare & Beauty E-Store",
    page_icon="💄",
    layout="wide",
    initial_sidebar_state="expanded",
)

ARTIFACT_DIR = "artifacts"
DATA_URL = os.environ.get(
    "DATA_URL",
    "https://docs.google.com/spreadsheets/d/e/2PACX-1vQHKCQ5-LS4nL7Ger0OE2kxachW17ueOdGL0EwbDsNzctoxBWZdDCtnlCerAV7rFA/pub?output=csv",
)
CATEGORICAL_COLS = ["Segment", "Region", "Market", "Subcategory", "Category"]

# --------------------------------------------------------------------------
# TEMA WARNA & FONT (Cream - Abu - Merah, Times New Roman)
# --------------------------------------------------------------------------
COLOR_CREAM_BG = "#F3ECDD"
COLOR_CREAM_CARD = "#FAF5EA"
COLOR_GRAY_DARK = "#3A3835"
COLOR_GRAY_MED = "#7A746B"
COLOR_GRAY_BORDER = "#DCD3BE"
COLOR_RED = "#A4222F"
COLOR_RED_DARK = "#7C1A24"
COLOR_RED_SOFT = "#EFD8D2"

PLOTLY_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Times New Roman, Times, serif", color=COLOR_GRAY_DARK, size=14),
        paper_bgcolor=COLOR_CREAM_CARD,
        plot_bgcolor=COLOR_CREAM_CARD,
        colorway=[COLOR_RED, COLOR_GRAY_MED, COLOR_RED_DARK, "#C46A6A", "#A39B89", "#5C5750"],
        xaxis=dict(gridcolor=COLOR_GRAY_BORDER, linecolor=COLOR_GRAY_BORDER),
        yaxis=dict(gridcolor=COLOR_GRAY_BORDER, linecolor=COLOR_GRAY_BORDER),
        title=dict(font=dict(size=18, color=COLOR_RED)),
        legend=dict(bgcolor=COLOR_CREAM_CARD),
    )
)

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.cdnfonts.com/css/times-new-roman');

html, body, [class*="css"], .stMarkdown, .stText, p, span, label, li, div {{
    font-family: 'Times New Roman', Times, serif !important;
}}

.stApp {{
    background-color: {COLOR_CREAM_BG};
    color: {COLOR_GRAY_DARK};
}}

section[data-testid="stSidebar"] {{
    background-color: {COLOR_CREAM_CARD};
    border-right: 1px solid {COLOR_GRAY_BORDER};
}}

h1, h2, h3, h4 {{
    color: {COLOR_RED} !important;
    font-family: 'Times New Roman', Times, serif !important;
    font-weight: 700 !important;
}}

hr {{
    border-top: 1px solid {COLOR_GRAY_BORDER};
}}

/* Kartu metric bawaan Streamlit */
[data-testid="stMetric"] {{
    background-color: {COLOR_CREAM_CARD};
    border: 1px solid {COLOR_GRAY_BORDER};
    border-radius: 10px;
    padding: 14px 16px;
}}
[data-testid="stMetricValue"] {{
    color: {COLOR_RED} !important;
}}
[data-testid="stMetricLabel"] {{
    color: {COLOR_GRAY_MED} !important;
}}

/* Tombol */
.stButton > button, .stDownloadButton > button {{
    background-color: {COLOR_RED};
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 0.55em 1.4em;
    font-weight: 600;
    transition: background-color 0.2s ease-in-out;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    background-color: {COLOR_RED_DARK};
    color: #FFFFFF;
}}

/* Radio menu sidebar - dijadikan seperti baris navigasi */
div[role="radiogroup"] {{
    gap: 2px;
}}
div[role="radiogroup"] label {{
    color: {COLOR_GRAY_DARK};
    padding: 9px 10px;
    border-radius: 8px;
    width: 100%;
    transition: background-color 0.15s ease-in-out;
}}
div[role="radiogroup"] label:hover {{
    background-color: {COLOR_RED_SOFT};
}}
div[role="radiogroup"] input[type="radio"] {{
    accent-color: {COLOR_RED};
}}
div[role="radiogroup"] label:has(input:checked) {{
    background-color: {COLOR_RED_SOFT};
    font-weight: 700;
    border-left: 3px solid {COLOR_RED};
}}

/* Card kustom */
.custom-card {{
    background-color: {COLOR_CREAM_CARD};
    border: 1px solid {COLOR_GRAY_BORDER};
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}}

/* Badge hasil prediksi */
.badge {{
    display: inline-block;
    padding: 10px 22px;
    border-radius: 30px;
    font-size: 1.25rem;
    font-weight: 700;
    text-align: center;
}}
.badge-rugi {{
    background-color: {COLOR_RED};
    color: #FFFFFF;
    border: 2px solid {COLOR_RED_DARK};
}}
.badge-rendah {{
    background-color: {COLOR_GRAY_BORDER};
    color: {COLOR_GRAY_DARK};
    border: 2px solid {COLOR_GRAY_MED};
}}
.badge-tinggi {{
    background-color: {COLOR_CREAM_CARD};
    color: {COLOR_RED};
    border: 2px solid {COLOR_RED};
}}

.app-title {{
    color: {COLOR_RED};
    font-size: 2.3rem;
    font-weight: 700;
    margin-bottom: 0;
}}
.app-subtitle {{
    color: {COLOR_GRAY_MED};
    font-size: 1.05rem;
    margin-top: 4px;
}}

[data-testid="stDataFrame"] div {{
    font-family: 'Times New Roman', Times, serif !important;
}}

/* ====== RESPONSIVE: TAMPILAN DI HP ====== */
@media (max-width: 640px) {{
    .app-title {{
        font-size: 1.5rem !important;
    }}
    .app-subtitle {{
        font-size: 0.88rem !important;
    }}
    h1, h2, h3, h4 {{
        font-size: 1.15rem !important;
    }}
    .custom-card {{
        padding: 14px 16px !important;
    }}
    .badge {{
        font-size: 1rem !important;
        padding: 8px 16px !important;
    }}
    [data-testid="stMetric"] {{
        padding: 10px 12px !important;
    }}
    [data-testid="stMetricValue"] {{
        font-size: 1.3rem !important;
    }}
    .stButton > button, .stDownloadButton > button {{
        width: 100%;
        padding: 0.6em 1em;
    }}
}}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# LOAD ARTIFACT & DATA (CACHED)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model & artifact...")
def load_artifacts():
    model = joblib.load(os.path.join(ARTIFACT_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(ARTIFACT_DIR, "label_encoder.pkl"))
    feature_columns = joblib.load(os.path.join(ARTIFACT_DIR, "feature_columns.pkl"))

    with open(os.path.join(ARTIFACT_DIR, "ui_metadata.json")) as f:
        ui_metadata = json.load(f)

    model_comparison = pd.read_csv(os.path.join(ARTIFACT_DIR, "model_comparison.csv"))
    feature_importance = pd.read_csv(os.path.join(ARTIFACT_DIR, "feature_importance.csv"))

    return model, scaler, label_encoder, feature_columns, ui_metadata, model_comparison, feature_importance


@st.cache_data(show_spinner="Memuat data...", ttl=3600)
def load_raw_data():
    df = pd.read_csv(DATA_URL)
    df["Discount"] = pd.to_numeric(df["Discount"].astype(str).str.replace(",", "."), errors="coerce")
    df["Profit"] = pd.to_numeric(df["Profit"].astype(str).str.replace(",", "."), errors="coerce")
    df = df.drop_duplicates().reset_index(drop=True)

    median_profit = df["Profit"].median()

    def label_profit(p):
        if p <= 0:
            return "Rugi"
        elif p <= median_profit:
            return "Untung Rendah"
        return "Untung Tinggi"

    df["Profit_Category"] = df["Profit"].apply(label_profit)
    return df


def build_input_vector(raw_dict: dict, feature_columns: list) -> pd.DataFrame:
    """Susun 1 baris input mentah menjadi vektor fitur yang sama persis
    strukturnya dengan data training (urutan kolom & one-hot encoding)."""
    input_df = pd.DataFrame([raw_dict])
    input_df = pd.get_dummies(input_df, columns=CATEGORICAL_COLS, drop_first=True)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    return input_df


try:
    model, scaler, label_encoder, feature_columns, ui_meta, model_comparison, feature_importance = load_artifacts()
    ARTIFACTS_READY = True
except FileNotFoundError:
    ARTIFACTS_READY = False


# --------------------------------------------------------------------------
# SIDEBAR NAVIGASI
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"""
        <div style='border-left:4px solid {COLOR_RED}; padding-left:12px; margin-bottom:2px;'>
            <span style='color:{COLOR_RED}; font-size:1.5rem; font-weight:700; letter-spacing:0.5px;'>
                SKINCARE ANALYTICS
            </span>
        </div>
        <p style='color:{COLOR_GRAY_MED}; margin:4px 0 0 16px; font-size:0.92rem; font-style:italic;'>
            Profit Category Classifier
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    menu = st.radio(
        "Navigasi",
        ["Dashboard", "Eksplorasi Data", "Prediksi Profit", "Perbandingan Model"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        f"<p style='color:{COLOR_GRAY_MED}; font-size:0.82rem; line-height:1.6;'>"
        "<b style='color:{0};'>Model</b><br>Random Forest (tuned)<br><br>"
        "<b style='color:{0};'>Dataset</b><br>Global Skincare &amp; Beauty E-Store</p>".format(COLOR_GRAY_DARK),
        unsafe_allow_html=True,
    )

if not ARTIFACTS_READY:
    st.error(
        "Artifact model belum ditemukan di folder `artifacts/`. "
        "Jalankan `python train_model.py` terlebih dahulu sebelum menjalankan aplikasi ini."
    )
    st.stop()


# --------------------------------------------------------------------------
# HALAMAN 1: DASHBOARD
# --------------------------------------------------------------------------
def page_dashboard():
    st.markdown("<p class='app-title'>Dashboard Klasifikasi Profit</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='app-subtitle'>Ringkasan data transaksi & performa model "
        "Global Skincare &amp; Beauty E-Store</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    df = load_raw_data()
    best_row = model_comparison.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transaksi", f"{df.shape[0]:,}")
    col2.metric("Total Sales", f"${df['Sales'].sum():,.0f}")
    col3.metric("Rata-rata Profit", f"${df['Profit'].mean():,.1f}")
    col4.metric("Akurasi Model Terbaik", f"{best_row['Test Accuracy']*100:.2f}%", best_row["Model"])

    st.write("")
    left, right = st.columns([1.3, 1])

    with left:
        st.markdown("#### Distribusi Kategori Profit")
        dist = df["Profit_Category"].value_counts().reindex(
            ["Rugi", "Untung Rendah", "Untung Tinggi"]
        )
        fig = px.bar(
            x=dist.index, y=dist.values, text=dist.values,
            labels={"x": "Kategori Profit", "y": "Jumlah Transaksi"},
        )
        fig.update_traces(marker_color=[COLOR_RED, COLOR_GRAY_MED, COLOR_RED_DARK], textposition="outside")
        fig.update_layout(template=PLOTLY_TEMPLATE, height=380, showlegend=False)
        st.plotly_chart(fig, width='stretch')

    with right:
        st.markdown("#### Proporsi Kategori Profit")
        fig2 = px.pie(
            names=dist.index, values=dist.values, hole=0.45,
            color=dist.index,
            color_discrete_map={
                "Rugi": COLOR_RED, "Untung Rendah": COLOR_GRAY_MED, "Untung Tinggi": COLOR_RED_DARK,
            },
        )
        fig2.update_layout(template=PLOTLY_TEMPLATE, height=380)
        st.plotly_chart(fig2, width='stretch')

    st.markdown("#### Contoh Data Transaksi")
    st.dataframe(df.head(10), width='stretch')


# --------------------------------------------------------------------------
# HALAMAN 2: EKSPLORASI DATA
# --------------------------------------------------------------------------
def page_eda():
    st.markdown("<p class='app-title'>Eksplorasi Data</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='app-subtitle'>Pola & hubungan antar variabel pada data transaksi</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    df = load_raw_data()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Korelasi Variabel Numerik terhadap Profit")
        numeric_df = df[["Sales", "Quantity", "Discount", "Profit"]]
        corr = numeric_df.corr()["Profit"].drop("Profit").sort_values(ascending=False)
        fig = px.bar(x=corr.values, y=corr.index, orientation="h",
                     labels={"x": "Korelasi dengan Profit", "y": ""})
        fig.update_traces(marker_color=[COLOR_RED if v > 0 else COLOR_GRAY_DARK for v in corr.values])
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320)
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("#### Rata-rata Profit per Segment")
        seg = df.groupby("Segment")["Profit"].mean().sort_values(ascending=False)
        fig = px.bar(x=seg.index, y=seg.values, labels={"x": "Segment", "y": "Rata-rata Profit"})
        fig.update_traces(marker_color=COLOR_RED)
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320)
        st.plotly_chart(fig, width='stretch')

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Jumlah Transaksi per Category")
        cat = df["Category"].value_counts()
        fig = px.bar(x=cat.index, y=cat.values, labels={"x": "Category", "y": "Jumlah Transaksi"})
        fig.update_traces(marker_color=COLOR_GRAY_MED)
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320)
        st.plotly_chart(fig, width='stretch')

    with col4:
        st.markdown("#### Discount vs Profit")
        sample = df.sample(min(3000, len(df)), random_state=42)
        fig = px.scatter(sample, x="Discount", y="Profit", opacity=0.5)
        fig.update_traces(marker=dict(color=COLOR_RED, size=5))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320)
        st.plotly_chart(fig, width='stretch')

    st.markdown("#### Rata-rata Profit berdasarkan Market")
    mkt = df.groupby("Market")["Profit"].mean().sort_values(ascending=False)
    fig = px.bar(x=mkt.index, y=mkt.values, labels={"x": "Market", "y": "Rata-rata Profit"}, text=mkt.values.round(1))
    fig.update_traces(marker_color=COLOR_RED_DARK, textposition="outside")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=350)
    st.plotly_chart(fig, width='stretch')


# --------------------------------------------------------------------------
# HALAMAN 3: PREDIKSI
# --------------------------------------------------------------------------
def page_prediction():
    st.markdown("<p class='app-title'>Prediksi Kategori Profit</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='app-subtitle'>Masukkan detail transaksi untuk memprediksi "
        "apakah transaksi berpotensi Rugi, Untung Rendah, atau Untung Tinggi</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            segment = st.selectbox("Segment", ui_meta["segments"])
            region = st.selectbox("Region", ui_meta["regions"])
            market = st.selectbox("Market", ui_meta["markets"])

        with c2:
            category = st.selectbox("Category", ui_meta["categories"])
            subcategory_options = ui_meta["category_subcategory_map"].get(category, [])
            subcategory = st.selectbox("Subcategory", subcategory_options)
            quantity = st.number_input(
                "Quantity", min_value=ui_meta["quantity_min"], max_value=ui_meta["quantity_max"], value=4
            )

        with c3:
            sales = st.number_input(
                "Sales ($)", min_value=ui_meta["sales_min"], max_value=ui_meta["sales_max"], value=100.0, step=1.0
            )
            discount = st.slider(
                "Discount", min_value=0.0, max_value=ui_meta["discount_max"], value=0.1, step=0.01
            )
            order_date = st.date_input("Order Date")

        submitted = st.form_submit_button("🔮 Prediksi Sekarang")

    if submitted:
        raw_input = {
            "Segment": segment,
            "Region": region,
            "Market": market,
            "Subcategory": subcategory,
            "Category": category,
            "Quantity": quantity,
            "Sales": sales,
            "Discount": discount,
            "Order_Year": order_date.year,
            "Order_Month": order_date.month,
            "Order_DayOfWeek": order_date.weekday(),
        }

        input_vector = build_input_vector(raw_input, feature_columns)
        input_scaled = scaler.transform(input_vector)

        pred_encoded = model.predict(input_scaled)[0]
        pred_label = label_encoder.inverse_transform([pred_encoded])[0]
        pred_proba = model.predict_proba(input_scaled)[0]
        proba_df = pd.DataFrame({
            "Kategori": label_encoder.classes_,
            "Probabilitas": pred_proba,
        }).sort_values("Probabilitas", ascending=False)

        st.write("")
        badge_class = {
            "Rugi": "badge-rugi",
            "Untung Rendah": "badge-rendah",
            "Untung Tinggi": "badge-tinggi",
        }[pred_label]

        result_col, proba_col = st.columns([1, 1.3])

        with result_col:
            st.markdown("#### Hasil Prediksi")
            st.markdown(
                f"<span class='badge {badge_class}'>{pred_label}</span>",
                unsafe_allow_html=True,
            )
            st.write("")
            st.markdown(
                f"<div class='custom-card'>Model memprediksi transaksi ini sebagai "
                f"<b>{pred_label}</b> dengan tingkat keyakinan "
                f"<b>{proba_df.iloc[0]['Probabilitas']*100:.1f}%</b>.</div>",
                unsafe_allow_html=True,
            )

        with proba_col:
            st.markdown("#### Distribusi Probabilitas")
            fig = px.bar(
                proba_df, x="Probabilitas", y="Kategori", orientation="h",
                text=proba_df["Probabilitas"].apply(lambda v: f"{v*100:.1f}%"),
            )
            fig.update_traces(marker_color=[COLOR_RED, COLOR_GRAY_MED, COLOR_RED_DARK], textposition="outside")
            fig.update_layout(template=PLOTLY_TEMPLATE, height=280, xaxis_range=[0, 1])
            st.plotly_chart(fig, width='stretch')


# --------------------------------------------------------------------------
# HALAMAN 4: PERBANDINGAN MODEL
# --------------------------------------------------------------------------
def page_comparison():
    st.markdown("<p class='app-title'>Perbandingan Model</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='app-subtitle'>Performa 4 algoritma klasifikasi pada data test</p>",
        unsafe_allow_html=True,
    )
    st.write("")

    display_df = model_comparison.copy()
    for col in ["Train Accuracy", "Test Accuracy", "Precision", "Recall", "F1-Score"]:
        display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"

    st.markdown("#### Tabel Metrik Evaluasi")
    st.dataframe(display_df, width='stretch', hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Test Accuracy per Model")
        fig = px.bar(
            model_comparison.sort_values("Test Accuracy"),
            x="Test Accuracy", y="Model", orientation="h",
            text=(model_comparison.sort_values("Test Accuracy")["Test Accuracy"] * 100).round(2).astype(str) + "%",
        )
        fig.update_traces(marker_color=COLOR_RED, textposition="outside")
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320, xaxis_range=[0, 1.05])
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.markdown("#### Gap Train vs Test Accuracy (Indikasi Overfitting)")
        gap_df = model_comparison.copy()
        gap_df["Gap"] = gap_df["Train Accuracy"] - gap_df["Test Accuracy"]
        fig = px.bar(
            gap_df.sort_values("Gap"), x="Gap", y="Model", orientation="h",
            text=(gap_df.sort_values("Gap")["Gap"] * 100).round(2).astype(str) + "%",
        )
        fig.update_traces(marker_color=COLOR_GRAY_MED, textposition="outside")
        fig.update_layout(template=PLOTLY_TEMPLATE, height=320)
        st.plotly_chart(fig, width='stretch')

    st.markdown("#### Feature Importance — Model Utama (Random Forest Tuned)")
    fig = px.bar(
        feature_importance.sort_values("Importance"),
        x="Importance", y="Feature", orientation="h",
    )
    fig.update_traces(marker_color=COLOR_RED_DARK)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=480)
    st.plotly_chart(fig, width='stretch')

    with st.expander("ℹ️ Detail Hyperparameter Model Terbaik"):
        st.json(ui_meta.get("best_rf_params", {}))
        st.write(f"Cross-validation score: **{ui_meta.get('best_rf_cv_score', 0)*100:.2f}%**")
        st.write(f"Test accuracy (final): **{ui_meta.get('best_rf_test_accuracy', 0)*100:.2f}%**")


# --------------------------------------------------------------------------
# ROUTER
# --------------------------------------------------------------------------
if menu == "Dashboard":
    page_dashboard()
elif menu == "Eksplorasi Data":
    page_eda()
elif menu == "Prediksi Profit":
    page_prediction()
elif menu == "Perbandingan Model":
    page_comparison()

st.markdown("---")
st.markdown(
    f"<p style='text-align:center; color:{COLOR_GRAY_MED}; font-size:0.85rem;'>"
    "Klasifikasi Profit — Global Skincare &amp; Beauty E-Store | Dibangun dengan Streamlit</p>",
    unsafe_allow_html=True,
)
