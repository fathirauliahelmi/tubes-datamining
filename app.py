import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

@st.cache_data
def load_model_result():
    return pd.read_csv("hasil_model.csv")

hasil_model = load_model_result()

@st.cache_data
def load_cluster():
    return pd.read_csv("hasil_cluster.csv")

df = load_cluster()

hasil_model = pd.read_csv(
    "hasil_model.csv"
)

# CONFIG


st.set_page_config(
    page_title="Dashboard Kecanduan Media Sosial",
    page_icon="📱",
    layout="wide"
)


# LOAD DATA


df = pd.read_csv("hasil_cluster.csv")


# FITUR CLUSTERING


fitur = [
    'daily_screen_time_hours',
    'doomscrolling_frequency',
    'notification_checks_per_day',
    'ai_recommendation_exposure',
    'productivity_loss_pct',
    'digital_detox_attempts'
]


# HITUNG SILHOUETTE SCORE OTOMATIS


scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df[fitur]
)

score = 0.12509860883361332


# SIDEBAR


menu = st.sidebar.radio(
    "Pilih Menu",
    [
        "Ringkasan Dataset",
        "Clustering",
        "Silhouette Score",
        "Perbandingan Model"
    ]
)


# MENU 1


if menu == "Ringkasan Dataset":

    st.title(
        "Analisis Segmentasi dan Klasifikasi Risiko Kecanduan Media Sosial"
    )

    st.subheader(
        "Menggunakan Algoritma K-Means, Logistic Regression, dan Naive Bayes"
    )

    st.markdown("---")

    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric(
            "Jumlah Record",
            len(df)
        )

    with col2:
        st.metric(
            "Jumlah Variabel",
            len(df.columns)
        )

    with col3:
        st.metric(
            "Jumlah Cluster",
            df['cluster'].nunique()
        )

    st.markdown("---")

    st.subheader(
        "Preview Dataset"
    )

    st.dataframe(
        df.head()
    )

    st.markdown("---")

    st.subheader(
        "Informasi Variabel"
    )

    info_df = pd.DataFrame({
        "Variabel": df.columns,
        "Tipe Data": df.dtypes.astype(str)
    })

    st.dataframe(info_df)

    st.markdown("---")

    st.subheader(
        "Statistik Deskriptif"
    )

    st.dataframe(
        df.describe()
    )


# MENU 2


elif menu == "Clustering":

    st.title(
        "Hasil Clustering K-Means"
    )

    st.subheader(
        "Distribusi Cluster"
    )

    cluster_count = (
        df['cluster']
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots()

    cluster_count.plot(
        kind='bar',
        ax=ax
    )

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Jumlah Data")

    st.pyplot(fig)

    st.markdown("---")

    st.subheader(
        "Profil Cluster"
    )

    cluster_profile = (
        df.groupby('cluster')[fitur]
        .mean()
    )

    st.dataframe(
        cluster_profile.round(2)
    )

    fig2, ax2 = plt.subplots(
        figsize=(10,5)
    )

    cluster_profile.T.plot(
        kind='bar',
        ax=ax2
    )

    ax2.set_title(
        "Profil Rata-rata Tiap Cluster"
    )

    st.pyplot(fig2)

    st.markdown("---")

    st.subheader(
        "Filter Data Berdasarkan Cluster"
    )

    pilih_cluster = st.selectbox(
        "Pilih Cluster",
        sorted(df['cluster'].unique())
    )

    st.dataframe(
        df[
            df['cluster']
            == pilih_cluster
        ]
        .head(100)
    )




# MENU 3


elif menu == "Silhouette Score":

    st.title(
        "Silhouette Score"
    )

    st.metric(
        "",
        round(score,3)
    )

    st.markdown("---")

    st.subheader(
        "Jumlah Anggota Tiap Cluster"
    )

    st.dataframe(
        df['cluster']
        .value_counts()
        .reset_index()
        .rename(
            columns={
                'index':'Cluster',
                'cluster':'Jumlah Data'
            }
        )
    )

# MENU 4


elif menu == "Perbandingan Model":

    st.title(
        "Perbandingan Model Klasifikasi"
    )

    st.subheader(
        "Logistic Regression vs Naive Bayes"
    )

    st.dataframe(
        hasil_model
    )

    st.markdown("---")

    st.subheader(
        "Grafik Perbandingan Accuracy"
    )

    fig1, ax1 = plt.subplots(
        figsize=(8,4)
    )

    ax1.bar(
        hasil_model["Model"],
        hasil_model["Accuracy"]
    )

    ax1.set_ylabel(
        "Accuracy"
    )

    ax1.set_title(
        "Perbandingan Accuracy"
    )

    st.pyplot(fig1)

    st.markdown("---")

    st.subheader(
        "Grafik Seluruh Metrik"
    )

    fig2, ax2 = plt.subplots(
        figsize=(10,5)
    )

    hasil_model.set_index(
        "Model"
    )[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ].plot(
        kind="bar",
        ax=ax2
    )

    ax2.set_title(
        "Perbandingan Logistic Regression dan Naive Bayes"
    )

    ax2.set_ylabel(
        "Nilai"
    )

    st.pyplot(fig2)

    st.markdown("---")

    model_terbaik = hasil_model.loc[
        hasil_model["Accuracy"].idxmax(),
        "Model"
    ]

    akurasi_terbaik = hasil_model["Accuracy"].max()

    st.success(
        f"Model terbaik adalah {model_terbaik} dengan Accuracy = {akurasi_terbaik:.4f}"
    )

