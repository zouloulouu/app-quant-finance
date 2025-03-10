import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator

# 🎨 Configuration de l'application
st.set_page_config(page_title="Analyse Boursière", layout="wide")

# 📌 Titre et introduction
st.title("📈 Analyse Boursière Interactive")
st.write("Entrez les tickers des actions pour afficher leur analyse technique.")

# 🔍 Saisie des tickers
tickers_input = st.text_input("Entrez les tickers des actions (séparés par des virgules)", "MSFT, AAPL, TSLA")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

# 📊 Récupération des données
df = yf.download(tickers, period="2y")
st.sidebar.write("Données téléchargées avec succès !")

# 📌 Ajout des indicateurs techniques
for ticker in tickers:
    df[("SMA_50", ticker)] = df["Close"][ticker].rolling(window=50).mean()
    df[("SMA_200", ticker)] = df["Close"][ticker].rolling(window=200).mean()

    bollinger = BollingerBands(df["Close"][ticker], window=20, window_dev=2)
    df[("Bollinger High", ticker)] = bollinger.bollinger_hband()
    df[("Bollinger Low", ticker)] = bollinger.bollinger_lband()

    df[("RSI", ticker)] = RSIIndicator(df["Close"][ticker], window=14).rsi()

# 📊 Affichage des graphiques
for ticker in tickers:
    st.subheader(f"📉 Graphique Bougies Japonaises - {ticker}")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"][ticker],
        high=df["High"][ticker],
        low=df["Low"][ticker],
        close=df["Close"][ticker],
        name="Bougies Japonaises"
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df[("SMA_50", ticker)],
        mode="lines", name="Moyenne Mobile 50j",
        line=dict(color="blue", width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df[("SMA_200", ticker)],
        mode="lines", name="Moyenne Mobile 200j",
        line=dict(color="red", width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df[("Bollinger High", ticker)],
        mode="lines", name="Bande Haute",
        line=dict(color="purple", width=1, dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df[("Bollinger Low", ticker)],
        mode="lines", name="Bande Basse",
        line=dict(color="green", width=1, dash="dot")
    ))

    fig.update_layout(
        title=f"Cours de {ticker} avec Moyennes Mobiles & Bandes de Bollinger",
        xaxis_title="Date",
        yaxis_title="Prix ($)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

# 📌 Score & Stratégie
st.subheader("📊 Stratégie basée sur les indicateurs")
strategy = {}

for ticker in tickers:
    score = (
        (df[("SMA_50", ticker)] > df[("SMA_200", ticker)]).astype(int) +
        (df["Close"][ticker] <= df[("Bollinger Low", ticker)]).astype(int) +
        (df[("RSI", ticker)] < 30).astype(int)
    )

    latest_score = score.iloc[-1]
    if latest_score >= 2:
        strategy[ticker] = "🟢 Achat 📈"
    elif latest_score == 0:
        strategy[ticker] = "🔴 Vente 📉"
    else:
        strategy[ticker] = "🟡 Conserver"

# 📊 Affichage de la stratégie
st.write(pd.DataFrame.from_dict(strategy, orient="index", columns=["Stratégie"]))
