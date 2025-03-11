import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import scipy.optimize as sco  # Pour l'optimisation de portefeuille
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator

# 🎨 Configuration de l'application
st.set_page_config(page_title="Analyse Boursière", layout="wide")

# 📌 Titre et introduction
st.title("📈 Analyse Boursière Interactive")
st.write("Entrez les tickers des actions pour afficher leur analyse technique et optimiser un portefeuille.")

# 🔍 Saisie des tickers
tickers_input = st.text_input("Entrez les tickers des actions (séparés par des virgules)", "MSFT, AAPL, TSLA")
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

# 📊 Récupération des données
df = yf.download(tickers, period="2y")
st.sidebar.write("Données téléchargées avec succès !")

# 📊 Statistiques Descriptives
st.subheader("📊 Statistiques des Actions")

stats_data = {}
for ticker in tickers:
    stock = yf.Ticker(ticker)
    info = stock.info

    stats_data[ticker] = {
        "PER": info.get("trailingPE", np.nan),
        "Rendement Cumulé (%)": (df["Close"][ticker].iloc[-1] / df["Close"][ticker].iloc[0] - 1) * 100,
        "Plus Haut": df["High"][ticker].max(),
        "Plus Bas": df["Low"][ticker].min(),
        "Espérance Rendements (%)": df["Close"][ticker].pct_change().mean() * 252 * 100,
        "Écart-Type (%)": df["Close"][ticker].pct_change().std() * np.sqrt(252) * 100,
        "Volatilité (%)": df["Close"][ticker].pct_change().std() * np.sqrt(252) * 100,
        "Beta": info.get("beta", np.nan),
        "Capitalisation (B$)": info.get("marketCap", np.nan) / 1e9
    }

df_stats = pd.DataFrame(stats_data).T
st.write(df_stats)

# 📌 Ajout des indicateurs techniques
for ticker in tickers:
    df[("SMA_50", ticker)] = df["Close"][ticker].rolling(window=50).mean()
    df[("SMA_200", ticker)] = df["Close"][ticker].rolling(window=200).mean()

    bollinger = BollingerBands(df["Close"][ticker], window=20, window_dev=2)
    df[("Bollinger High", ticker)] = bollinger.bollinger_hband()
    df[("Bollinger Low", ticker)] = bollinger.bollinger_lband()

    df[("RSI", ticker)] = RSIIndicator(df["Close"][ticker], window=14).rsi()

# 📊 Calcul des Corrélations et Variance
st.subheader("📈 Analyse de Corrélation et Risque")

returns = df["Close"].pct_change().dropna()
correlation_matrix = returns.corr()

# Affichage de la matrice de corrélation
st.write("### 📌 Matrice de Corrélation")
st.write(correlation_matrix)

# 📌 Optimisation du Portefeuille avec Markowitz
st.subheader("📊 Optimisation du Portefeuille")

# Fonction de minimisation du risque
def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

# Calcul de la matrice de covariance
cov_matrix = returns.cov()

# Nombre d'actifs
num_assets = len(tickers)

# Contraintes : Somme des poids = 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Poids initiaux égaux
init_weights = np.ones(num_assets) / num_assets

# Bornes (chaque poids entre 0% et 100%)
bounds = tuple((0, 1) for asset in range(num_assets))

# Optimisation
optimal = sco.minimize(portfolio_volatility, init_weights, args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=constraints)

# Résultats de l'optimisation
optimal_weights = optimal.x

# Affichage des poids optimaux
portfolio_df = pd.DataFrame(optimal_weights, index=tickers, columns=["Allocation Optimale"])
st.write(portfolio_df)

# 📊 Tracé de la Frontière d’Efficience
st.subheader("📉 Frontière d’Efficience du Portefeuille")

num_portfolios = 5000
results = np.zeros((3, num_portfolios))
risk_free_rate = 0.02

for i in range(num_portfolios):
    weights = np.random.dirichlet(np.ones(num_assets), size=1).flatten()
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    results[0, i] = portfolio_return
    results[1, i] = portfolio_volatility
    results[2, i] = sharpe_ratio

fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(results[1, :], results[0, :], c=results[2, :], cmap="coolwarm", marker="o", edgecolors="black")
ax.set_xlabel("Risque (Volatilité)")
ax.set_ylabel("Rendement Attendu")
ax.set_title("Frontière d’Efficience")
fig.colorbar(scatter, label="Ratio de Sharpe")

st.pyplot(fig)

# 📌 Score & Stratégie
st.subheader("📊 Stratégie basée sur les indicateurs et l'optimisation")

strategy = {}

for ticker in tickers:
    score = (
        (df[("SMA_50", ticker)] > df[("SMA_200", ticker)]).astype(int) +
        (df["Close"][ticker] <= df[("Bollinger Low", ticker)]).astype(int) +
        (df[("RSI", ticker)] < 30).astype(int)
    )

    latest_score = score.iloc[-1] if not score.dropna().empty else 0

    # Ajout des statistiques au scoring
    per = df_stats.loc[ticker, "PER"]
    rendement_cumule = df_stats.loc[ticker, "Rendement Cumulé (%)"]

    if per < df_stats["PER"].median():
        latest_score += 1  # PER faible = potentiellement sous-évalué

    if rendement_cumule > df_stats["Rendement Cumulé (%)"].median():
        latest_score -= 1  # Déjà bien monté, possible correction

    if latest_score >= 2:
        strategy[ticker] = "🟢 Achat 📈"
    elif latest_score <= 0:
        strategy[ticker] = "🔴 Vente 📉"
    else:
        strategy[ticker] = "🟡 Conserver"

# 📊 Affichage de la stratégie
st.write(pd.DataFrame.from_dict(strategy, orient="index", columns=["Stratégie"]))
