import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.optimize as sco
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

# 📌 Saisie du montant d'investissement et du profil de risque
investment_amount = st.number_input("💰 Montant à investir (€)", min_value=100, value=1000, step=100)
risk_profile = st.selectbox("🎯 Sélectionnez votre profil de risque", ["Faible", "Moyen", "Élevé"])

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
    df[(ticker, "SMA_50")] = df["Close"][ticker].rolling(window=50).mean()
    df[(ticker, "SMA_200")] = df["Close"][ticker].rolling(window=200).mean()

    bollinger = BollingerBands(df["Close"][ticker], window=20, window_dev=2)
    df[(ticker, "Bollinger High")] = bollinger.bollinger_hband()
    df[(ticker, "Bollinger Low")] = bollinger.bollinger_lband()

    df[(ticker, "RSI")] = RSIIndicator(df["Close"][ticker], window=14).rsi()

# 📊 Calcul des Corrélations et Variance
st.subheader("📈 Analyse de Corrélation et Risque")

returns = df["Close"].pct_change().dropna()
correlation_matrix = returns.corr()

# Calcul de la matrice de covariance
cov_matrix = returns.cov()

# 📊 Affichage de la matrice de corrélation avec plotly
fig_corr = px.imshow(correlation_matrix,
                     labels=dict(color="Corrélation"),
                     x=correlation_matrix.columns,
                     y=correlation_matrix.columns,
                     color_continuous_scale="RdBu_r",  # Rouge pour corrélation positive, Bleu pour négative
                     aspect="auto")  # Ajustement automatique de l'aspect

fig_corr.update_layout(
    title="Matrice de Corrélation",
    width=800,
    height=600
)

# Ajout des valeurs numériques sur la matrice
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        fig_corr.add_annotation(
            x=i,
            y=j,
            text=f"{correlation_matrix.iloc[j, i]:.2f}",
            showarrow=False,
            font=dict(color="black", size=10)
        )

st.plotly_chart(fig_corr)

# 📌 Fonction d'allocation du portefeuille selon le profil de risque
def get_portfolio_allocation(risk_profile, returns, cov_matrix):
    num_assets = len(tickers)

    if risk_profile == "Faible":
        def objective(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))  # Minimisation de la volatilité

    elif risk_profile == "Moyen":
        risk_free_rate = 0.02
        def objective(weights):
            port_return = np.sum(returns.mean() * weights) * 252
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return - (port_return - risk_free_rate) / port_volatility  # Maximisation de Sharpe

    else:  # "Élevé"
        def objective(weights):
            return -np.sum(returns.mean() * weights) * 252  # Maximisation du rendement

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    init_weights = np.ones(num_assets) / num_assets
    bounds = tuple((0, 1) for _ in range(num_assets))

    optimal = sco.minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    return optimal.x

# 📌 Allocation optimisée selon le profil
optimal_weights = get_portfolio_allocation(risk_profile, returns, cov_matrix)

# 📊 Création du dataframe des allocations optimales
portfolio_df = pd.DataFrame(optimal_weights, index=tickers, columns=["Allocation Optimale (%)"])
portfolio_df["Allocation Optimale (%)"] *= 100  # Convertir en pourcentage
st.subheader(f"📌 Allocation Optimale pour un profil {risk_profile}")
st.write(portfolio_df)

# 📊 Calcul des rendements et volatilité du portefeuille optimisé
expected_return = np.sum(returns.mean() * optimal_weights) * 252
expected_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))

# 📌 Définition des scénarios de risque
risk_multipliers = {"Faible": 0.5, "Moyen": 1, "Élevé": 1.5}

# 📌 Calcul des gains et pertes potentiels
expected_gain = investment_amount * (expected_return / 100)
potential_loss = investment_amount * (expected_volatility / 100) * risk_multipliers[risk_profile]

# 📊 Affichage des résultats d'investissement
st.subheader("📊 Résultats d'Investissement selon votre profil")

col1, col2 = st.columns(2)
with col1:
    st.metric(label="📈 Rendement Attendu (€)", value=f"{expected_gain:,.2f} €")
with col2:
    st.metric(label="📉 Perte Potentielle (€)", value=f"{-potential_loss:,.2f} €")

st.write("Ces estimations sont basées sur la volatilité et les rendements historiques des actions sélectionnées et ajustées au profil de risque choisi.")

# 📊 Tracé de la Frontière d'Efficience avec Plotly
st.subheader("📉 Frontière d'Efficience du Portefeuille")

num_portfolios = 5000
results = np.zeros((3, num_portfolios))
risk_free_rate = 0.02

for i in range(num_portfolios):
    weights = np.random.dirichlet(np.ones(len(tickers)), size=1).flatten()
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility

    results[0, i] = portfolio_return
    results[1, i] = portfolio_volatility
    results[2, i] = sharpe_ratio

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=results[1, :],
    y=results[0, :],
    mode='markers',
    marker=dict(
        size=5,
        color=results[2, :],
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="Ratio de Sharpe")
    ),
    name='Portefeuilles'
))

fig.update_layout(
    title="Frontière d'Efficience",
    xaxis_title="Risque (Volatilité)",
    yaxis_title="Rendement Attendu",
    showlegend=True
)

st.plotly_chart(fig)

# 📌 Graphiques des cours avec indicateurs techniques
st.subheader("📈 Graphiques des Cours avec Indicateurs Techniques")

for ticker in tickers:
    fig = go.Figure()

    # Tracé du cours de clôture
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"][ticker],
        mode='lines',
        name=f'Cours de {ticker}'
    ))

    # Tracé des moyennes mobiles
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[(ticker, "SMA_50")],
        mode='lines',
        name=f'SMA 50 de {ticker}',
        line=dict(color='orange')
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[(ticker, "SMA_200")],
        mode='lines',
        name=f'SMA 200 de {ticker}',
        line=dict(color='purple')
    ))

    # Tracé des bandes de Bollinger
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[(ticker, "Bollinger High")],
        mode='lines',
        name=f'Bollinger High de {ticker}',
        line=dict(width=0),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[(ticker, "Bollinger Low")],
        mode='lines',
        fill='tonexty',
        name=f'Bollinger Band de {ticker}',
        line=dict(width=0),
        fillcolor='rgba(128, 128, 128, 0.2)'
    ))

    fig.update_layout(
        title=f"Cours de {ticker} avec Indicateurs Techniques",
        xaxis_title="Date",
        yaxis_title="Prix",
        showlegend=True
    )

    st.plotly_chart(fig)

# 📌 Score & Stratégie
st.subheader("📊 Stratégie basée sur les indicateurs et l'optimisation")

strategy = {}

for ticker in tickers:
    score = (
        (df[(ticker, "SMA_50")] > df[(ticker, "SMA_200")]).astype(int) +
        (df["Close"][ticker] <= df[(ticker, "Bollinger Low")]).astype(int) +
        (df[(ticker, "RSI")] < 30).astype(int)
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
