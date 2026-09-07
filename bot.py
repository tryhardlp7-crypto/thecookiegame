import os
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION DU BOT (MODIFIABLE)
# ==========================================
TICKER = "BTC-USD"        # Actif à analyser (Ex: AAPL, TSLA, ETH-USD)
START_DATE = "2020-01-01"  # Début de l'historique
END_DATE = "2026-01-01"    # Fin de l'historique (Modifiable selon l'année en cours)
TAKE_PROFIT = 0.05         # Objectif de gain (+5%)
STOP_LOSS = -0.02          # Sécurité en cas de baisse (-2%)

print(f"📥 Téléchargement des données pour {TICKER}...")
data = yf.download(TICKER, start=START_DATE, end=END_DATE)

if data.empty:
    raise ValueError("Erreur : Impossible de récupérer les données. Vérifie le Ticker.")

# Suppression des multi-index si présents avec yfinance
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# ==========================================
# CREATION DES INDICATEURS (LES SIGNAUX)
# ==========================================
print("⚙️ Calcul des indicateurs techniques...")
# 1. Moyennes Mobiles (Tendances)
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# 2. RSI (Force du marché / Surchauffe)
delta = data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
data['RSI'] = 100 - (100 / (1 + rs))

# 3. Target (Ce que l'IA doit deviner : le prix monte-t-il le lendemain ?)
data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

# Nettoyage des lignes vides créées par les calculs
data.dropna(inplace=True)

# Définition des variables d'apprentissage
features = ['MA10', 'MA50', 'RSI']
X = data[features]
y = data['Target']

# ==========================================
# ENTRAÎNEMENT DE L'IA (MACHINE LEARNING)
# ==========================================
print("🧠 Entraînement de l'Intelligence Artificielle...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Utilisation d'un modèle Random Forest (Gratuit et performant)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédiction sur l'ensemble de test
data.loc[X_test.index, 'Prediction'] = model.predict(X_test)

# ==========================================
# SIMULATION DE TRADING (BACKTESTING)
# ==========================================
print("📈 Simulation de la stratégie de placement...")
test_data = data.loc[X_test.index].copy()

capital_ia = 1000.0  # Capital de départ fictif ($)
capital_bh = 1000.0  # Stratégie classique (Acheter et ne rien toucher)
en_position = False
prix_achat = 0.0

historique_ia = []
historique_bh = []

# Prix initial pour la stratégie "Hold"
prix_initial_bh = test_data['Close'].iloc[0]

for i in range(len(test_data)):
    current_price = test_data['Close'].iloc[i]
    signal_ia = test_data['Prediction'].iloc[i]
    
    # Évolution du Buy & Hold (Acheter et garder)
    rendement_bh = current_price / prix_initial_bh
    historique_bh.append(1000.0 * rendement_bh)
    
    if en_position:
        # Calcul de la performance actuelle de notre ligne
        variation = (current_price - prix_achat) / prix_achat
        
        # SÉCURITÉ 1 : Quand retirer pour encaisser les profits (Take Profit)
        # SÉCURITÉ 2 : Quand retirer pour stopper la casse (Stop Loss)
        if variation >= TAKE_PROFIT or variation <= STOP_LOSS:
            capital_ia = capital_ia * (1 + variation)
            en_position = False
            print(f"🔴 VENTE au prix de {current_price:.2f} | Rendement opération : {variation*100:.2f}% | Nouveau Capital IA : {capital_ia:.2f}$")
    else:
        # SIGNAL D'ACHAT : L'IA prédit une hausse et le marché n'est pas en surchauffe (RSI < 70)
        if signal_ia == 1 and test_data['RSI'].iloc[i] < 70:
            prix_achat = current_price
            en_position = True
            print(f"🟢 ACHAT (Où placer l'argent) au prix de {current_price:.2f}")
            
    if en_position:
        variation_actuelle = (current_price - prix_achat) / prix_achat
        historique_ia.append(capital_ia * (1 + variation_actuelle))
    else:
        historique_ia.append(capital_ia)

# ==========================================
# BILAN DES RÉSULTATS
# ==========================================
print("\n" + "="*40)
print("🏆 BILAN FINAL DE LA SIMULATION")
print("="*40)
print(f"💰 Stratégie Classique (Buy & Hold) : {historique_bh[-1]:.2f}$")
print(f"🤖 Stratégie pilotée par ton IA   : {capital_ia:.2f}$")
print("="*40)

# Message d'analyse amical
if capital_ia > historique_bh[-1]:
    print("Félicitations ! Ton IA a battu le marché sur cette période grâce à ses règles de sortie.")
else:
    print("Le marché a été plus fort. L'IA a peut-être coupé trop de positions à cause du Stop Loss.")
