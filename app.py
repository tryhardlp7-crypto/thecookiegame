import streamlit as pd_st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go

# Configuration de la page Web
pd_st.set_page_config(page_title="IA Trading Bot", layout="wide")
pd_st.title("🤖 Application Web IA : Où investir & Quand retirer")
pd_st.write("Cette IA analyse les marchés, s'entraîne en temps réel et simule une stratégie de placement.")

# ==========================================
# BARRE LATÉRALE DE CONFIGURATION (INTERFACE)
# ==========================================
pd_st.sidebar.header("⚙️ Paramètres de l'IA")
ticker = pd_st.sidebar.text_input("Symbole de l'actif (Ticker Yahoo Finance)", value="BTC-USD")
start_date = pd_st.sidebar.date_input("Date de début de l'historique", pd.to_datetime("2021-01-01"))
end_date = pd_st.sidebar.date_input("Date de fin de l'historique", pd.to_datetime("2026-01-01"))

pd_st.sidebar.header("💰 Gestion des Risques")
take_profit = pd_st.sidebar.slider("Objectif de gain (Take Profit en %)", 1, 20, 5) / 100
stop_loss = pd_st.sidebar.slider("Sécurité baisse (Stop Loss en %)", -20, -1, -2) / 100

if pd_st.sidebar.button("🧠 Entraîner l'IA et Lancer le Bot"):
    
    # CHARGEMENT DES DONNÉES
    with pd_st.spinner("📥 Téléchargement des données financières en cours..."):
        data = yf.download(ticker, start=start_date, end=end_date)
    
    if data.empty:
        pd_st.error("Erreur : Impossible de récupérer les données pour ce symbole.")
    else:
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # CALCUL DES INDICATEURS
        data['MA10'] = data['Close'].rolling(window=10).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
        data.dropna(inplace=True)
        
        features = ['MA10', 'MA50', 'RSI']
        X = data[features]
        y = data['Target']
        
        # ENTRAÎNEMENT IA
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        data.loc[X_test.index, 'Prediction'] = model.predict(X_test)
        test_data = data.loc[X_test.index].copy()
        
        # SIMULATION / BACKTEST
        capital_ia = 1000.0
        prix_initial_bh = test_data['Close'].iloc[0]
        en_position = False
        prix_achat = 0.0
        
        historique_ia = []
        historique_bh = []
        transactions = []
        
        for i in range(len(test_data)):
            current_price = test_data['Close'].iloc[i]
            current_date = test_data.index[i].strftime('%Y-%m-%d')
            signal_ia = test_data['Prediction'].iloc[i]
            
            # Evolution Buy & Hold
            historique_bh.append(1000.0 * (current_price / prix_initial_bh))
            
            if en_position:
                variation = (current_price - prix_achat) / prix_achat
                if variation >= take_profit or variation <= stop_loss:
                    capital_ia = capital_ia * (1 + variation)
                    en_position = False
                    transactions.append(f"🔴 Vente le {current_date} à {current_price:.2f}$ ({variation*100:.2f}%)")
            else:
                if signal_ia == 1 and test_data['RSI'].iloc[i] < 70:
                    prix_achat = current_price
                    en_position = True
                    transactions.append(f"🟢 Achat le {current_date} à {current_price:.2f}$")
            
            if en_position:
                variation_actuelle = (current_price - prix_achat) / prix_achat
                historique_ia.append(capital_ia * (1 + variation_actuelle))
            else:
                historique_ia.append(capital_ia)
                
        # AFFICHAGE DES RÉSULTATS SUR LE SITE
        col1, col2 = pd_st.columns(2)
        with col1:
            pd_st.metric("Gain Final IA", f"{capital_ia:.2f} $", f"{((capital_ia-1000)/1000)*100:.2f} % depuis 1000$")
        with col2:
            pd_st.metric("Gain Marché Classique (Buy & Hold)", f"{historique_bh[-1]:.2f} $", f"{((historique_bh[-1]-1000)/1000)*100:.2f} %")
            
        # GRAPHIQUE INTERACTIF
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=test_data.index, y=historique_bh, name="Marché Classique", line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=test_data.index, y=historique_ia, name="Notre IA de Placement", line=dict(color='green', width=2)))
        fig.update_layout(title="Comparaison des Performances ($)", template="plotly_white", xaxis_title="Date", yaxis_title="Valeur du Portefeuille")
        pd_st.plotly_chart(fig, use_container_width=True)
        
        # HISTORIQUE DES ORDRES
        pd_st.subheader("📜 Journal des actions de l'IA")
        for t in transactions[-10:]:  # Affiche les 10 dernières actions
            pd_st.write(t)
