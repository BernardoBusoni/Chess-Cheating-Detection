#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script per addestrare e valutare il modello Random Forest.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

def train_model(input_csv, output_model):
    """
    Addestra un modello Random Forest per rilevare il cheating.
    
    Args:
        input_csv (str): Percorso del file CSV con le feature etichettate
        output_model (str): Percorso dove salvare il modello addestrato
    """
    if not os.path.exists(input_csv):
        print(f"Il file {input_csv} non esiste.")
        sys.exit(1)
    
    try:
        # Carica il dataset
        print(f"Caricamento del dataset da {input_csv}...")
        df = pd.read_csv(input_csv, sep=';', skiprows=1)
        
        # Verifica che ci siano abbastanza dati
        if len(df) < 10:
            print("Dataset troppo piccolo per l'addestramento.")
            sys.exit(1)
        
        # Seleziona le feature e l'etichetta target
        features = [
            #'time_control',
            'stockfish_match_percentage', 
            'avg_centipawn_loss', 
            'avg_move_time', 
            'move_time_variance',
            'total_moves',
        ]
        
        X = df[features]
        y = df['cheating_suspected'].astype(int)
        
        # Dividi il dataset in training e test set
        df = df.sample(frac=1, random_state=42).reset_index(drop=True) 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Normalizza le feature
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Addestra il modello Random Forest
        print("Addestramento del modello Random Forest...")
        
        # Parametri per la ricerca in griglia
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': [ 2, 3, 4]
        }
        
        # Inizializza il modello base
        rf = RandomForestClassifier(random_state=42)
        
        # Esegui la ricerca in griglia
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
        grid_search.fit(X_train_scaled, y_train)
        
        # Ottieni il miglior modello
        best_rf = grid_search.best_estimator_
        print(f"Migliori parametri: {grid_search.best_params_}")
        
        # Valuta il modello sul test set
        y_pred = best_rf.predict(X_test_scaled)
        
        # Calcola le metriche
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print("\nRisultati della valutazione:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Calcola la matrice di confusione
        cm = confusion_matrix(y_test, y_pred)
        print("\nMatrice di confusione:")
        print(cm)
        
        # Calcola la curva ROC e l'AUC
        y_pred_proba = best_rf.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Calcola l'importanza delle feature
        feature_importance = best_rf.feature_importances_
        feature_names = X.columns
        
        # Ordina le feature per importanza e salva i risultati in un file CSV
        sorted_idx = np.argsort(feature_importance)
        
        # Visualizza l'importanza delle feature
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('Importanza')
        plt.title('Importanza delle Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(output_model), 'feature_importance.png'))
        plt.close()

        # Salva il modello
        print(f"\nSalvataggio del modello in {output_model}...")
        with open(output_model, 'wb') as f:
            pickle.dump({
                'model': best_rf,
                'scaler': scaler,
                'features': features,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc
                }
            }, f)
        
        print("Modello salvato con successo!")
        
        return best_rf, scaler
    
    except Exception as e:
        print(f"Errore durante l'addestramento del modello: {e}")
        sys.exit(1)

def main():
    # Definisci i percorsi dei file
    data_dir = "data"
    model_dir = "models"
    input_csv = os.path.join(data_dir, "Analysed_games.csv")
    output_model = os.path.join(model_dir, "random_forest_model.pkl")
    
    # Verifica che le directory esistano
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Verifica che il file di input esista
    if not os.path.exists(input_csv):
        print(f"Il file {input_csv} non esiste. Esegui prima check_tosviolation.py.")
        sys.exit(1)
    
    # Addestra il modello
    print("Inizio addestramento del modello...")
    train_model(input_csv, output_model)
    
    print("\nAddestramento completato!")
    print(f"Modello salvato in {output_model}")
    print(f"Grafici salvati nella directory {model_dir}")

if __name__ == "__main__":
    main()