#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script per raccogliere partite di utenti specifici da Lichess e analizzarle.
Le partite vengono salvate in un dataset cumulativo chiamato "Cheaters_games".
"""

import os
import sys
import requests
import chess.pgn
import chess.engine
import io
import csv
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

# Percorso per il dataset cumulativo
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "Cheaters_games.csv")

# Percorso di Stockfish
STOCKFISH_PATH = "/Users/bernardobusoni/Downloads/stockfish/stockfish-macos-m1-apple-silicon"

def get_player_games(username, max_games=10):
    """
    Ottiene le partite di un giocatore da Lichess API.
    
    Args:
        username (str): Nome utente del giocatore
        max_games (int): Numero massimo di partite da scaricare
    
    Returns:
        str: Testo PGN con le partite del giocatore
    """
    url = f"https://lichess.org/api/games/user/{username}"
    params = {
        "max": max_games,
        "pgnInJson": "true",
        "clocks": "true",
        #"evals": "true",
        #"opening": "true"
    }
    
    try:
        response = requests.get(url, params=params, stream=True)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Errore nella richiesta per {username}: {response.status_code}")
            return None
    except Exception as e:
        print(f"Errore durante la richiesta API: {e}")
        return None

def analyze_with_stockfish(game, player_color, engine=None, depth=20):
    """
    Analizza una partita con Stockfish e calcola le metriche per il giocatore specificato.
    
    Args:
        game (chess.pgn.Game): Oggetto partita
        player_color (chess.Color): Colore del giocatore da analizzare
        engine: Engine Stockfish già configurato (opzionale)
        depth (int): Profondità di analisi
    
    Returns:
        tuple: (percentuale di corrispondenza con Stockfish, perdita media in centipawn)
    """
    # Se non viene fornito un engine, ne creiamo uno
    should_close_engine = False
    if engine is None:
        engine = chess.engine.SimpleEngine.popen_uci("/Users/bernardobusoni/Downloads/stockfish/stockfish-macos-m1-apple-silicon")
        engine.configure({"Threads": 1, "Skill Level": 20})
        should_close_engine = True
    
    board = game.board()
    
    # Inizializza le metriche
    stockfish_matches = 0
    total_moves = 0
    centipawn_losses = []
    
    try:
        # Analizza ogni mossa
        current_node = game
        while current_node.variations:
            next_node = current_node.variations[0]
            move = next_node.move
            
            # Verifica se è il turno del giocatore che stiamo analizzando
            is_player_move = (board.turn == chess.WHITE and player_color == chess.WHITE) or \
                            (board.turn == chess.BLACK and player_color == chess.BLACK)
            
            if is_player_move:
                # Analizza la posizione con Stockfish
                try:
                    result = engine.analyse(board, chess.engine.Limit(depth=depth))
                    best_move = result['pv'][0] if 'pv' in result and result['pv'] else None
                    
                    # Verifica se la mossa giocata corrisponde alla migliore mossa di Stockfish
                    if move == best_move:
                        stockfish_matches += 1
                    
                    # Calcola la perdita in centipawns
                    if 'score' in result:
                        score_before = result['score'].white().score(mate_score=10000)
                        
                        # Esegui la mossa
                        board.push(move)
                        
                        # Analizza la nuova posizione
                        result_after = engine.analyse(board, chess.engine.Limit(depth=depth))
                        if 'score' in result_after:
                            score_after = result_after['score'].white().score(mate_score=10000)
                            
                            # Calcola la perdita
                            if player_color == chess.WHITE:
                                loss = max(0, score_before - score_after)
                            else:
                                loss = max(0, score_after - score_before)
                            
                            centipawn_losses.append(loss)
                        else:
                            board.pop()
                            board.push(move)
                    else:
                        board.push(move)
                    
                    total_moves += 1
                except Exception as e:
                    print(f"Errore durante l'analisi: {e}")
                    board.push(move)
            else:
                # Se non è il turno del giocatore, esegui semplicemente la mossa
                board.push(move)
            
            current_node = next_node
        
        # Calcola le metriche finali
        stockfish_match_percentage = (stockfish_matches / total_moves * 100) if total_moves > 0 else 0
        avg_centipawn_loss = np.mean(centipawn_losses) if centipawn_losses else 0
        
        return stockfish_match_percentage, avg_centipawn_loss
    
    finally:
        if should_close_engine:
            engine.quit()

def calculate_move_times(game, player_color):
    """
    Calcola il tempo medio per mossa e la varianza.
    
    Args:
        game (chess.pgn.Game): Oggetto partita
        player_color (chess.Color): Colore del giocatore da analizzare
    
    Returns:
        tuple: (tempo medio per mossa, varianza del tempo per mossa)
    """
    # Estrai i tempi dalla partita
    times = []
    node = game
    while node.variations:
        next_node = node.variations[0]
        if hasattr(next_node, 'clock') and next_node.clock is not None:
            # Chiama il metodo clock() per ottenere il valore effettivo
            clock_value = next_node.clock()
            times.append(clock_value)
        node = next_node
    
    # Se non ci sono tempi, restituisci 0
    if not times:
        return 0.0, 0.0
    
    # Filtra i tempi per il giocatore
    if player_color == chess.BLACK:
        player_times = times[1::2]
    else:
        player_times = times[0::2]
    
    # Calcola il tempo per mossa
    move_times = []
    for i in range(1, len(player_times)):
        prev_time = player_times[i-1]
        curr_time = player_times[i]
        move_time = prev_time - curr_time
        move_times.append(move_time)
    
    # Calcola media e varianza
    if move_times:
        avg_move_time = sum(move_times) / len(move_times)
        variance = sum((t - avg_move_time) ** 2 for t in move_times) / len(move_times)
        return avg_move_time, variance
    else:
        return 0.0, 0.0

def analyze_game(pgn_text, username):
    """
    Analizza una partita PGN e estrae le feature.
    
    Args:
        pgn_text (str): Testo PGN della partita
        username (str): Nome utente del giocatore da analizzare
    
    Returns:
        dict: Dizionario con le feature estratte
    """
    # Crea un oggetto partita da PGN
    pgn = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn)
    
    if game is None:
        return None
    
    # Estrai le informazioni di base
    game_id = game.headers.get("Site", "").split("/")[-1]
    white_player = game.headers.get("White", "")
    black_player = game.headers.get("Black", "")
    time_control = game.headers.get("TimeControl", "")
    opening_name = game.headers.get("Opening", "?")
    termination = game.headers.get("Termination", "?")
    result = game.headers.get("Result", "?")
    
    # Determina il colore del giocatore
    if white_player.lower() == username.lower():
        player_color = chess.WHITE
        white_cheater = True
        black_cheater = False
    else:
        player_color = chess.BLACK
        white_cheater = False
        black_cheater = True
    
    # Calcola il numero totale di mosse
    board = game.board()
    moves = list(game.mainline_moves())
    total_moves = len(moves)/2
    
    # Calcola le metriche di analisi
    stockfish_match, avg_centipawn_loss = analyze_with_stockfish(game, player_color)
    avg_move_time, move_time_variance = calculate_move_times(game, player_color)
    
    # Crea il dizionario delle feature
    features = {
        "game_id": game_id,
        "white_player": white_player,
        "black_player": black_player,
        "time_control": time_control,
        "stockfish_match_percentage": stockfish_match,
        "avg_centipawn_loss": avg_centipawn_loss,
        "avg_move_time": avg_move_time,
        "move_time_variance": move_time_variance,
        "total_moves": total_moves,
        "opening_name": opening_name,
        "termination": termination,
        "result": result,
        "white_cheater": white_cheater,
        "black_cheater": black_cheater,
        "cheating_suspected": True  # Tutti gli utenti forniti sono sospettati di cheating
    }
    
    return features

def collect_player_games(username, num_games=10):
    """
    Raccoglie e analizza le partite di un giocatore.
    
    Args:
        username (str): Nome utente del giocatore
        num_games (int): Numero di partite da raccogliere
    
    Returns:
        list: Lista di dizionari con le feature estratte
    """
    print(f"Raccolta delle ultime {num_games} partite di {username}...")
    
    # Ottieni le partite del giocatore
    pgn_text = get_player_games(username, max_games=num_games)
    
    if not pgn_text:
        print(f"Nessuna partita trovata per {username}")
        return []
    
    # Dividi il testo PGN in singole partite
    games_text = pgn_text.split("\n\n\n")
    
    # Analizza ogni partita
    features_list = []
    
    for game_text in tqdm(games_text, desc=f"Analisi partite di {username}"):
        if game_text.strip():
            features = analyze_game(game_text, username)
            if features:
                features_list.append(features)
    
    print(f"Raccolte {len(features_list)} partite per {username}")
    return features_list

def save_to_dataset(features_list):
    """
    Salva le feature estratte nel dataset cumulativo.
    
    Args:
        features_list (list): Lista di dizionari con le feature estratte
    """
    # Crea la directory di output se non esiste
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Crea un DataFrame dalle feature
    df_new = pd.DataFrame(features_list)
    
    # Se il file esiste già, aggiungi le nuove partite
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        
        # Verifica se ci sono partite duplicate
        #existing_game_ids = set(df_existing["game_id"])
        #df_new = df_new[~df_new["game_id"].isin(existing_game_ids)]
        
        # Unisci i DataFrame
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    # Salva il DataFrame nel file CSV
    df_combined.to_csv(OUTPUT_FILE, index=False)
    
    print(f"Dataset aggiornato con {len(df_new)} nuove partite")
    print(f"Totale partite nel dataset: {len(df_combined)}")

def main():
    """
    Funzione principale dello script.
    """
    # Chiedi all'utente il nome utente e il numero di partite
    username = input("Inserisci il nome utente Lichess: ")
    
    try:
        num_games = int(input("Inserisci il numero di partite da raccogliere: "))
    except ValueError:
        print("Numero non valido, utilizzo il valore predefinito di 10 partite")
        num_games = 10
    
    # Raccogli e analizza le partite
    features_list = collect_player_games(username, num_games)
    
    # Salva le feature nel dataset
    if features_list:
        save_to_dataset(features_list)
    else:
        print("Nessuna partita da aggiungere al dataset")

if __name__ == "__main__":
    main()