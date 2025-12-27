#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script per estrarre feature dalle partite di scacchi.
"""

import os
import sys
import csv
import chess
import chess.pgn
import chess.engine
import numpy as np
import pandas as pd
from tqdm import tqdm

def analyze_game_with_stockfish(game, engine, depth=10):
    """
    Analizza una partita con Stockfish e calcola le metriche per ciascun giocatore.
    
    Args:
        game (chess.pgn.Game): Partita da analizzare
        engine (chess.engine.SimpleEngine): Motore Stockfish
        depth (int): Profondità di analisi
    
    Returns:
        tuple: Metriche per il giocatore bianco e nero
    """
    board = game.board()
    
    # Inizializza le metriche per ciascun giocatore
    white_metrics = {
        'stockfish_matches': 0,
        'total_moves': 0,
        'centipawn_losses': [],
        'move_times': []
    }
    
    black_metrics = {
        'stockfish_matches': 0,
        'total_moves': 0,
        'centipawn_losses': [],
        'move_times': []
    }
    
    # Analizza ogni mossa
    for node in game.mainline():
        move = node.move
        
        # Ottieni il tempo della mossa se disponibile
        move_time = None
        if node.comment and 'clk' in node.comment:
            try:
                # Estrai il tempo dalla notazione [%clk h:m:s]
                clk_str = node.comment.split('[%clk ')[1].split(']')[0]
                h, m, s = map(float, clk_str.split(':'))
                move_time = h * 3600 + m * 60 + s
            except:
                pass
        
        # Determina il giocatore corrente
        current_player = 'white' if board.turn == chess.WHITE else 'black'
        metrics = white_metrics if current_player == 'white' else black_metrics
        
        # Analizza la posizione con Stockfish
        try:
            result = engine.analyse(board, chess.engine.Limit(depth=depth))
            best_move = result['pv'][0]
            
            # Verifica se la mossa giocata corrisponde alla migliore mossa di Stockfish
            if move == best_move:
                metrics['stockfish_matches'] += 1
            
            # Calcola la perdita in centipawns
            if 'score' in result:
                score_before = result['score'].white().score(mate_score=10000)
                
                # Esegui la mossa
                board.push(move)
                
                # Analizza la nuova posizione
                result_after = engine.analyse(board, chess.engine.Limit(depth=depth))
                if 'score' in result_after:
                    score_after = result_after['score'].white().score(mate_score=10000)
                    
                    # Calcola la perdita (negativa per il nero)
                    centipawn_loss = score_before - score_after if current_player == 'white' else score_after - score_before
                    metrics['centipawn_losses'].append(max(0, centipawn_loss))  # Solo perdite positive
                    
                    # Aggiungi il tempo della mossa se disponibile
                    if move_time is not None:
                        metrics['move_times'].append(move_time)
                    
                    metrics['total_moves'] += 1
                else:
                    board.push(move)
            else:
                board.push(move)
        except Exception as e:
            print(f"Errore durante l'analisi: {e}")
            board.push(move)
    
    # Calcola le metriche finali
    white_stockfish_match_percentage = (white_metrics['stockfish_matches'] / white_metrics['total_moves'] * 100) if white_metrics['total_moves'] > 0 else 0
    white_avg_centipawn_loss = np.mean(white_metrics['centipawn_losses']) if white_metrics['centipawn_losses'] else 0
    white_avg_move_time = np.mean(white_metrics['move_times']) if white_metrics['move_times'] else 0
    white_move_time_variance = np.var(white_metrics['move_times']) if len(white_metrics['move_times']) > 1 else 0
    
    black_stockfish_match_percentage = (black_metrics['stockfish_matches'] / black_metrics['total_moves'] * 100) if black_metrics['total_moves'] > 0 else 0
    black_avg_centipawn_loss = np.mean(black_metrics['centipawn_losses']) if black_metrics['centipawn_losses'] else 0
    black_avg_move_time = np.mean(black_metrics['move_times']) if black_metrics['move_times'] else 0
    black_move_time_variance = np.var(black_metrics['move_times']) if len(black_metrics['move_times']) > 1 else 0
    
    white_result = {
        'stockfish_match_percentage': white_stockfish_match_percentage,
        'avg_centipawn_loss': white_avg_centipawn_loss,
        'avg_move_time': white_avg_move_time,
        'move_time_variance': white_move_time_variance,
        'total_moves': white_metrics['total_moves']
    }
    
    black_result = {
        'stockfish_match_percentage': black_stockfish_match_percentage,
        'avg_centipawn_loss': black_avg_centipawn_loss,
        'avg_move_time': black_avg_move_time,
        'move_time_variance': black_move_time_variance,
        'total_moves': black_metrics['total_moves']
    }
    
    return white_result, black_result

def extract_features(pgn_file, output_csv, max_games=500):
    """
    Estrae feature dalle partite di scacchi.
    
    Args:
        pgn_file (str): Percorso del file PGN
        output_csv (str): Percorso del file CSV di output
        max_games (int): Numero massimo di partite da analizzare
    """
    if not os.path.exists(pgn_file):
        print(f"Il file {pgn_file} non esiste.")
        sys.exit(1)
    
    # Inizializza Stockfish
    try:
        # Cerca Stockfish in diverse posizioni
        stockfish_path = None
        possible_paths = [
            "/Users/bernardobusoni/Downloads/stockfish/stockfish-macos-m1-apple-silicon"
        ]
        
        for path in possible_paths:
            try:
                engine = chess.engine.SimpleEngine.popen_uci(path)
                stockfish_path = path
                break
            except:
                continue
        
        if stockfish_path is None:
            print("Stockfish non trovato. Assicurati che sia installato e nel PATH.")
            sys.exit(1)
        
        print(f"Stockfish trovato in {stockfish_path}")
    except Exception as e:
        print(f"Errore durante l'inizializzazione di Stockfish: {e}")
        sys.exit(1)
    
    # Apri il file PGN
    try:
        pgn = open(pgn_file)
    except Exception as e:
        print(f"Errore durante l'apertura del file PGN: {e}")
        sys.exit(1)
    
    # Prepara il file CSV di output
    fieldnames = [
        'game_id', 'white_player', 'black_player', 'time_control',
        'white_stockfish_match_percentage', 'white_avg_centipawn_loss', 'white_avg_move_time', 'white_move_time_variance', 'white_total_moves',
        'black_stockfish_match_percentage', 'black_avg_centipawn_loss', 'black_avg_move_time', 'black_move_time_variance', 'black_total_moves',
        'opening_name', 'termination', 'result'
    ]
    
    try:
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Analizza le partite
            games_analyzed = 0
            
            with tqdm(total=max_games, desc="Analisi partite") as pbar:
                while games_analyzed < max_games:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
                    
                    # Estrai le informazioni di base
                    game_id = game.headers.get('Site', '').split('/')[-1]
                    white_player = game.headers.get('White', '')
                    black_player = game.headers.get('Black', '')
                    time_control = game.headers.get('TimeControl', '')
                    opening_name = game.headers.get('Opening', '?')
                    termination = game.headers.get('Termination', '')
                    result = game.headers.get('Result', '')
                    
                    # Salta le partite senza mosse
                    if not game.mainline_moves():
                        continue
                    
                    # Analizza la partita con Stockfish
                    try:
                        white_metrics, black_metrics = analyze_game_with_stockfish(game, engine)
                        
                        # Scrivi i risultati nel CSV
                        writer.writerow({
                            'game_id': game_id,
                            'white_player': white_player,
                            'black_player': black_player,
                            'time_control': time_control,
                            'white_stockfish_match_percentage': white_metrics['stockfish_match_percentage'],
                            'white_avg_centipawn_loss': white_metrics['avg_centipawn_loss'],
                            'white_avg_move_time': white_metrics['avg_move_time'],
                            'white_move_time_variance': white_metrics['move_time_variance'],
                            'white_total_moves': white_metrics['total_moves'],
                            'black_stockfish_match_percentage': black_metrics['stockfish_match_percentage'],
                            'black_avg_centipawn_loss': black_metrics['avg_centipawn_loss'],
                            'black_avg_move_time': black_metrics['avg_move_time'],
                            'black_move_time_variance': black_metrics['move_time_variance'],
                            'black_total_moves': black_metrics['total_moves'],
                            'opening_name': opening_name,
                            'termination': termination,
                            'result': result
                        })
                        
                        games_analyzed += 1
                        pbar.update(1)
                    except Exception as e:
                        print(f"Errore durante l'analisi della partita {game_id}: {e}")
    except Exception as e:
        print(f"Errore durante la scrittura del file CSV: {e}")
    finally:
        # Chiudi Stockfish
        engine.quit()
        pgn.close()
    
    print(f"Analisi completata. {games_analyzed} partite analizzate e salvate in {output_csv}")

def main():
    # Definisci i percorsi dei file
    data_dir = "data"
    pgn_file = os.path.join(data_dir, "clean_subset.pgn")
    output_csv = os.path.join(data_dir, "features.csv")
    
    # Verifica se esiste un dataset bilanciato
    balanced_dataset_file = os.path.join(data_dir, "balanced_dataset.pgn")
    if os.path.exists(balanced_dataset_file):
        print(f"Trovato dataset bilanciato in {balanced_dataset_file}")
        pgn_file = balanced_dataset_file
    
    # Estrai le feature
    print("Inizio estrazione delle feature...")
    extract_features(pgn_file, output_csv, max_games=500)  # Aumentato da 50 a 500

if __name__ == "__main__":
    main()