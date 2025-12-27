from flask import Flask, render_template, request, jsonify, url_for
import pickle
import os
#from pgn_analyzer import analyze_pgn
import logging
import pandas as pd
import numpy as np
import requests
import chess.pgn
import io
#from collect_cheater_games import analyze_with_stockfish, calculate_move_times
from sklearn.pipeline import Pipeline
import joblib
import hashlib

# Configura il logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path='/static', static_folder='static')

# Carica il modello addestrato
try:
    model_path = os.path.join('models', 'random_forest_model.pkl')
    logger.info(f"Tentativo di caricamento del modello da: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Il file del modello non esiste in: {model_path}")
    
    # Calcola l'hash del file del modello
    with open(model_path, 'rb') as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()
        logger.info(f"Hash del modello: {model_hash}")
    
    with open(model_path, 'rb') as f:
        model_data = joblib.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']
        logger.info("Modello caricato con successo")
except Exception as e:
    logger.error(f"Errore nel caricamento del modello: {str(e)}")
    raise

def get_last_game_between_players(player1, player2):
    """
    Ottiene l'ultima partita giocata tra due giocatori da Lichess.
    """
    url = f"https://lichess.org/api/games/user/{player1}"
    params = {
        "max": 200,  # Ottieni le ultime 100 partite
        "pgnInJson": "true",
        "clocks": "true",
        #"evals": "true",
        #"opening": "true"
    }
    
    try:
        response = requests.get(url, params=params, stream=True)
        if response.status_code != 200:
            raise Exception(f"Errore nella richiesta API: {response.status_code}")
        
        # Dividi il testo PGN in singole partite
        games_text = response.text.split("\n\n\n")
        
        # Cerca la prima partita contro player2
        for game_text in games_text:
            if game_text.strip():
                pgn = io.StringIO(game_text)
                game = chess.pgn.read_game(pgn)
                
                if game:
                    white_player = game.headers.get("White", "").lower()
                    black_player = game.headers.get("Black", "").lower()
                    
                    if (white_player == player1.lower() and black_player == player2.lower()) or \
                       (white_player == player2.lower() and black_player == player1.lower()):
                        return game_text
        
        raise Exception("Nessuna partita trovata tra i due giocatori")
        
    except Exception as e:
        logger.error(f"Errore durante il recupero della partita: {str(e)}")
        raise

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
    move_times = []
    
    try:
        # Analizza ogni mossa
        current_node = game
        while current_node.variations:
            next_node = current_node.variations[0]
            move = next_node.move
            
            # Verifica se è il turno del giocatore che stiamo analizzando
            is_player_move = (board.turn == chess.WHITE and player_color == chess.WHITE) or \
                            (board.turn == chess.BLACK and player_color == chess.BLACK)
            
            # Estrai il tempo della mossa se disponibile
            move_time = None
            if hasattr(next_node, 'clock') and next_node.clock is not None:
                move_time = next_node.clock()
            
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
                        
                        # Verifica se la mossa è legale
                        if move in board.legal_moves:
                            # Esegui la mossa
                            board.push(move)
                            
                            # Analizza la nuova posizione
                            result_after = engine.analyse(board, chess.engine.Limit(depth=depth))
                            if 'score' in result_after:
                                score_after = result_after['score'].white().score(mate_score=10000)
                                
                                # Calcola la perdita (negativa per il nero)
                                if player_color == chess.WHITE:
                                    loss = max(0, score_before - score_after)
                                else:
                                    loss = max(0, score_after - score_before)
                                
                                centipawn_losses.append(loss)
                                
                                # Aggiungi il tempo della mossa se disponibile
                                if move_time is not None:
                                    move_times.append(move_time)
                                
                                total_moves += 1
                            else:
                                board.pop()  # Annulla la mossa se non possiamo analizzare la nuova posizione
                        else:
                            logger.warning(f"Mossa non legale: {move}")
                    else:
                        if move in board.legal_moves:
                            board.push(move)  # Esegui la mossa anche se non abbiamo uno score
                        else:
                            logger.warning(f"Mossa non legale: {move}")
                    
                except Exception as e:
                    logger.error(f"Errore durante l'analisi: {e}")
                    if move in board.legal_moves:
                        board.push(move)
            else:
                # Se non è il turno del giocatore, esegui semplicemente la mossa
                if move in board.legal_moves:
                    board.push(move)
                else:
                    logger.warning(f"Mossa non legale: {move}")
            
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
    
def analyze_pgn(pgn_text):
    """
    Analizza una partita PGN e estrae le feature necessarie per il modello.
    
    Args:
        pgn_text (str): Testo PGN della partita
        
    Returns:
        dict: Dizionario con le feature estratte
    """
    try:
        # Carica la partita dal PGN
        pgn = StringIO(pgn_text)
        game = chess.pgn.read_game(pgn)
        
        if not game:
            raise ValueError("Impossibile leggere il PGN")
        
        # Inizializza il motore Stockfish
        engine = chess.engine.SimpleEngine.popen_uci("/Users/bernardobusoni/Downloads/stockfish/stockfish-macos-m1-apple-silicon")
        
        # Analizza la partita
        metrics = analyze_game_with_stockfish(game, engine)
        
        # Calcola le feature finali
        features = {
            'time_control': int(game.headers.get('TimeControl', '600')),
            'stockfish_match_percentage': (metrics['stockfish_matches'] / metrics['total_moves'] * 100) if metrics['total_moves'] > 0 else 0,
            'avg_centipawn_loss': float(np.mean(metrics['centipawn_losses'])) if metrics['centipawn_losses'] else 0,
            'avg_move_time': float(np.mean(metrics['move_times'])) if metrics['move_times'] else 0,
            'move_time_variance': float(np.var(metrics['move_times'])) if len(metrics['move_times']) > 1 else 0,
            'total_moves': metrics['total_moves']
        }
        
        engine.quit()
        logger.info(f"Feature estratte: {features}")
        return features
        
    except Exception as e:
        logger.error(f"Errore durante l'analisi del PGN: {str(e)}")
        if 'engine' in locals():
            engine.quit()
        raise 

# Configura Stockfish per essere deterministico
def configure_stockfish_engine():
    engine = chess.engine.SimpleEngine.popen_uci("/Users/bernardobusoni/Downloads/stockfish/stockfish-macos-m1-apple-silicon")
    engine.configure({"Threads": 1, "Skill Level": 20})
    return engine

# Variabile globale per tenere traccia delle feature precedenti
prev_features = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Ottieni il PGN dalla richiesta
        pgn_text = request.form['pgn']
        logger.info("PGN ricevuto, inizio analisi")
        
        # Analizza il PGN e estrai le feature
        game_features = analyze_pgn(pgn_text)
        logger.info(f"Feature estratte: {game_features}")
        
        # Prepara i dati per il modello
        X = pd.DataFrame([game_features])
        X = X[features]  # Assicurati che le colonne siano nello stesso ordine
        X_scaled = scaler.transform(X)
        
        # Fai la previsione
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        logger.info(f"Previsione completata: {prediction}, probabilità: {probability}")
        
        return jsonify({
            'success': True,
            'prediction': bool(prediction),
            'probability': float(probability),
            'features': {k: float(v) for k, v in game_features.items()}
        })
    
    except Exception as e:
        logger.error(f"Errore durante l'analisi: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/analyze_lichess', methods=['POST'])
def analyze_lichess():
    try:
        # Ottieni i nomi dei giocatori dalla richiesta
        player1 = request.form['player1']
        player2 = request.form['player2']
        
        logger.info(f"Analisi richiesta per la partita tra {player1} e {player2}")
        
        # Ottieni l'ultima partita tra i due giocatori
        pgn_text = get_last_game_between_players(player1, player2)
        
        # Analizza la partita
        pgn = io.StringIO(pgn_text)
        game = chess.pgn.read_game(pgn)
        
        # Determina il colore dell'avversario (player2)
        if game.headers.get("White", "").lower() == player2.lower():
            player_color = chess.WHITE
        else:
            player_color = chess.BLACK
        
        # Configura l'engine Stockfish
        engine = configure_stockfish_engine()
        
        # Calcola le metriche con profondità fissa
        stockfish_match, avg_centipawn_loss = analyze_with_stockfish(game, player_color, engine, depth=20)
        avg_move_time, move_time_variance = calculate_move_times(game, player_color)
        
        # Chiudi l'engine
        engine.quit()
        
        # Calcola il numero corretto di mosse
        total_moves = len(list(game.mainline_moves())) // 2
        
        # Crea il dizionario delle feature
        game_features = {
            "time_control": int(game.headers.get("TimeControl", "0").split("+")[0]),
            "stockfish_match_percentage": stockfish_match,
            "avg_centipawn_loss": avg_centipawn_loss,
            "avg_move_time": avg_move_time,
            "move_time_variance": move_time_variance,
            "total_moves": total_moves
        }
        
        # Verifica la stabilità delle feature
        game_key = f"{player1}_{player2}"
        if game_key in prev_features:
            if game_features != prev_features[game_key]:
                logger.warning(f"Feature cambiate per la stessa partita! Vecchie: {prev_features[game_key]}, Nuove: {game_features}")
        prev_features[game_key] = game_features
        
        logger.info(f"Feature estratte: {game_features}")
        
        # Prepara i dati per il modello
        X = pd.DataFrame([game_features])
        X = X[features]  # Assicurati che le colonne siano nello stesso ordine
        
        logger.info(f"Ordine delle colonne: {X.columns.tolist()}")
        
        X_scaled = scaler.transform(X)
        
        # Fai la previsione
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        logger.info(f"Previsione completata: {prediction}, probabilità: {probability}")
        
        return jsonify({
            'success': True,
            'prediction': bool(prediction),
            'probability': float(probability),
            'features': {k: float(v) for k, v in game_features.items()},
            'pgn': pgn_text  # Aggiungo il PGN alla risposta
        })
    
    except Exception as e:
        logger.error(f"Errore durante l'analisi: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True, port=5001)