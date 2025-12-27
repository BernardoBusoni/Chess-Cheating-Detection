import chess.pgn
import chess.engine
import numpy as np
from io import StringIO
import logging
import re

logger = logging.getLogger(__name__)

def analyze_game_with_stockfish(game, engine, depth=60):
    """
    Analizza una partita con Stockfish e calcola le metriche per ciascun giocatore.
    
    Args:
        game (chess.pgn.Game): Partita da analizzare
        engine (chess.engine.SimpleEngine): Motore Stockfish
        depth (int): Profondità di analisi
    
    Returns:
        dict: Metriche della partita
    """
    board = game.board()
    
    # Inizializza le metriche
    metrics = {
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
                    
                    # Calcola la perdita
                    centipawn_loss = abs(score_before - score_after)
                    metrics['centipawn_losses'].append(max(0, centipawn_loss))
                    
                    # Aggiungi il tempo della mossa se disponibile
                    if move_time is not None:
                        metrics['move_times'].append(move_time)
                    
                    metrics['total_moves'] += 1
                else:
                    board.push(move)
            else:
                board.push(move)
        except Exception as e:
            logger.error(f"Errore durante l'analisi: {e}")
            board.push(move)
    
    return metrics

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