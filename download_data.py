#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script per scaricare e preparare i dati di Lichess.
"""

import os
import requests
import zstandard as zstd
import io

def download_lichess_data(year=2025, month=3, output_dir='data'):
    """
    Scarica il database di partite di Lichess per un mese specifico.
    
    Args:
        year (int): Anno del database
        month (int): Mese del database (1-12)
        output_dir (str): Directory di output
    """
    # Crea la directory di output se non esiste
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # URL del database Lichess
    url = f"https://database.lichess.org/standard/lichess_db_standard_rated_{year}-{month:02d}.pgn.zst"
    
    # Nome del file di output
    output_file = os.path.join(output_dir, f"lichess_db_standard_rated_{year}-{month:02d}.pgn.zst")
    
    # Scarica il file
    print(f"Scaricamento del database da {url}...")
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        # Ottieni la dimensione totale del file
        total_size = int(response.headers.get('content-length', 0))
        
        # Mostra una barra di progresso
        from tqdm import tqdm
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Download")
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # filtra i keep-alive chunks
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
        print(f"Database scaricato in {output_file}")
    else:
        print(f"Errore durante il download: {response.status_code}")
        return None
    
    return output_file

def extract_subset(input_file, output_file, num_games=1000):
    """
    Estrae un sottoinsieme di partite dal database compresso.
    
    Args:
        input_file (str): Percorso del file di input (.pgn.zst)
        output_file (str): Percorso del file di output (.pgn)
        num_games (int): Numero di partite da estrarre
    """
    print(f"Estrazione di {num_games} partite da {input_file}...")
    
    # Apri il file compresso
    with open(input_file, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(f)
        text_stream = io.TextIOWrapper(reader, encoding='utf-8')
        
        # Apri il file di output
        with open(output_file, 'w', encoding='utf-8') as out_f:
            game_count = 0
            current_game = []
            in_game = False
            
            # Leggi il file riga per riga
            for line in text_stream:
                # Se troviamo l'inizio di una nuova partita
                if line.startswith('[Event '):
                    # Se stavamo già leggendo una partita, scriviamola nel file di output
                    if in_game:
                        out_f.write(''.join(current_game))
                        out_f.write('\n\n')
                        game_count += 1
                        
                        # Se abbiamo raggiunto il numero desiderato di partite, usciamo
                        if game_count >= num_games:
                            break
                    
                    # Inizia una nuova partita
                    current_game = [line]
                    in_game = True
                elif in_game:
                    # Aggiungi la riga alla partita corrente
                    current_game.append(line)
            
            # Scrivi l'ultima partita se necessario
            if in_game and game_count < num_games:
                out_f.write(''.join(current_game))
                game_count += 1
    
    print(f"Estratte {game_count} partite in {output_file}")

def clean_pgn(input_file, output_file):
    """
    Pulisce un file PGN mantenendo le partite complete.
    
    Args:
        input_file (str): Percorso del file di input (.pgn)
        output_file (str): Percorso del file di output (.pgn)
    """
    print(f"Preparazione del file PGN {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Dividi il contenuto in partite (una partita inizia con [Event e termina con un risultato)
    games = []
    current_game = ""
    lines = content.split('\n')
    
    for line in lines:
        if line.startswith('[Event '):
            if current_game:
                games.append(current_game.strip())
            current_game = line + '\n'
        else:
            current_game += line + '\n'
    
    # Aggiungi l'ultima partita
    if current_game:
        games.append(current_game.strip())
    
    # Scrivi il file pulito
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(games))
    
    print(f"File PGN preparato salvato in {output_file}")

def main():
    # Directory di output
    output_dir = 'data'
    
    # Scarica il database
    db_file = download_lichess_data(output_dir=output_dir)
    
    if db_file:
        # Estrai un sottoinsieme di partite
        subset_file = os.path.join(output_dir, 'subset.pgn')
        extract_subset(db_file, subset_file, num_games=1000)
        
        # Crea una versione pulita del sottoinsieme
        clean_subset_file = os.path.join(output_dir, 'clean_subset.pgn')
        clean_pgn(subset_file, clean_subset_file)

if __name__ == "__main__":
    main()