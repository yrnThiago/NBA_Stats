import warnings
import whisper
import pyttsx3
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import wave
import pyaudio
from fuzzywuzzy import process

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def scrape_nba_stats(url, save_path="nba_stats.csv"):
    if os.path.exists(save_path):
        print("Arquivo de estatísticas encontrado. Carregando dados do arquivo...")
        return pd.read_csv(save_path)
    print("Buscando estatísticas dos jogadores da NBA...")
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find("table", {"id": "per_game_stats"})
    df = pd.read_html(str(table))[0]
    df = df[df['Player'] != 'Player']
    df.to_csv(save_path, index=False)
    print("Estatísticas salvas no arquivo com sucesso.")
    return df

def record_audio(filename, duration=5, rate=44100, channels=1):
    print(f"Gravando áudio por {duration} segundos...")
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=channels, rate=rate, input=True, frames_per_buffer=1024)
    frames = []
    for _ in range(0, int(rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    audio.terminate()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    print("Gravação de áudio concluída.")

def transcribe_audio_to_text(audio_file, model):
    print("Transcrevendo áudio...")
    result = model.transcribe(audio_file)
    print("Transcrição concluída.")
    return result["text"].strip()

def get_player_stats(player_name, stats_df, player_list):
    print(f"Procurando estatísticas para o jogador: {player_name}...")
    matched_name = fuzzy_match_name(player_name, player_list)
    player_stats = stats_df[stats_df['Player'].str.contains(matched_name, case=False, na=False)]
    if player_stats.empty:
        print(f"Nenhuma estatística encontrada para {player_name} (melhor correspondência: {matched_name}).")
        return None
    print(f"Estatísticas encontradas para o jogador: {matched_name}.")
    return player_stats.iloc[0]

def fuzzy_match_name(name, player_list):
    best_match = process.extractOne(name, player_list)
    return best_match[0] if best_match[1] > 80 else name

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def speak_player_stats(player_stats):
    stat_labels = {
        'G': 'Jogos',
        'MP': 'Minutos Jogados',
        'FG%': 'Porcentagem de Acerto de Arremesso',
        'eFG%': 'Eficiência',
        'TRB': 'Rebotes',
        'AST': 'Assistências',
        'STL': 'Roubos',
        'BLK': 'Bloqueios',
        'PTS': 'Pontos'
    }
    selected_stats = ['G', 'MP', 'FG%', 'eFG%', 'TRB', 'AST', 'STL', 'BLK', 'PTS']
    stats_string = f"Estatísticas de {player_stats['Player']}: " + ", ".join(
        [f"{stat_labels[col]}: {player_stats[col]}" for col in selected_stats if col in player_stats])
    print(stats_string)
    speak_text(stats_string)

def main():
    print("Iniciando o assistente de estatísticas da NBA...")
    nba_url = "https://www.basketball-reference.com/leagues/NBA_2025_per_game.html"
    stats_df = scrape_nba_stats(nba_url)
    player_list = stats_df['Player'].tolist()
    audio_file = "player_name.wav"
    record_audio(audio_file, duration=5)
    whisper_model = whisper.load_model("base")
    player_name = transcribe_audio_to_text(audio_file, whisper_model)
    print(f"Nome do jogador reconhecido: {player_name}")
    player_stats = get_player_stats(player_name, stats_df, player_list)
    if player_stats is None:
        speak_text(f"Desculpe, não encontrei estatísticas para {player_name}.")
    else:
        speak_player_stats(player_stats)

if __name__ == "__main__":
    main()
