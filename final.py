from flask import Flask, request, render_template_string
import os, random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

app = Flask(__name__)

# === Spotify API ===
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET")
))

# === Load Dataset ===
df = pd.read_csv("dataset.csv")
audio_features = [
    "danceability", "energy", "valence", "tempo",
    "acousticness", "instrumentalness", "liveness", "speechiness"
]
df = df.dropna(subset=audio_features)
scaler = MinMaxScaler()
df[audio_features] = scaler.fit_transform(df[audio_features])

# === Utility Functions ===
def get_spotify_info(track, artist):
    results = sp.search(q=f"{track} {artist}", type='track', limit=1)
    items = results['tracks']['items']
    if items:
        item = items[0]
        return {
            "album_art": item['album']['images'][0]['url'] if item['album']['images'] else None,
            "uri": item['uri']
        }
    return {"album_art": None, "uri": None}

def build_song_data(indices):
    songs = []
    for i in indices:
        song = df.iloc[i]
        info = get_spotify_info(song['track_name'], song['artists'])
        songs.append({
            "track_name": song['track_name'],
            "artists": song['artists'],
            "track_genre": song['track_genre'],
            "album_art": info["album_art"] or "https://via.placeholder.com/150?text=No+Image",
            "spotify_uri": info["uri"]
        })
    return songs

def get_random_songs(n=20):
    return build_song_data(random.sample(range(len(df)), n))

def recommend_similar_by_track(track_name, top_k=10):
    row = df[df['track_name'] == track_name]
    if row.empty: return []
    target = row[audio_features].values[0]
    scores = cosine_similarity([target], df[audio_features])[0]
    indices = np.argsort(scores)[-top_k-1:][::-1]
    indices = [i for i in indices if df.iloc[i]['track_name'] != track_name][:top_k]
    return build_song_data(indices)

# === HTML Template ===
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
<title>Music Recommender</title>
<style>
    body { font-family: 'Segoe UI', sans-serif; background: black; color: white; margin: 0; padding: 0; }
    header { background: #111; padding: 20px; text-align: center; font-size: 28px; font-weight: bold; }
    .songs-grid { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; padding: 20px; }
    .song-card {
        background: #222; padding: 10px; border-radius: 12px; width: 200px;
        text-align: center; box-shadow: 0 0 10px #00ff00aa; transition: transform 0.3s; cursor: pointer;
    }
    .song-card:hover { transform: scale(1.05); box-shadow: 0 0 20px #00ff00cc; }
    .song-card img { width: 100%; border-radius: 12px; }
    .back-home { text-align: center; margin: 40px; }
</style>
</head>
<body>
<header>Music Recommendation System</header>

{% if selected_song %}
<h2 style="text-align:center;">Now Playing: {{ selected_song }}</h2>
{% if selected_uri %}
<div style="text-align:center;"><iframe src="https://open.spotify.com/embed/track/{{ selected_uri.split(':')[-1] }}" width="300" height="80" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe></div>
{% endif %}
{% else %}
<h2 style="text-align:center;">Featured Songs</h2>
{% endif %}

<div class="songs-grid">
    {% for song in results %}
    <div class="song-card" onclick="window.location.href='/song/{{ song.track_name }}';">
        <img src="{{ song.album_art }}">
        <div><strong>{{ song.track_name }}</strong></div>
        <div>{{ song.artists }}</div>
    </div>
    {% endfor %}
</div>

{% if request.path != '/' %}
<div class="back-home"><a href="/" style="color:lightgreen;">← Back to Home</a></div>
{% endif %}
</body>
</html>
'''

# === Routes ===
@app.route('/')
def home():
    songs = get_random_songs()
    return render_template_string(HTML_TEMPLATE, results=songs)

@app.route('/song/<track_name>')
def song(track_name):
    similar = recommend_similar_by_track(track_name)
    info = get_spotify_info(track_name, df[df['track_name'] == track_name]['artists'].values[0])
    return render_template_string(HTML_TEMPLATE, selected_song=track_name, selected_uri=info["uri"], results=similar)

# === For Deployment (Render, Railway, etc.) ===
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
