from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import csv


@dataclass
class Song:
    """Represents a song and its attributes."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    # Challenge 1: Advanced features
    popularity: int = 50
    release_decade: str = "2020s"
    mood_tags: str = ""


@dataclass
class UserProfile:
    """Represents a user's taste preferences."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


class Recommender:
    """OOP implementation of the recommendation logic."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        """Return top-k songs ranked by relevance to the user profile."""
        scored = []
        for song in self.songs:
            score = 0.0
            if song.genre == user.favorite_genre:
                score += 2.0
            if song.mood == user.favorite_mood:
                score += 1.0
            score += 1.0 - abs(song.energy - user.target_energy)
            if user.likes_acoustic:
                score += song.acousticness * 0.5
            scored.append((score, song))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [song for _, song in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        """Return a human-readable explanation of why a song was recommended."""
        reasons = []
        if song.genre == user.favorite_genre:
            reasons.append(f"genre match: {song.genre} (+2.0)")
        if song.mood == user.favorite_mood:
            reasons.append(f"mood match: {song.mood} (+1.0)")
        energy_score = 1.0 - abs(song.energy - user.target_energy)
        reasons.append(f"energy similarity: {energy_score:.2f}")
        if user.likes_acoustic:
            acoustic_score = song.acousticness * 0.5
            reasons.append(f"acoustic bonus: {acoustic_score:.2f}")
        return "; ".join(reasons)


# ---------------------------------------------------------------
# Challenge 2: Scoring Mode Presets (Strategy Pattern)
# ---------------------------------------------------------------

SCORING_MODES = {
    "balanced": {
        "genre_weight": 2.0,
        "mood_weight": 1.0,
        "energy_weight": 1.0,
        "danceability_weight": 0.5,
        "valence_weight": 0.5,
        "popularity_weight": 0.0,
        "decade_weight": 0.0,
        "mood_tags_weight": 0.0,
    },
    "genre-first": {
        "genre_weight": 3.0,
        "mood_weight": 0.5,
        "energy_weight": 0.5,
        "danceability_weight": 0.3,
        "valence_weight": 0.3,
        "popularity_weight": 0.0,
        "decade_weight": 0.0,
        "mood_tags_weight": 0.0,
    },
    "energy-first": {
        "genre_weight": 0.5,
        "mood_weight": 0.5,
        "energy_weight": 3.0,
        "danceability_weight": 1.0,
        "valence_weight": 0.5,
        "popularity_weight": 0.0,
        "decade_weight": 0.0,
        "mood_tags_weight": 0.0,
    },
    "mood-first": {
        "genre_weight": 0.5,
        "mood_weight": 3.0,
        "energy_weight": 0.5,
        "danceability_weight": 0.5,
        "valence_weight": 1.0,
        "popularity_weight": 0.0,
        "decade_weight": 0.0,
        "mood_tags_weight": 0.0,
    },
    "discovery": {
        "genre_weight": 0.5,
        "mood_weight": 1.0,
        "energy_weight": 1.0,
        "danceability_weight": 0.5,
        "valence_weight": 0.5,
        "popularity_weight": 0.3,
        "decade_weight": 0.2,
        "mood_tags_weight": 0.5,
    },
}


def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file and return a list of dictionaries."""
    songs = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["id"] = int(row["id"])
            row["energy"] = float(row["energy"])
            row["tempo_bpm"] = float(row["tempo_bpm"])
            row["valence"] = float(row["valence"])
            row["danceability"] = float(row["danceability"])
            row["acousticness"] = float(row["acousticness"])
            # Challenge 1: Parse advanced features with defaults
            row["popularity"] = int(row.get("popularity", 50))
            row["release_decade"] = row.get("release_decade", "2020s")
            row["mood_tags"] = row.get("mood_tags", "")
            songs.append(row)
    print(f"Loaded {len(songs)} songs from {csv_path}")
    return songs


def score_song(user_prefs: Dict, song: Dict, weights: Dict = None) -> Tuple[float, List[str]]:
    """Score a single song against user preferences. Returns (score, reasons)."""
    if weights is None:
        weights = SCORING_MODES["balanced"]

    score = 0.0
    reasons = []

    # Genre match
    genre_w = weights.get("genre_weight", 2.0)
    if song["genre"].lower() == user_prefs.get("genre", "").lower():
        score += genre_w
        reasons.append(f"genre match: {song['genre']} (+{genre_w:.1f})")

    # Mood match
    mood_w = weights.get("mood_weight", 1.0)
    if song["mood"].lower() == user_prefs.get("mood", "").lower():
        score += mood_w
        reasons.append(f"mood match: {song['mood']} (+{mood_w:.1f})")

    # Energy similarity
    energy_w = weights.get("energy_weight", 1.0)
    if "energy" in user_prefs:
        energy_sim = (1.0 - abs(song["energy"] - user_prefs["energy"])) * energy_w
        score += energy_sim
        reasons.append(f"energy: {energy_sim:.2f}")

    # Danceability similarity
    dance_w = weights.get("danceability_weight", 0.5)
    if "danceability" in user_prefs:
        dance_sim = (1.0 - abs(song["danceability"] - user_prefs["danceability"])) * dance_w
        score += dance_sim
        reasons.append(f"dance: {dance_sim:.2f}")

    # Valence similarity
    valence_w = weights.get("valence_weight", 0.5)
    if "valence" in user_prefs:
        valence_sim = (1.0 - abs(song["valence"] - user_prefs["valence"])) * valence_w
        score += valence_sim
        reasons.append(f"valence: {valence_sim:.2f}")

    # Challenge 1: Popularity bonus (normalized 0-100 -> 0.0-1.0)
    pop_w = weights.get("popularity_weight", 0.0)
    if pop_w > 0:
        pop_score = (song.get("popularity", 50) / 100.0) * pop_w
        score += pop_score
        reasons.append(f"popularity: {pop_score:.2f}")

    # Challenge 1: Decade preference
    decade_w = weights.get("decade_weight", 0.0)
    if decade_w > 0 and "preferred_decade" in user_prefs:
        if song.get("release_decade", "") == user_prefs["preferred_decade"]:
            score += decade_w
            reasons.append(f"decade match: {song['release_decade']} (+{decade_w:.1f})")

    # Challenge 1: Mood tags bonus
    tags_w = weights.get("mood_tags_weight", 0.0)
    if tags_w > 0 and "preferred_mood_tags" in user_prefs:
        song_tags = set(t.strip().lower() for t in song.get("mood_tags", "").split(",") if t.strip())
        user_tags = set(t.strip().lower() for t in user_prefs["preferred_mood_tags"])
        if song_tags and user_tags:
            overlap = len(song_tags & user_tags) / max(len(user_tags), 1)
            tag_score = overlap * tags_w
            score += tag_score
            if tag_score > 0:
                reasons.append(f"mood tags: {tag_score:.2f}")

    return score, reasons


def recommend_songs(
    user_prefs: Dict,
    songs: List[Dict],
    k: int = 5,
    mode: str = "balanced",
    diversity_penalty: bool = False,
) -> List[Tuple[Dict, float, str]]:
    """Score all songs and return top-k as (song_dict, score, explanation).

    Args:
        mode: One of 'balanced', 'genre-first', 'energy-first', 'mood-first', 'discovery'
        diversity_penalty: If True, penalize repeated artists/genres in top results
    """
    weights = SCORING_MODES.get(mode, SCORING_MODES["balanced"])

    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song, weights)
        explanation = "; ".join(reasons)
        scored.append((song, score, explanation))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Challenge 3: Diversity penalty - greedy re-ranking
    if diversity_penalty:
        selected = []
        seen_artists = {}
        seen_genres = {}
        for song, score, explanation in scored:
            artist = song["artist"]
            genre = song["genre"]
            penalty = 0.0
            penalty_reasons = []

            # Penalize repeated artist: -1.0 per previous appearance
            artist_count = seen_artists.get(artist, 0)
            if artist_count > 0:
                artist_pen = artist_count * 1.0
                penalty += artist_pen
                penalty_reasons.append(f"artist repeat penalty: -{artist_pen:.1f}")

            # Penalize genre if already appears 2+ times: -0.5 per extra
            genre_count = seen_genres.get(genre, 0)
            if genre_count >= 2:
                genre_pen = (genre_count - 1) * 0.5
                penalty += genre_pen
                penalty_reasons.append(f"genre diversity penalty: -{genre_pen:.1f}")

            adjusted_score = score - penalty
            if penalty_reasons:
                explanation += "; " + "; ".join(penalty_reasons)

            selected.append((song, adjusted_score, explanation))
            seen_artists[artist] = artist_count + 1
            seen_genres[genre] = genre_count + 1

        selected.sort(key=lambda x: x[1], reverse=True)
        return selected[:k]

    return scored[:k]
