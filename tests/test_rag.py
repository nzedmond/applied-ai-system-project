"""Tests for the RAG engine and LLM client."""

import sys
import os

# Ensure src/ is on the path so imports work when running pytest from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rag_engine import RAGEngine, _song_to_text
from llm_client import generate_recommendation, _fallback_response


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_SONGS = [
    {
        "id": 1, "title": "Sunrise City", "artist": "Neon Echo",
        "genre": "pop", "mood": "happy", "energy": 0.82, "tempo_bpm": 118,
        "valence": 0.84, "danceability": 0.79, "acousticness": 0.18,
        "popularity": 78, "release_decade": "2020s",
        "mood_tags": "euphoric,uplifting,driving",
    },
    {
        "id": 2, "title": "Midnight Coding", "artist": "LoRoom",
        "genre": "lofi", "mood": "chill", "energy": 0.42, "tempo_bpm": 78,
        "valence": 0.56, "danceability": 0.62, "acousticness": 0.71,
        "popularity": 65, "release_decade": "2020s",
        "mood_tags": "mellow,late-night,cozy",
    },
    {
        "id": 3, "title": "Storm Runner", "artist": "Voltline",
        "genre": "rock", "mood": "intense", "energy": 0.91, "tempo_bpm": 152,
        "valence": 0.48, "danceability": 0.66, "acousticness": 0.10,
        "popularity": 72, "release_decade": "2010s",
        "mood_tags": "aggressive,powerful,raw",
    },
    {
        "id": 4, "title": "Gym Hero", "artist": "Max Pulse",
        "genre": "pop", "mood": "intense", "energy": 0.93, "tempo_bpm": 132,
        "valence": 0.77, "danceability": 0.88, "acousticness": 0.05,
        "popularity": 85, "release_decade": "2020s",
        "mood_tags": "motivational,powerful,driving",
    },
]


def _make_engine() -> RAGEngine:
    """Create an in-memory RAG engine with sample songs."""
    engine = RAGEngine(persist_dir=None)  # ephemeral
    engine.index_songs(SAMPLE_SONGS)
    return engine


# ---------------------------------------------------------------------------
# RAGEngine tests
# ---------------------------------------------------------------------------


class TestSongToText:
    def test_contains_title_and_artist(self):
        text = _song_to_text(SAMPLE_SONGS[0])
        assert "Sunrise City" in text
        assert "Neon Echo" in text

    def test_contains_genre_and_mood(self):
        text = _song_to_text(SAMPLE_SONGS[0])
        assert "pop" in text
        assert "happy" in text

    def test_contains_mood_tags(self):
        text = _song_to_text(SAMPLE_SONGS[0])
        assert "euphoric" in text


class TestRAGEngineIndex:
    def test_index_songs_count(self):
        engine = _make_engine()
        assert engine.count() == len(SAMPLE_SONGS)

    def test_index_songs_skips_duplicates(self):
        engine = _make_engine()
        added = engine.index_songs(SAMPLE_SONGS)  # index again
        assert added == 0
        assert engine.count() == len(SAMPLE_SONGS)

    def test_index_empty_list(self):
        engine = _make_engine()
        engine.reset()  # start with a clean collection
        added = engine.index_songs([])
        assert added == 0
        assert engine.count() == 0


class TestRAGEngineSearch:
    def test_search_returns_results(self):
        engine = _make_engine()
        results = engine.search("chill lofi music for studying")
        assert len(results) > 0

    def test_search_returns_correct_fields(self):
        engine = _make_engine()
        results = engine.search("intense rock music", k=1)
        assert len(results) == 1
        result = results[0]
        assert "title" in result
        assert "artist" in result
        assert "genre" in result
        assert "distance" in result
        assert "document" in result

    def test_search_ranks_relevant_first(self):
        engine = _make_engine()
        results = engine.search("relaxing chill lofi beats for late night coding", k=2)
        # Midnight Coding should rank higher than Storm Runner
        titles = [r["title"] for r in results]
        assert "Midnight Coding" in titles

    def test_search_respects_k(self):
        engine = _make_engine()
        results = engine.search("music", k=2)
        assert len(results) == 2

    def test_search_k_larger_than_collection(self):
        engine = _make_engine()
        results = engine.search("music", k=100)
        assert len(results) == len(SAMPLE_SONGS)

    def test_search_empty_query(self):
        engine = _make_engine()
        results = engine.search("")
        assert results == []

    def test_search_empty_collection(self):
        engine = _make_engine()
        engine.reset()  # clear all documents
        results = engine.search("anything")
        assert results == []


class TestRAGEngineReset:
    def test_reset_clears_collection(self):
        engine = _make_engine()
        assert engine.count() == len(SAMPLE_SONGS)
        engine.reset()
        assert engine.count() == 0


# ---------------------------------------------------------------------------
# LLM client tests
# ---------------------------------------------------------------------------


class TestFallbackResponse:
    def test_fallback_contains_query(self):
        response = _fallback_response("chill vibes", SAMPLE_SONGS[:2])
        assert "chill vibes" in response

    def test_fallback_contains_song_titles(self):
        response = _fallback_response("anything", SAMPLE_SONGS[:2])
        assert "Sunrise City" in response
        assert "Midnight Coding" in response

    def test_fallback_empty_songs(self):
        response = _fallback_response("test query", [])
        assert "No songs found" in response

    def test_fallback_mentions_api_key(self):
        response = _fallback_response("test", SAMPLE_SONGS[:1])
        assert "OPENAI_API_KEY" in response


class TestGenerateRecommendation:
    def test_generates_without_api_key(self):
        """Should use fallback when no API key is set."""
        # Ensure no key is set for this test
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        try:
            response = generate_recommendation("chill music", SAMPLE_SONGS[:3])
            assert isinstance(response, str)
            assert len(response) > 0
            # Should contain fallback indicator
            assert "OPENAI_API_KEY" in response or "Sunrise City" in response
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_handles_empty_retrieved_songs(self):
        response = generate_recommendation("anything", [])
        assert "No songs found" in response
