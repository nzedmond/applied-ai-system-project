"""
Command line runner for the Music Recommender Simulation.
Supports multiple scoring modes, diversity penalty, tabulated output,
and RAG-powered natural language queries.
"""
import logging
import os
import sys

from recommender import load_songs, recommend_songs, SCORING_MODES
from rag_engine import RAGEngine, build_rag_engine, _song_to_text
from llm_client import generate_recommendation
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("recommender.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def print_recommendations(profile_name: str, user_prefs: dict, results: list) -> None:
    """Challenge 4: Display recommendations as a formatted table."""
    print(f"\n{'='*80}")
    print(f"  Profile: {profile_name}")
    prefs_str = ", ".join(f"{k}={v}" for k, v in user_prefs.items() if k != "preferred_mood_tags")
    print(f"  Preferences: {prefs_str}")
    print(f"{'='*80}\n")

    table_data = []
    for i, (song, score, explanation) in enumerate(results, 1):
        table_data.append([
            i,
            song["title"],
            song["artist"],
            song["genre"],
            f"{score:.2f}",
            explanation,
        ])

    headers = ["#", "Title", "Artist", "Genre", "Score", "Reasons"]
    print(tabulate(table_data, headers=headers, tablefmt="rounded_grid", maxcolwidths=[3, 20, 15, 12, 6, 50]))
    print()


def main() -> None:
    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
    songs = load_songs(csv_path)

    # --- Build RAG engine (indexes songs on first run, reuses on subsequent) ---
    persist_dir = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
    rag = RAGEngine(persist_dir=persist_dir)
    rag.index_songs(songs)
    logger.info("RAG engine ready with %d songs", rag.count())

    # --- Core profiles ---
    profiles = {
        "High-Energy Pop Fan": {
            "genre": "pop", "mood": "happy", "energy": 0.85,
            "danceability": 0.8, "valence": 0.8,
        },
        "Chill Lofi Listener": {
            "genre": "lofi", "mood": "chill", "energy": 0.35,
            "danceability": 0.55, "valence": 0.6,
        },
        "Deep Intense Rock": {
            "genre": "rock", "mood": "intense", "energy": 0.9,
            "danceability": 0.7, "valence": 0.5,
        },
        "Mellow Jazz Lover": {
            "genre": "jazz", "mood": "relaxed", "energy": 0.3,
            "danceability": 0.45, "valence": 0.7,
        },
    }

    # -------------------------------------------------------
    # Part A: Default balanced mode (original scoring)
    # -------------------------------------------------------
    print("\n" + "="*80)
    print("  SCORING MODE: balanced (default)")
    print("="*80)
    for name, prefs in profiles.items():
        results = recommend_songs(prefs, songs, k=5, mode="balanced")
        print_recommendations(name, prefs, results)

    # -------------------------------------------------------
    # Part B: Challenge 2 — Compare scoring modes for one profile
    # -------------------------------------------------------
    test_profile_name = "High-Energy Pop Fan"
    test_prefs = profiles[test_profile_name]

    print("\n" + "#"*80)
    print("  CHALLENGE 2: Comparing Scoring Modes")
    print(f"  Profile: {test_profile_name}")
    print("#"*80)

    for mode_name in ["balanced", "genre-first", "energy-first", "mood-first"]:
        print(f"\n--- Mode: {mode_name} ---")
        results = recommend_songs(test_prefs, songs, k=5, mode=mode_name)
        print_recommendations(test_profile_name, test_prefs, results)

    # -------------------------------------------------------
    # Part C: Challenge 3 — Diversity penalty demo
    # -------------------------------------------------------
    print("\n" + "#"*80)
    print("  CHALLENGE 3: Diversity Penalty Comparison")
    print("#"*80)

    print("\n--- WITHOUT diversity penalty ---")
    results_no_div = recommend_songs(test_prefs, songs, k=5, mode="balanced", diversity_penalty=False)
    print_recommendations(test_profile_name, test_prefs, results_no_div)

    print("\n--- WITH diversity penalty ---")
    results_div = recommend_songs(test_prefs, songs, k=5, mode="balanced", diversity_penalty=True)
    print_recommendations(test_profile_name, test_prefs, results_div)

    # -------------------------------------------------------
    # Part D: Challenge 1 — Discovery mode with mood tags
    # -------------------------------------------------------
    print("\n" + "#"*80)
    print("  CHALLENGE 1: Discovery Mode with Advanced Features")
    print("#"*80)

    discovery_profile = {
        "genre": "electronic", "mood": "energetic", "energy": 0.85,
        "danceability": 0.8, "valence": 0.7,
        "preferred_decade": "2020s",
        "preferred_mood_tags": ["euphoric", "driving", "uplifting"],
    }
    results_disc = recommend_songs(discovery_profile, songs, k=5, mode="discovery")
    print_recommendations("Discovery Explorer", discovery_profile, results_disc)

    # -------------------------------------------------------
    # Part E: RAG-powered natural language recommendations
    # -------------------------------------------------------
    print("\n" + "#"*80)
    print("  RAG-POWERED NATURAL LANGUAGE RECOMMENDATIONS")
    print("#"*80)

    demo_queries = [
        "something chill and mellow for studying late at night",
        "high energy workout music that makes me want to run faster",
        "feel-good songs for a sunny road trip with friends",
    ]

    for query in demo_queries:
        print(f"\n{'─'*80}")
        print(f"  Query: \"{query}\"")
        print(f"{'─'*80}")
        retrieved = rag.search(query, k=5)
        logger.info("Retrieved %d songs for query: %s", len(retrieved), query)
        response = generate_recommendation(query, retrieved)
        print(f"\n{response}\n")

    # -------------------------------------------------------
    # Part F: Interactive RAG query loop
    # -------------------------------------------------------
    print("\n" + "#"*80)
    print("  INTERACTIVE MODE — Ask for music in plain English")
    print("  Type 'quit' or 'exit' to stop.")
    print("#"*80)

    while True:
        try:
            user_query = input("\n🎵 What kind of music are you looking for? > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_query:
            continue
        if user_query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        logger.info("User query: %s", user_query)
        retrieved = rag.search(user_query, k=5)
        response = generate_recommendation(user_query, retrieved)
        print(f"\n{response}")


if __name__ == "__main__":
    main()

