"""
LLM Client for the Music Recommender.

Uses OpenAI to generate natural-language recommendations from RAG-retrieved songs.
Falls back to a structured template when no API key is configured.
"""

import logging
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env file if present
load_dotenv()


def _build_system_prompt() -> str:
    """Return the system prompt that shapes the LLM's behaviour."""
    return (
        "You are a friendly music recommendation assistant. "
        "You will receive a user's natural-language music request and a set of "
        "songs retrieved from a catalog based on semantic similarity.\n\n"
        "Your job:\n"
        "1. Analyse the retrieved songs and pick the best matches for the user's request.\n"
        "2. Rank them from most to least relevant.\n"
        "3. For each song, explain in one sentence WHY it fits the request, "
        "referencing specific attributes (genre, mood, energy, tempo, tags, etc.).\n"
        "4. If none of the songs are a great fit, say so honestly.\n\n"
        "Keep your response concise and conversational. "
        "Do NOT invent songs that are not in the provided list."
    )


def _build_user_prompt(query: str, retrieved_songs: List[Dict]) -> str:
    """Build the user message containing the query and retrieved context."""
    songs_text = "\n".join(
        f"- {s.get('document', 'No description available')}"
        for s in retrieved_songs
    )
    return (
        f"User request: \"{query}\"\n\n"
        f"Retrieved songs from catalog:\n{songs_text}\n\n"
        "Based on the songs above, give me your ranked recommendations "
        "with a brief explanation for each."
    )


def _fallback_response(query: str, retrieved_songs: List[Dict]) -> str:
    """Generate a structured recommendation without calling the LLM.

    This ensures the system always produces useful output, even without
    an API key.
    """
    if not retrieved_songs:
        return f"No songs found matching: \"{query}\""

    lines = [f"Based on your request: \"{query}\"\n"]
    lines.append("Here are the best matches from our catalog:\n")

    for i, song in enumerate(retrieved_songs, 1):
        title = song.get("title", "Unknown")
        artist = song.get("artist", "Unknown")
        genre = song.get("genre", "")
        mood = song.get("mood", "")
        energy = song.get("energy", "")
        distance = song.get("distance", 0)
        similarity = max(0, 1 - distance) * 100

        lines.append(
            f"  {i}. \"{title}\" by {artist} — {mood} {genre}, "
            f"energy: {energy}, similarity: {similarity:.0f}%"
        )

    lines.append(
        "\n(Powered by semantic search. Set OPENAI_API_KEY in .env "
        "for AI-generated explanations.)"
    )
    return "\n".join(lines)


def generate_recommendation(
    query: str,
    retrieved_songs: List[Dict],
    model: str = "gpt-3.5-turbo",
) -> str:
    """Generate a natural-language recommendation using the LLM.

    If the OpenAI API key is not set or the call fails, falls back to
    a structured template response.

    Args:
        query: The user's natural-language music request.
        retrieved_songs: Songs retrieved by the RAG engine.
        model: OpenAI model to use.

    Returns:
        A string containing the recommendation text.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key or api_key == "your-api-key-here":
        logger.info("No OpenAI API key configured — using fallback response")
        return _fallback_response(query, retrieved_songs)

    try:
        import openai

        openai.api_key = api_key

        system_msg = _build_system_prompt()
        user_msg = _build_user_prompt(query, retrieved_songs)

        logger.info("Calling OpenAI %s with %d retrieved songs", model, len(retrieved_songs))

        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=600,
        )

        result = response.choices[0].message["content"].strip()
        logger.info("LLM response received (%d chars)", len(result))
        return result

    except Exception as e:
        logger.error("OpenAI API call failed: %s — using fallback", e)
        return _fallback_response(query, retrieved_songs)
