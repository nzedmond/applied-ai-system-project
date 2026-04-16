"""
RAG Engine for the Music Recommender.

Embeds song metadata into a ChromaDB vector store using sentence-transformers,
then supports natural-language semantic search to retrieve the most relevant songs.
"""

import logging
import os
from typing import Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

from recommender import load_songs

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_COLLECTION_NAME = "songs"
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # lightweight, fast, good quality


def _song_to_text(song: Dict) -> str:
    """Convert a song dict into a descriptive text string for embedding.

    The richer the text, the better the semantic matches will be.
    """
    tags = song.get("mood_tags", "")
    decade = song.get("release_decade", "unknown")
    popularity = song.get("popularity", "unknown")

    text = (
        f'"{song["title"]}" by {song["artist"]} is a {song["mood"]} {song["genre"]} track. '
        f"Energy: {song['energy']}, tempo: {song['tempo_bpm']} BPM, "
        f"danceability: {song['danceability']}, valence: {song['valence']}, "
        f"acousticness: {song['acousticness']}. "
        f"Released in the {decade}, popularity: {popularity}/100."
    )
    if tags:
        text += f" Mood tags: {tags}."
    return text


class RAGEngine:
    """Manages the ChromaDB vector store for song retrieval."""

    def __init__(self, persist_dir: Optional[str] = None):
        """Initialise the RAG engine.

        Args:
            persist_dir: Directory for persisting ChromaDB data.
                         If None, uses an in-memory store (useful for tests).
        """
        self._embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=_EMBEDDING_MODEL,
        )

        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_dir)
            logger.info("ChromaDB persistent client at %s", persist_dir)
        else:
            self._client = chromadb.EphemeralClient()
            logger.info("ChromaDB ephemeral (in-memory) client")

        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._embed_fn,
        )
        logger.info(
            "Collection '%s' ready (%d documents)",
            _COLLECTION_NAME,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_songs(self, songs: List[Dict]) -> int:
        """Embed and store songs in the vector store.

        Skips songs that are already indexed (based on song id).
        Returns the number of newly added songs.
        """
        existing_ids = set(self._collection.get()["ids"])
        new_docs, new_ids, new_metas = [], [], []

        for song in songs:
            song_id = str(song["id"])
            if song_id in existing_ids:
                continue
            doc_text = _song_to_text(song)
            new_docs.append(doc_text)
            new_ids.append(song_id)
            # Store core metadata for later use
            new_metas.append({
                "title": song["title"],
                "artist": song["artist"],
                "genre": song["genre"],
                "mood": song["mood"],
                "energy": float(song["energy"]),
                "danceability": float(song["danceability"]),
                "valence": float(song["valence"]),
            })

        if new_docs:
            self._collection.add(
                documents=new_docs,
                ids=new_ids,
                metadatas=new_metas,
            )
            logger.info("Indexed %d new songs (skipped %d existing)", len(new_docs), len(existing_ids))
        else:
            logger.info("All %d songs already indexed — nothing to add", len(songs))

        return len(new_docs)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve the top-k songs most semantically similar to *query*.

        Args:
            query: A natural-language description of the desired music.
            k: Number of results to return.

        Returns:
            A list of dicts, each containing:
              - id, title, artist, genre, mood, energy, danceability, valence
              - document  (the descriptive text that was embedded)
              - distance  (lower = more similar)
        """
        if not query or not query.strip():
            logger.warning("Empty query received — returning no results")
            return []

        total = self._collection.count()
        if total == 0:
            logger.warning("Collection is empty — call index_songs() first")
            return []

        # Never ask for more results than exist
        k = min(k, total)

        results = self._collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved = []
        for i in range(len(results["ids"][0])):
            entry = {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "distance": results["distances"][0][i],
                **results["metadatas"][0][i],
            }
            retrieved.append(entry)

        logger.info(
            "Query: '%s' — retrieved %d songs (closest distance: %.4f)",
            query[:80],
            len(retrieved),
            retrieved[0]["distance"] if retrieved else float("inf"),
        )
        return retrieved

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of indexed songs."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete the collection and recreate it (useful for re-indexing)."""
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._embed_fn,
        )
        logger.info("Collection reset")


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def build_rag_engine(csv_path: str, persist_dir: str = "chroma_db") -> RAGEngine:
    """Load songs from CSV, index them, and return a ready-to-use RAGEngine."""
    logger.info("Building RAG engine from %s", csv_path)
    songs = load_songs(csv_path)
    engine = RAGEngine(persist_dir=persist_dir)
    added = engine.index_songs(songs)
    logger.info("RAG engine ready — %d songs indexed (%d new)", engine.count(), added)
    return engine
