# IMPLEMENTING RAG (RETRIEVAL-AUGMENTED GENERATION)

## Why RAG?

* The project already has songs described by rich metadata (genre, mood, energy, tags, etc.) — perfect candidates for embedding into a vector store. 
* RAG would let users query in natural language (e.g., "something upbeat for a road trip") instead of filling in rigid profile fields. 
* It replaces the manual weighted scoring with semantic similarity search, which is more flexible and scalable. 
* An LLM can generate richer, more natural explanations for why a song was recommended. It directly addresses a stated limitation: the system currently can't do cross-genre discovery well. 
* Semantic search would surface songs that feel similar even across genres.

