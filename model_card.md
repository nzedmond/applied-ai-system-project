# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

> VibeFinder 1.0

---

## 2. Goal/Task

> VibeFinder 1.0 suggests up to 5 songs from an 18-song catalog by comparing each song's attributes against a listener's stated genre, mood, and energy preferences. It is designed for classroom exploration of content-based recommendation concepts and is not intended for deployment with real users or production music services.

---

## 3. Algorithm summary

VibeFinder looks at six things about each song: its genre, its mood, how energetic it feels (a number from 0 to 1), how danceable it is, its emotional tone (called valence — whether it sounds happy or sad), and optionally its popularity, release decade, and descriptive mood tags.

A listener provides their preferences: a favorite genre, a preferred mood, and a target energy level. The system then goes through every song in the catalog and awards points based on how well the song matches those preferences. Genre and mood matches earn fixed bonuses, while energy, danceability, and valence earn partial credit the closer the song's values are to what the listener wants.

All 18 songs receive a total score, are sorted from highest to lowest, and the top results are returned as recommendations along with a short explanation of why each song was chosen.

---

## 4. Data used

- **Size:** 18 songs in `data/songs.csv`; no songs were added or removed from the original starter dataset.
- **Genres represented:** pop, lofi, rock, ambient, jazz, synthwave, electronic, indie pop, folk, classical, hip-hop — 11 distinct genres, though pop, lofi, and electronic appear most often.
- **Moods represented:** happy, chill, intense, relaxed, moody, focused, energetic, melancholy.
- **Whose taste it reflects:** The catalog skews toward contemporary (2020s) Western genres with relatively high energy and danceability scores. Quieter, classical, or non-English music styles are underrepresented, so the data mostly reflects the tastes of listeners familiar with mainstream streaming genres.

---

## 5. Strengths

- **Accurate for well-represented profiles:** Listeners whose preferences align with common catalog genres (pop, lofi, rock, jazz) consistently receive a top recommendation that matches both genre and mood — for example, the Chill Lofi Listener reliably gets Library Rain as its #1 pick with a near-perfect energy match.
- **Transparent scoring:** Every recommendation comes with a plain-language explanation showing exactly which features matched and how many points each contributed, making it easy to understand and debug.
- **Flexible strategies:** Five scoring modes (balanced, genre-first, energy-first, mood-first, discovery) let the same system adapt to different listening contexts without retraining.
- **Diversity control:** The optional penalty for repeated artists and overrepresented genres prevents monotonous results and better reflects expectations for a real playlist.

---

## 6. Observed weaknesses and bias

- **Small catalog:** With only 18 songs, some genres (classical, folk, ambient) have just one or two entries, so listeners who prefer those styles receive very limited variety.
- **Genre dominance:** In balanced mode, a genre match alone (+2.0) outweighs a perfect energy score (+1.0) and a mood match (+1.0) combined, meaning cross-genre discoveries are rare even when another song is a better emotional fit.
- **Uniform taste shape:** Every user is scored with the same fixed weights per mode. Real listeners prioritize features very differently — one person's dealbreaker (wrong genre) is another person's minor preference.
- **No collaborative signal:** The system has no knowledge of what other listeners enjoy, so it cannot surface surprising discoveries the way a platform like Spotify can by finding users with similar histories.
- **Static profiles:** Preferences never update. If a listener's mood changes mid-session, the recommendations stay fixed.
- **Fairness risk at scale:** If deployed on a real product, the genre-weight bias would systematically under-serve niche or non-mainstream genres, reinforcing the visibility of already-popular styles.

---

## 7. Evaluation process

- **Multi-profile sanity check:** Four distinct listener profiles (High-Energy Pop Fan, Chill Lofi Listener, Deep Intense Rock, Mellow Jazz Lover) were run in balanced mode. In every case the #1 recommendation matched the expected genre, mood, and approximate energy level, confirming the core scoring logic behaves as intended.
- **Scoring mode comparison:** The same Pop Fan profile was run across all five modes. Results changed meaningfully — energy-first promoted a hip-hop track that balanced mode ranked lower — which confirmed that weight differences produce real, observable output changes.
- **Weight sensitivity test:** Halving the genre weight (2.0 → 1.0) caused Gym Hero to drop from #2 to #4, demonstrating that the system responds predictably to parameter changes.
- **Diversity penalty check:** Running the Pop Fan profile with and without the penalty confirmed that enabling it prevents Max Pulse from occupying two top-five slots.
- **Automated tests:** Unit tests in `tests/test_recommender.py` verify the scoring function and recommendation output for defined inputs.

---

## 8. Ideas for improvement

- **Larger, richer catalog:** Expanding beyond 18 songs — especially adding more entries for underrepresented genres — would make the recommendations meaningfully more varied.
- **Collaborative filtering layer:** Tracking which songs multiple users listen to together would allow the system to suggest tracks the listener has never heard but that users with similar taste consistently enjoy.
- **Dynamic user profiles:** Updating preferences based on recent session behavior (skips, replays, session time) would let recommendations adapt to changing moods rather than relying on a fixed setup.
- **Tempo-range and lyric-theme features:** Using `tempo_bpm` as a range preference (e.g., 100–130 BPM for running) and adding topic/lyric embeddings would capture dimensions of vibe that current numeric features miss.
- **Group vibe recommendations:** Supporting multiple listeners simultaneously and finding songs that satisfy the overlap of all their preference profiles would be a useful social feature.
- **Learned weights:** Instead of hand-tuning mode weights, a simple model trained on user feedback (skips and replays) could discover which features actually matter most for each listener.

---
