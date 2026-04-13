# 🎵 Music Recommender Simulation

## Project Summary

This project models a content-based music recommendation engine. Working from a library of 18 songs—each described by attributes such as genre, mood, energy, danceability, valence, popularity, release decade, and mood tags—the system compares every song to a listener's taste profile, then returns a ranked shortlist of suggestions along with brief explanations. It offers several scoring strategies, a diversity-aware re-ranking step, and a clean table-based output.

---

## How The System Works

### Song Features
Every song is described by: genre, mood, energy (0.0–1.0), tempo_bpm, valence (0.0–1.0), danceability (0.0–1.0), acousticness (0.0–1.0), popularity (0–100), release_decade, and mood_tags (comma-separated labels such as "euphoric,driving").

### User Profile
Each listener profile captures a preferred genre, mood, and target energy level, plus optional fields for danceability, valence, a preferred decade, and preferred mood tags.

### Scoring Algorithm ("Algorithm Recipe")
Under the default "balanced" strategy, each song receives a score built from five components:

- **Genre match:** +2.0 points when the song's genre aligns with the listener's preference
- **Mood match:** +1.0 point when the song's mood aligns
- **Energy proximity:** up to 1.0 points, computed as `1.0 - abs(song_energy - target_energy)`
- **Danceability proximity:** up to 0.5 points based on how close the values are
- **Valence proximity:** up to 0.5 points based on how close the values are

All songs are ordered from highest to lowest score, and the top-k results are returned with short explanations.

### Multiple Scoring Modes (Challenge 2)
Five scoring strategies are provided, each distributing weights differently across features:

| Mode | Genre | Mood | Energy | Dance | Valence | Use Case |
|------|-------|------|--------|-------|---------|----------|
| balanced | 2.0 | 1.0 | 1.0 | 0.5 | 0.5 | Default, well-rounded |
| genre-first | 3.0 | 0.5 | 0.5 | 0.3 | 0.3 | When genre matters most |
| energy-first | 0.5 | 0.5 | 3.0 | 1.0 | 0.5 | Workout/activity playlists |
| mood-first | 0.5 | 3.0 | 0.5 | 0.5 | 1.0 | Emotional/atmosphere playlists |
| discovery | 0.5 | 1.0 | 1.0 | 0.5 | 0.5 | Uses popularity, decade, mood tags |

### Diversity Penalty (Challenge 3)
When activated, the system docks points for artist repetition (-1.0 per prior appearance) and genre overrepresentation (-0.5 once a genre shows up 3 or more times). This prevents any single artist from taking over the top results.

### Advanced Features (Challenge 1)
Three extra columns exist in the dataset: popularity (0–100), release_decade, and mood_tags. The "discovery" scoring mode draws on all three, rewarding well-known songs, decade alignment, and matching mood descriptors.

### Potential Biases
- Genre match carries the highest point value in balanced mode, making cross-genre discoveries uncommon
- The catalog is small (18 songs) with an uneven spread across genres
- Every listener is evaluated with the same mode weights — actual listeners prioritize features differently

---

## Getting Started

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the app

```bash
python -m src.main
```

### Run tests

```bash
pytest
```

---

## Experiments

### Experiment 1: Four User Profiles (Balanced Mode)

| Profile | #1 Recommendation | Score | Key Match |
|---------|-------------------|-------|-----------|
| High-Energy Pop Fan | Sunrise City | 4.95 | genre + mood + energy |
| Chill Lofi Listener | Library Rain | 4.99 | genre + mood + near-perfect energy |
| Deep Intense Rock | Storm Runner | 4.96 | genre + mood + energy |
| Mellow Jazz Lover | Sunday Morning Tea | 4.90 | genre + mood + energy |


### Experiment 2: Scoring Mode Comparison (Pop Fan Profile)

| Mode | #1 Song | #2 Song | Key Observation |
|------|---------|---------|-----------------|
| balanced | Sunrise City (4.95) | Gym Hero (3.86) | Genre carries the most weight |
| genre-first | Sunrise City (4.57) | Gym Hero (4.03) | Pop tracks become even more dominant |
| energy-first | Sunrise City (5.38) | Downtown Bounce (4.90) | A hip-hop track climbs because its energy is an exact match |
| mood-first | Sunrise City (5.44) | Downtown Bounce (4.95) | Upbeat songs rise to the top regardless of genre |

### Experiment 3: Genre Weight Sensitivity
Reducing the genre weight by half (2.0 → 1.0) pushed Gym Hero from #2 down to #4 for the Pop Fan profile, with mood-matching songs from other genres moving ahead of it.

### Experiment 4: Diversity Penalty
After turning on the diversity penalty, Max Pulse's Downtown Bounce lost 1.0 point for artist repetition—Max Pulse had already appeared through Gym Hero—slipping from #3 to #4.

### Screenshots

![Screenshot 1](screenshots/Screenshot%202026-04-12%20at%209.55.41%20PM.png)

![Screenshot 2](screenshots/Screenshot%202026-04-12%20at%209.55.57%20PM.png)

![Screenshot 3](screenshots/Screenshot%202026-04-12%20at%209.56.22%20PM.png)

![Screenshot 4](screenshots/Screenshot%202026-04-12%20at%209.56.41%20PM.png)

![Screenshot 5](screenshots/Screenshot%202026-04-12%20at%209.57.03%20PM.png)

![Screenshot 6](screenshots/Screenshot%202026-04-12%20at%209.57.13%20PM.png)

---

## Limitations and Risks

- **Tiny catalog:** Only 18 songs — production systems work with millions
- **No audio analysis:** Relies entirely on metadata, not raw audio characteristics
- **Genre dominance:** In balanced mode, a genre match alone outscores all numeric features combined
- **No collaborative filtering:** The system has no awareness of what other listeners enjoy
- **Fixed preferences:** Listener profiles are static and do not adapt over time
- **Diversity is opt-in:** The diversity penalty must be deliberately switched on

---

## Reflection

[**Model Card**](model_card.md)

Working through this recommender made clear that even straightforward weighted scoring can yield impressively sensible results — every profile landed on a top pick that felt appropriate. The mode comparison, however, exposed how much weight choices are really decisions about values: energy-first mode surfaces a hip-hop track for a pop fan purely because the energy aligns, whereas genre-first mode would never bring it up.

The diversity penalty experiment was the most instructive — without it, Max Pulse fills two of the Pop Fan's top five slots, which would feel repetitive in a real product. The penalty solves the problem simply enough, but it surfaces a harder design question: should a recommender optimize for the best match, or for a varied listening experience?
