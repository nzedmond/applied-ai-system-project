import os
from typing import Dict, List, Tuple

import streamlit as st

from llm_client import generate_recommendation
from rag_engine import RAGEngine
from recommender import load_songs


st.set_page_config(
    page_title="PulseDeck",
    page_icon=":notes:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(255, 145, 77, 0.16), transparent 30%),
                    radial-gradient(circle at 85% 15%, rgba(245, 90, 74, 0.18), transparent 26%),
                    linear-gradient(160deg, #140d0d 0%, #231414 45%, #0e0f16 100%);
                color: #f7efe6;
            }

            [data-testid="stHeader"] {
                background: transparent;
            }

            [data-testid="stAppViewContainer"] > .main {
                background: transparent;
            }

            .hero-shell {
                position: relative;
                overflow: hidden;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 28px;
                padding: 2.5rem;
                background: linear-gradient(145deg, rgba(28, 19, 19, 0.92), rgba(17, 18, 25, 0.88));
                box-shadow: 0 30px 80px rgba(0, 0, 0, 0.35);
            }

            .hero-shell::after {
                content: "";
                position: absolute;
                inset: auto -80px -110px auto;
                width: 280px;
                height: 280px;
                border-radius: 50%;
                background: radial-gradient(circle, rgba(255, 184, 107, 0.55) 0%, rgba(255, 184, 107, 0.1) 35%, transparent 60%);
                filter: blur(6px);
            }

            .eyebrow {
                display: inline-block;
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                font-size: 0.72rem;
                font-weight: 700;
                color: #ffd9bc;
                background: rgba(255, 126, 95, 0.16);
                border: 1px solid rgba(255, 160, 122, 0.2);
            }

            .hero-title {
                margin: 1rem 0 0.6rem;
                font-size: 3.6rem;
                line-height: 0.95;
                font-weight: 800;
                letter-spacing: -0.04em;
                color: #fff5eb;
            }

            .hero-copy {
                max-width: 42rem;
                font-size: 1.04rem;
                line-height: 1.7;
                color: rgba(247, 239, 230, 0.78);
            }

            .vinyl-badge {
                width: 220px;
                height: 220px;
                margin: 0 auto;
                border-radius: 50%;
                background:
                    radial-gradient(circle at center, #1d1717 0 12%, #f0a86a 12.5% 15%, #171013 15.5% 29%, #241619 30% 43%, #19141b 44% 57%, #2d191b 58% 70%, #120d10 71% 100%);
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 24px 70px rgba(0, 0, 0, 0.35);
            }

            .panel {
                border-radius: 24px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                background: rgba(20, 18, 24, 0.8);
                padding: 1.1rem 1.2rem;
                box-shadow: 0 18px 50px rgba(0, 0, 0, 0.2);
            }

            .panel-title {
                margin: 0;
                font-size: 0.95rem;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                color: #ffbf90;
            }

            .panel-copy {
                margin: 0.5rem 0 0;
                color: rgba(247, 239, 230, 0.75);
                line-height: 1.6;
            }

            .result-card {
                border-radius: 22px;
                padding: 1.2rem;
                min-height: 215px;
                background: linear-gradient(160deg, rgba(43, 23, 23, 0.96), rgba(20, 21, 31, 0.92));
                border: 1px solid rgba(255, 255, 255, 0.08);
                box-shadow: 0 18px 48px rgba(0, 0, 0, 0.28);
            }

            .result-rank {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 2rem;
                height: 2rem;
                border-radius: 999px;
                background: linear-gradient(135deg, #ff9a5a, #ff6f61);
                color: #190f0f;
                font-weight: 800;
            }

            .song-title {
                margin: 0.9rem 0 0.1rem;
                font-size: 1.35rem;
                font-weight: 700;
                color: #fff7ef;
            }

            .song-artist {
                margin: 0;
                color: rgba(247, 239, 230, 0.72);
                font-size: 0.98rem;
            }

            .song-meta {
                margin-top: 0.9rem;
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
            }

            .song-pill {
                display: inline-block;
                padding: 0.3rem 0.65rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 700;
                color: #ffdcc1;
                background: rgba(255, 122, 89, 0.14);
                border: 1px solid rgba(255, 160, 122, 0.16);
            }

            .song-copy {
                margin-top: 1rem;
                color: rgba(247, 239, 230, 0.76);
                line-height: 1.55;
                font-size: 0.94rem;
            }

            .narrative-box {
                border-radius: 24px;
                padding: 1.4rem;
                background: linear-gradient(160deg, rgba(255, 244, 230, 0.92), rgba(255, 234, 214, 0.86));
                color: #2f1d16;
                border: 1px solid rgba(255, 255, 255, 0.35);
            }

            .metric-strip {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 1rem;
                margin-top: 1rem;
            }

            .metric-card {
                border-radius: 18px;
                padding: 1rem;
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.08);
            }

            .metric-label {
                margin: 0;
                color: rgba(247, 239, 230, 0.7);
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-size: 0.7rem;
            }

            .metric-value {
                margin: 0.4rem 0 0;
                font-size: 1.4rem;
                font-weight: 800;
                color: #fff7ef;
            }

            .stTextArea textarea {
                background: rgba(20, 18, 24, 0.92);
                color: #fff5eb;
                border-radius: 18px;
                border: 1px solid rgba(255, 255, 255, 0.1);
                min-height: 135px;
            }

            .stButton > button {
                width: 100%;
                border: none;
                border-radius: 999px;
                padding: 0.85rem 1.1rem;
                font-weight: 800;
                background: linear-gradient(135deg, #ff9a5a, #ff6f61);
                color: #1c1111;
                box-shadow: 0 12px 30px rgba(255, 111, 97, 0.28);
            }

            @media (max-width: 900px) {
                .hero-title {
                    font-size: 2.6rem;
                }

                .metric-strip {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def _load_system() -> Tuple[List[Dict], RAGEngine]:
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "..", "data", "songs.csv")
    persist_dir = os.path.join(base_dir, "..", "chroma_db")

    songs = load_songs(csv_path)
    rag = RAGEngine(persist_dir=persist_dir)
    rag.index_songs(songs)
    return songs, rag


def _estimate_similarity(distance: float) -> int:
    similarity = max(0.0, 1.0 - distance)
    return int(round(similarity * 100))


def _render_hero(song_count: int) -> None:
    left_col, right_col = st.columns([1.45, 0.75], gap="large")

    with left_col:
        st.markdown(
            f"""
            <section class="hero-shell">
                <span class="eyebrow">Music Discovery Studio</span>
                <h1 class="hero-title">Describe the vibe.<br/>Get a shortlist that fits.</h1>
                <p class="hero-copy">
                    PulseDeck turns plain-language requests into semantic music search.
                    Ask for late-night study tracks, road-trip anthems, or something warm and analog.
                    The engine searches {song_count} songs and explains why the closest matches fit.
                </p>
                <div class="metric-strip">
                    <div class="metric-card">
                        <p class="metric-label">Catalog</p>
                        <p class="metric-value">{song_count} songs</p>
                    </div>
                    <div class="metric-card">
                        <p class="metric-label">Search Mode</p>
                        <p class="metric-value">Semantic RAG</p>
                    </div>
                    <div class="metric-card">
                        <p class="metric-label">Output</p>
                        <p class="metric-value">Ranked picks</p>
                    </div>
                </div>
            </section>
            """,
            unsafe_allow_html=True,
        )

    with right_col:
        st.markdown("<div class=\"vinyl-badge\"></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="panel" style="margin-top: 1.2rem;">
                <p class="panel-title">Prompt ideas</p>
                <p class="panel-copy">
                    Try: "airy songs for a sunrise commute", "brooding guitar tracks with momentum",
                    or "soft focus beats for studying after midnight".
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_result_card(song: Dict, rank: int) -> None:
    similarity = _estimate_similarity(song.get("distance", 1.0))
    st.markdown(
        f"""
        <article class="result-card">
            <span class="result-rank">{rank}</span>
            <h3 class="song-title">{song.get("title", "Unknown title")}</h3>
            <p class="song-artist">{song.get("artist", "Unknown artist")}</p>
            <div class="song-meta">
                <span class="song-pill">{song.get("genre", "Unknown genre")}</span>
                <span class="song-pill">{song.get("mood", "Unknown mood")}</span>
                <span class="song-pill">Energy {song.get("energy", "-")}</span>
                <span class="song-pill">Match {similarity}%</span>
            </div>
            <p class="song-copy">{song.get("document", "No description available.")}</p>
        </article>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_styles()
    songs, rag = _load_system()
    _render_hero(len(songs))

    st.write("")
    query_col, info_col = st.columns([1.3, 0.7], gap="large")

    default_query = "I want warm, mellow songs for studying late at night"

    with query_col:
        st.markdown(
            """
            <div class="panel">
                <p class="panel-title">Describe the song you want</p>
                <p class="panel-copy">
                    Mention mood, tempo, energy, occasion, or anything else you want the system to pick up.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        query = st.text_area(
            "Your request",
            value=default_query,
            label_visibility="collapsed",
            placeholder="Example: cinematic instrumentals with a slow build and a reflective mood",
        )
        search_clicked = st.button("Find recommendations")

    with info_col:
        st.markdown(
            """
            <div class="panel">
                <p class="panel-title">How this works</p>
                <p class="panel-copy">
                    The app embeds song descriptions, retrieves the closest matches for your request,
                    then writes a short recommendation summary grounded in the retrieved songs.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not search_clicked:
        return

    cleaned_query = query.strip()
    if not cleaned_query:
        st.warning("Enter a music description before searching.")
        return

    with st.spinner("Searching the catalog and composing recommendations..."):
        results = rag.search(cleaned_query, k=5)
        narrative = generate_recommendation(cleaned_query, results)

    if not results:
        st.error("No matching songs were found for that description.")
        return

    st.write("")
    st.markdown(
        f"""
        <div class="narrative-box">
            <h3 style="margin-top: 0;">Why these songs fit</h3>
            <p style="white-space: pre-line; margin-bottom: 0; line-height: 1.7;">{narrative}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.subheader("Top matches")
    result_columns = st.columns(len(results))
    for index, song in enumerate(results):
        with result_columns[index]:
            _render_result_card(song, index + 1)


if __name__ == "__main__":
    main()