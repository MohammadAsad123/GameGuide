"""
GameGuide — Similarity Engine Module (Module 9)
==============================================
This module provides content-based game recommendation using TF-IDF
vectorization of weighted metadata (genres, tags, categories, descriptions),
cosine similarity (linear_kernel), popularity blending, explainability tags,
and franchise deduplication.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import time
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

try:
    from src import config
    from src.info_engine import get_game_card, format_list
    from src.search_engine import deduplicate_franchise, get_name_stem
except ImportError:
    import config
    from info_engine import get_game_card, format_list
    from search_engine import deduplicate_franchise, get_name_stem


def build_and_save_tfidf(
    df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    max_features: int = 50000,
    min_df: int = 3
) -> Tuple[TfidfVectorizer, Any]:
    """
    Fit TF-IDF on the weighted metadata soup and serialize (tfidf, X) to disk.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with 'soup' column.
    save_path : str or Path, optional
        Path to models/tfidf_matrix.joblib.
        
    Returns:
    --------
    Tuple[TfidfVectorizer, scipy.sparse.csr_matrix] : Fitted vectorizer and matrix X.
    """
    if save_path is None:
        base_dir = Path(__file__).resolve().parents[1]
        save_path = base_dir / "models" / "tfidf_matrix.joblib"
        
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=min_df
    )
    
    soup_text = df["soup"].fillna("").astype(str)
    X = tfidf.fit_transform(soup_text)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((tfidf, X), save_path)
    return tfidf, X


def load_tfidf_matrix(
    model_path: Optional[Union[str, Path]] = None
) -> Tuple[TfidfVectorizer, Any]:
    """
    Load precomputed TF-IDF vectorizer and sparse feature matrix from disk.
    """
    if model_path is None:
        base_dir = Path(__file__).resolve().parents[1]
        model_path = base_dir / "models" / "tfidf_matrix.joblib"
    return joblib.load(model_path)


def why_similar(src_row: Union[pd.Series, dict], other_row: Union[pd.Series, dict]) -> str:
    """
    Generate an explainability summary detailing shared genres and tags.
    
    Parameters:
    -----------
    src_row : pd.Series or dict
        Source game metadata.
    other_row : pd.Series or dict
        Candidate similar game metadata.
        
    Returns:
    --------
    str : Formatted explainability explanation.
    """
    def to_set(val):
        if isinstance(val, (list, tuple, np.ndarray, set)):
            return set(str(x).strip() for x in val if str(x).strip())
        elif isinstance(val, str) and val.strip():
            return set(x.strip() for x in val.split(",") if x.strip())
        return set()

    src_g = to_set(src_row.get("genres"))
    oth_g = to_set(other_row.get("genres"))
    shared_g = sorted(list(src_g & oth_g))

    src_t = to_set(src_row.get("tags"))
    oth_t = to_set(other_row.get("tags"))
    shared_t = sorted(list(src_t & oth_t))

    g_str = ", ".join(shared_g) if shared_g else "-"
    t_str = ", ".join(shared_t[:5]) if shared_t else "-"
    
    return f"Shared genres: {g_str} | Shared tags: {t_str}"


def find_similar_games(
    game_idx: int,
    df: pd.DataFrame,
    tfidf: Optional[TfidfVectorizer] = None,
    X: Optional[Any] = None,
    top_n: int = config.TOP_N_DEFAULT,
    pop_weight: float = 0.20,
    min_reviews: int = config.MIN_REVIEWS
) -> pd.DataFrame:
    """
    Query-time content-based similarity calculation for a given game index.
    
    Parameters:
    -----------
    game_idx : int
        Row index in df for the source game.
    df : pd.DataFrame
        Catalog DataFrame.
    tfidf : TfidfVectorizer, optional
        Fitted TF-IDF vectorizer.
    X : sparse matrix, optional
        Precomputed TF-IDF sparse matrix.
    top_n : int
        Number of top similar games to return.
    pop_weight : float
        Weight assigned to popularity blending (1 - pop_weight for cosine similarity).
    min_reviews : int
        Review count floor to filter out obscure noise.
        
    Returns:
    --------
    pd.DataFrame : Top N similar games with 'sim', 'final', and 'why' columns.
    """
    if X is None:
        _, X = load_tfidf_matrix()

    if game_idx not in df.index:
        return pd.DataFrame()

    # 1. Compute cosine similarity row slice (TF-IDF rows are L2-normalized)
    sims = linear_kernel(X[game_idx], X).ravel()
    sims[game_idx] = -1.0  # Exclude source game itself

    # 2. Over-fetch top pool to allow post-filtering
    fetch_count = min(len(df), max(100, top_n * 20))
    top_indices = np.argsort(sims)[::-1][:fetch_count]

    pool_df = df.iloc[top_indices].assign(sim=sims[top_indices]).copy()

    # 3. Filter minimum review threshold
    pool_df = pool_df[pool_df["num_reviews_total"].fillna(0) >= min_reviews].copy()

    if pool_df.empty:
        return pool_df

    # 4. Popularity blending
    p_series = pool_df["popularity_score"].fillna(0)
    p_range = p_series.max() - p_series.min()
    p_norm = (p_series - p_series.min()) / (p_range + 1e-9)

    pool_df["final"] = (1.0 - pop_weight) * pool_df["sim"] + pop_weight * p_norm

    # 5. Drop direct same-franchise clones (keep distinct titles/studios)
    src_row = df.loc[game_idx]
    src_stem = get_name_stem(src_row["name"])
    
    # Exclude exact franchise name stem
    is_diff_franchise = pool_df["name"].apply(lambda n: get_name_stem(n) != src_stem)
    pool_filtered = pool_df[is_diff_franchise].copy()
    if pool_filtered.empty:
        pool_filtered = pool_df

    # Deduplicate candidate franchise multiples
    ranked_pool = pool_filtered.sort_values("final", ascending=False)
    deduped = deduplicate_franchise(ranked_pool)

    final_results = deduped.head(top_n).copy()

    # 6. Attach explainability tags
    final_results["why"] = [why_similar(src_row, r) for _, r in final_results.iterrows()]

    return final_results


def format_similarity_response(
    game_idx: int,
    similar_df: pd.DataFrame,
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Format similar game results into the standardized GameGuide chatbot response payload.
    """
    if game_idx not in df.index or similar_df.empty:
        return {
            "type": "similar_games",
            "text": "I couldn't find close recommendations for that game in the catalogue.",
            "games": [],
            "suggestions": ["Recommend top RPGs", "Show free PC games", "Best story games"]
        }

    src_name = df.loc[game_idx, "name"]
    header_text = f"If you enjoyed **{src_name}**, here are {len(similar_df)} similar titles you might like:"

    cards = []
    for _, row in similar_df.iterrows():
        card = get_game_card(row)
        card["why"] = row.get("why", "")
        cards.append(card)

    top_rec = cards[0]["name"] if cards else "this game"
    suggestions = [
        f"Tell me about {top_rec}",
        f"Who developed {top_rec}?",
        f"Is {top_rec} on Mac?"
    ]

    return {
        "type": "similar_games",
        "text": header_text,
        "games": cards,
        "suggestions": suggestions
    }
