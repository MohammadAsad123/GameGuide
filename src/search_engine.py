"""
GameGuide — Search and Recommendation Engine Module (Module 8)
==============================================================
This module provides multi-attribute filtering, Bayesian-weighted ranking,
franchise deduplication, automatic filter relaxation, and structured
response generation for game search and recommendation queries.
"""

from typing import Dict, Any, List, Optional, Tuple
import re
import numpy as np
import pandas as pd

try:
    from src import config
    from src.info_engine import get_game_card
except ImportError:
    import config
    from info_engine import get_game_card


def filter_games(
    df: pd.DataFrame,
    entities: Dict[str, Any],
    min_reviews: int = config.MIN_REVIEWS
) -> pd.DataFrame:
    """
    Apply active entity filters to the game catalog.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned and engineered game dataset (games_features.parquet).
    entities : dict
        Structured entity dictionary containing any of:
        'genres', 'tags', 'category', 'platform', 'max_price',
        'year_after', 'year_before', 'min_rating'.
    min_reviews : int
        Minimum review threshold floor.
        
    Returns:
    --------
    pd.DataFrame : Filtered candidate games.
    """
    mask = df["num_reviews_total"].fillna(0) >= min_reviews
    
    # 1. Genres filter (OR match across requested genres)
    genres = entities.get("genres") or []
    if genres:
        req_genres = [g.lower().strip().replace(" ", "_") for g in genres if str(g).strip()]
        if req_genres:
            if "genres_str" in df.columns:
                mask &= df["genres_str"].apply(lambda s: any(g in str(s) for g in req_genres))
            elif "genres" in df.columns:
                mask &= df["genres"].apply(lambda glist: any(g in [str(x).lower().replace(" ", "_") for x in glist] for g in req_genres))

    # 2. Tags filter
    tags = entities.get("tags") or []
    if tags:
        req_tags = [t.lower().strip().replace(" ", "_") for t in tags if str(t).strip()]
        if req_tags:
            if "tags_str" in df.columns:
                mask &= df["tags_str"].apply(lambda s: any(t in str(s) for t in req_tags))
            elif "tags" in df.columns:
                mask &= df["tags"].apply(lambda tlist: any(t in [str(x).lower().replace(" ", "_") for x in tlist] for t in req_tags))

    # 3. Category filter (singleplayer, multiplayer, co-op)
    cat = entities.get("category")
    if cat:
        c_lower = str(cat).lower()
        if "single" in c_lower and "is_singleplayer" in df.columns:
            mask &= df["is_singleplayer"].astype(bool)
        elif "multi" in c_lower and "is_multiplayer" in df.columns:
            mask &= df["is_multiplayer"].astype(bool)
        elif "co" in c_lower and "is_coop" in df.columns:
            mask &= df["is_coop"].astype(bool)

    # 4. Platform filter (windows, mac, linux)
    platform = entities.get("platform")
    if platform:
        p_clean = str(platform).lower().strip()
        if p_clean in ("pc", "win"):
            p_clean = "windows"
        elif p_clean in ("macos", "osx"):
            p_clean = "mac"
            
        if p_clean in ("windows", "mac", "linux") and p_clean in df.columns:
            mask &= df[p_clean].astype(bool)

    # 5. Price filter (INR)
    max_price = entities.get("max_price")
    if max_price is not None:
        try:
            p_val = float(max_price)
            if p_val == 0:
                mask &= (df["is_free"].astype(bool) | (df["price_inr"] == 0))
            else:
                mask &= (df["price_inr"].fillna(0) <= p_val)
        except (ValueError, TypeError):
            pass

    # 6. Year filters
    year_after = entities.get("year_after")
    if year_after is not None:
        try:
            mask &= (df["release_year"].fillna(0) > int(year_after))
        except (ValueError, TypeError):
            pass

    year_before = entities.get("year_before")
    if year_before is not None:
        try:
            mask &= (df["release_year"].fillna(9999) < int(year_before))
        except (ValueError, TypeError):
            pass

    # 7. Minimum Rating filter (positive review percentage)
    min_rating = entities.get("min_rating")
    if min_rating is not None:
        try:
            mask &= (df["pct_pos_total"].fillna(0) >= float(min_rating))
        except (ValueError, TypeError):
            pass

    return df[mask].copy()


def rank_candidates(
    candidates: pd.DataFrame,
    entities: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Rank candidate games using a weighted composite score of Bayesian rating,
    popularity, and recency.
    
    Parameters:
    -----------
    candidates : pd.DataFrame
        Filtered candidate rows.
    entities : dict
        User query entities (used for dynamic weight adjustment).
    weights : dict, optional
        Base ranking weights (default from config.RANK_WEIGHTS).
        
    Returns:
    --------
    pd.DataFrame : Ranked candidates with 'score' column in descending order.
    """
    if candidates.empty:
        return candidates

    w = dict(weights or config.RANK_WEIGHTS)
    
    # If the user explicitly stressed quality / rating, boost rating weight
    if entities.get("min_rating") is not None:
        w["rating"] = min(0.80, w.get("rating", 0.5) + 0.15)
        w["popularity"] = max(0.10, w.get("popularity", 0.3) - 0.15)

    # Min-max normalization helper
    def norm_series(s: pd.Series) -> pd.Series:
        s_clean = s.fillna(s.min() if not pd.isna(s.min()) else 0)
        rng = s_clean.max() - s_clean.min()
        return (s_clean - s_clean.min()) / (rng + 1e-9)

    # Compute normalized composite signals
    r_norm = norm_series(candidates["rating_score"] if "rating_score" in candidates.columns else candidates["pct_pos_total"])
    p_norm = norm_series(candidates["popularity_score"] if "popularity_score" in candidates.columns else candidates["num_reviews_total"])
    
    if "recency" in candidates.columns:
        y_norm = norm_series(candidates["recency"])
    else:
        y_norm = norm_series(candidates["release_year"])

    score = (
        w.get("rating", 0.5) * r_norm
        + w.get("popularity", 0.3) * p_norm
        + w.get("recency", 0.2) * y_norm
    )

    return candidates.assign(score=score).sort_values("score", ascending=False)


def get_name_stem(name: str) -> str:
    """Extract base franchise name stem for deduplication."""
    s = str(name).strip()
    # Remove subtitles after colon, dash, or parenthesis
    s = re.sub(r"[:\-–—\(].*", "", s)
    # Remove edition keywords and symbols
    s = re.sub(r"\b(edition|remastered|goty|deluxe|bundle|hd|vr|legacy|collection|director's cut)\b", "", s, flags=re.I)
    s = re.sub(r"[^a-zA-Z0-9\s]", "", s).strip().lower()
    return s if s else str(name).lower()


def deduplicate_franchise(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Remove near-duplicate editions, bundles, and sequels from the same franchise,
    keeping only the top-ranked title.
    """
    if candidates.empty or len(candidates) <= 1:
        return candidates
        
    stems = candidates["name"].apply(get_name_stem)
    unique_indices = stems.drop_duplicates().index
    return candidates.loc[unique_indices]


def search_and_recommend(
    df: pd.DataFrame,
    entities: Dict[str, Any],
    top_n: int = config.TOP_N_DEFAULT
) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Execute search, ranking, franchise deduplication, and filter relaxation if empty.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Game dataset.
    entities : dict
        Extracted search criteria.
    top_n : int
        Number of top results to return.
        
    Returns:
    --------
    Tuple[pd.DataFrame, Optional[str]] : (Top N ranked games DataFrame, relaxation message)
    """
    e = dict(entities)
    cands = filter_games(df, e)
    relaxation_msg = None

    # Step 10.4: Progressive Filter Relaxation Order (rating -> tags -> price -> year)
    if cands.empty:
        if e.get("min_rating") is not None:
            e["min_rating"] = None
            relaxation_msg = "Relaxed rating requirement to find matching games."
            cands = filter_games(df, e)

    if cands.empty:
        if e.get("tags"):
            e["tags"] = []
            relaxation_msg = "Broadened tags filter to find matching games."
            cands = filter_games(df, e)

    if cands.empty:
        if e.get("max_price") is not None:
            e["max_price"] = None
            relaxation_msg = "Relaxed price constraint to find matching games."
            cands = filter_games(df, e)

    if cands.empty:
        if e.get("year_after") is not None or e.get("year_before") is not None:
            e["year_after"] = None
            e["year_before"] = None
            relaxation_msg = "Relaxed release year filter to find matching games."
            cands = filter_games(df, e)

    if cands.empty:
        # Final fallback: return top rated games in catalog
        cands = df[df["num_reviews_total"].fillna(0) >= config.MIN_REVIEWS].copy()
        relaxation_msg = "No exact matches found. Here are popular top-rated recommendations:"

    # Rank and deduplicate
    ranked = rank_candidates(cands, entities=e)
    deduped = deduplicate_franchise(ranked)
    
    requested_n = int(entities.get("top_n") or top_n or config.TOP_N_DEFAULT)
    return deduped.head(requested_n), relaxation_msg


def format_search_response(
    results_df: pd.DataFrame,
    entities: Dict[str, Any],
    relaxation_msg: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format search and recommendation results into standardized GameGuide payload.
    """
    if results_df.empty:
        return {
            "type": "search_recommend",
            "text": "I couldn't find any games matching your specific criteria.",
            "games": [],
            "suggestions": ["Recommend popular RPG games", "Show free PC games", "Best story games"]
        }

    # Build descriptive response heading
    desc_parts = []
    if entities.get("category"):
        desc_parts.append(str(entities["category"]))
    if entities.get("tags"):
        desc_parts.append(", ".join(entities["tags"]))
    if entities.get("genres"):
        desc_parts.append(", ".join(entities["genres"]))
    if entities.get("platform"):
        desc_parts.append(f"for {entities['platform'].title()}")
    if entities.get("max_price") is not None:
        if entities["max_price"] == 0:
            desc_parts.append("that are Free")
        else:
            desc_parts.append(f"under ₹{entities['max_price']}")
    if entities.get("min_rating") is not None:
        desc_parts.append("with high ratings")

    summary_criteria = " ".join(desc_parts).strip()
    if not summary_criteria:
        summary_criteria = "top-rated"

    header_text = f"Here are the top recommended **{summary_criteria}** games:"
    if relaxation_msg:
        header_text = f"_{relaxation_msg}_\n\n{header_text}"

    # Generate game cards
    cards = [get_game_card(row) for _, row in results_df.iterrows()]

    # Dynamic suggestions based on top result
    top_name = cards[0]["name"] if cards else "Elden Ring"
    suggestions = [
        f"Tell me about {top_name}",
        f"What games are similar to {top_name}?",
        "Show cheaper alternatives"
    ]

    return {
        "type": "search_recommend",
        "text": header_text,
        "games": cards,
        "suggestions": suggestions
    }
