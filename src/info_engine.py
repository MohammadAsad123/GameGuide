"""
GameGuide — Game Information Engine Module (Module 7)
=====================================================
This module provides structured game cards, natural-language attribute
lookups, and robust edge-case handling for game information queries.
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np


def format_list(val: Any) -> str:
    """Helper to convert list or array of strings into a clean comma-separated string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if isinstance(val, (list, tuple, np.ndarray, set)):
        items = [str(x).strip() for x in val if str(x).strip() and str(x).strip().lower() != "nan"]
        return ", ".join(items)
    s = str(val).strip()
    return s if s.lower() != "nan" else ""


def get_game_card(row: Union[pd.Series, dict]) -> Dict[str, Any]:
    """
    Generate a standardized game card dictionary for UI rendering.
    
    Parameters:
    -----------
    row : pd.Series or dict
        A game record from games_features.parquet.
        
    Returns:
    --------
    dict : Standardized game card.
    """
    name = str(row.get("name", "Unknown Game"))
    
    # Release year
    rel_year = row.get("release_year")
    if pd.notna(rel_year) and str(rel_year).strip():
        try:
            year_str = str(int(float(rel_year)))
        except (ValueError, TypeError):
            year_str = "N/A"
    else:
        year_str = "N/A"

    # Developers & Publishers
    devs = format_list(row.get("developers")) or "Unknown"
    pubs = format_list(row.get("publishers")) or "Unknown"
    genres = format_list(row.get("genres")) or "General"
    
    # Price
    is_free = bool(row.get("is_free", False))
    price_inr = row.get("price_inr", 0.0)
    if is_free or (pd.notna(price_inr) and float(price_inr) == 0.0):
        price_str = "Free to Play"
    elif pd.notna(price_inr):
        price_str = f"₹{float(price_inr):,.0f}"
    else:
        price_str = "N/A"

    # Platforms
    platforms = []
    for p in ["windows", "mac", "linux"]:
        if bool(row.get(p, False)):
            platforms.append(p.title())
            
    # Ratings
    pct_pos = row.get("pct_pos_total", np.nan)
    num_rev = row.get("num_reviews_total", 0)
    if pd.notna(pct_pos) and pd.notna(num_rev) and float(num_rev) > 0:
        rating_str = f"{float(pct_pos):.0f}% positive ({int(float(num_rev)):,} reviews)"
    else:
        rating_str = "No reviews yet"

    # Description & Image
    desc = str(row.get("short_description", "") or "").strip()
    if len(desc) > 300:
        desc = desc[:297].rstrip() + "..."
        
    header_img = str(row.get("header_image", "") or "").strip()

    return {
        "name": name,
        "year": year_str,
        "developers": devs,
        "publishers": pubs,
        "genres": genres,
        "price": price_str,
        "platforms": platforms,
        "rating": rating_str,
        "description": desc,
        "header_image": header_img
    }


def get_attribute(row: Union[pd.Series, dict], attribute: str, platform: Optional[str] = None) -> str:
    """
    Retrieve and format a specific game attribute into a natural language sentence.
    
    Parameters:
    -----------
    row : pd.Series or dict
        A game record from games_features.parquet.
    attribute : str
        The requested attribute key ('developer', 'publisher', 'genre',
        'release_date', 'platform', 'price', 'rating', 'languages').
    platform : str, optional
        Specific platform to query ('windows', 'mac', 'linux').
        
    Returns:
    --------
    str : A natural-language answer sentence.
    """
    game_name = str(row.get("name", "This game"))
    attr = str(attribute).lower().strip() if attribute else ""

    if attr in ("developer", "developers"):
        devs = format_list(row.get("developers"))
        if devs:
            return f"{game_name} was developed by {devs}."
        return f"I don't have developer information for {game_name}."

    elif attr in ("publisher", "publishers"):
        pubs = format_list(row.get("publishers"))
        if pubs:
            return f"{game_name} was published by {pubs}."
        return f"I don't have publisher information for {game_name}."

    elif attr in ("genre", "genres"):
        genres = format_list(row.get("genres"))
        if genres:
            return f"{game_name} belongs to the {genres} genre(s)."
        return f"I don't have genre information for {game_name}."

    elif attr in ("release_date", "date", "release_year", "released"):
        rel_date = row.get("release_date")
        if pd.notna(rel_date):
            try:
                date_ts = pd.to_datetime(rel_date)
                date_formatted = date_ts.strftime("%d %B %Y")
                return f"{game_name} was released on {date_formatted}."
            except Exception:
                pass
        rel_year = row.get("release_year")
        if pd.notna(rel_year):
            return f"{game_name} was released in {int(float(rel_year))}."
        return f"I don't have the release date for {game_name}."

    elif attr in ("platform", "platforms"):
        if platform:
            plat_clean = platform.lower().strip()
            if plat_clean in ("pc", "win"):
                plat_clean = "windows"
            elif plat_clean in ("macos", "osx"):
                plat_clean = "mac"
                
            if plat_clean in ("windows", "mac", "linux"):
                is_avail = bool(row.get(plat_clean, False))
                plat_name = "macOS" if plat_clean == "mac" else plat_clean.title()
                if is_avail:
                    return f"Yes, {game_name} is available on {plat_name}."
                else:
                    return f"No, {game_name} is not available on {plat_name}."
                    
        # General platforms list
        supported = [p.title() if p != "mac" else "macOS" for p in ["windows", "mac", "linux"] if bool(row.get(p, False))]
        if supported:
            return f"{game_name} is available on {', '.join(supported)}."
        return f"I don't have platform support details for {game_name}."

    elif attr in ("price", "cost"):
        is_free = bool(row.get("is_free", False))
        price_inr = row.get("price_inr")
        price_usd = row.get("price")
        
        if is_free or (pd.notna(price_inr) and float(price_inr) == 0.0):
            return f"{game_name} is free to play."
        elif pd.notna(price_inr) and float(price_inr) > 0:
            usd_str = f" (${float(price_usd):.2f})" if pd.notna(price_usd) else ""
            return f"{game_name} costs ₹{float(price_inr):,.0f}{usd_str} on Steam."
        return f"I don't have pricing information for {game_name}."

    elif attr in ("rating", "reviews", "score"):
        pct_pos = row.get("pct_pos_total")
        num_rev = row.get("num_reviews_total")
        if pd.notna(pct_pos) and pd.notna(num_rev) and float(num_rev) > 0:
            return f"{float(pct_pos):.0f}% of {int(float(num_rev)):,} user reviews for {game_name} are positive."
        return f"I don't have enough review data for {game_name}."

    elif attr in ("language", "languages", "subtitles"):
        langs = row.get("supported_languages")
        if isinstance(langs, (list, tuple, np.ndarray, set)) and len(langs) > 0:
            lang_list = [str(x) for x in langs[:8]]
            suffix = f" (and {len(langs)-8} more)" if len(langs) > 8 else ""
            return f"Supported languages for {game_name} include {', '.join(lang_list)}{suffix}."
        elif isinstance(langs, str) and langs.strip():
            return f"Supported languages for {game_name} include {langs}."
        return f"I don't have language support information for {game_name}."

    else:
        return f"I don't have information on '{attribute}' for {game_name}."


def check_other_editions(df: pd.DataFrame, game_idx: int) -> List[str]:
    """
    Check if there are other editions, bundles, or variations of the same title.
    """
    if game_idx not in df.index:
        return []
    base_name = str(df.loc[game_idx, "name_clean"]).split(":")[0].strip()
    if len(base_name) < 4:
        return []
    
    matches = df[
        (df["name_clean"].str.startswith(base_name)) & 
        (df.index != game_idx) &
        (df["num_reviews_total"] >= 10)
    ]
    return matches["name"].head(3).tolist()


def handle_game_information(
    game_idx: int,
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Format a complete game information response for the chatbot.
    """
    if game_idx not in df.index:
        return {
            "type": "fallback",
            "text": "I could not find that game in the catalogue.",
            "games": [],
            "suggestions": []
        }
        
    row = df.loc[game_idx]
    card = get_game_card(row)
    name = card["name"]
    
    editions = check_other_editions(df, game_idx)
    edition_note = f" (Note: Other editions include: {', '.join(editions)})" if editions else ""
    
    intro_text = f"Here is the game overview for **{name}** ({card['year']}){edition_note}:"
    
    suggestions = [
        f"Who developed {name}?",
        f"What games are similar to {name}?",
        f"Is {name} on Mac?"
    ]
    
    return {
        "type": "game_information",
        "text": intro_text,
        "games": [card],
        "suggestions": suggestions
    }


def handle_specific_game_info(
    game_idx: int,
    attribute: str,
    platform: Optional[str],
    df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Format a specific attribute lookup response for the chatbot.
    """
    if game_idx not in df.index:
        return {
            "type": "fallback",
            "text": "I could not find that game in the catalogue.",
            "games": [],
            "suggestions": []
        }
        
    row = df.loc[game_idx]
    answer_text = get_attribute(row, attribute, platform=platform)
    card = get_game_card(row)
    name = card["name"]
    
    suggestions = [
        f"Tell me more about {name}",
        f"Games similar to {name}",
        f"What is the price of {name}?"
    ]
    
    return {
        "type": "specific_game_info",
        "text": answer_text,
        "games": [card],
        "suggestions": suggestions
    }
