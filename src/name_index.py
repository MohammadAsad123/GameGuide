"""
GameGuide — Game Name Index & Fuzzy Lookup Module
=================================================
Fast exact, substring, and typo-tolerant fuzzy game name matching.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
from rapidfuzz import process, fuzz

try:
    from src import config
except ImportError:
    import config


def build_name_index(df: pd.DataFrame) -> Tuple[Dict[str, int], List[str]]:
    """
    Build popularity-prioritized name index dictionary (name_clean -> row index).
    """
    order = df.sort_values("num_reviews_total", ascending=False, na_position="last")
    name_index = {}
    for idx, name in zip(order.index, order["name_clean"]):
        if name and name not in name_index:
            name_index[name] = idx
    names_list = list(name_index.keys())
    return name_index, names_list


def find_game(
    text: str,
    name_index: Dict[str, int],
    names_list: List[str],
    cutoff: int = config.FUZZY_CUTOFF
) -> Tuple[Optional[int], float]:
    """
    Find best matching game index in df by exact match or WRatio fuzzy match.
    """
    t = text.lower().strip()
    if not t:
        return None, 0.0
    if t in name_index:
        return name_index[t], 100.0
    match = process.extractOne(t, names_list, scorer=fuzz.WRatio, score_cutoff=cutoff)
    return (name_index[match[0]], float(match[1])) if match else (None, 0.0)


def suggest(
    text: str,
    name_index: Dict[str, int],
    names_list: List[str],
    df: pd.DataFrame,
    n: int = 3
) -> List[str]:
    """
    Return top N candidate game title suggestions for misspelled queries.
    """
    t = text.lower().strip()
    if not t:
        return []
    matches = process.extract(t, names_list, scorer=fuzz.WRatio, limit=n)
    return [str(df.loc[name_index[m[0]], "name"]) for m in matches] if matches else []
