"""
GameGuide — Entity Extraction & Pronoun Resolution Module (Module 7)
===================================================================
Rule-based extraction using dictionaries, regex patterns, sliding n-grams,
and fuzzy matching for games, genres, tags, categories, platforms, prices,
years, ratings, and attributes.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import re
import pandas as pd

try:
    from src import config
    from src.preprocessing import preprocess
    from src.name_index import find_game
except ImportError:
    import config
    from preprocessing import preprocess
    from name_index import find_game

# Regular Expressions
PRICE_RE = re.compile(r"(?:under|below|less than|within|<=?|max|around)\s*[₹$]?\s*(\d+)")
YEAR_AFTER_RE = re.compile(r"(?:after|since|from|post)\s*(\d{4})\b")
YEAR_BEFORE_RE = re.compile(r"(?:before|until|prior to|pre)\s*(\d{4})\b")
YEAR_EXACT_RE = re.compile(r"(?:released in|in year|games of)\s*(\d{4})\b")
COUNT_RE = re.compile(r"\b(?:top|show|give me|recommend|suggest|find)\s+(\d{1,2})\b|\b(\d{1,2})\s+(?:games|titles|recommendations)\b")

TRIGGERS = [
    "what games are similar to", "what games are like", "games similar to",
    "games like", "recommend games similar to", "something like",
    "tell me about", "give me details of", "give me info on",
    "who is the developer of", "who developed", "who made", "who created",
    "who is the publisher of", "who published",
    "when was", "what is the release date of", "what year did",
    "what genre is", "what are the genres of", "what type of game is",
    "is", "can i play", "does", "available on", "runs on",
    "how much does", "what is the price of", "what is the cost of",
    "what is the rating of", "how good are the reviews for",
    "what languages does", "does", "support subtitles", "info on", "details of"
]

GENRE_MAP = {
    "action": "Action", "rpg": "RPG", "role playing": "RPG", "role-playing": "RPG", "rpgs": "RPG",
    "adventure": "Adventure", "strategy": "Strategy", "strategic": "Strategy",
    "simulation": "Simulation", "simulator": "Simulation", "indie": "Indie",
    "casual": "Casual", "racing": "Racing", "sports": "Sports", "sport": "Sports",
    "massively multiplayer": "Massively Multiplayer", "mmorpg": "Massively Multiplayer", "mmo": "Massively Multiplayer",
    "free to play": "Free To Play", "early access": "Early Access", "shooter": "Action", "fps": "Action",
    "horror": "Action", "puzzle": "Strategy"
}

TAG_MAP = {
    "open world": "Open World", "open-world": "Open World", "story rich": "Story Rich", "story-rich": "Story Rich",
    "souls-like": "Souls-like", "soulslike": "Souls-like", "souls like": "Souls-like",
    "roguelike": "Rogue-like", "rogue-like": "Rogue-like", "roguelite": "Rogue-lite", "rogue-lite": "Rogue-lite",
    "pixel graphics": "Pixel Graphics", "atmospheric": "Atmospheric", "sci-fi": "Sci-fi", "cyberpunk": "Cyberpunk",
    "dark fantasy": "Dark Fantasy", "first-person": "First-Person", "third-person": "Third-Person",
    "tactical": "Tactical", "turn-based": "Turn-Based", "turn based": "Turn-Based",
    "zombie": "Zombies", "zombies": "Zombies", "post-apocalyptic": "Post-apocalyptic",
    "base building": "Base Building", "deckbuilding": "Deckbuilding", "deck building": "Deckbuilding",
    "metroidvania": "Metroidvania", "hack and slash": "Hack and Slash", "detective": "Detective",
    "mystery": "Mystery", "space": "Space", "relaxing": "Relaxing", "anime": "Anime",
    "psychological horror": "Psychological Horror", "crafting": "Crafting", "stealth": "Stealth",
    "survival": "Survival", "sandbox": "Sandbox", "fighting": "Fighting", "puzzle": "Puzzle",
    "shooter": "Shooter", "fps": "FPS", "vr": "VR"
}

CATEGORY_MAP = {
    "single player": "singleplayer", "single-player": "singleplayer", "singleplayer": "singleplayer", "solo": "singleplayer",
    "multiplayer": "multiplayer", "multi-player": "multiplayer", "online": "multiplayer", "pvp": "multiplayer",
    "co-op": "co-op", "coop": "co-op", "co operative": "co-op", "cooperative": "co-op"
}

PLATFORM_MAP = {
    "windows": "windows", "win": "windows", "pc": "windows",
    "mac": "mac", "macos": "mac", "osx": "mac", "apple": "mac",
    "linux": "linux", "ubuntu": "linux", "steamdeck": "linux", "steam deck": "linux"
}

ATTRIBUTE_MAP = {
    "developer": ["developer", "developers", "developed", "who developed", "who made", "creator", "studio", "made by"],
    "publisher": ["publisher", "publishers", "published", "who published", "publishing company"],
    "genre": ["genre", "genres", "type of game", "what kind of game", "category"],
    "release_date": ["release date", "released", "release year", "when was", "when did", "came out", "launch date"],
    "platform": ["platform", "platforms", "available on", "runs on", "playable on", "support mac", "support linux", "on pc", "on mac", "on windows"],
    "price": ["price", "cost", "how much", "how expensive", "free to play", "rupees", "inr", "costs"],
    "rating": ["rating", "ratings", "reviews", "score", "how good", "positive reviews", "steam rating"],
    "languages": ["language", "languages", "subtitles", "audio", "supported languages", "translation", "english"]
}

STOP_WORDS_COMMON = {
    "what", "game", "games", "like", "who", "when", "show", "find", "tell", "give",
    "which", "how", "does", "can", "about", "for", "and", "the", "with", "under",
    "from", "after", "before", "rpg", "best", "top", "free", "play", "good", "cheap",
    "indie", "pc", "mac", "it", "this", "that", "one", "all", "is", "are", "was", "were",
    "where", "why", "to", "in", "on", "at", "by", "an", "a", "of", "some", "any", "more",
    "titles", "title", "similar", "same", "such", "info", "details", "overview"
}


def extract_game(q: str, name_index: Dict[str, int], names_list: List[str]) -> Tuple[Optional[int], float]:
    """
    Extract game entity via sliding n-grams and fuzzy trigger stripping.
    """
    tokens = q.split()
    # 1. Sliding n-gram window (6 words down to 1)
    for n in range(min(6, len(tokens)), 0, -1):
        for i in range(len(tokens) - n + 1):
            cand = " ".join(tokens[i:i + n])
            if cand in name_index and len(cand) >= 3:
                # Disallow common single stop/command words matching obscure single-word game titles
                if n == 1 and cand.lower() in STOP_WORDS_COMMON:
                    continue
                return name_index[cand], 100.0

    # 2. Strip known trigger phrases and test fuzzy matching on the residual
    stripped = q
    for t in sorted(TRIGGERS, key=len, reverse=True):
        stripped = stripped.replace(t, " ")
    stripped = re.sub(r"\s+", " ", stripped).strip()

    if stripped and len(stripped) >= 3 and stripped.lower() not in STOP_WORDS_COMMON:
        idx, score = find_game(stripped, name_index, names_list, cutoff=85)
        if idx is not None:
            return idx, score

    return None, 0.0


def extract_entities(
    q: str,
    name_index: Dict[str, int],
    names_list: List[str]
) -> Dict[str, Any]:
    """
    Extract all structured entities from a preprocessed query.
    """
    clean_q = preprocess(q)
    game_idx, _ = extract_game(clean_q, name_index, names_list)

    entities: Dict[str, Any] = {
        "game": game_idx,
        "genres": [],
        "tags": [],
        "category": None,
        "platform": None,
        "max_price": None,
        "min_rating": None,
        "year_after": None,
        "year_before": None,
        "attribute": None,
        "top_n": config.TOP_N_DEFAULT
    }

    # 1. Price extraction
    if m := PRICE_RE.search(clean_q):
        entities["max_price"] = int(m.group(1))
    elif "free" in clean_q or "free to play" in clean_q:
        entities["max_price"] = 0
    elif "cheap" in clean_q or "budget" in clean_q:
        entities["max_price"] = 500

    # 2. Year extraction
    if m := YEAR_AFTER_RE.search(clean_q):
        entities["year_after"] = int(m.group(1))
    if m := YEAR_BEFORE_RE.search(clean_q):
        entities["year_before"] = int(m.group(1))
    if m := YEAR_EXACT_RE.search(clean_q):
        yr = int(m.group(1))
        entities["year_after"] = yr - 1
        entities["year_before"] = yr + 1

    # 3. Rating extraction
    if any(w in clean_q for w in ["highly rated", "top rated", "best rated", "good ratings", "great reviews", "critically acclaimed"]):
        entities["min_rating"] = config.HIGH_RATING

    # 4. Result count extraction
    if m := COUNT_RE.search(clean_q):
        cnt_val = m.group(1) or m.group(2)
        if cnt_val:
            entities["top_n"] = max(1, min(20, int(cnt_val)))

    # 5. Genre extraction
    for phrase, canonical in GENRE_MAP.items():
        if re.search(r"\b" + re.escape(phrase) + r"\b", clean_q):
            if canonical not in entities["genres"]:
                entities["genres"].append(canonical)

    # 6. Tag extraction
    for phrase, canonical in TAG_MAP.items():
        if re.search(r"\b" + re.escape(phrase) + r"\b", clean_q):
            if canonical not in entities["tags"]:
                entities["tags"].append(canonical)

    # 7. Category extraction
    for phrase, canonical in CATEGORY_MAP.items():
        if re.search(r"\b" + re.escape(phrase) + r"\b", clean_q):
            entities["category"] = canonical
            break

    # 8. Platform extraction
    for phrase, canonical in PLATFORM_MAP.items():
        if re.search(r"\b" + re.escape(phrase) + r"\b", clean_q):
            entities["platform"] = canonical
            break

    # 9. Attribute extraction
    for attr_name, triggers in ATTRIBUTE_MAP.items():
        if any(re.search(r"\b" + re.escape(t) + r"\b", clean_q) for t in triggers):
            entities["attribute"] = attr_name
            break

    return entities


def resolve_pronouns(
    q: str,
    ent: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Resolve conversational pronoun references ('it', 'this game', 'that one')
    by substituting the last active game from context memory.
    """
    ctx = context or {}
    last_game = ctx.get("last_game")
    
    if ent.get("game") is None and last_game is not None:
        q_lower = q.lower()
        pronoun_triggers = [" it", "it ", "this game", "that game", "the game", "this title", "that one", "about it", "developed it", "for it", "like it", "similar to it"]
        if any(p in q_lower for p in pronoun_triggers) or ent.get("attribute") is not None:
            ent["game"] = last_game
            
    return ent
