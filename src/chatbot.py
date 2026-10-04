"""
GameGuide — Main Chatbot Controller Module (Module 10)
=====================================================
Integrates NLP Preprocessing, Entity Extraction, Pronoun Resolution,
Intent Classification, Hybrid Guardrails, and Specialized Engines into
the unified GameGuideBot class.
"""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import os
import time
import datetime
import csv
import pandas as pd
import joblib

try:
    from src import config
    from src.preprocessing import preprocess
    from src.name_index import build_name_index, find_game, suggest
    from src.entity_extractor import extract_entities, resolve_pronouns
    from src.intent_classifier import load_intent_classifier, classify, apply_guardrails
    from src.info_engine import handle_game_information, handle_specific_game_info
    from src.search_engine import search_and_recommend, format_search_response
    from src.similarity_engine import load_tfidf_matrix, find_similar_games, format_similarity_response
    from src.response_generator import generate_canned_response
except ImportError:
    import config
    from preprocessing import preprocess
    from name_index import build_name_index, find_game, suggest
    from entity_extractor import extract_entities, resolve_pronouns
    from intent_classifier import load_intent_classifier, classify, apply_guardrails
    from info_engine import handle_game_information, handle_specific_game_info
    from search_engine import search_and_recommend, format_search_response
    from similarity_engine import load_tfidf_matrix, find_similar_games, format_similarity_response
    from response_generator import generate_canned_response


class GameGuideBot:
    """
    Unified Conversational Bot Controller for Game Discovery & Information.
    """

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        model_path: Optional[Union[str, Path]] = None,
        tfidf_path: Optional[Union[str, Path]] = None
    ):
        base_dir = Path(__file__).resolve().parents[1]
        self.data_path = Path(data_path) if data_path else base_dir / "data" / "processed" / "games_features.parquet"
        self.model_path = Path(model_path) if model_path else base_dir / "models" / "intent_clf.joblib"
        self.tfidf_path = Path(tfidf_path) if tfidf_path else base_dir / "models" / "tfidf_matrix.joblib"

        print(f"Loading GameGuide catalog from {self.data_path}...")
        self.df = pd.read_parquet(self.data_path)
        
        print("Building name index...")
        self.name_index, self.names_list = build_name_index(self.df)
        
        print(f"Loading intent classifier from {self.model_path}...")
        self.clf = load_intent_classifier(str(self.model_path))
        
        print(f"Loading TF-IDF similarity matrix from {self.tfidf_path}...")
        self.tfidf, self.X = load_tfidf_matrix(str(self.tfidf_path))
        
        self.n_genres = len(set(g for glist in self.df["genres"].dropna() for g in glist if str(g).strip()))
        self.log_file = base_dir / "data" / "processed" / "query_logs.csv"
        
        print(f"GameGuideBot initialized successfully! ({len(self.df):,} games loaded)")

    def log_interaction(
        self,
        raw_query: str,
        clean_query: str,
        intent: str,
        conf: float,
        entities: Dict[str, Any],
        response_type: str
    ):
        """Append user interaction data to CSV query log."""
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            file_exists = self.log_file.exists()
            with open(self.log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp", "raw_query", "clean_query", "intent", "confidence", "entities", "response_type"])
                writer.writerow([
                    datetime.datetime.now().isoformat(),
                    raw_query,
                    clean_query,
                    intent,
                    f"{conf:.4f}",
                    str(entities),
                    response_type
                ])
        except Exception:
            pass  # Logging should never break the chatbot response

    # --- Intent Handlers ---

    def greet(self, ent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return generate_canned_response("greeting")

    def thanks(self, ent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return generate_canned_response("thanks")

    def bye(self, ent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return generate_canned_response("goodbye")

    def fallback(self, ent: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return generate_canned_response("fallback")

    def info(self, ent: Dict[str, Any]) -> Dict[str, Any]:
        game_idx = ent.get("game")
        if game_idx is None:
            return {
                "type": "fallback",
                "text": "Which game would you like to know about? For example, try asking: *'Tell me about Elden Ring'*. 😊",
                "games": [],
                "suggestions": ["Tell me about Elden Ring", "Tell me about Cyberpunk 2077", "Tell me about Hades"]
            }
        return handle_game_information(game_idx, self.df)

    def attribute(self, ent: Dict[str, Any]) -> Dict[str, Any]:
        game_idx = ent.get("game")
        attr = ent.get("attribute")
        plat = ent.get("platform")
        if game_idx is None:
            return {
                "type": "fallback",
                "text": "Which game are you asking about? For example: *'Who developed Baldur's Gate 3?'* or *'Is Stardew Valley on Mac?'*",
                "games": [],
                "suggestions": ["Who developed Elden Ring?", "When was Witcher 3 released?", "Is Hollow Knight on Mac?"]
            }
        if not attr and plat:
            attr = "platform"
        elif not attr:
            return self.info(ent)
            
        return handle_specific_game_info(game_idx, attr, plat, self.df)

    def search(self, ent: Dict[str, Any]) -> Dict[str, Any]:
        results, rel_msg = search_and_recommend(self.df, ent, top_n=ent.get("top_n", config.TOP_N_DEFAULT))
        return format_search_response(results, ent, relaxation_msg=rel_msg)

    def similar(self, ent: Dict[str, Any]) -> Dict[str, Any]:
        game_idx = ent.get("game")
        if game_idx is None:
            return {
                "type": "fallback",
                "text": "Which game would you like to find recommendations similar to? For example: *'Games similar to Elden Ring'* 🎮",
                "games": [],
                "suggestions": ["Games similar to Elden Ring", "Games similar to Stardew Valley", "Games similar to Portal 2"]
            }
        similar_df = find_similar_games(game_idx, self.df, tfidf=self.tfidf, X=self.X, top_n=ent.get("top_n", config.TOP_N_DEFAULT))
        return format_similarity_response(game_idx, similar_df, self.df)

    # --- Main Respond Interface ---

    def respond(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Processes user query through the full NLP and recommendation pipeline.
        
        Parameters:
        -----------
        query : str
            Raw natural-language user query.
        context : dict, optional
            Session conversation state (tracks 'last_game').
            
        Returns:
        --------
        dict : Structured payload {type, text, games, suggestions}.
        """
        ctx = context if context is not None else {}
        
        try:
            # 1. Input sanitization & truncation
            raw_q = str(query or "").strip()
            if len(raw_q) > 300:
                raw_q = raw_q[:300]
                
            q = preprocess(raw_q)
            if not q:
                return {
                    "type": "fallback",
                    "text": "Please type a question about games! You can ask for recommendations, game details, or similar titles.",
                    "games": [],
                    "suggestions": ["Recommend some RPG games", "Tell me about Elden Ring", "Games similar to Hades"]
                }

            # 2. Extract entities
            ent = extract_entities(q, self.name_index, self.names_list)

            # 3. Resolve pronouns using context memory
            ent = resolve_pronouns(q, ent, ctx)

            # 4. Intent classification
            raw_intent, conf = classify(q, self.clf)

            # 5. Apply hybrid guardrails
            intent = apply_guardrails(raw_intent, conf, q, ent)

            # 6. Route to intent handler
            handlers = {
                "greeting": self.greet,
                "thanks": self.thanks,
                "goodbye": self.bye,
                "game_information": self.info,
                "specific_game_info": self.attribute,
                "search_recommend": self.search,
                "similar_games": self.similar,
                "fallback": self.fallback
            }
            handler = handlers.get(intent, self.fallback)
            reply = handler(ent)

            # 7. Update conversation memory only for game-specific contexts
            if intent in ("game_information", "specific_game_info", "similar_games") and ent.get("game") is not None:
                ctx["last_game"] = ent["game"]

            # 8. Log interaction
            self.log_interaction(raw_q, q, intent, conf, ent, reply.get("type", "unknown"))

            return reply

        except Exception as e:
            # Resilient fallback so chatbot never crashes with a raw stack trace
            return {
                "type": "fallback",
                "text": "I encountered an issue processing your request. Please try asking in another way, or ask for game recommendations!",
                "games": [],
                "suggestions": ["Recommend some RPG games", "Tell me about Elden Ring", "Games like Stardew Valley"]
            }
