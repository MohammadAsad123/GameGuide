"""
GameGuide — Intent Classification & Hybrid Guardrails Module (Module 6)
======================================================================
This module provides functions to load the trained intent classifier,
predict intent probabilities with confidence thresholding, and apply
rule-based hybrid guardrails for reliable conversational dialogue routing.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib

try:
    from src import config
except ImportError:
    import config


def load_intent_classifier(model_path: Optional[str] = None):
    """
    Load the serialized intent classification pipeline.
    """
    if model_path is None:
        base_dir = Path(__file__).resolve().parents[1]
        model_path = base_dir / "models" / "intent_clf.joblib"
    return joblib.load(model_path)


def classify(
    query: str,
    clf=None,
    threshold: float = config.INTENT_CONF_THRESHOLD
) -> Tuple[str, float]:
    """
    Predict the intent of a preprocessed user query with confidence thresholding.
    
    Parameters:
    -----------
    query : str
        The preprocessed user query string.
    clf : Pipeline, optional
        The fitted scikit-learn pipeline. If None, loads from default path.
    threshold : float
        Minimum confidence threshold. Below this, intent defaults to 'fallback'.
        
    Returns:
    --------
    Tuple[str, float] : (predicted_intent, confidence_score)
    """
    if clf is None:
        clf = load_intent_classifier()
        
    if not query or not str(query).strip():
        return "fallback", 0.0
        
    proba = clf.predict_proba([query])[0]
    k = proba.argmax()
    intent, conf = clf.classes_[k], float(proba[k])
    
    if conf < threshold:
        return "fallback", conf
    return intent, conf


def apply_guardrails(
    intent: str,
    conf: float,
    query: str,
    entities: Optional[Dict[str, Any]] = None
) -> str:
    """
    Apply rule-based safety nets to override or refine ML predictions.
    
    Parameters:
    -----------
    intent : str
        Predicted intent from the ML model.
    conf : float
        Confidence score of the ML prediction.
    query : str
        The preprocessed user query string.
    entities : dict, optional
        Extracted entities dictionary from entity extractor.
        
    Returns:
    --------
    str : The final guarded intent label.
    """
    ent = entities or {}
    has_game = ent.get("game") is not None
    has_attribute = ent.get("attribute") is not None
    q = query.lower()

    if has_game:
        # Rule 1: Similarity keywords with a game detected -> similar_games
        if any(w in q for w in ["similar", "like ", "such as", "same as", "alternatives to", "resemble"]):
            return "similar_games"
            
        # Rule 2: Specific attribute keyword with a game detected -> specific_game_info
        if has_attribute:
            return "specific_game_info"

    # Rule 3: Game-specific intent predicted but NO game entity was extracted and confidence is low
    if not has_game and intent in ("game_information", "specific_game_info", "similar_games"):
        if conf < 0.70:
            return "fallback"

    return intent


def predict_intent(
    query: str,
    entities: Optional[Dict[str, Any]] = None,
    clf=None,
    threshold: float = config.INTENT_CONF_THRESHOLD
) -> Tuple[str, float, str]:
    """
    High-level helper: classifies query and applies hybrid guardrails in one call.
    
    Returns:
    --------
    Tuple[str, float, str] : (raw_intent, confidence, guarded_intent)
    """
    raw_intent, conf = classify(query, clf=clf, threshold=threshold)
    guarded_intent = apply_guardrails(raw_intent, conf, query, entities=entities)
    return raw_intent, conf, guarded_intent
