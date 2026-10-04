"""
GameGuide — Response Templates & Canned Messages Module
=======================================================
Provides diverse conversational responses and follow-up suggestion chips.
"""

import random
from typing import Dict, Any, List


GREETING_TEMPLATES = [
    "Hello! 👋 I'm GameGuide, your gaming recommendation and discovery assistant. What kind of games are you in the mood for today?",
    "Hey there! 🎮 Welcome to GameGuide. Ask me to recommend games, look up details for a specific game, or find titles similar to your favorites!",
    "Greetings! I'm GameGuide. Whether you're looking for RPGs under ₹1000, co-op multiplayer titles, or game details, I'm here to help!"
]

THANKS_TEMPLATES = [
    "You're very welcome! Let me know if you need any more recommendations or details. Happy gaming! 🎮",
    "Glad I could help! Enjoy your gaming session, and feel free to ask about other games anytime! ✨",
    "Anytime! Hope you discover your next favorite game. Let me know if you need anything else! 🚀"
]

GOODBYE_TEMPLATES = [
    "Goodbye! Have a great gaming time, and come back whenever you're searching for new adventures! 👋",
    "See you later! Take care and happy gaming! 🎮✨",
    "Bye! Feel free to chat with me anytime you need game recommendations or info! 🕹️"
]

FALLBACK_TEMPLATES = [
    "I'm specialized in discovering PC and console games from Steam. You can ask me to **recommend games**, **tell you about a game**, or **find similar titles**.",
    "I didn't quite catch that. Try asking me for game recommendations (e.g. *'Open-world RPGs under ₹1000'*) or details about a game (e.g. *'Who developed Elden Ring?'*)."
]

DEFAULT_SUGGESTIONS = [
    "Recommend some RPG games",
    "Open-world RPG under ₹1000",
    "Tell me about Elden Ring",
    "Games similar to Hades"
]


def generate_canned_response(intent: str) -> Dict[str, Any]:
    """
    Return a randomized conversational response payload with suggested follow-ups.
    """
    if intent == "greeting":
        text = random.choice(GREETING_TEMPLATES)
        suggestions = [
            "Recommend some RPG games",
            "Open-world games under ₹1000",
            "Tell me about Elden Ring"
        ]
    elif intent == "thanks":
        text = random.choice(THANKS_TEMPLATES)
        suggestions = [
            "Show games similar to Witcher 3",
            "Highly rated multiplayer games",
            "Goodbye!"
        ]
    elif intent == "goodbye":
        text = random.choice(GOODBYE_TEMPLATES)
        suggestions = [
            "Hello again!",
            "Recommend top indie games"
        ]
    else:
        text = random.choice(FALLBACK_TEMPLATES)
        suggestions = list(DEFAULT_SUGGESTIONS)

    return {
        "type": intent,
        "text": text,
        "games": [],
        "suggestions": suggestions
    }
