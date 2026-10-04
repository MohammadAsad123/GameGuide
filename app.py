"""
GameGuide — Streamlit Frontend Application (Module 11)
======================================================
Interactive conversational interface for game discovery, recommendations,
attribute lookups, and similarity matching.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np

from src import config
from src.chatbot import GameGuideBot

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GameGuide — Gaming Discovery Chatbot",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Right-Aligned User Chat and Left-Aligned Bot Chat
st.markdown("""
<style>
    /* Global clean font and spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Right-align User Chat Messages */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse !important;
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%) !important;
        border: 1px solid #3b82f6 !important;
        border-radius: 18px 18px 4px 18px !important;
        margin-left: 20% !important;
        margin-right: 0px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25) !important;
    }
    
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
        font-size: 0.98rem !important;
        text-align: left !important;
    }

    /* Left-align Bot Assistant Messages */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        flex-direction: row !important;
        background-color: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 18px 18px 18px 4px !important;
        margin-right: 15% !important;
        margin-left: 0px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }
    
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) [data-testid="stMarkdownContainer"] {
        color: #f1f5f9 !important;
        font-size: 0.98rem !important;
    }

    /* Game Card Styling */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #111827 !important;
        border: 1px solid #374151 !important;
        border-radius: 12px !important;
        margin-top: 8px !important;
        margin-bottom: 8px !important;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: #60a5fa !important;
        transform: translateY(-2px);
    }

    /* Suggestion Chips Button Styling */
    button[kind="secondary"] {
        border-radius: 20px !important;
        border: 1px solid #3b82f6 !important;
        color: #93c5fd !important;
        background-color: rgba(59, 130, 246, 0.08) !important;
        transition: all 0.2s ease !important;
    }

    button[kind="secondary"]:hover {
        background-color: #2563eb !important;
        color: #ffffff !important;
        border-color: #2563eb !important;
        transform: scale(1.02) !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. Cached Bot Loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="🚀 Loading GameGuide models & catalog...")
def load_gameguide_bot():
    return GameGuideBot()

bot = load_gameguide_bot()

# -----------------------------------------------------------------------------
# 3. Session State Initialization
# -----------------------------------------------------------------------------
WELCOME_MESSAGE = (
    "Hey! 🎮 I'm **GameGuide**, your gaming recommendation and discovery assistant.\n\n"
    "I can help you **find game recommendations**, **look up game details & attributes**, "
    "or **discover games similar to your favorites**.\n\n"
    "What kind of games are you looking for today?"
)

DEFAULT_EXAMPLES = [
    "Recommend some RPG games",
    "Open-world RPG for Windows under ₹1000",
    "Tell me about Elden Ring",
    "Who developed Elden Ring?",
    "What games are similar to Elden Ring?",
    "Show highly rated Linux games",
    "Recommend farming games like Stardew Valley",
    "Free-to-play shooters on Windows"
]

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": WELCOME_MESSAGE,
        "games": [],
        "suggestions": [
            "Recommend some RPG games",
            "Open-world RPG under ₹1000",
            "Tell me about Elden Ring",
            "Games similar to Hades"
        ]
    }]

if "context" not in st.session_state:
    st.session_state.context = {}

if "pending" not in st.session_state:
    st.session_state.pending = None


# -----------------------------------------------------------------------------
# 4. Rendering Functions
# -----------------------------------------------------------------------------
def render_game_cards(games: list):
    """Render clean, responsive game cards with metadata badges and images."""
    if not games:
        return
        
    for g in games:
        with st.container(border=True):
            col_img, col_info = st.columns([1, 2.8])
            
            with col_img:
                header_url = g.get("header_image")
                if header_url and str(header_url).startswith("http"):
                    try:
                        st.image(header_url, use_container_width=True)
                    except Exception:
                        st.caption("🎮 Game")
                else:
                    st.caption("🎮 Game")

            with col_info:
                name_str = g.get("name", "Unknown Game")
                year_str = f" ({g.get('year')})" if g.get("year") and str(g.get("year")) != "N/A" else ""
                st.markdown(f"### {name_str}{year_str}")
                
                # Metadata subtitle / tags
                meta_parts = []
                if g.get("genres"):
                    meta_parts.append(f"🏷️ **{g['genres']}**")
                if g.get("price"):
                    meta_parts.append(f"💰 **{g['price']}**")
                if g.get("rating"):
                    meta_parts.append(f"⭐ **{g['rating']}**")
                
                if meta_parts:
                    st.markdown(" • ".join(meta_parts))
                
                # Description
                desc = g.get("description")
                if desc:
                    st.write(desc)
                    
                # Explainability badge for similarity recommendations
                if g.get("why"):
                    st.info(f"💡 **Why similar:** {g['why']}")
                    
                # Extra Platforms & Developer info
                extra_meta = []
                if g.get("developers") and g.get("developers") != "Unknown":
                    extra_meta.append(f"**Developer:** {g['developers']}")
                if g.get("platforms"):
                    extra_meta.append(f"**Platforms:** {', '.join(g['platforms'])}")
                    
                if extra_meta:
                    st.caption(" | ".join(extra_meta))


# -----------------------------------------------------------------------------
# 5. Sidebar Setup & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🎮 GameGuide")
    st.caption("AI Gaming Recommendation & Discovery Chatbot")
    st.write("Ask me to **recommend**, **describe**, or **find similar** games!")
    
    st.divider()
    st.subheader("💡 Try Asking")
    for ex in DEFAULT_EXAMPLES:
        if st.button(ex, key=f"btn_sidebar_{ex}", use_container_width=True):
            st.session_state.pending = ex
            st.rerun()

    st.divider()
    st.subheader("📊 Catalog Metrics")
    c1, c2 = st.columns(2)
    c1.metric("Games", f"{len(bot.df):,}")
    c2.metric("Genres", f"{bot.n_genres}")
    
    st.subheader("⚙️ Settings")
    top_n_setting = st.slider("Results Count", min_value=3, max_value=10, value=config.TOP_N_DEFAULT)

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": WELCOME_MESSAGE,
            "games": [],
            "suggestions": [
                "Recommend some RPG games",
                "Open-world RPG under ₹1000",
                "Tell me about Elden Ring",
                "Games similar to Hades"
            ]
        }]
        st.session_state.context = {}
        st.rerun()


# -----------------------------------------------------------------------------
# 6. Main Chat Area
# -----------------------------------------------------------------------------
st.title("🎮 GameGuide: Gaming Recommendation Chatbot")

# 1. Render Chat History (User on Right, Bot on Left)
for msg in st.session_state.messages:
    is_user = (msg["role"] == "user")
    with st.chat_message(msg["role"], avatar="👤" if is_user else "🎮"):
        st.markdown(msg["content"])
        if msg.get("games"):
            render_game_cards(msg["games"])

# 2. Check for button click inputs (pending state)
user_input = None
if st.session_state.pending:
    user_input = st.session_state.pending
    st.session_state.pending = None

# 3. Chat Input Box
chat_input_val = st.chat_input("Ask about games, e.g. 'Recommend an RPG under ₹1000' or 'Tell me about Elden Ring'...")
if chat_input_val:
    user_input = chat_input_val

# 4. Process Query
if user_input:
    # Append and display user message on the right
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "games": [],
        "suggestions": []
    })
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # Generate bot response on the left
    with st.chat_message("assistant", avatar="🎮"):
        with st.spinner("Thinking..."):
            reply = bot.respond(user_input, st.session_state.context)
            st.markdown(reply["text"])
            if reply.get("games"):
                render_game_cards(reply["games"])

    # Save assistant message to state
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply["text"],
        "games": reply.get("games", []),
        "suggestions": reply.get("suggestions", [])
    })
    
    st.rerun()

# 5. Render interactive quick-reply suggestion chips below the latest response
if st.session_state.messages:
    latest_msg = st.session_state.messages[-1]
    if latest_msg["role"] == "assistant" and latest_msg.get("suggestions"):
        st.markdown("###### 💬 Suggested follow-ups:")
        suggs = latest_msg["suggestions"]
        cols = st.columns(min(len(suggs), 4))
        for idx, s_text in enumerate(suggs[:4]):
            with cols[idx]:
                if st.button(f"👉 {s_text}", key=f"sug_btn_{len(st.session_state.messages)}_{idx}", use_container_width=True):
                    st.session_state.pending = s_text
                    st.rerun()
