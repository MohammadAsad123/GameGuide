# GameGuide 🎮 — Gaming Recommendation & Information Chatbot

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Steam%20Games%20Dataset-89%2C550%20Records-000000.svg?logo=steam&logoColor=white)](https://store.steampowered.com/)
[![Macro-F1](https://img.shields.io/badge/Intent%20Macro--F1-98.96%25-brightgreen.svg)]()
[![Latency](https://img.shields.io/badge/Similarity%20Latency-58ms-success.svg)]()

> **GameGuide** is an intelligent conversational discovery assistant that helps gamers find recommendations, explore detailed game overviews, look up specific attributes, and discover titles similar to their favorites. Built on a dataset of **89,550 Steam games**, GameGuide combines natural language processing, machine learning intent classification, Bayesian rating scores, sub-second content-based cosine similarity, and an interactive Streamlit frontend.

---

## 📑 Table of Contents

- [Key Features & Strengths](#-key-features--strengths)
- [System Architecture](#-system-architecture)
- [Supported Conversational Intents](#-supported-conversational-intents)
- [Data Science Pipeline & Features](#-data-science-pipeline--features)
- [Core Engines & Methodologies](#-core-engines--methodologies)
  - [1. Intent Classifier & Hybrid Guardrails](#1-intent-classifier--hybrid-guardrails)
  - [2. Entity Extractor & Pronoun Resolver](#2-entity-extractor--pronoun-resolver)
  - [3. Game Information Engine](#3-game-information-engine)
  - [4. Search & Recommendation Engine](#4-search--recommendation-engine)
  - [5. Content-Based Similarity Engine](#5-content-based-similarity-engine)
- [Streamlit Chatbot UI](#-streamlit-chatbot-ui)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Setup](#-installation--setup)
- [Sample Conversation Walkthrough](#-sample-conversation-walkthrough)
- [Evaluation & Benchmark Results](#-evaluation--benchmark-results)
- [Authors & License](#-authors--license)

---

## 🌟 Key Features & Strengths

1. **Conversational Game Discovery vs. Traditional Filters**: Instead of clicking through dozens of rigid store filters, users can express multi-faceted preferences in natural language (e.g., *"Show me open-world RPGs under ₹1000 for Windows"*).
2. **Explainable Recommendations**: Every similar-game recommendation provides transparent reasoning by displaying shared genre tags, mechanics, and themes.
3. **Multi-Turn Context & Pronoun Resolution**: Supports natural follow-up questions using pronouns (e.g., *"Tell me about Elden Ring"* followed by *"Who developed it?"* and *"What games are like it?"*).
4. **Bayesian Weighted Quality Floor**: Prevents obscure games with single 100% positive reviews from outranking universally acclaimed titles with tens of thousands of reviews.
5. **Sub-Second Query Latency**: Avoids precomputing an intractable $89,550 \times 89,550$ dense matrix by computing single-row linear kernel slices at runtime in under **60 ms**.
6. **Robust Hybrid Safety Guardrails**: Combines statistical ML probabilities with deterministic rule-based safety nets for 100% reliable dialogue routing.
7. **Clean Streamlit Frontend**: Modern chat interface featuring right-aligned user messages, left-aligned bot responses, rich game cards with Steam images, and interactive follow-up chips.

---

## 🏗️ System Architecture

```
                                  +-------------------------------------------------------------+
                                  |                     STREAMLIT CHAT UI                       |
                                  |     (User on Right, Bot on Left, Game Cards, Suggestions)   |
                                  +------------------------------+------------------------------+
                                                                 | User Query
                                                                 v
                                  +-------------------------------------------------------------+
                                  |                    TEXT PREPROCESSING                       |
                                  |      (Lowercase, ₹ Currency Normalization, Noise Removal)   |
                                  +------------------------------+------------------------------+
                                                                 |
                                  +------------------------------+------------------------------+
                                  |                                                             |
                                  v                                                             v
                  +-------------------------------+                             +-------------------------------+
                  |       ENTITY EXTRACTOR        |                             |       INTENT CLASSIFIER       |
                  | Dictionaries, Sliding N-Grams,|                             | TF-IDF (1,2) + LogReg (C=5)   |
                  | RapidFuzz Typo-Tolerant Match |                             | Confidence Threshold >= 0.45  |
                  +---------------+---------------+                             +---------------+---------------+
                                  |                                                             |
                                  v                                                             |
                  +-------------------------------+                                             |
                  |       PRONOUN RESOLVER        |                                             |
                  |  ("it", "this game" -> active)|                                             |
                  +---------------+---------------+                                             |
                                  |                                                             |
                                  +------------------------------+------------------------------+
                                                                 |
                                                                 v
                                  +-------------------------------------------------------------+
                                  |                      HYBRID GUARDRAILS                      |
                                  |         (ML Predictions + Deterministic Rule Safety Nets)   |
                                  +------------------------------+------------------------------+
                                                                 |
                                                                 v
                                  +-------------------------------------------------------------+
                                  |                DIALOGUE ROUTER & DISPATCHER                 |
                                  +---+--------------+--------------+-------------+-----------+-+
                                      |              |              |             |           |
                     +----------------+   +----------+---+   +------+------+  +---+------+    +---------------+
                     | Canned Replies |   | Game Info    |   | Specific    |  | Search & |    | Similarity    |
                     | (Greet/Thanks) |   | Engine       |   | Attribute   |  | Ranking  |    | Engine        |
                     +----------------+   +--------------+   +-------------+  +----------+    +---------------+
                                      \              |              |             |           /
                                       \             |              |             |          /
                                  +-----v------------v--------------v-------------v---------v---+
                                  |                 STRUCTURED RESPONSE GENERATOR               |
                                  |             { type, text, games: [...], suggestions: [...] }|
                                  +------------------------------+------------------------------+
                                                                 |
                                  +------------------------------+------------------------------+
                                  |                              |                              |
                                  v                              v                              v
                  +-------------------------------+ +-------------------------+ +-------------------------------+
                  | CONVERSATION MEMORY (Context) | | STREAMLIT CARD RENDERER | | QUERY LOGGER (query_logs.csv) |
                  +-------------------------------+ +-------------------------+ +-------------------------------+
```

---

## 🎯 Supported Conversational Intents

| Intent | Example User Query | Handling Subsystem | Response Type |
| :--- | :--- | :--- | :--- |
| **`greeting`** | *"Hello GameGuide!"* | Response Generator | Conversational opening with suggestions |
| **`thanks`** | *"Thank you for the help!"* | Response Generator | Friendly acknowledgement |
| **`goodbye`** | *"Bye, see you later!"* | Response Generator | Conversation closing |
| **`game_information`** | *"Tell me about Elden Ring"* | Information Engine | Comprehensive game card + overview |
| **`specific_game_info`** | *"Who developed Baldur's Gate 3?"* | Information Engine | Single-sentence verified attribute lookup |
| **`search_recommend`** | *"Open-world RPG for Windows under ₹1000"* | Search Engine | Filtered & Bayesian-ranked game cards |
| **`similar_games`** | *"What games are similar to Hades?"* | Similarity Engine | Cosine-similar cards + explainability badges |
| **`fallback`** | *"What is the weather today in Paris?"* | Response Generator | Helpful guidance prompt & examples |

---

## 📊 Data Science Pipeline & Features

### 1. Dataset Cleaning & Parsing (`games_features.parquet`)
- Raw records: **89,625 games across 46 columns** (`games_march2025_full.csv`).
- Removed non-game entries (DLCs, soundtracks, playtests, wallpapers, artbooks).
- Parsed stringified JSON structures into native Python arrays (`genres`, `tags`, `categories`, `developers`, `publishers`, `supported_languages`).
- Converted USD prices to Indian Rupees using documented rate ($1\text{ USD} = ₹96.12$).
- Handled missing values, standardized release dates to datetime, and created clean text descriptions.

### 2. Bayesian (Weighted) Rating Score
To prevent games with only 3 reviews (100% positive) from outranking games with 50,000 reviews (95% positive), we compute:
$$WR = \left(\frac{v}{v+m}\right) \cdot R + \left(\frac{m}{v+m}\right) \cdot C$$
Where:
- $v =$ Total review count of the game (`num_reviews_total`).
- $m =$ Minimum review count threshold (70th percentile across the catalog).
- $R =$ Positive review ratio of the game ($\text{positive} / (\text{positive} + \text{negative})$).
- $C =$ Global mean positive review ratio across all games.

### 3. Weighted Text Soup Construction
For content-based similarity, game metadata is assembled into a weighted textual representation:
$$\text{Soup} = 3 \times (\text{Genres}) + 3 \times (\text{Tags}) + 1 \times (\text{Categories}) + 1 \times (\text{Short Description})$$
Multi-word tokens are joined with underscores (e.g., `open_world`, `story_rich`, `dark_fantasy`) so TF-IDF preserves thematic unity.

---

## ⚙️ Core Engines & Methodologies

### 1. Intent Classifier & Hybrid Guardrails
- **Pipeline**: Scikit-Learn `Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)), ('model', LogisticRegression(C=5, max_iter=1000))])`.
- **Preserving Stop Words**: Conversational words (*"about"*, *"who"*, *"when"*, *"like"*, *"under"*) carry critical intent boundaries. Removing them reduced test accuracy from **98.96% to 96.88%**.
- **Confidence Cutoff** (`INTENT_CONF_THRESHOLD = 0.45`): Predictions with maximum probability $< 0.45$ safely fall back to `"fallback"`.
- **Hybrid Guardrails**:
  1. Game detected + similarity keywords (`"similar"`, `"like"`, `"such as"`) $\rightarrow$ `"similar_games"`.
  2. Game detected + attribute keywords (`"who developed"`, `"when released"`) $\rightarrow$ `"specific_game_info"`.
  3. No game detected + game-specific intent with confidence $< 0.70$ $\rightarrow$ `"fallback"`.

### 2. Entity Extractor & Pronoun Resolver
- **Game Lookup**: Combines sliding n-gram windows (6 words down to 1) over a normalized name index with RapidFuzz $W\text{Ratio}$ fuzzy matching ($85\%$ cutoff).
- **Slot Regular Expressions**:
  - Price: `(?:under|below|less than|within)\s*₹?(\d+)` + keywords `free` / `cheap`.
  - Year: `(?:after|since|from)\s*(\d{4})` and `(?:before|until)\s*(\d{4})`.
  - Result Count: `(?:top|show|recommend)\s*(\d+)`.
- **Dictionaries**: Extensive keyword maps for 26 genres, 64 tags, categories (singleplayer, multiplayer, co-op), platforms (Windows, Mac, Linux), and 8 attribute types.
- **Pronoun Resolution**: Automatically maps references like *"it"*, *"this game"*, or *"that one"* to the active game in `context['last_game']`.

### 3. Game Information Engine
Provides verified natural-language answers and structured game cards across **8 core attributes**:
- **`developer`**: *"{game} was developed by {developers}."*
- **`publisher`**: *"{game} was published by {publishers}."*
- **`genre`**: *"{game} belongs to the {genres} genre(s)."*
- **`release_date`**: *"{game} was released on {formatted_date}."*
- **`platform`**: *"Yes/No, {game} is (not) available on {platform}."*
- **`price`**: *"{game} is free to play."* / *"{game} costs ₹{price} on Steam."*
- **`rating`**: *"{pct_pos}% of {reviews} user reviews for {game} are positive."*
- **`languages`**: *"Supported languages for {game} include {languages}."*

### 4. Search & Recommendation Engine
1. **Multi-Attribute Filtering**: Simultaneously filters on genres, tags, category, platform, max price in INR, release year bounds, and minimum rating floor (`MIN_REVIEWS >= 30`).
2. **Composite Multi-Criteria Ranking**:
   $$\text{Score} = w_{\text{rating}} \cdot \text{norm}(\text{rating\_score}) + w_{\text{pop}} \cdot \text{norm}(\text{popularity\_score}) + w_{\text{rec}} \cdot \text{norm}(\text{recency})$$
   - Default Weights: `0.50 Rating`, `0.30 Popularity`, `0.20 Recency`.
   - Quality-Stressed Queries: `0.65 Rating`, `0.15 Popularity`, `0.20 Recency`.
3. **Franchise Deduplication**: Strips edition subtitles (*Remastered*, *VR*, *GOTY*, *Bundle*) to guarantee diversity in top recommendations.
4. **Progressive Filter Relaxation**: When criteria yield 0 matches, systematically relaxes constraints ($\text{Rating} \rightarrow \text{Tags} \rightarrow \text{Price} \rightarrow \text{Year}$) and notifies the user.

### 5. Content-Based Similarity Engine
1. **TF-IDF Matrix**: Sparse matrix ($89,550 \times 50,000$ features, $99.90\%$ sparsity) persisted to [`models/tfidf_matrix.joblib`](file:///d:/GameGuide/models/tfidf_matrix.joblib).
2. **Query-Time Cosine Similarity**: Evaluates row slice dot product `X[idx] @ X.T` via `linear_kernel` in **~58 ms**.
3. **Popularity Blending**: Blends content similarity with log-popularity ($0.80 \cdot \text{sim} + 0.20 \cdot \text{norm}(\text{popularity})$) to balance niche similarity with quality consensus.
4. **Explainability Badges**: Automatically extracts shared genres and mechanics tags (e.g. `💡 Why similar: Shared genres: Action, RPG | Shared tags: Dark Fantasy, Difficult, Action RPG`).

---

## 💻 Streamlit Chatbot UI

The frontend application ([`app.py`](file:///d:/GameGuide/app.py)) is built with Streamlit:
- **Right-Aligned User Messages**: Blue gradient chat bubbles aligned to the right.
- **Left-Aligned Bot Responses**: Dark slate bubbles aligned to the left with assistant avatar (`🎮`).
- **Interactive Game Cards**: Displays header image, release year, price in INR, rating badges, descriptions, and similarity explanations.
- **Follow-Up Suggestions**: Clickable quick-reply suggestion chips underneath bot replies.
- **Sidebar Features**: "Try Asking" example buttons, real-time catalog metrics (`89,550` games, `26` genres), top-N results slider, and chat reset button.

---

## 📁 Project Directory Structure

```
GameGuide/
├── data/
│   ├── raw/
│   │   └── games_march2025_full.csv        # Original Steam dataset (89,625 rows)
│   └── processed/
│       ├── games_clean.parquet             # Cleaned dataset
│       ├── games_features.parquet          # Engineered features + weighted soup (89,550 rows)
│       ├── intent_train.csv                # 1,920 balanced training queries
│       ├── intent_test.csv                 # 96 hand-written disjoint test queries
│       └── query_logs.csv                  # Real-time interaction logging
├── models/
│   ├── intent_clf.joblib                   # Serialized intent classification pipeline
│   ├── tfidf_matrix.joblib                 # Serialized TF-IDF vectorizer + sparse matrix
│   └── intent_confusion_matrix.png         # High-resolution confusion matrix
├── notebooks/
│   ├── 01_data_understanding.ipynb         # Step 1: Inspection & column analysis
│   ├── 02_cleaning.ipynb                   # Step 2: Data cleaning & type parsing
│   ├── 03_name_index.ipynb                 # Step 3: Game name index & fuzzy lookup
│   ├── 04_eda.ipynb                        # Step 4: Exploratory data analysis
│   ├── 05_feature_engineering.ipynb        # Step 5: Bayesian ratings & text soup
│   ├── 06_intent_dataset.ipynb             # Step 6: NLP preprocessing & intent dataset
│   ├── 07_entity_extraction.ipynb          # Step 7: Entity extraction & regex rules
│   ├── 08_intent_classifier.ipynb          # Step 8: Intent model training & evaluation
│   ├── 09_information_engine.ipynb         # Step 9: Game information & attribute engine
│   ├── 10_search_engine.ipynb              # Step 10: Search, ranking & relaxation
│   ├── 11_similarity_engine.ipynb          # Step 11: Content-based similarity engine
│   └── 12_chatbot_integration.ipynb        # Step 12: Unified GameGuideBot testing
├── src/
│   ├── config.py                           # Tunable project constants & thresholds
│   ├── preprocessing.py                    # NLP text normalization
│   ├── name_index.py                       # Exact & fuzzy game name lookup
│   ├── entity_extractor.py                 # Rule-based entity & pronoun extraction
│   ├── intent_classifier.py                # Intent prediction & hybrid guardrails
│   ├── info_engine.py                      # Game cards & attribute lookups
│   ├── search_engine.py                    # Multi-attribute filtering & composite ranking
│   ├── similarity_engine.py                # TF-IDF cosine similarity & explainability
│   ├── response_generator.py               # Canned response templates & fallbacks
│   └── chatbot.py                          # Unified GameGuideBot controller class
├── app.py                                  # Primary Streamlit application
├── requirements.txt                        # Pinned project dependencies
└── README.md                               # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/GameGuide.git
cd GameGuide
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit App
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 💬 Sample Conversation Walkthrough

```text
User:       "Hello!"
GameGuide:  "Hey there! 🎮 Welcome to GameGuide. Ask me to recommend games, look up details for a specific game, or find titles similar to your favorites!"

User:       "Open-world RPG for Windows under ₹1000"
GameGuide:  "Here are the top recommended Open World RPG for Windows under ₹1000 games:"
            [Cards: Terraria, Horizon Zero Dawn, South Park: The Stick of Truth, Supraland, Kingdom Come: Deliverance]

User:       "Tell me about Elden Ring"
GameGuide:  "Here is the game overview for ELDEN RING (2022):"
            [Card: ELDEN RING | Developer: FromSoftware, Inc. | Rating: 92% positive (749,606 reviews) | Price: ₹5,766]

User:       "Who developed it?"
GameGuide:  "ELDEN RING was developed by FromSoftware, Inc.."

User:       "What games are like it?"
GameGuide:  "If you enjoyed ELDEN RING, here are 5 similar titles you might like:"
            [Cards: Nioh: Complete Edition, DARK SOULS II, Codex Lost, DARK SOULS: REMASTERED, Mortal Shell]
            [💡 Why: Shared genres: Action, RPG | Shared tags: Dark Fantasy, Difficult, Action RPG]

User:       "Show highly rated Linux games"
GameGuide:  "Here are the top recommended for Linux with high ratings games:"
            [Cards: Stardew Valley, Tiny Glade, shapez 2, Slay the Spire, RimWorld]

User:       "Thanks, bye!"
GameGuide:  "Bye! Feel free to chat with me anytime you need game recommendations or info! 🕹️"
```

---

## 📈 Evaluation & Benchmark Results

### 1. Intent Classification Model Comparison
Evaluated on the unseen **96-query hand-written test set** (12 non-template queries per intent):

| Model Configuration | 5-Fold CV Macro-F1 | Test Accuracy | Test Macro-F1 | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **TF-IDF (1,1) + Logistic Regression** | 0.9937 | 0.9896 | 0.9896 | Unigram baseline |
| **TF-IDF (1,2) + Logistic Regression (Selected)** | **0.9958** | **0.9896** | **0.9896** | **Top performance & calibrated probabilities** |
| **TF-IDF (1,2) + Linear SVM** | 0.9969 | 0.9896 | 0.9896 | Strong decision boundary |
| **TF-IDF (1,2) + Multinomial Naive Bayes** | 0.9911 | 0.9896 | 0.9896 | Fast baseline |
| **TF-IDF (1,2) + Random Forest** | 0.9844 | 0.9896 | 0.9896 | Non-linear ensemble |
| **TF-IDF (1,2) [Stop Words Removed] + LogReg** | 0.9828 | 0.9688 | 0.9686 | *Ablation proof*: removing stop words degrades accuracy |

### 2. Similarity Engine Benchmark (Precision@5)
Evaluated across 10 diverse reference titles (*Elden Ring*, *Stardew Valley*, *Portal*, *The Witcher 3*, *Cyberpunk 2077*, *Hollow Knight*, *Hades*, *Terraria*, *Rust*, *Baldur's Gate 3*):

| Variant | Features Used | Popularity Blend | Precision@5 | Latency |
| :--- | :--- | :---: | :---: | :---: |
| **Variant A** | Description only | No | 0.800 | ~55 ms |
| **Variant B** | Genres + Tags only | No | 1.000 | ~48 ms |
| **Variant C** | Weighted Soup (3x Genres, 3x Tags, Desc) | No | 0.900 | ~58 ms |
| **Variant D (Selected)** | **Weighted Soup + Popularity** | **Yes (0.8/0.2)** | **0.900** | **~58 ms** |

### 3. Summary of Component Performance Against Targets

| Component | Target Metric | Achieved Result | Status |
| :--- | :--- | :--- | :---: |
| **Intent Classification** | Macro-F1 $\ge 90\%$ | **98.96% Macro-F1** | **PASS** |
| **Similarity Query Latency** | Latency $< 1,000\text{ ms}$ | **58.04 ms** | **PASS** |
| **Search Filter Correctness** | 100% active filter satisfaction | **100% (20/20 query suites)** | **PASS** |
| **Attribute Lookup Accuracy** | 100% ground-truth match | **100% (40/40 checks)** | **PASS** |
| **Conversational Memory** | Multi-turn pronoun resolution | **100% verified** | **PASS** |

---

## 👤 Author & Acknowledgements

- **Developer**: Mohammad Asad
- **Course**: Data Science Lab Mini-Project
- **Dataset Source**: Steam Games Dataset 2025
- **Technology Stack**: Python, Pandas, Scikit-Learn, RapidFuzz, Joblib, Streamlit

---
*Developed with ❤️ for discovering the best gaming experiences on Steam.*
