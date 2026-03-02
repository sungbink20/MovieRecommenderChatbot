🎬 Mello, the Moviebot

A Hybrid Rule-Based + LLM Movie Recommendation Chatbot

Mello, the moviebot (internally named “Having A Sungderful Time”) is a conversational movie recommendation system built for Stanford CS124.

The chatbot collects user opinions on movies, analyzes sentiment, and generates personalized recommendations using item-item collaborative filtering. It also supports an optional LLM-powered personality mode for richer, character-driven interactions.

🚀 Features
🎥 Movie Preference Extraction

Detects movie titles in quotation marks (e.g., "Titanic")

Extracts sentiment (positive / negative / neutral)

Handles negation (e.g., "did not like", "never enjoyed")

Tracks number of unique movie preferences

🤝 Collaborative Filtering

Uses item-item cosine similarity

Binarizes ratings matrix (> 2.5 = positive, ≤ 2.5 = negative)

Recommends unseen movies based on similarity-weighted scores

Excludes movies the user already rated

🧠 Optional LLM Mode

Persona-based conversation: Mello, the moviebot

Emotion detection using structured JSON extraction

Foreign movie title translation support

Controlled conversational flow via system prompting

🧮 Recommendation Algorithm

The recommendation system uses:

Binarized ratings matrix

Cosine similarity

Item-item collaborative filtering

No mean-centering

No normalization of final scores

Score calculation:

score(movie_i) = Σ similarity(movie_i, movie_j) × user_rating(movie_j)

Top-K unseen movies are returned as recommendations.

🛠 Tech Stack

Python

NumPy

Regular Expressions

Porter Stemmer

Pydantic (structured LLM outputs)

Optional LLM integration

🧩 Conversation Flow

User provides opinions on movies (must be in quotes).

Chatbot extracts:

Movie titles

Sentiment

After 5 unique preferences:

Chatbot asks if the user wants recommendations.

Provides recommendations iteratively until user declines.
