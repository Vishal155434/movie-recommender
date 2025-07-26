# 📽️ Movie Recommendation System

A content-based movie recommender system using TF-IDF vectorization and cosine similarity built with Python and Streamlit.

---

## 🚀 Demo

*GitHub Repo: https://github.com/Vishal155434/movie-recommender

---

## 📊 Dataset

- Source: [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- Size: ~10,000+ movies
- Files:
  - tmdb_5000_movies.csv
  - tmdb_5000_credits.csv

---

## 🧠 How It Works

This is a *content-based recommendation engine* that:
- Extracts text metadata from each movie (overview, cast, genre, director, keywords).
- Combines them into a single tags field.
- Uses *TF-IDF Vectorization* to numerically represent the tags.
- Measures similarity between movies using *Cosine Similarity*.
- Returns top 5 similar movies for any input title.

---

## 🛠️ Tech Stack

- *Python*
- *Pandas* for data manipulation
- *Scikit-learn* for TF-IDF & similarity
- *Streamlit* for building the web app

---
