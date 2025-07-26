import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# -----------------------------
# 1. Load Data
# -----------------------------
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge on title
movies = movies.merge(credits, on='title')

# Select relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

# -----------------------------
# 2. Helper Functions
# -----------------------------
def convert(obj):
    try:
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L
    except:
        return []

def get_top_3_cast(obj):
    try:
        L = []
        for i in ast.literal_eval(obj)[:3]:
            L.append(i['name'])
        return L
    except:
        return []

def get_director(obj):
    try:
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
        return L
    except:
        return []

# -----------------------------
# 3. Apply transformations
# -----------------------------
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(get_top_3_cast)
movies['crew'] = movies['crew'].apply(get_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Combine all text into one 'tags' column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x))
new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# -----------------------------
# 4. Text Vectorization
# -----------------------------
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(new_df['tags']).toarray()

# Cosine similarity
similarity = cosine_similarity(vectors)

# -----------------------------
# 5. Recommend function
# -----------------------------
def recommend(movie):
    movie = movie.lower()
    if movie not in new_df['title'].str.lower().values:
        return "Movie not found."

    index = new_df[new_df['title'].str.lower() == movie].index[0]
    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    result = []
    for i in movies_list:
        result.append(new_df.iloc[i[0]].title)
    return result

# -----------------------------
# 6. Streamlit App
# -----------------------------
st.title("🎬 Movie Recommendation System")
st.markdown("Get similar movie suggestions based on your favorite!")

movie_input = st.text_input("Enter a movie title")

if st.button("Recommend"):
    if not movie_input:
        st.warning("Please enter a movie title.")
    else:
        output = recommend(movie_input)
        if isinstance(output, str):
            st.error(output)
        else:
            st.success("Top 5 Recommended Movies:")
            for i, title in enumerate(output, 1):
                st.write(f"{i}. {title}")