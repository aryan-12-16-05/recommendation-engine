import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('../data/movies.csv')

# Convert genres into usable text
movies['genres'] = movies['genres'].str.replace('|', ' ')

# TF-IDF vectorization
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)

def recommend(title, num_recommendations=5):
    idx = movies[movies['title'] == title].index[0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    movie_indices = [i[0] for i in scores]
    return movies['title'].iloc[movie_indices]

if __name__ == "__main__":
    print("Similar Movies:")
    print(recommend("Adventures of Rocky and Bullwinkle, The (2000)"))