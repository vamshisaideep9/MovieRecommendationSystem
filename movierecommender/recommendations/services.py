import numpy as np
from django.core.exceptions import ObjectDoesNotExist
from recommendations.ai_engine import get_bert_embedding
from recommendations.models import MovieEmbedding
from movies.models import Movie
from sklearn.metrics.pairwise import cosine_similarity
import pickle


from django.core.cache import cache

CACHE_TIMEOUT = 60 * 60

def generate_and_store_embeddings():
    """
    Generate BERT embeddings for all movies and store them in the database
    """
    movies = Movie.objects.all()
    embeddings_to_update = []
    embeddings_to_create = []

    for movie in movies:
        text_data = f"{movie.genres} {movie.overview} {movie.tagline} {movie.production}"

        # Generate embedding
        embedding_vector = get_bert_embedding(text_data)
        embedding_binary = embedding_vector.tobytes()

        try:
            # Check if embedding already exists
            movie_embedding = MovieEmbedding.objects.get(movie=movie)
            movie_embedding.embedding = embedding_binary
            embeddings_to_update.append(movie_embedding)
        except ObjectDoesNotExist:
            # If embedding doesn't exist, create a new one
            embeddings_to_create.append(MovieEmbedding(movie=movie, embedding=embedding_binary))

    # Bulk update existing embeddings
    if embeddings_to_update:
        MovieEmbedding.objects.bulk_update(embeddings_to_update, ["embedding"])

    # Bulk create new embeddings (this is needed to insert new records)
    if embeddings_to_create:
        MovieEmbedding.objects.bulk_create(embeddings_to_create)




def get_movie_embedding(movie_id):
    """
    Retrieve the stored embedding for a given movie
    """

    try:
        embedding_record = MovieEmbedding.objects.get(movie_id=movie_id)
        return np.frombuffer(embedding_record.embedding, dtype=np.float32)
    except MovieEmbedding.DoesNotExist:
        return None
    



def get_all_embeddings():
    """
    Load all movie Embeddings and their IDs
    """

    movies = MovieEmbedding.objects.select_related('movie').all()
    movie_ids = []
    embeddings = []


    for movie in movies:
        embedding_vector = np.frombuffer(movie.embedding, dtype=np.float32)
        embeddings.append(embedding_vector)
        movie_ids.append(movie.movie.id)

    return np.array(embeddings), movie_ids



def recommend_similar_movies(movie_id, top_n=5):
    """
    Find top-N similar movies based on cosine similarity
    
    """

    #check cache first
    cache_key = f"movie_recommendations_{movie_id}"
    cache_recommendations = cache.get(cache_key)

    if cache_recommendations:
        return Movie.objects.filter(id__in=cache_recommendations)

    target_embedding = get_movie_embedding(movie_id)
    if target_embedding is None:
        return []
    

    embeddings, movie_ids = get_all_embeddings()

    similarities = cosine_similarity([target_embedding], embeddings)[0]

    similar_indices = np.argsort(similarities)[::-1][1:top_n+1] #sort in descending order
    similar_movie_ids = [movie_ids[i] for i in similar_indices]

    cache.set(cache_key, similar_movie_ids, timeout=CACHE_TIMEOUT)


    return Movie.objects.filter(id__in=similar_movie_ids)




##LLM

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from movies.models import Movie
import os
from dotenv import load_dotenv
load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
LANGSMITH_TRACING=True
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"]=os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"]=os.getenv("LANGSMITH_PROJECT")


def get_movie_recommendation_with_explanation(movie_id):
    """
    Get similar movies along with an AI-generated Explanation.
    """

    recommended_movies = recommend_similar_movies(movie_id)
    if not recommended_movies:
        return [], "No similar movies found."
    

    movie = Movie.objects.get(id=movie_id)
    recommend_titles = [m.title for m in recommended_movies]

    #prompt
    prompt_template = PromptTemplate.from_template(
        "I am recommending movies similar to '{title}' based on their genres, keywords, and storyline."
        "The recommended movies are: {recommended_movies}. Explain why these movies are similar."
    )

    prompt = prompt_template.format(title=movie.title, recommended_movies=", ".join(recommend_titles))

    model = Ollama(model="llama3.2")
    response = model.invoke(prompt)

    return recommended_movies, response





## Collaborative Filtering
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from movies.models import Movie


def get_numerical_features():
    movies = Movie.objects.all().values("id", "budget", "popularity", "runtime", "vote_average", "vote_count")
    df = pd.DataFrame(movies)

    scaler = MinMaxScaler()
    df[["budget", "popularity", "runtime", "vote_average", "vote_count"]] = scaler.fit_transform(
        df[["budget", "popularity", "runtime", "vote_average", "vote_count"]]
    )

    return df



def recommend_collaborative(movie_id, top_n=5):

    df = get_numerical_features()

    x = df.drop(columns=['id']).values

    knn = NearestNeighbors(n_neighbors=top_n+1, metric="euclidean")
    knn.fit(x)

    movie_idx = df[df["id"] == movie_id].index[0]

    distances, indices = knn.kneighbors([x[movie_idx]])

    recommended_movie_ids = df.iloc[indices[0][1:]]["id"].tolist()

    return Movie.objects.filter(id__in=recommended_movie_ids)

