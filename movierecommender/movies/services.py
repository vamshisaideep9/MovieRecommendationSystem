import pandas as pd 
from .models import Movie


def load_cleaned_movie_data(csv_path):
    df = pd.read_csv(csv_path)

    movies = []

    for _, row in df.iterrows():
        movie = Movie(
            tmdb_id = row['id'],
            title = row['title'],
            genres = row['genres'],
            keywords = row['keywords'],
            overview = row['overview'],
            popularity = row['popularity'],
            production = row['production'],
            runtime = row['runtime'] if pd.notna(row['runtime']) else None,
            tagline = row['tagline'] if pd.notna(row['tagline']) else None,
            vote_average = row['vote_average'],
            vote_count = row['vote_count'],
            budget = row['budget']
        )
        movies.append(movie)

    return movies