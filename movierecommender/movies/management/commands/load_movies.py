from django.core.management.base import BaseCommand
from movies.services import load_cleaned_movie_data
from movies.models import Movie


class Command(BaseCommand):
    help = "Load cleaned movie data into PostgreSQL"

    def handle(self, *args, **kwargs):
        movies = load_cleaned_movie_data('c:/Users/vamsh/OneDrive/Desktop/movierecommender/MovieRecommendationSystem/movierecommender/cleaned data/cleaned_movie_dataset.csv')
        Movie.objects.bulk_create(movies, ignore_conflicts=True)
        self.stdout.write(self.style.SUCCESS("cleaned Movie Data Loaded Successfully."))