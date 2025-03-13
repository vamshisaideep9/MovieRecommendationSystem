from rest_framework.response import Response
from rest_framework.views import APIView
from recommendations.services import recommend_similar_movies
from recommendations.services import get_movie_recommendation_with_explanation
from movies.serializers import MovieSerializer
from movies.models import Movie

# Create your views here.

class MovieRecommendationView(APIView):

    """
    API endpoint for getting movie recommendations
    """


    def get(self, request, movie_id):
        recommended_movies = recommend_similar_movies(movie_id)
        serialized_movies = MovieSerializer(recommended_movies, many=True)
        return Response(serialized_movies.data)


class ContextualRecommendationView(APIView):
    """API to get personalized recommendations with explanations."""

    def get(self, request, movie_id):
        recommended_movies, explanation = get_movie_recommendation_with_explanation(movie_id)
        serialized_movies = MovieSerializer(recommended_movies, many=True)

        return Response({
            "recommended_movies": serialized_movies.data,
            "explanation": explanation
        })

