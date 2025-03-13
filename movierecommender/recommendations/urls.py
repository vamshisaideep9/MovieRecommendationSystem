from django.urls import path
from recommendations.views import MovieRecommendationView, ContextualRecommendationView

urlpatterns = [
    path('recommend/<int:movie_id>/', MovieRecommendationView.as_view(), name='movie-recommendation'),
     path('contextual_recommend/<int:movie_id>/', ContextualRecommendationView.as_view(), name='contextual-movie-recommendation'),
]
