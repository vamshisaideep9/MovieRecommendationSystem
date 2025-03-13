from django.urls import path
from recommendations.views import MovieRecommendationView, ContextualRecommendationView
from recommendations.views import CollaborativeRecommendView

urlpatterns = [
    path('recommend/<int:movie_id>/', MovieRecommendationView.as_view(), name='movie-recommendation'),
     path('contextual_recommend/<int:movie_id>/', ContextualRecommendationView.as_view(), name='contextual-movie-recommendation'),
     path("recommend/collaborative/<int:movie_id>/", CollaborativeRecommendView.as_view(), name="collaborative_recommend")
]
