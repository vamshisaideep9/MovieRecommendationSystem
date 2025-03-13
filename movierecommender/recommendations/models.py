from django.db import models
from movies.models import Movie
# Create your models here.


class MovieEmbedding(models.Model):
    movie = models.OneToOneField(Movie, on_delete=models.CASCADE, related_name="embedding")
    embedding = models.BinaryField()

    class Meta:
        indexes = [
            models.Index(fields=['movie']), #indexing on movie foreign key
        ]

    def __str__(self):
        return f"Embedding for {self.movie.title}"
