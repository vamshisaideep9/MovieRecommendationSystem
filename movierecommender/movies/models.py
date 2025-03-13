from django.db import models

# Create your models here.


class Movie(models.Model):
    tmdb_id = models.IntegerField(unique=True)
    title = models.CharField(max_length=255)
    genres = models.TextField()
    keywords = models.TextField()
    overview = models.TextField()
    popularity = models.FloatField()
    production = models.TextField()
    runtime = models.FloatField(null=True, blank=True)
    tagline = models.TextField(null=True, blank=True)
    vote_average = models.FloatField()
    vote_count = models.IntegerField()
    budget = models.BigIntegerField()


    def __str__(self):
        return self.title
    
