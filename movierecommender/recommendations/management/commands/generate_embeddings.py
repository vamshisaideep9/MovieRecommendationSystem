from django.core.management.base import BaseCommand
from recommendations.services import generate_and_store_embeddings


class Command(BaseCommand):
    help = "Generate and store BERT embeddings for movies"

    def handle(self, *args, **kwargs):
        generate_and_store_embeddings()
        self.stdout.write(self.style.SUCCESS("Movie Embeddings Generated Succesfully!"))