import os
import torch
import hashlib
import json
from pathlib import Path


class CLIPTextCache:
    def __init__(self, cache_dir="./clip_cache"):
        """Initialize the cache with a directory to store embeddings."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a metadata file to store text-to-filename mappings
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """Load existing metadata or create new if doesn't exist."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)
    
    def _generate_cache_key(self, text_queries):
        """Generate a unique cache key for the text queries."""
        # Sort to ensure same queries in different order get same key
        if isinstance(text_queries, list):
            text_queries = sorted(text_queries)
            text_data = json.dumps(text_queries, sort_keys=True).encode()
        else:
            text_data = str(text_queries).encode()
        return hashlib.md5(text_data).hexdigest()
    
    def get_cached_embeddings(self, text_queries):
        """Retrieve cached embeddings if they exist."""
        cache_key = self._generate_cache_key(text_queries)
        
        if cache_key in self.metadata:
            cache_path = self.cache_dir / f"{cache_key}.pt"
            if cache_path.exists():
                return torch.load(cache_path)
        return None
    
    def cache_embeddings(self, text_queries, embeddings):
        """Cache the computed embeddings."""
        cache_key = self._generate_cache_key(text_queries)
        cache_path = self.cache_dir / f"{cache_key}.pt"
        
        # Save the embeddings
        torch.save(embeddings, cache_path)
        
        # Update metadata
        self.metadata[cache_key] = {
            'queries': text_queries,
            'shape': list(embeddings.shape)
        }
        self._save_metadata()
