"""
Persistent Memory Layer - FAISS vector store + JSON persistence
Stores script history, character metadata, and image references
"""
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

from src.schema import MemoryEntry

class MemoryStore:
    def __init__(self, storage_dir: str = "outputs/memory"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.entries_file = self.storage_dir / "memory_entries.json"
        self.index_file = self.storage_dir / "faiss_index.bin"
        
        self.entries: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
        self.faiss_index: Optional[Any] = None
        
        self._load_from_disk()

    def _load_from_disk(self):
        """Load memory from JSON and FAISS index."""
        if self.entries_file.exists():
            with open(self.entries_file, "r") as f:
                data = json.load(f)
                self.entries = data.get("entries", [])
                self.embeddings = data.get("embeddings", [])
        
        if faiss and self.index_file.exists():
            try:
                self.faiss_index = faiss.read_index(str(self.index_file))
            except Exception as e:
                print(f"Warning: Could not load FAISS index: {e}")
                self.faiss_index = None

    def _save_to_disk(self):
        """Persist memory to JSON and FAISS index."""
        # Save entries and embeddings as JSON
        with open(self.entries_file, "w") as f:
            json.dump({
                "entries": self.entries,
                "embeddings": self.embeddings,
                "saved_at": datetime.now().isoformat()
            }, f, indent=2)
        
        # Save FAISS index if available
        if faiss and self.faiss_index and self.embeddings:
            try:
                faiss.write_index(self.faiss_index, str(self.index_file))
            except Exception as e:
                print(f"Warning: Could not save FAISS index: {e}")

    def commit(self, entry_type: str, content: Dict[str, Any]) -> str:
        """
        Store a new memory entry.
        Returns entry ID.
        """
        entry_id = f"{entry_type}_{len(self.entries)}_{datetime.now().timestamp()}"
        
        # Create simple embedding from content (word frequency hash)
        embedding = self._create_embedding(json.dumps(content))
        
        entry = {
            "id": entry_id,
            "entry_type": entry_type,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "embedding": embedding
        }
        
        self.entries.append(entry)
        self.embeddings.append(embedding)
        
        # Update FAISS index if available
        if faiss and embedding:
            self._update_faiss_index()
        
        self._save_to_disk()
        return entry_id

    def query(self, entry_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Query memory entries by type."""
        results = self.entries
        if entry_type:
            results = [e for e in results if e["entry_type"] == entry_type]
        return results[-limit:]

    def query_by_similarity(self, query_embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Query memory by vector similarity (requires FAISS)."""
        if not faiss or not self.faiss_index or not query_embedding:
            return []
        
        try:
            query_vec = np.array([query_embedding], dtype=np.float32)
            distances, indices = self.faiss_index.search(query_vec, limit)
            results = [self.entries[i] for i in indices[0] if i < len(self.entries)]
            return results
        except Exception as e:
            print(f"Warning: FAISS search failed: {e}")
            return []

    def get_entry(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific memory entry by ID."""
        for entry in self.entries:
            if entry["id"] == entry_id:
                return entry
        return None

    def _create_embedding(self, text: str) -> List[float]:
        """Create simple embedding from text (word frequency hash)."""
        # For simplicity, create a fixed-size vector based on content hash
        words = text.lower().split()
        embedding = [0.0] * 128  # Fixed 128-dim embedding
        
        for i, word in enumerate(words[:128]):
            hash_val = hash(word) % 128
            embedding[hash_val] += 1.0 / (i + 1)  # Inverse position weighting
        
        # Normalize
        norm = sum(e**2 for e in embedding)**0.5
        if norm > 0:
            embedding = [e / norm for e in embedding]
        
        return embedding

    def _update_faiss_index(self):
        """Rebuild FAISS index from current embeddings."""
        if not faiss or not self.embeddings:
            return
        
        try:
            embeddings_array = np.array(self.embeddings, dtype=np.float32)
            dimension = embeddings_array.shape[1] if embeddings_array.ndim > 1 else 128
            
            self.faiss_index = faiss.IndexFlatL2(dimension)
            if embeddings_array.ndim == 1:
                embeddings_array = embeddings_array.reshape(1, -1)
            self.faiss_index.add(embeddings_array)
        except Exception as e:
            print(f"Warning: Could not update FAISS index: {e}")

    def clear(self):
        """Clear all memory entries."""
        self.entries = []
        self.embeddings = []
        self.faiss_index = None
        self._save_to_disk()

    def export_summary(self, filepath: str):
        """Export memory summary as JSON."""
        summary = {
            "total_entries": len(self.entries),
            "by_type": {},
            "entries": self.entries
        }
        
        for entry in self.entries:
            entry_type = entry["entry_type"]
            summary["by_type"][entry_type] = summary["by_type"].get(entry_type, 0) + 1
        
        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)
