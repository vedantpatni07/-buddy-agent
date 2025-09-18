# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ultra-lightweight search implementation for the RAG Retriever Sub-Agent.

This module provides a memory-efficient, crash-safe document search using only Python standard library.
Designed to work on systems with limited memory resources.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class LightweightSearchEngine:
    """Ultra-lightweight search engine using only Python standard library."""
    
    def __init__(self, collection_name: str = "lightweight_documents"):
        """Initialize the lightweight search engine.
        
        Args:
            collection_name: Name of the document collection
        """
        self.collection_name = collection_name
        self.documents = {}  # document_id -> document data
        self.chunks = {}  # chunk_id -> chunk data
        self.inverted_index = defaultdict(set)  # word -> set of chunk_ids
        self.logger = logging.getLogger(__name__)
        
        # Disable disk operations to prevent crashes
        self.disk_operations_enabled = False
        
        self.logger.info(f"✅ Lightweight Search Engine initialized: {collection_name}")
    
    def chunk_document(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, Any]]:
        """Chunk document text into small, overlapping segments.
        
        Args:
            text: Document text to chunk
            chunk_size: Maximum characters per chunk (smaller for memory efficiency)
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of document chunks with metadata
        """
        try:
            self.logger.info(f"📄 Chunking document: {len(text)} characters")
            
            chunks = []
            start = 0
            chunk_id = 0
            
            while start < len(text):
                # Calculate end position
                end = min(start + chunk_size, len(text))
                
                # Extract chunk text
                chunk_text = text[start:end].strip()
                
                if chunk_text:  # Only add non-empty chunks
                    chunks.append({
                        "id": f"chunk_{chunk_id}",
                        "text": chunk_text,
                        "start_pos": start,
                        "end_pos": end,
                        "chunk_size": len(chunk_text),
                        "chunk_index": chunk_id
                    })
                    chunk_id += 1
                
                # Move start position with overlap
                start = end - overlap
                if start >= len(text):
                    break
                
                # Safety check to prevent infinite loops
                if chunk_id > 1000:  # Max 1000 chunks per document
                    self.logger.warning("⚠️ Reached maximum chunk limit (1000)")
                    break
            
            self.logger.info(f"✅ Created {len(chunks)} chunks from document")
            return chunks
            
        except Exception as e:
            self.logger.error(f"❌ Error chunking document: {str(e)}")
            return []
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for indexing.
        
        Args:
            text: Input text
            
        Returns:
            List of processed words
        """
        # Convert to lowercase and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _build_inverted_index(self, chunk_id: str, text: str):
        """Build inverted index for a chunk.
        
        Args:
            chunk_id: Unique identifier for the chunk
            text: Chunk text content
        """
        words = self._preprocess_text(text)
        for word in words:
            self.inverted_index[word].add(chunk_id)
    
    def add_document(self, document_id: str, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Add a document to the search engine.
        
        Args:
            document_id: Unique identifier for the document
            text: Document text content
            metadata: Additional document metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info(f"📚 Adding document to search engine: {document_id}")
            
            # Limit document size to prevent memory issues
            if len(text) > 100000:  # Max 100KB per document
                self.logger.warning(f"⚠️ Document too large ({len(text)} chars), truncating to 100KB")
                text = text[:100000]
            
            # Chunk the document
            chunks = self.chunk_document(text)
            if not chunks:
                self.logger.warning("⚠️ No chunks created from document")
                return False
            
            # Store document metadata (minimal)
            self.documents[document_id] = {
                "text": text,
                "metadata": metadata or {},
                "chunk_count": len(chunks),
                "total_chars": len(text)
            }
            
            # Process each chunk
            for chunk in chunks:
                chunk_id = f"{document_id}_{chunk['id']}"
                
                # Store chunk data (minimal)
                self.chunks[chunk_id] = {
                    "document_id": document_id,
                    "text": chunk["text"],
                    "chunk_index": chunk["chunk_index"],
                    "start_pos": chunk["start_pos"],
                    "end_pos": chunk["end_pos"],
                    "chunk_size": chunk["chunk_size"],
                    "metadata": metadata or {}
                }
                
                # Build inverted index
                self._build_inverted_index(chunk_id, chunk["text"])
            
            # Skip disk operations to prevent crashes
            if self.disk_operations_enabled:
                self._save_to_disk()
            
            self.logger.info(f"✅ Successfully added document {document_id} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error adding document {document_id}: {str(e)}")
            return False
    
    def search_similar(self, query: str, n_results: int = 3, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Search for similar document chunks.
        
        Args:
            query: Search query text
            n_results: Number of results to return (limited for memory efficiency)
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar document chunks with scores
        """
        try:
            self.logger.info(f"🔍 Searching for similar content: '{query[:50]}...'")
            
            # Limit results to prevent memory issues
            n_results = min(n_results, 5)
            
            # Preprocess query
            query_words = set(self._preprocess_text(query))
            if not query_words:
                return []
            
            # Calculate scores for each chunk
            chunk_scores = {}
            
            for chunk_id, chunk_data in self.chunks.items():
                chunk_text = chunk_data["text"]
                chunk_words = set(self._preprocess_text(chunk_text))
                
                # Calculate Jaccard similarity
                intersection = len(query_words.intersection(chunk_words))
                union = len(query_words.union(chunk_words))
                similarity_score = intersection / union if union > 0 else 0
                
                # Also check for exact phrase matches
                phrase_bonus = 0
                query_lower = query.lower()
                chunk_lower = chunk_text.lower()
                
                if query_lower in chunk_lower:
                    phrase_bonus = 0.2  # Bonus for exact phrase match
                
                # Combined score
                final_score = similarity_score + phrase_bonus
                
                if final_score >= threshold:
                    chunk_scores[chunk_id] = {
                        "similarity_score": final_score,
                        "jaccard_score": similarity_score,
                        "phrase_bonus": phrase_bonus
                    }
            
            # Sort by score and return top results
            sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1]["similarity_score"], reverse=True)
            
            similar_chunks = []
            for i, (chunk_id, scores) in enumerate(sorted_chunks[:n_results]):
                chunk_data = self.chunks[chunk_id]
                similar_chunks.append({
                    "document_id": chunk_data["document_id"],
                    "chunk_text": chunk_data["text"],
                    "similarity_score": scores["similarity_score"],
                    "jaccard_score": scores["jaccard_score"],
                    "phrase_bonus": scores["phrase_bonus"],
                    "metadata": chunk_data["metadata"],
                    "rank": i + 1,
                    "chunk_index": chunk_data["chunk_index"]
                })
            
            self.logger.info(f"✅ Found {len(similar_chunks)} similar chunks (threshold: {threshold})")
            return similar_chunks
            
        except Exception as e:
            self.logger.error(f"❌ Error searching similar content: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            total_chunks = len(self.chunks)
            total_documents = len(self.documents)
            total_words = len(self.inverted_index)
            
            return {
                "total_chunks": total_chunks,
                "total_documents": total_documents,
                "total_words": total_words,
                "collection_name": self.collection_name,
                "search_engine": "Lightweight Text Similarity",
                "memory_optimized": True
            }
        except Exception as e:
            self.logger.error(f"❌ Error getting collection stats: {str(e)}")
            return {}
    
    def clear_collection(self):
        """Clear all data from memory to free up space."""
        try:
            self.documents.clear()
            self.chunks.clear()
            self.inverted_index.clear()
            self.logger.info("🧹 Collection cleared from memory")
        except Exception as e:
            self.logger.error(f"❌ Error clearing collection: {str(e)}")


# Global lightweight search engine instance
_lightweight_engine = None


def get_lightweight_engine() -> LightweightSearchEngine:
    """Get the global lightweight search engine instance."""
    global _lightweight_engine
    if _lightweight_engine is None:
        _lightweight_engine = LightweightSearchEngine()
    return _lightweight_engine

