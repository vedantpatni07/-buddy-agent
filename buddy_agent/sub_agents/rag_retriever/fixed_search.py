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

"""Fixed search implementation for the RAG Retriever Sub-Agent.

This module provides an improved search algorithm with better similarity matching.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class FixedSearchEngine:
    """Fixed search engine with improved similarity matching."""
    
    def __init__(self, collection_name: str = "fixed_documents"):
        """Initialize the fixed search engine.
        
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
        
        self.logger.info(f"✅ Fixed Search Engine initialized: {collection_name}")
    
    def chunk_document(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, Any]]:
        """Chunk document text into small, overlapping segments."""
        try:
            self.logger.info(f"📄 Chunking document: {len(text)} characters")
            
            chunks = []
            start = 0
            chunk_id = 0
            
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunk_text = text[start:end].strip()
                
                if chunk_text:
                    chunks.append({
                        "id": f"chunk_{chunk_id}",
                        "text": chunk_text,
                        "start_pos": start,
                        "end_pos": end,
                        "chunk_size": len(chunk_text),
                        "chunk_index": chunk_id
                    })
                    chunk_id += 1
                
                start = end - overlap
                if start >= len(text):
                    break
                
                if chunk_id > 1000:
                    self.logger.warning("⚠️ Reached maximum chunk limit (1000)")
                    break
            
            self.logger.info(f"✅ Created {len(chunks)} chunks from document")
            return chunks
            
        except Exception as e:
            self.logger.error(f"❌ Error chunking document: {str(e)}")
            return []
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for indexing."""
        # Convert to lowercase and split into words
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _build_inverted_index(self, chunk_id: str, text: str):
        """Build inverted index for a chunk."""
        words = self._preprocess_text(text)
        for word in words:
            self.inverted_index[word].add(chunk_id)
    
    def add_document(self, document_id: str, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Add a document to the search engine."""
        try:
            self.logger.info(f"📚 Adding document to search engine: {document_id}")
            
            # Limit document size to prevent memory issues
            if len(text) > 100000:
                self.logger.warning(f"⚠️ Document too large ({len(text)} chars), truncating to 100KB")
                text = text[:100000]
            
            # Chunk the document
            chunks = self.chunk_document(text)
            if not chunks:
                self.logger.warning("⚠️ No chunks created from document")
                return False
            
            # Store document metadata
            self.documents[document_id] = {
                "text": text,
                "metadata": metadata or {},
                "chunk_count": len(chunks),
                "total_chars": len(text)
            }
            
            # Process each chunk
            for chunk in chunks:
                chunk_id = f"{document_id}_{chunk['id']}"
                
                # Store chunk data
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
            
            self.logger.info(f"✅ Successfully added document {document_id} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error adding document {document_id}: {str(e)}")
            return False
    
    def search_similar(self, query: str, n_results: int = 3, threshold: float = 0.01) -> List[Dict[str, Any]]:
        """Search for similar document chunks with improved matching."""
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
                jaccard_score = intersection / union if union > 0 else 0
                
                # Calculate word overlap score
                word_overlap = intersection / len(query_words) if len(query_words) > 0 else 0
                
                # Check for exact phrase matches
                phrase_bonus = 0
                query_lower = query.lower()
                chunk_lower = chunk_text.lower()
                
                if query_lower in chunk_lower:
                    phrase_bonus = 0.5  # Higher bonus for exact phrase match
                
                # Check for partial phrase matches
                query_words_list = query_words
                chunk_words_list = list(chunk_words)
                
                partial_match = 0
                for q_word in query_words_list:
                    for c_word in chunk_words_list:
                        if q_word in c_word or c_word in q_word:
                            partial_match += 0.1
                
                # Combined score with multiple factors
                final_score = (jaccard_score * 0.4) + (word_overlap * 0.3) + phrase_bonus + partial_match
                
                if final_score >= threshold:
                    chunk_scores[chunk_id] = {
                        "similarity_score": final_score,
                        "jaccard_score": jaccard_score,
                        "word_overlap": word_overlap,
                        "phrase_bonus": phrase_bonus,
                        "partial_match": partial_match
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
                    "word_overlap": scores["word_overlap"],
                    "phrase_bonus": scores["phrase_bonus"],
                    "partial_match": scores["partial_match"],
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
        """Get statistics about the collection."""
        try:
            total_chunks = len(self.chunks)
            total_documents = len(self.documents)
            total_words = len(self.inverted_index)
            
            return {
                "total_chunks": total_chunks,
                "total_documents": total_documents,
                "total_words": total_words,
                "collection_name": self.collection_name,
                "search_engine": "Fixed Text Similarity",
                "memory_optimized": True
            }
        except Exception as e:
            self.logger.error(f"❌ Error getting collection stats: {str(e)}")
            return {}


# Global fixed search engine instance
_fixed_engine = None


def get_fixed_engine() -> FixedSearchEngine:
    """Get the global fixed search engine instance."""
    global _fixed_engine
    if _fixed_engine is None:
        _fixed_engine = FixedSearchEngine()
    return _fixed_engine
