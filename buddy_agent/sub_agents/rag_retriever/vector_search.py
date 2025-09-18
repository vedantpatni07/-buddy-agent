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

"""Vector search implementation for the RAG Retriever Sub-Agent.

This module handles document chunking, embedding generation, and vector similarity search.
Uses Google Vertex AI embeddings instead of local sentence-transformers to avoid dependency issues.
"""

import logging
import uuid
import os
from typing import List, Dict, Any, Optional
import chromadb
from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)

# Try to import Vertex AI, fallback to simple text similarity if not available
try:
    from google.cloud import aiplatform
    from vertexai.preview.language_models import TextEmbeddingModel
    VERTEX_AI_AVAILABLE = True
    logger.info("✅ Vertex AI available for embeddings")
except ImportError:
    VERTEX_AI_AVAILABLE = False
    logger.warning("⚠️ Vertex AI not available, using simple text similarity")


class VectorSearchEngine:
    """Vector search engine for document retrieval using Google Vertex AI embeddings."""
    
    def __init__(self, collection_name: str = "buddy_agent_documents", project_id: str = None):
        """Initialize the vector search engine.
        
        Args:
            collection_name: Name of the ChromaDB collection
            project_id: Google Cloud project ID for Vertex AI
        """
        self.collection_name = collection_name
        self.project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize ChromaDB and embedding model."""
        try:
            # Initialize ChromaDB client
            self.chroma_client = chromadb.Client()
            logger.info("✅ ChromaDB client initialized")
            
            # Initialize embedding model
            if VERTEX_AI_AVAILABLE and self.project_id:
                try:
                    # Initialize Vertex AI
                    aiplatform.init(project=self.project_id, location="us-central1")
                    self.embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@001")
                    logger.info("✅ Vertex AI Text Embedding model loaded")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize Vertex AI: {e}")
                    self.embedding_model = None
            else:
                logger.info("ℹ️ Using simple text similarity (no embeddings)")
                self.embedding_model = None
            
            # Create or get collection
            try:
                self.collection = self.chroma_client.get_collection(self.collection_name)
                logger.info(f"✅ Retrieved existing collection: {self.collection_name}")
            except ValueError:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Buddy Agent document collection"}
                )
                logger.info(f"✅ Created new collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"❌ Error initializing vector search engine: {str(e)}")
            raise
    
    def chunk_document(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Chunk document text into overlapping segments.
        
        Args:
            text: Document text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of document chunks with metadata
        """
        try:
            logger.info(f"📄 Chunking document: {len(text)} characters")
            
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
            
            logger.info(f"✅ Created {len(chunks)} chunks from document")
            return chunks
            
        except Exception as e:
            logger.error(f"❌ Error chunking document: {str(e)}")
            return []
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            logger.info(f"🔢 Generating embeddings for {len(texts)} texts")
            
            if self.embedding_model and VERTEX_AI_AVAILABLE:
                # Use Vertex AI embeddings
                embeddings = self.embedding_model.get_embeddings(texts)
                embeddings_list = [emb.values for emb in embeddings]
                logger.info(f"✅ Generated {len(embeddings_list)} Vertex AI embeddings (dimension: {len(embeddings_list[0])})")
                return embeddings_list
            else:
                # Fallback: Use simple text similarity (no real embeddings)
                # Create dummy embeddings for ChromaDB compatibility
                dummy_dimension = 384  # Standard dimension
                embeddings_list = []
                for text in texts:
                    # Create a simple hash-based "embedding" for basic similarity
                    import hashlib
                    hash_obj = hashlib.md5(text.encode())
                    hash_bytes = hash_obj.digest()
                    # Convert to float vector
                    embedding = [float(b) / 255.0 for b in hash_bytes[:dummy_dimension]]
                    # Pad or truncate to desired dimension
                    while len(embedding) < dummy_dimension:
                        embedding.append(0.0)
                    embedding = embedding[:dummy_dimension]
                    embeddings_list.append(embedding)
                
                logger.info(f"✅ Generated {len(embeddings_list)} simple hash-based embeddings (dimension: {dummy_dimension})")
                return embeddings_list
            
        except Exception as e:
            logger.error(f"❌ Error generating embeddings: {str(e)}")
            return []
    
    def add_document(self, document_id: str, text: str, metadata: Dict[str, Any] = None) -> bool:
        """Add a document to the vector database.
        
        Args:
            document_id: Unique identifier for the document
            text: Document text content
            metadata: Additional document metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"📚 Adding document to vector database: {document_id}")
            
            # Chunk the document
            chunks = self.chunk_document(text)
            if not chunks:
                logger.warning("⚠️ No chunks created from document")
                return False
            
            # Prepare data for ChromaDB
            chunk_texts = [chunk["text"] for chunk in chunks]
            chunk_ids = [f"{document_id}_{chunk['id']}" for chunk in chunks]
            chunk_metadata = []
            
            for chunk in chunks:
                chunk_meta = {
                    "document_id": document_id,
                    "chunk_index": chunk["chunk_index"],
                    "start_pos": chunk["start_pos"],
                    "end_pos": chunk["end_pos"],
                    "chunk_size": chunk["chunk_size"]
                }
                if metadata:
                    chunk_meta.update(metadata)
                chunk_metadata.append(chunk_meta)
            
            # Generate embeddings
            embeddings = self.generate_embeddings(chunk_texts)
            if not embeddings:
                logger.error("❌ Failed to generate embeddings")
                return False
            
            # Add to ChromaDB collection
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=chunk_metadata
            )
            
            logger.info(f"✅ Successfully added document {document_id} with {len(chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding document {document_id}: {str(e)}")
            return False
    
    def search_similar(self, query: str, n_results: int = 5, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Search for similar document chunks.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar document chunks with scores
        """
        try:
            logger.info(f"🔍 Searching for similar content: '{query[:50]}...'")
            
            if self.embedding_model and VERTEX_AI_AVAILABLE:
                # Use vector similarity search
                query_embedding = self.embedding_model.get_embeddings([query])
                query_embedding_list = [emb.values for emb in query_embedding]
                
                # Search in ChromaDB
                results = self.collection.query(
                    query_embeddings=query_embedding_list,
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
                
                # Process results
                similar_chunks = []
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    # Convert distance to similarity score (1 - distance)
                    similarity_score = 1 - distance
                    
                    if similarity_score >= threshold:
                        similar_chunks.append({
                            "document_id": metadata["document_id"],
                            "chunk_text": doc,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "metadata": metadata,
                            "rank": i + 1
                        })
            else:
                # Fallback: Simple text-based search
                logger.info("ℹ️ Using simple text-based search (no embeddings)")
                similar_chunks = self._simple_text_search(query, n_results, threshold)
            
            logger.info(f"✅ Found {len(similar_chunks)} similar chunks (threshold: {threshold})")
            return similar_chunks
            
        except Exception as e:
            logger.error(f"❌ Error searching similar content: {str(e)}")
            return []
    
    def _simple_text_search(self, query: str, n_results: int = 5, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """Simple text-based search as fallback when embeddings are not available.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar document chunks with scores
        """
        try:
            # Get all documents from collection
            all_docs = self.collection.get(include=["documents", "metadatas"])
            
            if not all_docs or not all_docs["documents"]:
                return []
            
            # Simple keyword-based scoring
            query_words = set(query.lower().split())
            scored_chunks = []
            
            for i, (doc, metadata) in enumerate(zip(all_docs["documents"], all_docs["metadatas"])):
                doc_words = set(doc.lower().split())
                
                # Calculate simple Jaccard similarity
                intersection = len(query_words.intersection(doc_words))
                union = len(query_words.union(doc_words))
                similarity_score = intersection / union if union > 0 else 0
                
                if similarity_score >= threshold:
                    scored_chunks.append({
                        "document_id": metadata["document_id"],
                        "chunk_text": doc,
                        "similarity_score": similarity_score,
                        "distance": 1 - similarity_score,
                        "metadata": metadata,
                        "rank": len(scored_chunks) + 1
                    })
            
            # Sort by similarity score and return top results
            scored_chunks.sort(key=lambda x: x["similarity_score"], reverse=True)
            return scored_chunks[:n_results]
            
        except Exception as e:
            logger.error(f"❌ Error in simple text search: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "embedding_model": "Vertex AI Text Embedding" if (self.embedding_model and VERTEX_AI_AVAILABLE) else "Simple Text Similarity",
                "embedding_dimension": 768 if (self.embedding_model and VERTEX_AI_AVAILABLE) else 384,
                "vertex_ai_available": VERTEX_AI_AVAILABLE
            }
        except Exception as e:
            logger.error(f"❌ Error getting collection stats: {str(e)}")
            return {}


# Global vector search engine instance
_vector_engine = None


def get_vector_engine() -> VectorSearchEngine:
    """Get the global vector search engine instance."""
    global _vector_engine
    if _vector_engine is None:
        _vector_engine = VectorSearchEngine()
    return _vector_engine
