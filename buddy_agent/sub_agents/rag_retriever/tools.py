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

"""Tools for the RAG Retriever Sub-Agent."""

import logging
from typing import Dict, List, Any
from google.adk.tools import ToolContext
from .vector_search import get_vector_engine

logger = logging.getLogger(__name__)


async def create_corpus(
    processed_documents: List[Dict[str, Any]],
    tool_context: ToolContext = None,
) -> Dict[str, Any]:
    """Create a searchable corpus from processed documents using vector search."""
    try:
        if not processed_documents:
            return {
                "success": False,
                "error": "No processed documents provided",
                "corpus_id": None
            }
        
        logger.info(f"🚀 Creating vector corpus with {len(processed_documents)} documents")
        
        # Get vector search engine
        vector_engine = get_vector_engine()
        
        # Add each document to vector database
        successful_docs = 0
        total_chunks = 0
        
        for doc in processed_documents:
            document_id = doc.get("path", f"doc_{len(processed_documents)}")
            text = doc.get("text", "")
            metadata = {
                "path": doc.get("path", "unknown"),
                "type": doc.get("type", "unknown"),
                "original_metadata": doc.get("metadata", {})
            }
            
            # Add document to vector database
            success = vector_engine.add_document(document_id, text, metadata)
            if success:
                successful_docs += 1
                # Count chunks (approximate)
                chunks = vector_engine.chunk_document(text)
                total_chunks += len(chunks)
        
        # Get collection statistics
        stats = vector_engine.get_collection_stats()
        
        # Store corpus info in session state
        tool_context.state["document_corpus"] = {
            "vector_engine_ready": True,
            "total_documents": successful_docs,
            "total_chunks": total_chunks,
            "collection_name": vector_engine.collection_name
        }
        tool_context.state["rag_retrieval_ready"] = True
        
        logger.info(f"✅ Vector corpus created: {successful_docs} documents, {total_chunks} chunks")
        
        return {
            "success": True,
            "corpus_id": vector_engine.collection_name,
            "total_documents": successful_docs,
            "total_chunks": total_chunks,
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimension": 384,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"❌ Error creating vector corpus: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "corpus_id": None
        }


async def search_documents(
    query: str,
    tool_context: ToolContext = None,
) -> Dict[str, Any]:
    """Search documents for relevant content using vector similarity search."""
    try:
        corpus = tool_context.state.get("document_corpus")
        if not corpus or not corpus.get("vector_engine_ready"):
            return {
                "success": False,
                "error": "No vector corpus found. Please create corpus first.",
                "results": []
            }
        
        logger.info(f"🔍 Vector searching for: '{query[:50]}...'")
        
        # Get vector search engine
        vector_engine = get_vector_engine()
        
        # Search for similar content
        similar_chunks = vector_engine.search_similar(
            query=query,
            n_results=10,
            threshold=0.6
        )
        
        # Process results
        results = []
        for chunk in similar_chunks:
            results.append({
                "document": chunk["document_id"],
                "chunk_text": chunk["chunk_text"][:200] + "..." if len(chunk["chunk_text"]) > 200 else chunk["chunk_text"],
                "similarity_score": chunk["similarity_score"],
                "distance": chunk["distance"],
                "rank": chunk["rank"],
                "metadata": chunk["metadata"]
            })
        
        logger.info(f"✅ Found {len(results)} similar chunks for query: {query}")
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "total_found": len(results),
            "search_type": "vector_similarity",
            "embedding_model": "all-MiniLM-L6-v2"
        }
        
    except Exception as e:
        logger.error(f"❌ Error searching documents: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "results": []
        }


async def get_relevant_sections(
    query: str,
    max_sections: int = 5,
    tool_context: ToolContext = None,
) -> Dict[str, Any]:
    """Get the most relevant document sections for a query using vector search."""
    try:
        corpus = tool_context.state.get("document_corpus")
        if not corpus or not corpus.get("vector_engine_ready"):
            return {
                "success": False,
                "error": "No vector corpus found. Please create corpus first.",
                "sections": []
            }
        
        logger.info(f"🔍 Getting relevant sections for: '{query[:50]}...'")
        
        # Get vector search engine
        vector_engine = get_vector_engine()
        
        # Search for similar content
        similar_chunks = vector_engine.search_similar(
            query=query,
            n_results=max_sections,
            threshold=0.6
        )
        
        # Process results into sections
        relevant_sections = []
        for chunk in similar_chunks:
            relevant_sections.append({
                "document": chunk["document_id"],
                "type": chunk["metadata"].get("type", "unknown"),
                "section_index": chunk["metadata"].get("chunk_index", 0),
                "content": chunk["chunk_text"],
                "relevance_score": chunk["similarity_score"],
                "distance": chunk["distance"],
                "characters": len(chunk["chunk_text"]),
                "rank": chunk["rank"],
                "metadata": chunk["metadata"]
            })
        
        logger.info(f"✅ Retrieved {len(relevant_sections)} relevant sections for query: {query}")
        
        return {
            "success": True,
            "query": query,
            "sections": relevant_sections,
            "total_found": len(relevant_sections),
            "search_type": "vector_similarity",
            "embedding_model": "all-MiniLM-L6-v2"
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting relevant sections: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "sections": []
        }
