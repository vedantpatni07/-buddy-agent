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

"""Tools for the Q&A Responder Sub-Agent."""

import logging
from typing import Dict, List, Any
from google.adk.tools import ToolContext

logger = logging.getLogger(__name__)


async def generate_comprehensive_answer(
    question: str,
    relevant_sections: List[Dict[str, Any]],
    tool_context: ToolContext = None,
) -> Dict[str, Any]:
    """Generate a comprehensive answer from relevant document sections."""
    try:
        if not relevant_sections:
            return {
                "success": False,
                "answer": "I couldn't find relevant information in the processed documents to answer your question.",
                "sources": [],
                "confidence": "Low"
            }
        
        # Combine relevant sections for context
        context_parts = []
        sources = []
        
        for section in relevant_sections:
            context_parts.append(f"**From {section.get('document', 'Unknown Document')}:**\n{section.get('content', '')}")
            sources.append({
                "document": section.get('document', 'Unknown Document'),
                "relevance_score": section.get('relevance_score', 0.0),
                "section_index": section.get('section_index', 0)
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate answer using the context
        answer = f"""
Based on the processed documents, here's what I found regarding your question: "{question}"

{context}

Please note that this answer is based on the content available in the processed documents. If you need more specific information, please let me know and I can help you find additional details.
        """.strip()
        
        # Calculate confidence based on number and quality of relevant sections
        avg_relevance = sum(s.get('relevance_score', 0) for s in relevant_sections) / len(relevant_sections)
        confidence = "High" if avg_relevance > 0.7 and len(relevant_sections) >= 2 else "Medium" if avg_relevance > 0.4 else "Low"
        
        logger.info(f"Generated comprehensive answer for question: {question}")
        
        return {
            "success": True,
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "sections_used": len(relevant_sections),
            "average_relevance": avg_relevance
        }
        
    except Exception as e:
        logger.error(f"Error generating comprehensive answer: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "answer": "I encountered an error while generating an answer. Please try again.",
            "sources": [],
            "confidence": "Low"
        }


async def format_response(
    answer: str,
    sources: List[Dict[str, Any]],
    confidence: str,
    tool_context: ToolContext = None,
) -> Dict[str, Any]:
    """Format the response in a clear, structured way."""
    try:
        # Create formatted response
        formatted_response = f"""
## **Answer**
{answer}

## **Source**
- **Documents**: {', '.join(set(s.get('document', 'Unknown') for s in sources))}
- **Confidence**: {confidence}
- **Sections Used**: {len(sources)}

## **Additional Context**
This information is based on the processed company documents. If you need more specific details or have follow-up questions, please let me know.
        """.strip()
        
        logger.info("Formatted response successfully")
        
        return {
            "success": True,
            "formatted_response": formatted_response,
            "sources": sources,
            "confidence": confidence
        }
        
    except Exception as e:
        logger.error(f"Error formatting response: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "formatted_response": answer,  # Return original answer if formatting fails
            "sources": sources,
            "confidence": confidence
        }


async def validate_answer(
    answer: str,
    question: str,
    sources: List[Dict[str, Any]],
    tool_context: ToolContext = None,
) -> Dict[str, Any]:
    """Validate the quality and accuracy of the generated answer."""
    try:
        validation_results = {
            "is_complete": len(answer.strip()) > 50,
            "has_sources": len(sources) > 0,
            "is_relevant": any(keyword in answer.lower() for keyword in question.lower().split()),
            "is_helpful": "based on" in answer.lower() or "according to" in answer.lower(),
            "issues": []
        }
        
        # Check for common issues
        if not validation_results["is_complete"]:
            validation_results["issues"].append("Answer is too short or incomplete")
        
        if not validation_results["has_sources"]:
            validation_results["issues"].append("No source documents cited")
        
        if not validation_results["is_relevant"]:
            validation_results["issues"].append("Answer may not be directly relevant to the question")
        
        if not validation_results["is_helpful"]:
            validation_results["issues"].append("Answer lacks helpful context or explanation")
        
        # Overall validation score
        validation_score = sum([
            validation_results["is_complete"],
            validation_results["has_sources"],
            validation_results["is_relevant"],
            validation_results["is_helpful"]
        ]) / 4
        
        validation_results["validation_score"] = validation_score
        validation_results["is_valid"] = validation_score >= 0.75
        
        logger.info(f"Answer validation completed. Score: {validation_score:.2f}")
        
        return {
            "success": True,
            "validation_results": validation_results,
            "is_valid": validation_results["is_valid"],
            "validation_score": validation_score
        }
        
    except Exception as e:
        logger.error(f"Error validating answer: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "validation_results": {"is_valid": False, "issues": ["Validation error"]},
            "is_valid": False,
            "validation_score": 0.0
        }

