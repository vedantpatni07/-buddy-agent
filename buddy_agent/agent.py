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

"""Main Buddy Agent for Process Document Q&A

This agent serves as a "buddy" for employees by answering process-related questions
based on company documents. It processes PDF and Word documents, creates searchable
knowledge bases, and provides accurate answers based solely on document content.
"""

import os
from datetime import date

from google.genai import types
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import load_artifacts

from .sub_agents import document_processor, rag_retriever, qa_responder
from .prompts import return_instructions_root
from .tools import (
    process_document,
    create_document_corpus,
    query_documents,
    generate_answer
)

date_today = date.today()


def setup_before_agent_call(callback_context: CallbackContext):
    """Setup the Buddy Agent with document processing capabilities."""
    
    # Initialize document processing state
    if "processed_documents" not in callback_context.state:
        callback_context.state["processed_documents"] = []
    
    if "document_corpus" not in callback_context.state:
        callback_context.state["document_corpus"] = None
    
    if "rag_retrieval_ready" not in callback_context.state:
        callback_context.state["rag_retrieval_ready"] = False


root_agent = Agent(
    model=os.getenv("BUDDY_AGENT_MODEL", "gemini-2.5-flash"),
    name="buddy_agent",
    instruction=return_instructions_root(),
    global_instruction=(
        f"""
        You are a Buddy Agent - a helpful assistant for employees who need answers 
        to process-related questions based on company documents.
        
        **Your Mission:**
        Help employees find accurate, document-based answers to their questions about
        company processes, policies, procedures, and guidelines.
        
        **Core Capabilities:**
        - **Document Processing**: Extract text from PDF and Word documents
        - **Knowledge Base Creation**: Build searchable document corpora using RAG
        - **Intelligent Retrieval**: Find relevant document sections for any question
        - **Contextual Answers**: Provide accurate answers based solely on document content
        
        **Workflow:**
        1. **Process Documents**: Use `process_document` to extract text from uploaded files
        2. **Create Knowledge Base**: Use `create_document_corpus` to index documents for search
        3. **Answer Questions**: Use `query_documents` to find relevant content and `generate_answer` to respond
        
        **Important Guidelines:**
        - Only provide answers based on processed document content
        - If information is not in the documents, clearly state this
        - Always cite which document and section your answer comes from
        - Be helpful and conversational, like a knowledgeable colleague
        
        Today's date: {date_today}
        """
    ),
    sub_agents=[document_processor, rag_retriever, qa_responder],
    tools=[
        process_document,
        create_document_corpus,
        query_documents,
        generate_answer,
        load_artifacts,
    ],
    before_agent_callback=setup_before_agent_call,
    generate_content_config=types.GenerateContentConfig(temperature=0.1),
)
