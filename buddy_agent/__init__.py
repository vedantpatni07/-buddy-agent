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

"""Buddy Agent - Process Document Q&A Assistant

This agent helps employees find answers to process-related questions by:
1. Processing PDF and Word documents from various sources
2. Creating searchable knowledge bases using RAG
3. Providing accurate, document-based answers to user queries

Architecture:
- Main Agent: Orchestrates document processing and Q&A workflow
- Document Processor Sub-Agent: Handles PDF/Word parsing and text extraction
- RAG Retriever Sub-Agent: Manages document indexing and retrieval
- Q&A Responder Sub-Agent: Generates contextual answers from retrieved content
"""

from .agent import root_agent

__all__ = ["root_agent"]

