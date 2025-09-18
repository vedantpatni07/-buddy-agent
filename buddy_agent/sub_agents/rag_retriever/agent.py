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

"""RAG Retriever Sub-Agent

Specialized agent for creating document corpora and retrieving relevant content.
"""

import os
from google.adk.agents import Agent
from .prompts import return_instructions_rag_retriever
from .tools import create_corpus, search_documents, get_relevant_sections

root_agent = Agent(
    model=os.getenv("RAG_RETRIEVER_MODEL", "gemini-2.5-flash"),
    name="rag_retriever",
    instruction=return_instructions_rag_retriever(),
    tools=[
        create_corpus,
        search_documents,
        get_relevant_sections,
    ],
)

