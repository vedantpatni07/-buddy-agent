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

"""Sub-agents for the Buddy Agent system.

This module contains specialized sub-agents for different aspects of document processing
and question answering.
"""

from .document_processor import root_agent as document_processor
from .rag_retriever import root_agent as rag_retriever
from .qa_responder import root_agent as qa_responder

__all__ = ["document_processor", "rag_retriever", "qa_responder"]

