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

"""Document Processor Sub-Agent

Specialized agent for processing PDF and Word documents.
"""

import os
from google.adk.agents import Agent
from .prompts import return_instructions_document_processor
from .tools import extract_pdf_text, extract_word_text, validate_document

root_agent = Agent(
    model=os.getenv("DOCUMENT_PROCESSOR_MODEL", "gemini-2.5-flash"),
    name="document_processor",
    instruction=return_instructions_document_processor(),
    tools=[
        extract_pdf_text,
        extract_word_text,
        validate_document,
    ],
)
