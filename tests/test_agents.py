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

"""Tests for the Buddy Agent system."""

import pytest
from buddy_agent import root_agent


def test_agent_initialization():
    """Test that the main agent initializes correctly."""
    assert root_agent is not None
    assert root_agent.name == "buddy_agent"


def test_agent_has_required_tools():
    """Test that the agent has the required tools."""
    tool_names = [tool.name for tool in root_agent.tools]
    
    required_tools = [
        "process_document",
        "create_document_corpus", 
        "query_documents",
        "generate_answer"
    ]
    
    for tool_name in required_tools:
        assert tool_name in tool_names


def test_agent_has_sub_agents():
    """Test that the agent has the required sub-agents."""
    sub_agent_names = [agent.name for agent in root_agent.sub_agents]
    
    expected_sub_agents = [
        "document_processor",
        "rag_retriever", 
        "qa_responder"
    ]
    
    for sub_agent_name in expected_sub_agents:
        assert sub_agent_name in sub_agent_names


if __name__ == "__main__":
    pytest.main([__file__])

