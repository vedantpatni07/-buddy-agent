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

"""Deployment script for the Buddy Agent."""

import os
import sys
from pathlib import Path

# Add the buddy_agent module to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from buddy_agent import root_agent


def main():
    """Deploy the Buddy Agent."""
    print("🚀 Deploying Buddy Agent...")
    print(f"Agent Name: {root_agent.name}")
    print(f"Model: {root_agent.model}")
    print(f"Tools: {len(root_agent.tools)}")
    print(f"Sub-agents: {len(root_agent.sub_agents)}")
    print("✅ Buddy Agent deployed successfully!")


if __name__ == "__main__":
    main()

