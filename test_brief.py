#!/usr/bin/env python3
"""Test the brief_task compound learning tool."""
import asyncio
import os
import sys

# Load .env file first, then override with actual env vars if they exist
from dotenv import load_dotenv
load_dotenv(override=False)

# The .env has ${OPENROUTER_API_KEY} as placeholder - we need the actual key
# Check if already set from parent shell
parent_key = os.environ.get('OPENROUTER_API_KEY', '')
if parent_key and not parent_key.startswith('${'):
    pass  # Already have real key
else:
    # Try to get from shell environment by reading it
    print("Note: OPENROUTER_API_KEY needs to be set. Run with: OPENROUTER_API_KEY=your_key python test_brief.py", file=sys.stderr)

async def test():
    print(f"OPENROUTER_API_KEY set: {bool(os.getenv('OPENROUTER_API_KEY') and not os.getenv('OPENROUTER_API_KEY', '').startswith('$'))}", file=sys.stderr)
    
    from src.a_mem.main import call_tool
    
    # First, create a test note so we have something to retrieve
    print('Creating test note...', file=sys.stderr)
    result = await call_tool('create_atomic_note', {
        'content': 'Memento V2 is an agentic memory system based on A-MEM (NeurIPS research). It uses ChromaDB for vector storage, NetworkX for graph relationships, and supports 19 MCP tools including compound learning tools like brief_task, debrief_task, track_error, and promote_pattern.',
        'source': 'test'
    })
    for r in result:
        print(r.text)
    
    print('\nTesting brief_task: "Brief me on this project"...', file=sys.stderr)
    result = await call_tool('brief_task', {
        'task_description': 'Brief me on this project',
        'include_errors': True,
        'include_skills': True
    })
    for r in result:
        print(r.text)

if __name__ == "__main__":
    asyncio.run(test())
