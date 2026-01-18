#!/usr/bin/env python3
"""Test script for Google AI Studio provider in memento-v2."""

import os
import sys
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv()

async def test_google_provider():
    """Test the Google AI Studio provider."""
    print("=" * 60)
    print("Testing memento-v2 Google AI Studio Provider")
    print("=" * 60)
    
    # Check environment
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.startswith("your_"):
        # Try to get from shell env
        api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("❌ ERROR: No GOOGLE_API_KEY or GEMINI_API_KEY found!")
        print("   Set one of these environment variables to your Google AI Studio key.")
        return False
    
    # Set it for the test
    os.environ["GOOGLE_API_KEY"] = api_key
    print(f"✓ API Key found: {api_key[:10]}...{api_key[-4:]}")
    
    # Check provider setting
    provider = os.getenv("LLM_PROVIDER", "ollama")
    print(f"✓ LLM_PROVIDER: {provider}")
    
    if provider != "google":
        print("⚠️  Warning: LLM_PROVIDER is not set to 'google'. Setting it now...")
        os.environ["LLM_PROVIDER"] = "google"
    
    # Reload config after env changes
    from a_mem import config
    config.settings = config.Config()
    
    # Test LLM Service
    print("\n--- Testing LLM Service ---")
    from a_mem.utils.llm import LLMService
    
    llm = LLMService()
    print(f"✓ Provider: {llm.provider}")
    print(f"✓ LLM Model: {llm.google_llm_model}")
    print(f"✓ Embedding Model: {llm.google_embedding_model}")
    print(f"✓ Embedding Dimensions: {llm.google_embedding_dimensions}")
    
    # Test embedding
    print("\n--- Testing Embedding ---")
    test_text = "This is a test memory about Python programming and software development."
    try:
        embedding = llm.get_embedding(test_text)
        print(f"✓ Embedding generated successfully!")
        print(f"  - Dimensions: {len(embedding)}")
        print(f"  - First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test LLM generation
    print("\n--- Testing LLM Generation ---")
    try:
        response = llm._call_llm("Say 'Hello from Gemini!' in exactly those words.")
        print(f"✓ LLM response: {response[:100]}...")
    except Exception as e:
        print(f"❌ LLM generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test metadata extraction
    print("\n--- Testing Metadata Extraction ---")
    try:
        metadata = llm.extract_metadata(test_text)
        print(f"✓ Metadata extracted:")
        print(f"  - Summary: {metadata.get('summary')}")
        print(f"  - Keywords: {metadata.get('keywords')}")
        print(f"  - Tags: {metadata.get('tags')}")
        print(f"  - Type: {metadata.get('type')}")
    except Exception as e:
        print(f"❌ Metadata extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test full note creation via logic
    print("\n--- Testing Full Note Creation ---")
    try:
        from a_mem.core.logic import MemoryController
        from a_mem.models.note import NoteInput
        
        controller = MemoryController()
        note_input = NoteInput(content="Test note: Google AI Studio integration is working correctly!")
        result = await controller.create_note(note_input)
        print(f"✓ Note created successfully!")
        print(f"  - Result: {result[:100]}...")
    except Exception as e:
        print(f"❌ Note creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Google AI Studio provider is working!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_google_provider())
    sys.exit(0 if success else 1)
