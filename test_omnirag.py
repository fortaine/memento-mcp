#!/usr/bin/env python3
"""
Test StrategyBuilder OmniRAG Intent Detection

Tests the rule-based and LLM-based strategy routing.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def test_strategy_builder_rules():
    """Test rule-based pattern matching."""
    print("=" * 60)
    print("Testing StrategyBuilder Rule-Based Patterns")
    print("=" * 60)
    
    from a_mem.omnirag.strategy_builder import StrategyBuilder, Strategy
    
    builder = StrategyBuilder()  # No LLM = rules only
    
    # DB pattern tests
    db_queries = [
        "lookup Flask",
        "find error pattern",
        "get note abc-123",
        "fetch all procedures",
        "list all rules",
        "search by tag python",
        "retrieve specific item",
    ]
    
    print("\n📦 DB Strategy Queries:")
    for q in db_queries:
        result = builder.determine(q)
        status = "✅" if result.strategy == Strategy.DB else "❌"
        print(f"  {status} '{q}' → {result.strategy.value} ({result.algorithm}, conf: {result.confidence:.2f})")
    
    # Vector pattern tests
    vector_queries = [
        "what is memory evolution?",
        "how to brief a task?",
        "explain the A-MEM pattern",
        "similar to this code snippet",
        "describe the compound learning workflow",
        "summarize the memento architecture",
    ]
    
    print("\n🔍 Vector Strategy Queries:")
    for q in vector_queries:
        result = builder.determine(q)
        status = "✅" if result.strategy == Strategy.VECTOR else "❌"
        print(f"  {status} '{q}' → {result.strategy.value} ({result.algorithm}, conf: {result.confidence:.2f})")
    
    # Graph pattern tests
    graph_queries = [
        "what is related to memento-v2?",
        "dependencies of the LLM service",
        "show connections between A and B",
        "notes that lead to this concept",
        "hierarchy of note types",
        "what supports this rule?",
        "what contradicts this pattern?",
    ]
    
    print("\n🌐 Graph Strategy Queries:")
    for q in graph_queries:
        result = builder.determine(q)
        status = "✅" if result.strategy == Strategy.GRAPH else "❌"
        print(f"  {status} '{q}' → {result.strategy.value} ({result.algorithm}, conf: {result.confidence:.2f})")
    
    # Ambiguous queries (should default to vector)
    ambiguous_queries = [
        "memento",
        "help me with this",
        "tell me about errors",
    ]
    
    print("\n❓ Ambiguous Queries (expect vector default):")
    for q in ambiguous_queries:
        result = builder.determine(q)
        status = "✅" if result.strategy == Strategy.VECTOR else "⚠️"
        print(f"  {status} '{q}' → {result.strategy.value} ({result.algorithm}, conf: {result.confidence:.2f})")
    
    print("\n" + "=" * 60)
    print("✅ Rule-based tests complete!")
    print("=" * 60)


async def test_strategy_builder_with_llm():
    """Test with LLM fallback for complex queries."""
    print("\n" + "=" * 60)
    print("Testing StrategyBuilder with LLM Fallback")
    print("=" * 60)
    
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  Skipping LLM tests - no API key found")
        return
    
    os.environ["LLM_PROVIDER"] = "google"
    
    from a_mem.utils.llm import LLMService
    from a_mem.omnirag.strategy_builder import StrategyBuilder, Strategy
    
    llm = LLMService()
    builder = StrategyBuilder(llm)
    
    # Complex queries that might need LLM
    complex_queries = [
        "I need to understand the relationship between notes and their evolution",  # graph
        "Can you find memories about Python best practices?",  # vector
        "Get me the exact configuration for the LLM provider",  # db
    ]
    
    print("\n🤖 Complex Queries (LLM may be used):")
    for q in complex_queries:
        result = builder.determine(q)
        print(f"  → '{q[:50]}...'")
        print(f"    Strategy: {result.strategy.value}, Algorithm: {result.algorithm}, Confidence: {result.confidence:.2f}")
    
    print("\n" + "=" * 60)
    print("✅ LLM tests complete!")
    print("=" * 60)


async def test_rag_data_service():
    """Test the full RAGDataService with OmniRAG routing."""
    print("\n" + "=" * 60)
    print("Testing RAGDataService OmniRAG Orchestration")
    print("=" * 60)
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  Skipping RAG tests - no API key found")
        return
    
    os.environ["LLM_PROVIDER"] = "google"
    
    from a_mem.core.logic import MemoryController
    from a_mem.omnirag.rag_data_service import RAGDataService
    from a_mem.models.note import NoteInput
    
    controller = MemoryController()
    rag_service = RAGDataService(controller)
    
    # Create test notes
    print("\n📝 Creating test notes...")
    test_notes = [
        NoteInput(
            content="Python is a high-level programming language known for readability.",
            source="test",
        ),
        NoteInput(
            content="Flask is a micro web framework written in Python.",
            source="test",
        ),
        NoteInput(
            content="The LLM service connects to Google AI Studio for embeddings.",
            source="test",
        ),
    ]
    
    for note in test_notes:
        await controller.create_note(note)
    print("  ✅ Created 3 test notes")
    
    # Test different query types
    queries = [
        ("lookup Flask", "db"),
        ("what is Python?", "vector"),
        ("related to LLM service", "graph"),
    ]
    
    print("\n🔄 Testing OmniRAG Routing:")
    for query, expected in queries:
        result = await rag_service.get_rag_data(query, max_results=3)
        status = "✅" if result.strategy.strategy.value == expected else "⚠️"
        print(f"\n  {status} Query: '{query}'")
        print(f"    Strategy: {result.strategy.strategy.value} (expected: {expected})")
        print(f"    Documents: {len(result.documents)}")
        print(f"    Fallback: {result.fallback_used}")
        if result.documents:
            print(f"    Top result: {result.documents[0].content[:60]}...")
    
    # Test route_query helper
    print("\n🧭 Testing route_query helper:")
    route = await rag_service.route_query("how to use memory enzymes?")
    print(f"  Query: 'how to use memory enzymes?'")
    print(f"  Route: {route.to_dict()}")
    
    print("\n" + "=" * 60)
    print("✅ RAGDataService tests complete!")
    print("=" * 60)


def main():
    """Run all tests."""
    # Rule-based tests (no LLM needed)
    test_strategy_builder_rules()
    
    # LLM and RAG tests (require API key)
    asyncio.run(test_strategy_builder_with_llm())
    asyncio.run(test_rag_data_service())
    
    print("\n" + "=" * 60)
    print("🎉 All OmniRAG tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
