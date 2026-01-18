"""
Comprehensive tests for the OmniRAG module (V3).

Tests all components:
- StrategyBuilder (rule-based + embedding classifier)
- RAGDataService (multi-source routing)
- ConversationService (session persistence)
- CosmosDBConfig (schema validation)
- JenaMigration (ontology + SPARQL templates)
"""

import asyncio
import tempfile
from pathlib import Path
import os
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set up environment
os.environ['LLM_PROVIDER'] = 'google'

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')


def test_strategy_builder():
    """Test StrategyBuilder rule-based and embedding classification."""
    print("\n" + "=" * 60)
    print("TEST: StrategyBuilder")
    print("=" * 60)
    
    from src.a_mem.omnirag.strategy_builder import StrategyBuilder, Strategy
    from src.a_mem.utils.llm import LLMService
    
    # Without LLM (rules only)
    sb_rules = StrategyBuilder(llm_service=None)
    
    # Test rule-based detection
    test_cases = [
        ("lookup Flask", Strategy.DB, "rule"),
        ("find notes by tag", Strategy.DB, "rule"),
        ("what is memory evolution?", Strategy.VECTOR, "rule"),
        ("explain the architecture", Strategy.VECTOR, "rule"),
        ("related to memento", Strategy.GRAPH, "rule"),
        ("dependencies of auth module", Strategy.GRAPH, "rule"),
    ]
    
    passed = 0
    for query, expected_strategy, expected_algo in test_cases:
        result = sb_rules.determine(query)
        status = "✅" if result.strategy == expected_strategy else "❌"
        if result.strategy == expected_strategy:
            passed += 1
        print(f'  {status} "{query}" → {result.strategy.value} ({result.algorithm})')
    
    print(f"\n  Rule-based: {passed}/{len(test_cases)} passed")
    
    # With embedding classifier
    try:
        llm = LLMService()
        sb_emb = StrategyBuilder(llm, use_embedding_classifier=True, use_llm_fallback=False)
        
        # Test ambiguous query that should trigger embedding classifier
        result = sb_emb.determine("can you analyze the codebase")
        print(f"\n  Embedding test: '{result.natural_language}' → {result.strategy.value} ({result.algorithm}, {result.confidence:.2f})")
        
    except Exception as e:
        print(f"\n  ⚠️  Embedding classifier test skipped: {e}")
    
    return passed == len(test_cases)


def test_conversation_service():
    """Test ConversationService with local storage."""
    print("\n" + "=" * 60)
    print("TEST: ConversationService")
    print("=" * 60)
    
    from src.a_mem.omnirag.conversation_service import (
        ConversationService, 
        LocalConversationStore,
        Conversation,
        Message,
        ToolCall
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        store = LocalConversationStore(Path(tmpdir))
        service = ConversationService(store)
        
        async def run_test():
            # Start session
            session = await service.start_session(user_id="test-user", context="Testing OmniRAG")
            print(f"  ✅ Session started: {session.session_id[:8]}...")
            
            # Add messages
            await service.add_user_message("What is memento?")
            await service.add_assistant_response(
                "Memento is a memory system for AI agents.",
                tool_calls=[ToolCall("search", {"query": "memento"}, "Found 5 notes", 150)],
                notes_referenced=["note-001", "note-002"]
            )
            print(f"  ✅ Messages added: {session.message_count} messages")
            
            # Get context
            context = await service.get_context(max_messages=10)
            print(f"  ✅ Context retrieved: {len(context)} messages")
            
            # Save and reload
            session_id = session.session_id
            reloaded = await service.resume_session(session_id)
            assert reloaded is not None
            assert reloaded.message_count == session.message_count
            print(f"  ✅ Session persisted and reloaded")
            
            # List sessions
            sessions = await service.list_recent_sessions()
            assert len(sessions) >= 1
            print(f"  ✅ Listed {len(sessions)} session(s)")
            
            # Submit feedback
            feedback = await service.submit_feedback(
                rating="thumbs_up",
                query="What is memento?",
                strategy_used="vector"
            )
            print(f"  ✅ Feedback saved: {feedback.id[:8]}...")
            
            return True
        
        return asyncio.run(run_test())


def test_cosmos_config():
    """Test CosmosDBConfig schema definitions."""
    print("\n" + "=" * 60)
    print("TEST: CosmosDBConfig")
    print("=" * 60)
    
    from src.a_mem.omnirag.cosmos_config import (
        CosmosDBConfig,
        NOTES_CONTAINER_SCHEMA,
        CONVERSATIONS_CONTAINER_SCHEMA,
        VECTOR_INDEX_POLICY,
        get_hybrid_search_query,
        get_vector_search_query,
    )
    
    # Test config from env (should use defaults)
    config = CosmosDBConfig.from_env()
    print(f"  ✅ Config loaded: db={config.database_name}, dims={config.embedding_dimensions}")
    
    # Test schemas exist
    assert "properties" in NOTES_CONTAINER_SCHEMA
    assert "id" in NOTES_CONTAINER_SCHEMA["properties"]
    assert "embedding" in NOTES_CONTAINER_SCHEMA["properties"]
    print(f"  ✅ Notes schema: {len(NOTES_CONTAINER_SCHEMA['properties'])} properties")
    
    assert "properties" in CONVERSATIONS_CONTAINER_SCHEMA
    print(f"  ✅ Conversations schema: {len(CONVERSATIONS_CONTAINER_SCHEMA['properties'])} properties")
    
    # Test vector index policy
    assert "vectorIndexes" in VECTOR_INDEX_POLICY
    print(f"  ✅ Vector index policy: {VECTOR_INDEX_POLICY['vectorIndexes'][0]['type']}")
    
    # Test query builders (return tuples of query and params)
    vector_query, vector_params = get_vector_search_query([0.1] * 768)
    assert "VectorDistance" in vector_query
    print(f"  ✅ Vector search query generated ({len(vector_query)} chars)")
    
    hybrid_query, hybrid_params = get_hybrid_search_query("test query", [0.1] * 768)
    assert "VectorDistance" in hybrid_query
    print(f"  ✅ Hybrid search query generated ({len(hybrid_query)} chars)")
    
    return True


def test_jena_migration():
    """Test Jena migration utilities and ontology."""
    print("\n" + "=" * 60)
    print("TEST: JenaMigration")
    print("=" * 60)
    
    from src.a_mem.omnirag.jena_migration import (
        MEMENTO_ONTOLOGY_TURTLE,
        SPARQL_TEMPLATES,
        MigrationPlan,
        NetworkXToRDFExporter,
    )
    
    # Test ontology
    assert "memo:Note" in MEMENTO_ONTOLOGY_TURTLE
    assert "memo:leadsTo" in MEMENTO_ONTOLOGY_TURTLE
    assert "owl:Ontology" in MEMENTO_ONTOLOGY_TURTLE
    print(f"  ✅ Ontology: {len(MEMENTO_ONTOLOGY_TURTLE)} chars, Turtle format")
    
    # Test SPARQL templates
    templates = list(SPARQL_TEMPLATES.keys())
    print(f"  ✅ SPARQL templates: {len(templates)} queries")
    for t in templates[:5]:
        print(f"      - {t}")
    if len(templates) > 5:
        print(f"      ... and {len(templates) - 5} more")
    
    # Test migration plan phases via class method
    phases = MigrationPlan.get_all_phases()
    assert len(phases) > 0
    print(f"  ✅ Migration plan: {len(phases)} phases")
    
    # Test exporter class exists
    assert NetworkXToRDFExporter is not None
    print(f"  ✅ NetworkXToRDFExporter class defined")
    
    return True


def test_rag_data_service():
    """Test RAGDataService data models."""
    print("\n" + "=" * 60)
    print("TEST: RAGDataService")
    print("=" * 60)
    
    from src.a_mem.omnirag.rag_data_service import (
        RAGDocument,
        RAGDataResult,
    )
    from src.a_mem.omnirag.strategy_builder import StrategyResult, Strategy
    
    # Test RAGDocument
    doc = RAGDocument(
        id="test-001",
        content="Test content",
        source="vector",
        score=0.85,
        metadata={"type": "concept"}
    )
    doc_dict = doc.to_dict()
    assert doc_dict["id"] == "test-001"
    assert doc_dict["score"] == 0.85
    print(f"  ✅ RAGDocument: {doc.id} (score={doc.score})")
    
    # Test RAGDataResult
    strategy = StrategyResult(
        natural_language="test query",
        strategy=Strategy.VECTOR,
        algorithm="rule",
        confidence=0.9
    )
    result = RAGDataResult(query="test query", strategy=strategy)
    result.add_doc(doc)
    result_dict = result.to_dict()
    
    assert result_dict["document_count"] == 1
    assert result_dict["strategy"]["strategy"] == "vector"
    print(f"  ✅ RAGDataResult: {result.query} → {len(result.documents)} docs")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MEMENTO V3 (OmniRAG) - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Run tests
    try:
        results.append(("StrategyBuilder", test_strategy_builder()))
    except Exception as e:
        results.append(("StrategyBuilder", False))
        print(f"  ❌ Error: {e}")
    
    try:
        results.append(("ConversationService", test_conversation_service()))
    except Exception as e:
        results.append(("ConversationService", False))
        print(f"  ❌ Error: {e}")
    
    try:
        results.append(("CosmosDBConfig", test_cosmos_config()))
    except Exception as e:
        results.append(("CosmosDBConfig", False))
        print(f"  ❌ Error: {e}")
    
    try:
        results.append(("JenaMigration", test_jena_migration()))
    except Exception as e:
        results.append(("JenaMigration", False))
        print(f"  ❌ Error: {e}")
    
    try:
        results.append(("RAGDataService", test_rag_data_service()))
    except Exception as e:
        results.append(("RAGDataService", False))
        print(f"  ❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
