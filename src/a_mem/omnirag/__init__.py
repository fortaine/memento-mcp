"""
OmniRAG Module - V3 Multi-Source Retrieval with AI Intent Routing

Based on CosmosAIGraph OmniRAG pattern:
- StrategyBuilder: Intent detection (db/vector/graph routing)
- RAGDataService: Unified retrieval orchestration
- ConversationService: Session persistence
- CosmosDBConfig: Azure Cosmos DB configuration and schemas
- JenaMigration: NetworkX → Apache Jena migration utilities
"""

from .strategy_builder import StrategyBuilder, Strategy, StrategyResult
from .rag_data_service import RAGDataService, RAGDocument, RAGDataResult
from .cosmos_config import (
    CosmosDBConfig,
    AuthMechanism,
    NOTES_CONTAINER_SCHEMA,
    CONVERSATIONS_CONTAINER_SCHEMA,
    FEEDBACK_CONTAINER_SCHEMA,
    VECTOR_INDEX_POLICY,
    get_hybrid_search_query,
    get_vector_search_query,
)
from .conversation_service import (
    ConversationService,
    Conversation,
    Message,
    ToolCall,
    Feedback,
    LocalConversationStore,
)
from .jena_migration import (
    MEMENTO_ONTOLOGY_TURTLE,
    SPARQL_TEMPLATES,
    MigrationPlan,
    NetworkXToRDFExporter,
    JenaConfig,
    JenaService,
    NL2SPARQLService,
)

__all__ = [
    # Strategy Builder
    "StrategyBuilder",
    "Strategy",
    "StrategyResult",
    
    # RAG Data Service
    "RAGDataService",
    "RAGDocument",
    "RAGDataResult",
    
    # Cosmos DB
    "CosmosDBConfig",
    "AuthMechanism",
    "NOTES_CONTAINER_SCHEMA",
    "CONVERSATIONS_CONTAINER_SCHEMA",
    "FEEDBACK_CONTAINER_SCHEMA",
    "VECTOR_INDEX_POLICY",
    "get_hybrid_search_query",
    "get_vector_search_query",
    
    # Conversation Service
    "ConversationService",
    "Conversation",
    "Message",
    "ToolCall",
    "Feedback",
    "LocalConversationStore",
    
    # Jena Migration
    "MEMENTO_ONTOLOGY_TURTLE",
    "SPARQL_TEMPLATES",
    "MigrationPlan",
    "NetworkXToRDFExporter",
    "JenaConfig",
    "JenaService",
    "NL2SPARQLService",
]
