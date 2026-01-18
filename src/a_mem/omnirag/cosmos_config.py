"""
Azure Cosmos DB Configuration and Schema for Memento V3

Defines the container schemas, vector index policies, and connection settings
for the OmniRAG architecture.
"""

import os
from dataclasses import dataclass
from typing import Optional, Literal
from enum import Enum


class AuthMechanism(str, Enum):
    """Authentication mechanism for Cosmos DB."""
    KEY = "key"         # Primary/Secondary key authentication
    RBAC = "rbac"       # Azure AD RBAC authentication


@dataclass
class CosmosDBConfig:
    """Configuration for Azure Cosmos DB connection and containers."""
    
    # Connection
    endpoint: str = ""
    key: str = ""
    auth_mechanism: AuthMechanism = AuthMechanism.KEY
    
    # Database
    database_name: str = "memento"
    
    # Containers
    notes_container: str = "notes"
    config_container: str = "config"
    conversations_container: str = "conversations"
    feedback_container: str = "feedback"
    
    # Vector Index Settings
    embedding_dimensions: int = 768
    embedding_field: str = "embedding"
    distance_function: Literal["cosine", "euclidean", "dotproduct"] = "cosine"
    
    # Full-text Search Fields
    fulltext_fields: Optional[list] = None
    
    # Partition Key
    partition_key: str = "/partitionKey"
    default_partition: str = "memento"  # Or use user_id for multi-tenant
    
    @classmethod
    def from_env(cls) -> "CosmosDBConfig":
        """Load configuration from environment variables."""
        return cls(
            # Connection
            endpoint=os.getenv("CAIG_COSMOSDB_NOSQL_URI", ""),
            key=os.getenv("CAIG_COSMOSDB_NOSQL_KEY", ""),
            auth_mechanism=AuthMechanism(
                os.getenv("CAIG_COSMOSDB_NOSQL_AUTH_MECHANISM", "key").lower()
            ),
            
            # Database
            database_name=os.getenv("CAIG_GRAPH_SOURCE_DB", "memento"),
            
            # Containers
            notes_container=os.getenv("CAIG_GRAPH_SOURCE_CONTAINER", "notes"),
            config_container=os.getenv("CAIG_CONFIG_CONTAINER", "config"),
            conversations_container=os.getenv("CAIG_CONVERSATIONS_CONTAINER", "conversations"),
            feedback_container=os.getenv("CAIG_FEEDBACK_CONTAINER", "feedback"),
            
            # Vector Index
            embedding_dimensions=int(os.getenv("CAIG_EMBEDDING_DIMENSIONS", "768")),
            embedding_field=os.getenv("CAIG_EMBEDDING_FIELD_NAME", "embedding"),
            
            # Full-text Search
            fulltext_fields=os.getenv(
                "CAIG_FULLTEXT_SEARCH_FIELDS", 
                "content,contextual_summary,tags"
            ).split(","),
            
            # Partition
            partition_key=os.getenv("CAIG_PARTITION_KEY", "/partitionKey"),
            default_partition=os.getenv("CAIG_DEFAULT_PARTITION", "memento"),
        )
    
    def to_env_template(self) -> str:
        """Generate .env template for Cosmos DB configuration."""
        return """# ============================================
# Memento V3 - Azure Cosmos DB Configuration
# ============================================

# Connection
CAIG_COSMOSDB_NOSQL_URI=https://your-account.documents.azure.com:443/
CAIG_COSMOSDB_NOSQL_KEY=your-primary-key-here
CAIG_COSMOSDB_NOSQL_AUTH_MECHANISM=key  # or rbac

# Database and Containers
CAIG_GRAPH_SOURCE_DB=memento
CAIG_GRAPH_SOURCE_CONTAINER=notes
CAIG_CONFIG_CONTAINER=config
CAIG_CONVERSATIONS_CONTAINER=conversations
CAIG_FEEDBACK_CONTAINER=feedback

# Vector Index Settings
CAIG_EMBEDDING_DIMENSIONS=768
CAIG_EMBEDDING_FIELD_NAME=embedding

# Full-text Search
CAIG_FULLTEXT_SEARCH_FIELDS=content,contextual_summary,tags

# Partition Key Strategy
CAIG_PARTITION_KEY=/partitionKey
CAIG_DEFAULT_PARTITION=memento
"""


# ============================================
# Container Schema Definitions (JSON Schema)
# ============================================

NOTES_CONTAINER_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["id", "content", "partitionKey"],
    "properties": {
        # Core Identity
        "id": {
            "type": "string",
            "description": "Unique note identifier (UUID)"
        },
        "partitionKey": {
            "type": "string",
            "description": "Partition key (default: 'memento' or user_id for multi-tenant)"
        },
        
        # Content
        "content": {
            "type": "string",
            "description": "Original note content"
        },
        "contextual_summary": {
            "type": "string",
            "description": "LLM-generated summary (max 200 chars)"
        },
        
        # Classification
        "type": {
            "type": "string",
            "enum": ["rule", "procedure", "concept", "tool", "reference", "integration"],
            "description": "Note type classification"
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3-5 key terms for search"
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "2-3 category tags"
        },
        
        # Vector Embedding (DiskANN)
        "embedding": {
            "type": "array",
            "items": {"type": "number"},
            "description": "768-dimensional L2-normalized embedding vector"
        },
        
        # Graph Edges (denormalized for fast access)
        "graph_edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "Target note ID"},
                    "type": {
                        "type": "string",
                        "enum": ["LEADS_TO", "DEPENDS_ON", "CONTRADICTS", "SUPPORTS", "SUPERSEDES", "PART_OF"]
                    },
                    "weight": {"type": "number", "minimum": 0, "maximum": 1}
                }
            },
            "description": "Outgoing edges to other notes"
        },
        
        # Metadata
        "metadata": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "Origin of the note"},
                "created_at": {"type": "string", "format": "date-time"},
                "updated_at": {"type": "string", "format": "date-time"},
                "access_count": {"type": "integer", "minimum": 0},
                "priority_score": {"type": "number", "minimum": 0, "maximum": 1}
            }
        },
        
        # System fields (Cosmos DB managed)
        "_ts": {"type": "integer", "description": "Timestamp (Cosmos managed)"},
        "_etag": {"type": "string", "description": "ETag (Cosmos managed)"}
    }
}

CONVERSATIONS_CONTAINER_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["id", "partitionKey", "session_id"],
    "properties": {
        "id": {"type": "string"},
        "partitionKey": {"type": "string"},
        "session_id": {
            "type": "string",
            "description": "Unique session identifier"
        },
        "user_id": {
            "type": "string",
            "description": "User identifier (for multi-tenant)"
        },
        "messages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "role": {"type": "string", "enum": ["user", "assistant", "system"]},
                    "content": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "tool_calls": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "tool_name": {"type": "string"},
                                "arguments": {"type": "object"},
                                "result": {"type": "string"}
                            }
                        }
                    },
                    "notes_referenced": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Note IDs referenced in this message"
                    }
                }
            }
        },
        "metadata": {
            "type": "object",
            "properties": {
                "created_at": {"type": "string", "format": "date-time"},
                "updated_at": {"type": "string", "format": "date-time"},
                "message_count": {"type": "integer"},
                "context": {"type": "string", "description": "Session context/topic"}
            }
        }
    }
}

FEEDBACK_CONTAINER_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["id", "partitionKey", "session_id", "rating"],
    "properties": {
        "id": {"type": "string"},
        "partitionKey": {"type": "string"},
        "session_id": {"type": "string"},
        "message_id": {"type": "string", "description": "Specific message ID (optional)"},
        "user_id": {"type": "string"},
        "rating": {
            "type": "string",
            "enum": ["thumbs_up", "thumbs_down", "1", "2", "3", "4", "5"],
            "description": "User rating"
        },
        "feedback_text": {"type": "string"},
        "notes_referenced": {
            "type": "array",
            "items": {"type": "string"}
        },
        "query": {"type": "string", "description": "Original query"},
        "strategy_used": {
            "type": "string",
            "enum": ["db", "vector", "graph"],
            "description": "Which OmniRAG strategy was used"
        },
        "timestamp": {"type": "string", "format": "date-time"}
    }
}


# ============================================
# Vector Index Policy (DiskANN)
# ============================================

VECTOR_INDEX_POLICY = {
    "vectorIndexes": [
        {
            "path": "/embedding",
            "type": "DiskANN",
            "dimensions": 768,
            "distanceFunction": "cosine"
        }
    ]
}

# Full container policy including vector indexes
NOTES_CONTAINER_POLICY = {
    "indexingMode": "consistent",
    "automatic": True,
    "includedPaths": [
        {"path": "/*"}
    ],
    "excludedPaths": [
        {"path": "/embedding/*"}  # Exclude vector from regular index
    ],
    "vectorIndexes": VECTOR_INDEX_POLICY["vectorIndexes"]
}


# ============================================
# Helper Functions
# ============================================

def get_cosmos_connection_string(config: CosmosDBConfig) -> str:
    """Build connection string from config."""
    return f"AccountEndpoint={config.endpoint};AccountKey={config.key};"


def get_hybrid_search_query(
    embedding: list,
    search_text: str,
    limit: int = 10,
    min_score: float = 0.0
) -> tuple:
    """
    Generate SQL query for hybrid search (vector + full-text).
    
    Returns:
        Tuple of (query_string, parameters)
    """
    query = """
    SELECT TOP @limit 
        c.id, 
        c.content, 
        c.contextual_summary, 
        c.tags, 
        c.type,
        c.keywords,
        c.metadata,
        VectorDistance(c.embedding, @embedding) as similarity_score
    FROM c
    WHERE (
        CONTAINS(LOWER(c.content), LOWER(@search_text), true)
        OR CONTAINS(LOWER(c.contextual_summary), LOWER(@search_text), true)
        OR ARRAY_CONTAINS(c.tags, @search_text)
        OR ARRAY_CONTAINS(c.keywords, @search_text)
    )
    ORDER BY VectorDistance(c.embedding, @embedding)
    """
    
    parameters = [
        {"name": "@limit", "value": limit},
        {"name": "@embedding", "value": embedding},
        {"name": "@search_text", "value": search_text}
    ]
    
    return query, parameters


def get_vector_search_query(
    embedding: list,
    limit: int = 10,
    filter_type: Optional[str] = None
) -> tuple:
    """
    Generate SQL query for pure vector search.
    
    Returns:
        Tuple of (query_string, parameters)
    """
    if filter_type:
        query = """
        SELECT TOP @limit 
            c.id, 
            c.content, 
            c.contextual_summary,
            c.type,
            c.tags,
            c.keywords,
            c.metadata,
            VectorDistance(c.embedding, @embedding) as similarity_score
        FROM c
        WHERE c.type = @filter_type
        ORDER BY VectorDistance(c.embedding, @embedding)
        """
        parameters = [
            {"name": "@limit", "value": limit},
            {"name": "@embedding", "value": embedding},
            {"name": "@filter_type", "value": filter_type}
        ]
    else:
        query = """
        SELECT TOP @limit 
            c.id, 
            c.content, 
            c.contextual_summary,
            c.type,
            c.tags,
            c.keywords,
            c.metadata,
            VectorDistance(c.embedding, @embedding) as similarity_score
        FROM c
        ORDER BY VectorDistance(c.embedding, @embedding)
        """
        parameters = [
            {"name": "@limit", "value": limit},
            {"name": "@embedding", "value": embedding}
        ]
    
    return query, parameters
