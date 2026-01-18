"""
RAGDataService: OmniRAG Orchestration Layer

Routes queries to the optimal data source (db/vector/graph) based on
StrategyBuilder intent detection, then aggregates results.

Ported from CosmosAIGraph: https://github.com/AzureCosmosDB/CosmosAIGraph
"""

import sys
from dataclasses import dataclass, field
from typing import List, Optional, Any, TYPE_CHECKING

from .strategy_builder import StrategyBuilder, Strategy, StrategyResult

if TYPE_CHECKING:
    from ..core.logic import MemoryController


@dataclass
class RAGDocument:
    """A document retrieved by RAG."""
    id: str
    content: str
    source: str  # "db", "vector", or "graph"
    score: float = 0.0
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "score": self.score,
            "metadata": self.metadata
        }


@dataclass
class RAGDataResult:
    """Result container for OmniRAG retrieval."""
    query: str
    strategy: StrategyResult
    documents: List[RAGDocument] = field(default_factory=list)
    sparql: Optional[str] = None  # For graph queries
    fallback_used: bool = False
    error: Optional[str] = None
    
    def add_doc(self, doc: RAGDocument):
        """Add a document to results."""
        self.documents.append(doc)
    
    def set_sparql(self, sparql: str):
        """Set the SPARQL query used (for graph)."""
        self.sparql = sparql
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "strategy": self.strategy.to_dict(),
            "documents": [d.to_dict() for d in self.documents],
            "sparql": self.sparql,
            "fallback_used": self.fallback_used,
            "document_count": len(self.documents),
            "error": self.error
        }


class RAGDataService:
    """
    OmniRAG orchestration service.
    
    Routes queries to optimal data source based on intent:
    - DB: Exact match queries, lookups, fetch by ID/tag
    - Vector: Semantic similarity, open-ended questions
    - Graph: Relationship traversal, dependencies
    
    Falls back to vector search if primary source returns no results.
    """
    
    def __init__(
        self, 
        controller: "MemoryController",
        llm_service = None,
        enable_fallback: bool = True
    ):
        """
        Initialize RAGDataService.
        
        Args:
            controller: The MemoryController for data access
            llm_service: Optional LLM service for strategy classification
            enable_fallback: Whether to fall back to vector if primary fails
        """
        self.controller = controller
        self.llm = llm_service or (controller.llm if hasattr(controller, 'llm') else None)
        self.strategy_builder = StrategyBuilder(self.llm)
        self.enable_fallback = enable_fallback
    
    async def get_rag_data(
        self, 
        query: str, 
        max_results: int = 10,
        strategy_override: Optional[Strategy] = None
    ) -> RAGDataResult:
        """
        Get RAG data using OmniRAG pattern.
        
        Args:
            query: Natural language query
            max_results: Maximum documents to return
            strategy_override: Force a specific strategy (bypass auto-detection)
            
        Returns:
            RAGDataResult with documents from optimal source
        """
        # Determine strategy
        if strategy_override:
            strategy = StrategyResult(
                natural_language=query,
                strategy=strategy_override,
                algorithm="override"
            )
        else:
            strategy = self.strategy_builder.determine(query)
        
        result = RAGDataResult(query=query, strategy=strategy)
        
        try:
            # Route to appropriate handler
            if strategy.strategy == Strategy.DB:
                await self._get_db_rag_data(query, max_results, result)
            elif strategy.strategy == Strategy.GRAPH:
                await self._get_graph_rag_data(query, max_results, result)
            else:  # VECTOR (default)
                await self._get_vector_rag_data(query, max_results, result)
            
            # Fallback to vector if no results and fallback enabled
            if not result.documents and self.enable_fallback:
                if strategy.strategy != Strategy.VECTOR:
                    result.fallback_used = True
                    await self._get_vector_rag_data(query, max_results, result)
                    
        except Exception as e:
            result.error = str(e)
            print(f"RAG retrieval error: {e}", file=sys.stderr)
            
            # Try vector fallback on error
            if self.enable_fallback and strategy.strategy != Strategy.VECTOR:
                try:
                    result.fallback_used = True
                    await self._get_vector_rag_data(query, max_results, result)
                except Exception as fallback_error:
                    result.error = f"Primary: {e}, Fallback: {fallback_error}"
        
        return result
    
    async def _get_db_rag_data(
        self, 
        query: str, 
        max_results: int, 
        result: RAGDataResult
    ):
        """
        Retrieve data using exact match / lookup strategy.
        
        Uses controller's storage for direct lookups by:
        - Note ID
        - Tag match
        - Type filter
        - Keyword exact match
        """
        storage = self.controller.storage
        entity = result.strategy.entity_name
        
        # Try lookup by ID first
        if entity and len(entity) > 8:  # UUIDs are longer
            try:
                note = storage.vector.get(entity)
                if note:
                    result.add_doc(RAGDocument(
                        id=note.id,
                        content=note.content,
                        source="db",
                        score=1.0,
                        metadata={
                            "type": note.type,
                            "tags": note.tags,
                            "summary": note.contextual_summary
                        }
                    ))
                    return
            except Exception:
                pass  # Not a valid ID, continue with search
        
        # Search by tag/type if entity looks like a filter
        all_notes = storage.vector.list_all()
        
        for note in all_notes[:max_results * 2]:  # Get extra for filtering
            score = 0.0
            
            # Match by tag
            if entity and note.tags and entity.lower() in [t.lower() for t in note.tags]:
                score = 0.9
            
            # Match by type
            if entity and note.type and entity.lower() == note.type.lower():
                score = 0.85
            
            # Match by keyword
            if entity and note.keywords:
                if entity.lower() in [k.lower() for k in note.keywords]:
                    score = 0.8
            
            # Content contains entity
            if entity and entity.lower() in note.content.lower():
                score = max(score, 0.6)
            
            if score > 0:
                result.add_doc(RAGDocument(
                    id=note.id,
                    content=note.content,
                    source="db",
                    score=score,
                    metadata={
                        "type": note.type,
                        "tags": note.tags,
                        "summary": note.contextual_summary
                    }
                ))
        
        # Sort by score and limit
        result.documents = sorted(
            result.documents, 
            key=lambda d: d.score, 
            reverse=True
        )[:max_results]
    
    async def _get_vector_rag_data(
        self, 
        query: str, 
        max_results: int, 
        result: RAGDataResult
    ):
        """
        Retrieve data using semantic similarity (vector search).
        
        Uses controller's retrieve method which does:
        - Embedding generation
        - ChromaDB similarity search
        - Priority scoring
        """
        results = await self.controller.retrieve(query, n_results=max_results)
        
        for res in results:
            note = res.note
            result.add_doc(RAGDocument(
                id=note.id,
                content=note.content,
                source="vector",
                score=res.score,
                metadata={
                    "type": note.type,
                    "tags": note.tags,
                    "summary": note.contextual_summary,
                    "priority_score": res.priority_score if hasattr(res, 'priority_score') else None
                }
            ))
    
    async def _get_graph_rag_data(
        self, 
        query: str, 
        max_results: int, 
        result: RAGDataResult
    ):
        """
        Retrieve data using graph traversal.
        
        For V3: This will use SPARQL queries against Apache Jena.
        For now (V2 compatibility): Uses NetworkX graph traversal.
        """
        storage = self.controller.storage
        graph = storage.graph
        entity = result.strategy.entity_name
        
        # Find starting node(s) by entity name
        start_nodes = []
        
        if entity:
            # Search for nodes matching entity
            for node_id in graph.G.nodes():
                note = storage.vector.get(node_id)
                if note:
                    # Check if entity matches
                    if entity.lower() in note.content.lower():
                        start_nodes.append(node_id)
                    elif note.keywords and entity.lower() in [k.lower() for k in note.keywords]:
                        start_nodes.append(node_id)
        
        # If no specific entity, use highest-degree nodes
        if not start_nodes:
            # Get top connected nodes
            degrees = list(graph.G.degree())
            degrees.sort(key=lambda x: x[1], reverse=True)
            start_nodes = [n[0] for n in degrees[:3]]
        
        # Traverse from start nodes
        visited = set()
        for start_id in start_nodes[:3]:  # Limit starting points
            self._traverse_graph(
                graph=graph,
                storage=storage,
                node_id=start_id,
                depth=2,
                visited=visited,
                result=result,
                max_results=max_results
            )
            
            if len(result.documents) >= max_results:
                break
        
        # Sort by score and limit
        result.documents = sorted(
            result.documents, 
            key=lambda d: d.score, 
            reverse=True
        )[:max_results]
    
    def _traverse_graph(
        self,
        graph,
        storage,
        node_id: str,
        depth: int,
        visited: set,
        result: RAGDataResult,
        max_results: int
    ):
        """
        Recursively traverse graph from a starting node.
        
        Args:
            graph: Graph storage
            storage: Full storage for note retrieval
            node_id: Current node ID
            depth: Remaining depth to traverse
            visited: Set of visited node IDs
            result: Result container
            max_results: Max documents to return
        """
        if depth <= 0 or node_id in visited or len(result.documents) >= max_results:
            return
        
        visited.add(node_id)
        
        # Get the note for this node
        note = storage.vector.get(node_id)
        if not note:
            return
        
        # Calculate score based on depth (closer = higher)
        base_score = 0.8 if depth == 2 else (0.6 if depth == 1 else 0.4)
        
        # Boost score by edge count
        edge_count = graph.G.degree(node_id) if node_id in graph.G else 0
        edge_boost = min(edge_count * 0.05, 0.2)
        
        result.add_doc(RAGDocument(
            id=note.id,
            content=note.content,
            source="graph",
            score=base_score + edge_boost,
            metadata={
                "type": note.type,
                "tags": note.tags,
                "summary": note.contextual_summary,
                "edge_count": edge_count,
                "traversal_depth": 2 - depth + 1
            }
        ))
        
        # Traverse neighbors
        if node_id in graph.G:
            for neighbor_id in graph.G.neighbors(node_id):
                self._traverse_graph(
                    graph=graph,
                    storage=storage,
                    node_id=neighbor_id,
                    depth=depth - 1,
                    visited=visited,
                    result=result,
                    max_results=max_results
                )
    
    # Convenience methods
    
    async def hybrid_search(
        self, 
        query: str, 
        keyword: Optional[str] = None,
        max_results: int = 10
    ) -> RAGDataResult:
        """
        Perform hybrid search combining vector + keyword matching.
        
        Args:
            query: Natural language query for vector search
            keyword: Optional keyword for filtering
            max_results: Maximum results to return
            
        Returns:
            RAGDataResult with combined results
        """
        # Get vector results
        result = await self.get_rag_data(
            query, 
            max_results=max_results * 2,
            strategy_override=Strategy.VECTOR
        )
        
        # Filter by keyword if provided
        if keyword:
            filtered = []
            for doc in result.documents:
                if keyword.lower() in doc.content.lower():
                    # Boost score for keyword match
                    doc.score = min(doc.score + 0.1, 1.0)
                    filtered.append(doc)
            result.documents = filtered[:max_results]
        
        return result
    
    async def route_query(self, query: str) -> StrategyResult:
        """
        Just determine the strategy without executing retrieval.
        
        Args:
            query: Natural language query
            
        Returns:
            StrategyResult with routing decision
        """
        return self.strategy_builder.determine(query)
