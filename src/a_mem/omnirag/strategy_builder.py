"""
StrategyBuilder: AI-Powered Intent Detection for OmniRAG

Determines the optimal data source (db/vector/graph) for each query.
Uses a fast rule-based path for simple queries, with LLM fallback for complex ones.

Ported from CosmosAIGraph: https://github.com/AzureCosmosDB/CosmosAIGraph
"""

import re
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Literal, Tuple


class Strategy(str, Enum):
    """Data source strategies for OmniRAG."""
    DB = "db"           # Exact match, lookups, fetch by ID
    VECTOR = "vector"   # Semantic similarity, open-ended questions
    GRAPH = "graph"     # Relationship traversal, dependencies


@dataclass
class StrategyResult:
    """Result of strategy determination."""
    natural_language: str
    strategy: Strategy
    entity_name: Optional[str] = None
    algorithm: Literal["rule", "llm"] = "rule"
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            "natural_language": self.natural_language,
            "strategy": self.strategy.value,
            "entity_name": self.entity_name,
            "algorithm": self.algorithm,
            "confidence": self.confidence
        }


class StrategyBuilder:
    """
    Determines the optimal retrieval strategy for OmniRAG queries.
    
    Strategy Selection Matrix:
    - db:     lookup, find, fetch, get by id, search by tag
    - vector: what is, how to, similar to, semantic questions
    - graph:  related to, dependencies, connections, hierarchy
    
    Uses rule-based fast path first (no LLM cost), then LLM fallback.
    """
    
    # Rule-based patterns for fast detection (no LLM needed)
    DB_PATTERNS = [
        r"^(lookup|find|fetch|search|get|retrieve|show|list)\s+",  # Action verbs
        r"\bby\s+(id|tag|type|name|keyword)\b",                     # By field
        r"\b(exact|specific|particular)\b",                         # Exact match
        r"^get\s+note\s+",                                          # Get note X
        r"\b(all|every|each)\s+\w+s?\b",                           # List all
    ]
    
    GRAPH_PATTERNS = [
        r"\b(related|connected|linked)\s+to\b",                    # Relationships
        r"\bdependenc(y|ies)\b",                                   # Dependencies
        r"\b(path|route|connection)\s+between\b",                  # Paths
        r"\b(hierarchy|tree|network|graph)\b",                     # Structure
        r"\b(parent|child|sibling|ancestor|descendant)\b",         # Tree relations
        r"\bleads?\s+to\b",                                        # Edge types
        r"\bsupports?\b",
        r"\bcontradicts?\b",
        r"\bsupersedes?\b",
    ]
    
    VECTOR_PATTERNS = [
        r"^(what|how|why|when|where|who|explain|describe)\s+",     # Questions
        r"\bsimilar\s+to\b",                                       # Similarity
        r"\blike\s+this\b",                                        # Like
        r"\b(meaning|definition|concept)\b",                       # Semantics
        r"\b(summarize|summary|overview)\b",                       # Summaries
    ]
    
    # System prompt for LLM classification (when rules fail)
    LLM_CLASSIFICATION_PROMPT = """You are a query router for a memory system. Classify the data source for this query.

Options:
- db: For exact lookups, finding by ID/tag/type, listing items, fetching specific records
- vector: For semantic similarity, open-ended questions, "what is", "how to", conceptual queries
- graph: For relationship queries, dependencies, connections between entities, hierarchies

Examples:
- "lookup Flask" → db
- "get note abc-123" → db
- "list all procedures" → db
- "what is memory evolution?" → vector
- "how to brief a task?" → vector
- "explain the A-MEM pattern" → vector
- "what is related to memento-v2?" → graph
- "dependencies of the LLM service" → graph
- "show connections between A and B" → graph

Return ONLY one word: db, vector, or graph"""

    def __init__(self, llm_service=None):
        """
        Initialize StrategyBuilder.
        
        Args:
            llm_service: Optional LLM service for fallback classification.
                        If None, uses rule-based only (faster but less accurate).
        """
        self.llm = llm_service
        self._compiled_db = [re.compile(p, re.IGNORECASE) for p in self.DB_PATTERNS]
        self._compiled_graph = [re.compile(p, re.IGNORECASE) for p in self.GRAPH_PATTERNS]
        self._compiled_vector = [re.compile(p, re.IGNORECASE) for p in self.VECTOR_PATTERNS]
    
    def determine(self, natural_language: str) -> StrategyResult:
        """
        Determine the optimal retrieval strategy for a query.
        
        Args:
            natural_language: The user's query in natural language.
            
        Returns:
            StrategyResult with strategy, entity name, and algorithm used.
        """
        text = natural_language.strip()
        
        # Phase 1: Fast rule-based detection
        strategy, entity, confidence = self._check_rules(text)
        if strategy:
            return StrategyResult(
                natural_language=text,
                strategy=strategy,
                entity_name=entity,
                algorithm="rule",
                confidence=confidence
            )
        
        # Phase 2: LLM classification (if available)
        if self.llm:
            strategy = self._classify_with_llm(text)
            return StrategyResult(
                natural_language=text,
                strategy=strategy,
                entity_name=self._extract_entity(text),
                algorithm="llm",
                confidence=0.85  # LLM is confident but not 100%
            )
        
        # Default: Vector search (safest for unknown queries)
        return StrategyResult(
            natural_language=text,
            strategy=Strategy.VECTOR,
            entity_name=self._extract_entity(text),
            algorithm="rule",
            confidence=0.5  # Low confidence = used default
        )
    
    def _check_rules(self, text: str) -> Tuple[Optional[Strategy], Optional[str], float]:
        """
        Apply rule-based pattern matching.
        
        Returns:
            Tuple of (strategy, entity_name, confidence) or (None, None, 0)
        """
        # Check DB patterns
        for pattern in self._compiled_db:
            if pattern.search(text):
                entity = self._extract_entity(text)
                return (Strategy.DB, entity, 0.95)
        
        # Check Graph patterns
        for pattern in self._compiled_graph:
            if pattern.search(text):
                entity = self._extract_entity(text)
                return (Strategy.GRAPH, entity, 0.95)
        
        # Check Vector patterns
        for pattern in self._compiled_vector:
            if pattern.search(text):
                entity = self._extract_entity(text)
                return (Strategy.VECTOR, entity, 0.90)
        
        return (None, None, 0.0)
    
    def _classify_with_llm(self, text: str) -> Strategy:
        """
        Use LLM to classify the query when rules fail.
        
        Args:
            text: The query text
            
        Returns:
            Strategy enum value
        """
        try:
            # Use LOW thinking for classification (fast)
            if hasattr(self.llm, '_call_google'):
                response = self.llm._call_google(
                    prompt=text,
                    system=self.LLM_CLASSIFICATION_PROMPT,
                    thinking_level="LOW"
                )
            else:
                response = self.llm._call_llm(
                    prompt=text,
                    system=self.LLM_CLASSIFICATION_PROMPT
                )
            
            return self._normalize_strategy(response)
        except Exception as e:
            # On error, default to vector
            import sys
            print(f"LLM classification error: {e}", file=sys.stderr)
            return Strategy.VECTOR
    
    def _normalize_strategy(self, raw: str) -> Strategy:
        """
        Normalize LLM output to valid Strategy enum.
        
        Args:
            raw: Raw LLM response
            
        Returns:
            Strategy enum value
        """
        text = str(raw).strip().lower()
        
        if "graph" in text:
            return Strategy.GRAPH
        if "db" in text or "database" in text or "lookup" in text:
            return Strategy.DB
        if "vector" in text or "embedding" in text or "semantic" in text:
            return Strategy.VECTOR
        
        # Default to vector (safest)
        return Strategy.VECTOR
    
    def _extract_entity(self, text: str) -> Optional[str]:
        """
        Extract the main entity/subject from the query.
        
        Args:
            text: The query text
            
        Returns:
            Entity name or None
        """
        # Remove common action words
        action_words = [
            "lookup", "find", "fetch", "search", "get", "retrieve", "show", "list",
            "what", "how", "why", "when", "where", "who",
            "is", "are", "the", "a", "an", "this", "that",
            "related", "to", "connected", "with", "between"
        ]
        
        words = text.lower().split()
        
        # Filter out action words
        entities = [w for w in words if w not in action_words and len(w) > 2]
        
        if entities:
            # Return the last significant word (often the target)
            return entities[-1]
        
        return None
    
    # Convenience methods for common patterns
    
    def is_db_query(self, text: str) -> bool:
        """Check if query should use DB strategy."""
        result = self.determine(text)
        return result.strategy == Strategy.DB
    
    def is_vector_query(self, text: str) -> bool:
        """Check if query should use Vector strategy."""
        result = self.determine(text)
        return result.strategy == Strategy.VECTOR
    
    def is_graph_query(self, text: str) -> bool:
        """Check if query should use Graph strategy."""
        result = self.determine(text)
        return result.strategy == Strategy.GRAPH
