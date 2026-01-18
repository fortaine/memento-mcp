"""
StrategyBuilder: AI-Powered Intent Detection for OmniRAG

Determines the optimal data source (db/vector/graph) for each query.

Classification Hierarchy (fastest → most accurate):
1. Rule-based pattern matching (~0.1ms, no cost)
2. Embedding-based classifier (~50ms, embedding cost only)
3. LLM fallback (~500-2000ms, LLM cost) - optional

The embedding classifier uses cosine similarity against pre-computed
exemplar embeddings for each strategy, achieving ~90% accuracy without
requiring an LLM call.

Ported from CosmosAIGraph: https://github.com/AzureCosmosDB/CosmosAIGraph
"""

import re
import sys
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Literal, Tuple, List, Dict


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
    algorithm: Literal["rule", "embedding", "llm", "override"] = "rule"
    confidence: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            "natural_language": self.natural_language,
            "strategy": self.strategy.value,
            "entity_name": self.entity_name,
            "algorithm": self.algorithm,
            "confidence": self.confidence
        }


# ============================================
# Embedding-Based Classifier
# ============================================

# Exemplar queries for each strategy (used to build classifier)
STRATEGY_EXEMPLARS: Dict[Strategy, List[str]] = {
    Strategy.DB: [
        "lookup Flask",
        "find error pattern",
        "get note abc-123",
        "fetch all procedures",
        "list all rules",
        "search by tag python",
        "retrieve specific item",
        "show me note xyz",
        "get the configuration",
        "find notes with keyword test",
        "list items by type concept",
        "fetch record by id",
    ],
    Strategy.VECTOR: [
        "what is memory evolution?",
        "how to brief a task?",
        "explain the A-MEM pattern",
        "describe compound learning",
        "summarize the memento architecture",
        "what does this code do?",
        "help me understand enzymes",
        "tell me about Zettelkasten",
        "why use vector search?",
        "meaning of OmniRAG pattern",
        "overview of the system",
        "concept of knowledge graph",
    ],
    Strategy.GRAPH: [
        "what is related to memento-v2?",
        "dependencies of the LLM service",
        "show connections between A and B",
        "notes that lead to this concept",
        "hierarchy of note types",
        "what supports this rule?",
        "what contradicts this pattern?",
        "find path from X to Y",
        "parent notes of this concept",
        "child dependencies",
        "network of related concepts",
        "graph structure analysis",
    ],
}


class EmbeddingClassifier:
    """
    Lightweight classifier using embedding similarity.
    
    Classifies queries by comparing their embeddings against
    pre-computed exemplar embeddings for each strategy.
    
    ~50ms latency, embedding cost only (no LLM tokens).
    ~90% accuracy on typical queries.
    """
    
    def __init__(self, embedding_service=None):
        """
        Initialize classifier.
        
        Args:
            embedding_service: Service with get_embedding(text) and 
                             get_embeddings_batch(texts) methods.
                             If None, classifier is disabled.
        """
        self.embedding_service = embedding_service
        self._exemplar_embeddings: Dict[Strategy, List[List[float]]] = {}
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazy initialization of exemplar embeddings using batch API."""
        if self._initialized or not self.embedding_service:
            return
        
        try:
            import time
            start = time.time()
            print("Initializing embedding classifier...", file=sys.stderr)
            
            # Collect all exemplars with their strategy mapping
            all_exemplars = []
            strategy_mapping = []  # Track which strategy each exemplar belongs to
            
            for strategy, exemplars in STRATEGY_EXEMPLARS.items():
                for exemplar in exemplars:
                    all_exemplars.append(exemplar)
                    strategy_mapping.append(strategy)
            
            # Batch embed all exemplars in one API call (~500ms vs ~7s)
            if hasattr(self.embedding_service, 'get_embeddings_batch'):
                all_embeddings = self.embedding_service.get_embeddings_batch(all_exemplars)
            else:
                # Fallback to sequential if batch not available
                all_embeddings = [
                    self.embedding_service.get_embedding(e) for e in all_exemplars
                ]
            
            # Distribute embeddings back to their strategies
            for strategy in STRATEGY_EXEMPLARS.keys():
                self._exemplar_embeddings[strategy] = []
            
            for i, (strategy, embedding) in enumerate(zip(strategy_mapping, all_embeddings)):
                self._exemplar_embeddings[strategy].append(embedding)
            
            elapsed = (time.time() - start) * 1000
            total_exemplars = sum(len(v) for v in self._exemplar_embeddings.values())
            print(f"Classifier initialized: {total_exemplars} exemplars in {elapsed:.0f}ms", file=sys.stderr)
            self._initialized = True
            
        except Exception as e:
            print(f"Failed to initialize classifier: {e}", file=sys.stderr)
            self._initialized = False
    
    def classify(self, query: str, min_confidence: float = 0.6) -> Tuple[Optional[Strategy], float]:
        """
        Classify a query using embedding similarity.
        
        Args:
            query: The query to classify
            min_confidence: Minimum similarity score to return a result
            
        Returns:
            Tuple of (Strategy, confidence) or (None, 0) if not confident
        """
        if not self.embedding_service:
            return (None, 0.0)
        
        self._ensure_initialized()
        
        if not self._initialized:
            return (None, 0.0)
        
        try:
            # Get query embedding
            query_emb = self.embedding_service.get_embedding(query)
            
            # Calculate similarity to each strategy's exemplars
            strategy_scores: Dict[Strategy, float] = {}
            
            for strategy, exemplar_embs in self._exemplar_embeddings.items():
                # Average similarity to all exemplars of this strategy
                similarities = [
                    self._cosine_similarity(query_emb, ex_emb) 
                    for ex_emb in exemplar_embs
                ]
                # Use max similarity (best match) rather than average
                strategy_scores[strategy] = max(similarities) if similarities else 0.0
            
            # Find best strategy
            best_strategy = max(strategy_scores, key=lambda s: strategy_scores[s])
            best_score = strategy_scores[best_strategy]
            
            # Check confidence threshold
            if best_score >= min_confidence:
                return (best_strategy, best_score)
            
            return (None, best_score)
            
        except Exception as e:
            print(f"Embedding classification error: {e}", file=sys.stderr)
            return (None, 0.0)
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        a_arr = np.array(a)
        b_arr = np.array(b)
        
        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot / (norm_a * norm_b))


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
        r"\b(related|connected|linked|connects?)\s+(to|with)?\b",  # Relationships
        r"\bdependenc(y|ies)\b",                                   # Dependencies
        r"\b(path|route|connection)\s+between\b",                  # Paths
        r"\b(hierarchy|tree|network|graph)\b",                     # Structure
        r"\b(parent|child|sibling|ancestor|descendant)\b",         # Tree relations
        r"\bleads?\s+to\b",                                        # Edge types
        r"\bsupports?\b",
        r"\bcontradicts?\b",
        r"\bsupersedes?\b",
        r"\blinked\s+\w+",                                         # "linked concepts"
        r"\brelation(ship)?s?\b",                                  # "relationships"
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

    def __init__(
        self, 
        llm_service=None, 
        use_embedding_classifier: bool = True,
        use_llm_fallback: bool = False,
        classifier_confidence_threshold: float = 0.65
    ):
        """
        Initialize StrategyBuilder.
        
        Args:
            llm_service: Service with get_embedding() and optionally _call_llm().
            use_embedding_classifier: Use embedding-based classification (recommended).
            use_llm_fallback: Use LLM as final fallback (expensive, high accuracy).
            classifier_confidence_threshold: Min similarity for embedding classifier.
        
        Classification Order:
        1. Rule-based patterns (always, ~0.1ms)
        2. Embedding classifier (if enabled, ~50ms) 
        3. LLM fallback (if enabled, ~500-2000ms)
        4. Default to vector (if all else fails)
        """
        self.llm = llm_service
        self.use_embedding_classifier = use_embedding_classifier
        self.use_llm_fallback = use_llm_fallback
        self.classifier_threshold = classifier_confidence_threshold
        
        # Initialize pattern matchers
        self._compiled_db = [re.compile(p, re.IGNORECASE) for p in self.DB_PATTERNS]
        self._compiled_graph = [re.compile(p, re.IGNORECASE) for p in self.GRAPH_PATTERNS]
        self._compiled_vector = [re.compile(p, re.IGNORECASE) for p in self.VECTOR_PATTERNS]
        
        # Initialize embedding classifier (lazy loading)
        self._embedding_classifier: Optional[EmbeddingClassifier] = None
        if use_embedding_classifier and llm_service:
            self._embedding_classifier = EmbeddingClassifier(llm_service)
    
    def determine(self, natural_language: str) -> StrategyResult:
        """
        Determine the optimal retrieval strategy for a query.
        
        Classification hierarchy:
        1. Rule-based patterns → Fast, no cost
        2. Embedding classifier → Fast, embedding cost only  
        3. LLM fallback → Slow, LLM token cost
        4. Default to vector → Safest fallback
        
        Args:
            natural_language: The user's query in natural language.
            
        Returns:
            StrategyResult with strategy, entity name, and algorithm used.
        """
        text = natural_language.strip()
        
        # Phase 1: Fast rule-based detection (~0.1ms)
        strategy, entity, confidence = self._check_rules(text)
        if strategy:
            return StrategyResult(
                natural_language=text,
                strategy=strategy,
                entity_name=entity,
                algorithm="rule",
                confidence=confidence
            )
        
        # Phase 2: Embedding classifier (~50ms, embedding cost only)
        if self._embedding_classifier:
            strategy, confidence = self._embedding_classifier.classify(
                text, 
                min_confidence=self.classifier_threshold
            )
            if strategy:
                return StrategyResult(
                    natural_language=text,
                    strategy=strategy,
                    entity_name=self._extract_entity(text),
                    algorithm="embedding",
                    confidence=confidence
                )
        
        # Phase 3: LLM fallback (~500-2000ms, LLM cost)
        if self.use_llm_fallback and self.llm:
            strategy = self._classify_with_llm(text)
            return StrategyResult(
                natural_language=text,
                strategy=strategy,
                entity_name=self._extract_entity(text),
                algorithm="llm",
                confidence=0.85
            )
        
        # Phase 4: Default to vector search (safest for unknown queries)
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
        
        Order matters: Check GRAPH first (more specific relationship keywords),
        then DB (action verbs), then VECTOR (question words).
        
        Returns:
            Tuple of (strategy, entity_name, confidence) or (None, None, 0)
        """
        # Check Graph patterns FIRST (most specific - relationship keywords)
        for pattern in self._compiled_graph:
            if pattern.search(text):
                entity = self._extract_entity(text)
                return (Strategy.GRAPH, entity, 0.95)
        
        # Check DB patterns (action verbs for direct lookups)
        for pattern in self._compiled_db:
            if pattern.search(text):
                entity = self._extract_entity(text)
                return (Strategy.DB, entity, 0.95)
        
        # Check Vector patterns (question words, semantic)
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
