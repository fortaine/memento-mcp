"""
Apache Jena Migration Plan and OWL Ontology for Memento V3

This module defines:
1. The Memento ontology in OWL/RDF format
2. SPARQL query templates for common operations
3. Migration utilities from NetworkX to Jena
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


# ============================================
# Memento Ontology (Turtle/OWL Format)
# ============================================

MEMENTO_ONTOLOGY_TURTLE = """
@prefix memo: <http://memento.ai/ontology#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# ============================================
# Ontology Metadata
# ============================================

<http://memento.ai/ontology> rdf:type owl:Ontology ;
    rdfs:label "Memento Memory Ontology"@en ;
    rdfs:comment "OWL ontology for the Memento agentic memory system based on A-MEM research."@en ;
    owl:versionInfo "3.0.0" .

# ============================================
# Classes (Note Types)
# ============================================

memo:Note rdf:type owl:Class ;
    rdfs:label "Note"@en ;
    rdfs:comment "An atomic unit of knowledge in the Memento system"@en .

memo:Rule rdf:type owl:Class ;
    rdfs:subClassOf memo:Note ;
    rdfs:label "Rule"@en ;
    rdfs:comment "Imperative instructions: Never X, Always Y"@en .

memo:Procedure rdf:type owl:Class ;
    rdfs:subClassOf memo:Note ;
    rdfs:label "Procedure"@en ;
    rdfs:comment "Numbered steps or sequential instructions"@en .

memo:Concept rdf:type owl:Class ;
    rdfs:subClassOf memo:Note ;
    rdfs:label "Concept"@en ;
    rdfs:comment "Explanatory content, definitions, no commands"@en .

memo:Tool rdf:type owl:Class ;
    rdfs:subClassOf memo:Note ;
    rdfs:label "Tool"@en ;
    rdfs:comment "Functions, APIs, or utilities"@en .

memo:Reference rdf:type owl:Class ;
    rdfs:subClassOf memo:Note ;
    rdfs:label "Reference"@en ;
    rdfs:comment "Tables, comparison lists, cheatsheets"@en .

memo:Integration rdf:type owl:Class ;
    rdfs:subClassOf memo:Note ;
    rdfs:label "Integration"@en ;
    rdfs:comment "Connections between systems"@en .

# ============================================
# Object Properties (Edge Types)
# ============================================

memo:leadsTo rdf:type owl:ObjectProperty ;
    rdfs:domain memo:Note ;
    rdfs:range memo:Note ;
    rdfs:label "leads to"@en ;
    rdfs:comment "Indicates a logical or temporal progression"@en .

memo:dependsOn rdf:type owl:ObjectProperty ;
    rdfs:domain memo:Note ;
    rdfs:range memo:Note ;
    rdfs:label "depends on"@en ;
    rdfs:comment "Indicates a dependency relationship"@en .

memo:contradicts rdf:type owl:ObjectProperty ;
    rdfs:domain memo:Note ;
    rdfs:range memo:Note ;
    rdfs:label "contradicts"@en ;
    rdfs:comment "Indicates conflicting information"@en ;
    owl:inverseOf memo:contradicts .  # Symmetric

memo:supports rdf:type owl:ObjectProperty ;
    rdfs:domain memo:Note ;
    rdfs:range memo:Note ;
    rdfs:label "supports"@en ;
    rdfs:comment "Indicates supporting evidence or reinforcement"@en .

memo:supersedes rdf:type owl:ObjectProperty ;
    rdfs:domain memo:Note ;
    rdfs:range memo:Note ;
    rdfs:label "supersedes"@en ;
    rdfs:comment "Indicates an updated version replacing older content"@en .

memo:partOf rdf:type owl:ObjectProperty ;
    rdfs:domain memo:Note ;
    rdfs:range memo:Note ;
    rdfs:label "part of"@en ;
    rdfs:comment "Indicates a part-whole relationship"@en ;
    owl:inverseOf memo:hasPart .

memo:hasPart rdf:type owl:ObjectProperty ;
    rdfs:domain memo:Note ;
    rdfs:range memo:Note ;
    rdfs:label "has part"@en ;
    owl:inverseOf memo:partOf .

memo:relatedTo rdf:type owl:ObjectProperty ;
    rdfs:domain memo:Note ;
    rdfs:range memo:Note ;
    rdfs:label "related to"@en ;
    rdfs:comment "Generic relationship between notes"@en ;
    owl:inverseOf memo:relatedTo .  # Symmetric

# ============================================
# Data Properties (Note Attributes)
# ============================================

memo:content rdf:type owl:DatatypeProperty ;
    rdfs:domain memo:Note ;
    rdfs:range xsd:string ;
    rdfs:label "content"@en .

memo:contextualSummary rdf:type owl:DatatypeProperty ;
    rdfs:domain memo:Note ;
    rdfs:range xsd:string ;
    rdfs:label "contextual summary"@en .

memo:hasKeyword rdf:type owl:DatatypeProperty ;
    rdfs:domain memo:Note ;
    rdfs:range xsd:string ;
    rdfs:label "has keyword"@en .

memo:hasTag rdf:type owl:DatatypeProperty ;
    rdfs:domain memo:Note ;
    rdfs:range xsd:string ;
    rdfs:label "has tag"@en .

memo:createdAt rdf:type owl:DatatypeProperty ;
    rdfs:domain memo:Note ;
    rdfs:range xsd:dateTime ;
    rdfs:label "created at"@en .

memo:updatedAt rdf:type owl:DatatypeProperty ;
    rdfs:domain memo:Note ;
    rdfs:range xsd:dateTime ;
    rdfs:label "updated at"@en .

memo:accessCount rdf:type owl:DatatypeProperty ;
    rdfs:domain memo:Note ;
    rdfs:range xsd:integer ;
    rdfs:label "access count"@en .

memo:priorityScore rdf:type owl:DatatypeProperty ;
    rdfs:domain memo:Note ;
    rdfs:range xsd:float ;
    rdfs:label "priority score"@en .

memo:edgeWeight rdf:type owl:DatatypeProperty ;
    rdfs:domain owl:ObjectProperty ;
    rdfs:range xsd:float ;
    rdfs:label "edge weight"@en .

memo:source rdf:type owl:DatatypeProperty ;
    rdfs:domain memo:Note ;
    rdfs:range xsd:string ;
    rdfs:label "source"@en .
"""


# ============================================
# SPARQL Query Templates
# ============================================

SPARQL_TEMPLATES = {
    # Get all related notes (1 hop)
    "get_related": """
        PREFIX memo: <http://memento.ai/ontology#>
        
        SELECT ?related ?relationshipType ?weight
        WHERE {
            memo:$NOTE_ID ?relationshipType ?related .
            OPTIONAL { ?relationshipType memo:edgeWeight ?weight }
            FILTER (STRSTARTS(STR(?relationshipType), "http://memento.ai/ontology#"))
        }
    """,
    
    # Get dependencies (transitive)
    "get_dependencies": """
        PREFIX memo: <http://memento.ai/ontology#>
        
        SELECT ?dependency ?depth
        WHERE {
            {
                SELECT ?dependency (1 AS ?depth)
                WHERE {
                    memo:$NOTE_ID memo:dependsOn ?dependency .
                }
            }
            UNION
            {
                SELECT ?dependency (2 AS ?depth)
                WHERE {
                    memo:$NOTE_ID memo:dependsOn ?intermediate .
                    ?intermediate memo:dependsOn ?dependency .
                }
            }
        }
        ORDER BY ?depth
    """,
    
    # Find notes by type
    "find_by_type": """
        PREFIX memo: <http://memento.ai/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT ?note ?content ?summary
        WHERE {
            ?note rdf:type memo:$NOTE_TYPE .
            ?note memo:content ?content .
            OPTIONAL { ?note memo:contextualSummary ?summary }
        }
        LIMIT $LIMIT
    """,
    
    # Find notes by keyword
    "find_by_keyword": """
        PREFIX memo: <http://memento.ai/ontology#>
        
        SELECT ?note ?content
        WHERE {
            ?note memo:hasKeyword ?keyword .
            FILTER (LCASE(?keyword) = LCASE("$KEYWORD"))
            ?note memo:content ?content .
        }
    """,
    
    # Get connection path between two notes
    "find_path": """
        PREFIX memo: <http://memento.ai/ontology#>
        
        SELECT ?intermediate ?rel1 ?rel2
        WHERE {
            memo:$NOTE_A ?rel1 ?intermediate .
            ?intermediate ?rel2 memo:$NOTE_B .
            FILTER (STRSTARTS(STR(?rel1), "http://memento.ai/ontology#"))
            FILTER (STRSTARTS(STR(?rel2), "http://memento.ai/ontology#"))
        }
        LIMIT 10
    """,
    
    # Get notes that contradict
    "find_contradictions": """
        PREFIX memo: <http://memento.ai/ontology#>
        
        SELECT ?note1 ?note2 ?content1 ?content2
        WHERE {
            ?note1 memo:contradicts ?note2 .
            ?note1 memo:content ?content1 .
            ?note2 memo:content ?content2 .
        }
    """,
    
    # Get notes superseded (outdated)
    "find_superseded": """
        PREFIX memo: <http://memento.ai/ontology#>
        
        SELECT ?oldNote ?newNote ?oldContent
        WHERE {
            ?newNote memo:supersedes ?oldNote .
            ?oldNote memo:content ?oldContent .
        }
    """,
    
    # Get graph statistics
    "graph_stats": """
        PREFIX memo: <http://memento.ai/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        
        SELECT 
            (COUNT(DISTINCT ?note) AS ?noteCount)
            (COUNT(?rel) AS ?edgeCount)
        WHERE {
            ?note rdf:type/rdfs:subClassOf* memo:Note .
            OPTIONAL { ?note ?rel ?other . FILTER (STRSTARTS(STR(?rel), "http://memento.ai/ontology#")) }
        }
    """,
    
    # Insert a note with type
    "insert_note": """
        PREFIX memo: <http://memento.ai/ontology#>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        
        INSERT DATA {
            memo:$NOTE_ID rdf:type memo:$NOTE_TYPE ;
                memo:content "$CONTENT" ;
                memo:contextualSummary "$SUMMARY" ;
                memo:createdAt "$CREATED_AT"^^xsd:dateTime .
        }
    """,
    
    # Insert an edge
    "insert_edge": """
        PREFIX memo: <http://memento.ai/ontology#>
        
        INSERT DATA {
            memo:$SOURCE_ID memo:$EDGE_TYPE memo:$TARGET_ID .
        }
    """,
    
    # Delete a note
    "delete_note": """
        PREFIX memo: <http://memento.ai/ontology#>
        
        DELETE WHERE {
            memo:$NOTE_ID ?p ?o .
        };
        DELETE WHERE {
            ?s ?p memo:$NOTE_ID .
        }
    """
}


# ============================================
# Migration Plan: NetworkX → Jena
# ============================================

@dataclass
class MigrationPlan:
    """
    Migration plan from NetworkX to Apache Jena.
    
    Phases:
    1. Deploy Jena microservice (Docker or Azure Container App)
    2. Export NetworkX graph to RDF/Turtle format
    3. Load RDF into Jena
    4. Implement SPARQL query generation
    5. Dual-write mode (both backends)
    6. Read from Jena with NetworkX fallback
    7. Full cutover
    """
    
    phase: int
    description: str
    steps: List[str]
    estimated_hours: int
    dependencies: List[str]
    
    @classmethod
    def get_all_phases(cls) -> List["MigrationPlan"]:
        """Get all migration phases."""
        return [
            cls(
                phase=1,
                description="Deploy Apache Jena Microservice",
                steps=[
                    "Create Dockerfile for Jena Fuseki (SPARQL endpoint)",
                    "Configure TDB2 for persistent storage",
                    "Deploy to Azure Container App or local Docker",
                    "Set up health checks and monitoring",
                    "Configure CORS for API access"
                ],
                estimated_hours=4,
                dependencies=[]
            ),
            cls(
                phase=2,
                description="Define and Load Ontology",
                steps=[
                    "Finalize memento ontology (MEMENTO_ONTOLOGY_TURTLE)",
                    "Load ontology into Jena dataset",
                    "Verify ontology with test queries",
                    "Document ontology for team"
                ],
                estimated_hours=2,
                dependencies=["Phase 1"]
            ),
            cls(
                phase=3,
                description="Implement SPARQL Generation",
                steps=[
                    "Create NL2SPARQL service using LLM",
                    "Implement template-based SPARQL for common patterns",
                    "Add query validation and sanitization",
                    "Test with sample queries"
                ],
                estimated_hours=6,
                dependencies=["Phase 2"]
            ),
            cls(
                phase=4,
                description="Export NetworkX to RDF",
                steps=[
                    "Create export function: NetworkX → Turtle",
                    "Map edge types to ontology properties",
                    "Handle node attributes → data properties",
                    "Validate exported RDF"
                ],
                estimated_hours=4,
                dependencies=["Phase 2"]
            ),
            cls(
                phase=5,
                description="Dual-Write Implementation",
                steps=[
                    "Wrap graph operations to write both backends",
                    "Implement async Jena writes (non-blocking)",
                    "Add error handling for Jena failures",
                    "Monitor consistency between backends"
                ],
                estimated_hours=4,
                dependencies=["Phase 3", "Phase 4"]
            ),
            cls(
                phase=6,
                description="Read Preference Migration",
                steps=[
                    "Implement Jena-first read with NetworkX fallback",
                    "Add metrics for read source tracking",
                    "Validate result consistency",
                    "Gradually increase Jena read percentage"
                ],
                estimated_hours=4,
                dependencies=["Phase 5"]
            ),
            cls(
                phase=7,
                description="Full Cutover",
                steps=[
                    "Stop NetworkX writes",
                    "Remove fallback to NetworkX",
                    "Archive NetworkX data",
                    "Update documentation"
                ],
                estimated_hours=2,
                dependencies=["Phase 6"]
            )
        ]


# ============================================
# Graph Exporter (NetworkX → RDF)
# ============================================

class NetworkXToRDFExporter:
    """
    Export NetworkX graph to RDF/Turtle format for Jena import.
    """
    
    NAMESPACE = "http://memento.ai/ontology#"
    
    EDGE_TYPE_MAPPING = {
        "LEADS_TO": "leadsTo",
        "DEPENDS_ON": "dependsOn",
        "CONTRADICTS": "contradicts",
        "SUPPORTS": "supports",
        "SUPERSEDES": "supersedes",
        "PART_OF": "partOf",
        # Default fallback
        None: "relatedTo"
    }
    
    NOTE_TYPE_MAPPING = {
        "rule": "Rule",
        "procedure": "Procedure",
        "concept": "Concept",
        "tool": "Tool",
        "reference": "Reference",
        "integration": "Integration",
        # Default fallback
        None: "Note"
    }
    
    def __init__(self, graph, vector_storage):
        """
        Initialize exporter.
        
        Args:
            graph: NetworkX graph from memento storage
            vector_storage: Vector storage for note metadata
        """
        self.graph = graph
        self.vector_storage = vector_storage
    
    def export_to_turtle(self) -> str:
        """
        Export entire graph to Turtle format.
        
        Returns:
            RDF/Turtle string
        """
        lines = [
            "# Memento Graph Export",
            f"# Generated: {__import__('datetime').datetime.now().isoformat()}",
            "",
            f"@prefix memo: <{self.NAMESPACE}> .",
            "@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            ""
        ]
        
        # Export nodes
        for node_id in self.graph.G.nodes():
            note = self.vector_storage.get(node_id)
            if note:
                lines.extend(self._export_note(node_id, note))
        
        # Export edges
        for source, target, data in self.graph.G.edges(data=True):
            lines.extend(self._export_edge(source, target, data))
        
        return "\n".join(lines)
    
    def _export_note(self, node_id: str, note) -> List[str]:
        """Export a single note to Turtle triples."""
        lines = []
        
        # Determine note type
        note_type = self.NOTE_TYPE_MAPPING.get(
            note.type, 
            self.NOTE_TYPE_MAPPING[None]
        )
        
        # Escape content for Turtle
        content = self._escape_string(note.content)
        summary = self._escape_string(note.contextual_summary or "")
        
        lines.append(f"memo:{node_id}")
        lines.append(f"    rdf:type memo:{note_type} ;")
        lines.append(f'    memo:content "{content}" ;')
        
        if summary:
            lines.append(f'    memo:contextualSummary "{summary}" ;')
        
        # Keywords
        for keyword in (note.keywords or []):
            lines.append(f'    memo:hasKeyword "{self._escape_string(keyword)}" ;')
        
        # Tags
        for tag in (note.tags or []):
            lines.append(f'    memo:hasTag "{self._escape_string(tag)}" ;')
        
        # Source
        if note.source:
            lines.append(f'    memo:source "{self._escape_string(note.source)}" ;')
        
        # Created timestamp
        if hasattr(note, 'created_at') and note.created_at:
            lines.append(f'    memo:createdAt "{note.created_at}"^^xsd:dateTime ;')
        
        # Remove trailing semicolon, add period
        lines[-1] = lines[-1].rstrip(" ;") + " ."
        lines.append("")
        
        return lines
    
    def _export_edge(
        self, 
        source: str, 
        target: str, 
        data: dict
    ) -> List[str]:
        """Export a single edge to Turtle triple."""
        edge_type = data.get("type") or data.get("relation_type")
        rdf_property = self.EDGE_TYPE_MAPPING.get(
            edge_type,
            self.EDGE_TYPE_MAPPING[None]
        )
        
        return [f"memo:{source} memo:{rdf_property} memo:{target} ."]
    
    def _escape_string(self, s: str) -> str:
        """Escape string for Turtle literal."""
        if not s:
            return ""
        return (
            s.replace("\\", "\\\\")
             .replace('"', '\\"')
             .replace("\n", "\\n")
             .replace("\r", "\\r")
             .replace("\t", "\\t")
        )


# ============================================
# Jena Service Client (Future Implementation)
# ============================================

@dataclass
class JenaConfig:
    """Configuration for Apache Jena Fuseki connection."""
    endpoint: str = "http://localhost:3030"
    dataset: str = "memento"
    username: Optional[str] = None
    password: Optional[str] = None
    timeout_seconds: int = 30
    
    @property
    def query_url(self) -> str:
        return f"{self.endpoint}/{self.dataset}/query"
    
    @property
    def update_url(self) -> str:
        return f"{self.endpoint}/{self.dataset}/update"
    
    @property
    def data_url(self) -> str:
        return f"{self.endpoint}/{self.dataset}/data"


class JenaService:
    """
    Client for Apache Jena Fuseki SPARQL endpoint.
    
    NOTE: This is a placeholder for V3 implementation.
    The actual client will be implemented when Jena is deployed.
    """
    
    def __init__(self, config: JenaConfig):
        self.config = config
    
    async def query(self, sparql: str) -> Dict[str, Any]:
        """
        Execute a SPARQL query.
        
        Args:
            sparql: SPARQL query string
            
        Returns:
            Query results as dict
        """
        # TODO: Implement actual HTTP POST to Fuseki
        raise NotImplementedError("Jena client not yet implemented")
    
    async def update(self, sparql: str) -> bool:
        """
        Execute a SPARQL UPDATE.
        
        Args:
            sparql: SPARQL UPDATE string
            
        Returns:
            True if successful
        """
        # TODO: Implement actual HTTP POST to Fuseki
        raise NotImplementedError("Jena client not yet implemented")
    
    async def load_turtle(self, turtle: str) -> bool:
        """
        Load RDF/Turtle data into the dataset.
        
        Args:
            turtle: Turtle format RDF data
            
        Returns:
            True if successful
        """
        # TODO: Implement actual HTTP POST to Fuseki data endpoint
        raise NotImplementedError("Jena client not yet implemented")
    
    async def health_check(self) -> bool:
        """Check if Jena is available."""
        # TODO: Implement actual health check
        raise NotImplementedError("Jena client not yet implemented")


# ============================================
# NL2SPARQL Service
# ============================================

class NL2SPARQLService:
    """
    Convert natural language to SPARQL queries.
    
    Uses template matching for common patterns, LLM for complex queries.
    """
    
    PROMPT_TEMPLATE = """You are a SPARQL query generator for a memory knowledge graph.

Ontology summary:
- Classes: Note (subtypes: Rule, Procedure, Concept, Tool, Reference, Integration)
- Object Properties: leadsTo, dependsOn, contradicts, supports, supersedes, partOf, relatedTo
- Data Properties: content, contextualSummary, hasKeyword, hasTag, createdAt

Namespace: memo: <http://memento.ai/ontology#>

Generate a SPARQL query for this request: {query}

Return ONLY the SPARQL query, no explanation."""
    
    def __init__(self, llm_service=None):
        self.llm = llm_service
        self.templates = SPARQL_TEMPLATES
    
    def generate_sparql(self, natural_language: str) -> str:
        """
        Generate SPARQL from natural language.
        
        Args:
            natural_language: User's query in natural language
            
        Returns:
            SPARQL query string
        """
        # Try template matching first
        template_match = self._match_template(natural_language)
        if template_match:
            return template_match
        
        # Fall back to LLM generation
        if self.llm:
            return self._generate_with_llm(natural_language)
        
        # If no LLM, return generic query
        return self._generic_search_query(natural_language)
    
    def _match_template(self, text: str) -> Optional[str]:
        """Try to match a template-based query."""
        text_lower = text.lower()
        
        # Related notes
        if "related to" in text_lower:
            # Extract entity
            import re
            match = re.search(r"related to\s+(\w+)", text_lower)
            if match:
                entity = match.group(1)
                return self.templates["get_related"].replace("$NOTE_ID", entity)
        
        # Dependencies
        if "dependencies" in text_lower or "depends on" in text_lower:
            match = re.search(r"(?:dependencies of|depends on)\s+(\w+)", text_lower)
            if match:
                entity = match.group(1)
                return self.templates["get_dependencies"].replace("$NOTE_ID", entity)
        
        # Find by type
        for note_type in ["rule", "procedure", "concept", "tool", "reference"]:
            if note_type in text_lower:
                return (
                    self.templates["find_by_type"]
                    .replace("$NOTE_TYPE", note_type.capitalize())
                    .replace("$LIMIT", "20")
                )
        
        # Graph stats
        if "statistics" in text_lower or "how many" in text_lower:
            return self.templates["graph_stats"]
        
        return None
    
    def _generate_with_llm(self, text: str) -> str:
        """Generate SPARQL using LLM."""
        prompt = self.PROMPT_TEMPLATE.format(query=text)
        
        try:
            if hasattr(self.llm, '_call_google'):
                response = self.llm._call_google(prompt, thinking_level="LOW")
            else:
                response = self.llm._call_llm(prompt)
            
            # Clean up response
            sparql = response.strip()
            if sparql.startswith("```"):
                sparql = sparql.split("```")[1]
                if sparql.startswith("sparql"):
                    sparql = sparql[6:]
            
            return sparql.strip()
        except Exception:
            return self._generic_search_query(text)
    
    def _generic_search_query(self, text: str) -> str:
        """Generate a generic search query."""
        # Extract potential keywords
        words = [w for w in text.split() if len(w) > 3]
        keyword = words[0] if words else "note"
        
        return f"""
        PREFIX memo: <http://memento.ai/ontology#>
        
        SELECT ?note ?content
        WHERE {{
            ?note memo:content ?content .
            FILTER (CONTAINS(LCASE(?content), LCASE("{keyword}")))
        }}
        LIMIT 10
        """
