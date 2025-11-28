"""
Memory Enzymes: Autonome Hintergrund-Prozesse für Graph-Pflege

KISS-Approach: Kleine, unabhängige Module die den Graph automatisch optimieren.
- prune_links: Entfernt alte/schwache Links
- suggest_relations: Schlägt neue Verbindungen vor
- digest_node: Komprimiert überfüllte Nodes
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ..models.note import AtomicNote, NoteRelation
from ..storage.engine import GraphStore
from ..utils.llm import LLMService
from ..utils.priority import log_event


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Berechnet Cosine Similarity zwischen zwei Embeddings."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot_product = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def prune_links(
    graph: GraphStore,
    max_age_days: int = 90,
    min_weight: float = 0.3,
    min_usage: int = 0
) -> int:
    """
    Entfernt schwache oder alte Kanten aus dem Graph.
    
    Args:
        graph: GraphStore Instanz
        max_age_days: Maximale Alter in Tagen (default: 90)
        min_weight: Minimale Edge-Weight (default: 0.3)
        min_usage: Minimale Usage-Count (default: 0, da wir das noch nicht tracken)
    
    Returns:
        Anzahl entfernte Edges
    """
    now = datetime.utcnow()
    to_remove = []
    
    for source, target, data in graph.graph.edges(data=True):
        should_remove = False
        
        # CRITICAL: Orphaned Edge Check - Entferne Edges zu nicht existierenden Nodes
        if source not in graph.graph.nodes or target not in graph.graph.nodes:
            to_remove.append((source, target))
            log_event("RELATION_PRUNED", {
                "source": source,
                "target": target,
                "reason": "orphaned_edge_missing_node"
            })
            continue  # Skip weitere Checks, Edge wird entfernt
        
        # CRITICAL: Zombie Node Check - Entferne Edges zu Nodes ohne Content (gelöschte Nodes)
        source_node = graph.graph.nodes[source]
        target_node = graph.graph.nodes[target]
        
        # Prüfe ob Nodes leer sind (keine Keys) oder kein Content haben
        source_is_empty = len(source_node) == 0 or "content" not in source_node
        target_is_empty = len(target_node) == 0 or "content" not in target_node
        
        if not source_is_empty:
            source_content = source_node.get("content", "")
            source_has_content = source_content and len(str(source_content).strip()) > 0
        else:
            source_has_content = False
        
        if not target_is_empty:
            target_content = target_node.get("content", "")
            target_has_content = target_content and len(str(target_content).strip()) > 0
        else:
            target_has_content = False
        
        if not source_has_content or not target_has_content:
            to_remove.append((source, target))
            log_event("RELATION_PRUNED", {
                "source": source,
                "target": target,
                "reason": "orphaned_edge_zombie_node",
                "source_is_empty": source_is_empty,
                "target_is_empty": target_is_empty,
                "source_has_content": source_has_content,
                "target_has_content": target_has_content
            })
            continue  # Skip weitere Checks, Edge wird entfernt
        
        # Weight-Check: Schwache Verbindungen entfernen
        weight = data.get("weight", 1.0)
        if weight < min_weight:
            should_remove = True
        
        # Age-Check: Alte Verbindungen entfernen
        if "created_at" in data:
            try:
                # created_at ist als ISO string gespeichert
                edge_time = datetime.fromisoformat(data["created_at"])
                age_days = (now - edge_time).days
                if age_days > max_age_days:
                    should_remove = True
            except (ValueError, TypeError):
                # Fallback: Prüfe Node-Alter wenn Edge kein created_at hat
                if source in graph.graph.nodes and target in graph.graph.nodes:
                    source_node = graph.graph.nodes[source]
                    target_node = graph.graph.nodes[target]
                    
                    # Wenn beide Nodes alt sind und Edge schwach ist → entfernen
                    source_created = source_node.get("created_at")
                    target_created = target_node.get("created_at")
                    
                    if isinstance(source_created, str):
                        try:
                            source_created = datetime.fromisoformat(source_created)
                        except ValueError:
                            source_created = None
                    
                    if isinstance(target_created, str):
                        try:
                            target_created = datetime.fromisoformat(target_created)
                        except ValueError:
                            target_created = None
                    
                    if source_created and target_created:
                        source_age = (now - source_created).days
                        target_age = (now - target_created).days
                        if source_age > max_age_days and target_age > max_age_days and weight < 0.5:
                            should_remove = True
        
        if should_remove:
            to_remove.append((source, target))
    
    # Entferne Edges
    for source, target in to_remove:
        graph.graph.remove_edge(source, target)
    
    if to_remove:
        log_event("LINKS_PRUNED", {
            "count": len(to_remove),
            "max_age_days": max_age_days,
            "min_weight": min_weight
        })
    
    return len(to_remove)


def prune_zombie_nodes(graph: GraphStore) -> int:
    """
    Entfernt Zombie-Nodes (Nodes ohne Content) aus dem Graph.
    
    Args:
        graph: GraphStore Instanz
    
    Returns:
        Anzahl entfernte Nodes
    """
    to_remove = []
    
    for node_id in list(graph.graph.nodes()):
        node_data = graph.graph.nodes[node_id]
        
        # Prüfe ob Node leer ist (keine Keys) oder kein Content hat
        is_empty = len(node_data) == 0 or "content" not in node_data
        
        if not is_empty:
            content = node_data.get("content", "")
            has_content = content and len(str(content).strip()) > 0
        else:
            has_content = False
        
        if not has_content:
            to_remove.append(node_id)
            log_event("NODE_PRUNED", {
                "node_id": node_id,
                "reason": "zombie_node_no_content",
                "is_empty": is_empty
            })
    
    # Entferne Nodes
    for node_id in to_remove:
        graph.graph.remove_node(node_id)
    
    if to_remove:
        log_event("ZOMBIE_NODES_PRUNED", {
            "count": len(to_remove),
            "node_ids": to_remove[:10]  # Nur erste 10 loggen
        })
    
    return len(to_remove)


def suggest_relations(
    notes: Dict[str, AtomicNote],
    graph: GraphStore,
    llm_service: LLMService,
    threshold: float = 0.75,
    max_suggestions: int = 10
) -> List[Tuple[str, str, float]]:
    """
    Schlägt neue Beziehungen zwischen Notes vor basierend auf semantischer Ähnlichkeit.
    
    Args:
        notes: Dict von note_id -> AtomicNote
        llm_service: LLMService für Embedding-Berechnung
        threshold: Minimale Similarity (default: 0.75)
        max_suggestions: Maximale Anzahl Vorschläge (default: 10)
    
    Returns:
        Liste von (source_id, target_id, similarity) Tupeln
    """
    if len(notes) < 2:
        return []
    
    suggestions = []
    note_ids = list(notes.keys())
    
    # Embeddings einmal berechnen
    vectors = {}
    for note_id, note in notes.items():
        text = f"{note.content} {note.contextual_summary} {' '.join(note.keywords)}"
        embedding = llm_service.get_embedding(text)
        vectors[note_id] = embedding
    
    # Paarweiser Vergleich (nur wenn nicht bereits verbunden)
    for i in range(len(note_ids)):
        for j in range(i + 1, len(note_ids)):
            if len(suggestions) >= max_suggestions:
                break
            
            a_id, b_id = note_ids[i], note_ids[j]
            
            note_a = notes[a_id]
            note_b = notes[b_id]
            
            # Prüfe ob bereits verbunden
            if graph.graph.has_edge(a_id, b_id) or graph.graph.has_edge(b_id, a_id):
                continue  # Bereits verbunden, Skip
            
            # Pre-Filter: Wenn keine gemeinsamen Keywords/Tags → Skip
            common_keywords = set(note_a.keywords) & set(note_b.keywords)
            common_tags = set(note_a.tags) & set(note_b.tags)
            
            if not common_keywords and not common_tags:
                continue  # Zu unterschiedlich, Skip
            
            # Cosine Similarity berechnen
            similarity = cosine_similarity(vectors[a_id], vectors[b_id])
            
            if similarity >= threshold:
                suggestions.append((a_id, b_id, similarity))
    
    # Sortiere nach Similarity (höchste zuerst)
    suggestions.sort(key=lambda x: x[2], reverse=True)
    
    return suggestions[:max_suggestions]


def digest_node(
    node_id: str,
    child_notes: List[AtomicNote],
    llm_service: LLMService,
    max_children: int = 8
) -> Optional[str]:
    """
    Wenn ein Node zu viele Kinder hat, erzeugt eine kompakte Zusammenfassung.
    
    Args:
        node_id: ID des überfüllten Nodes
        child_notes: Liste der Child-Notes
        llm_service: LLMService für Zusammenfassung
        max_children: Maximale Anzahl Children bevor Digest nötig ist
    
    Returns:
        Zusammenfassungstext oder None wenn nicht nötig
    """
    if len(child_notes) <= max_children:
        return None
    
    # Sammle Content aller Children
    texts = "\n\n---\n\n".join([
        f"[{note.id}] {note.content}\nSummary: {note.contextual_summary}\nKeywords: {', '.join(note.keywords)}"
        for note in child_notes
    ])
    
    prompt = f"""Fasse folgende {len(child_notes)} Notizen prägnant zusammen.
Ziel: Eine abstrahierte, verdichtete Meta-Note die die Essenz aller Notizen erfasst.

Notizen:
{texts}

Erstelle eine kompakte Zusammenfassung (max 200 Wörter) die:
1. Die Hauptthemen zusammenfasst
2. Gemeinsame Patterns identifiziert
3. Wichtige Details bewahrt
4. Redundanzen eliminiert

Zusammenfassung:"""
    
    try:
        summary = llm_service._call_llm(prompt)
        log_event("NODE_DIGESTED", {
            "node_id": node_id,
            "children_count": len(child_notes),
            "summary_length": len(summary)
        })
        return summary
    except Exception as e:
        print(f"Digest Error für Node {node_id}: {e}")
        return None


def run_memory_enzymes(
    graph: GraphStore,
    llm_service: LLMService,
    prune_config: Optional[Dict[str, Any]] = None,
    suggest_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Führt alle Memory-Enzyme aus.
    
    Args:
        graph: GraphStore Instanz
        llm_service: LLMService Instanz
        prune_config: Config für prune_links (optional)
        suggest_config: Config für suggest_relations (optional)
    
    Returns:
        Dict mit Ergebnissen
    """
    results = {
        "pruned_count": 0,
        "zombie_nodes_removed": 0,
        "suggestions_count": 0,
        "digested_count": 0
    }
    
    # 1. Prune Links
    prune_params = prune_config or {}
    results["pruned_count"] = prune_links(graph, **prune_params)
    
    # 1.5. Prune Zombie Nodes (nach Links, damit keine orphaned Edges entstehen)
    results["zombie_nodes_removed"] = prune_zombie_nodes(graph)
    
    # 2. Suggest Relations
    # Sammle alle Notes
    notes = {}
    for node_id in graph.graph.nodes():
        node_data = graph.graph.nodes[node_id]
        try:
            # Versuche AtomicNote zu erstellen
            note = AtomicNote(**node_data)
            notes[node_id] = note
        except Exception:
            continue  # Skip invalid nodes
    
    if len(notes) >= 2:
        suggest_params = suggest_config or {}
        suggestions = suggest_relations(notes, graph, llm_service, **suggest_params)
        results["suggestions_count"] = len(suggestions)
        
        # Logge Suggestions (aber füge sie nicht automatisch hinzu - User entscheidet)
        if suggestions:
            log_event("RELATIONS_SUGGESTED", {
                "count": len(suggestions),
                "suggestions": [
                    {"from": s[0], "to": s[1], "similarity": s[2]}
                    for s in suggestions[:5]  # Nur erste 5 loggen
                ]
            })
    
    # 3. Digest Nodes (optional, für später)
    # Finde Nodes mit vielen Children
    for node_id in graph.graph.nodes():
        neighbors = graph.get_neighbors(node_id)
        if len(neighbors) > 8:  # max_children default
            # Konvertiere zu AtomicNote Liste
            child_notes = []
            for neighbor_data in neighbors:
                try:
                    child_note = AtomicNote(**neighbor_data)
                    child_notes.append(child_note)
                except Exception:
                    continue
            
            if child_notes:
                summary = digest_node(node_id, child_notes, llm_service)
                if summary:
                    results["digested_count"] += 1
    
    return results

