"""
Benchmark Script für Memory Enzymes Performance
Misst die Ausführungszeit der einzelnen Enzyme-Funktionen.
"""

import sys
import time
from pathlib import Path
import asyncio

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from a_mem.core.logic import MemoryController
from a_mem.utils.enzymes import prune_links, suggest_relations, digest_node, run_memory_enzymes
from a_mem.utils.llm import LLMService
from a_mem.utils.priority import log_event

async def benchmark_enzymes():
    """Benchmark für alle Enzyme-Funktionen."""
    print("\n" + "="*60)
    print("🔬 A-MEM Enzyme Performance Benchmark")
    print("="*60 + "\n")
    
    controller = MemoryController()
    graph_store = controller.storage.graph
    llm = controller.llm
    
    # Graph-Statistiken
    node_count = graph_store.graph.number_of_nodes()
    edge_count = graph_store.graph.number_of_edges()
    
    print(f"📊 Graph-Statistiken:")
    print(f"   Nodes: {node_count}")
    print(f"   Edges: {edge_count}")
    print()
    
    if node_count == 0:
        print("⚠️  Graph ist leer. Erstelle Test-Notes...")
        # Erstelle ein paar Test-Notes für Benchmark
        from a_mem.models.note import NoteInput
        for i in range(10):
            await controller.create_note(NoteInput(
                content=f"Test Note {i} für Benchmark: Dies ist eine Test-Notiz zur Performance-Messung der Enzyme.",
                source="benchmark"
            ))
        graph_store = controller.storage.graph
        node_count = graph_store.graph.number_of_nodes()
        edge_count = graph_store.graph.number_of_edges()
        print(f"✅ {node_count} Test-Notes erstellt\n")
    else:
        print(f"✅ Verwende echte Daten aus dem Graph ({node_count} Nodes, {edge_count} Edges)\n")
    
    results = {}
    
    # 1. Benchmark: prune_links
    print("🔬 Benchmark: prune_links")
    print("-" * 60)
    # Zähle Edges vorher
    edges_before = graph_store.graph.number_of_edges()
    print(f"   📊 Edges vorher: {edges_before}")
    
    start_time = time.perf_counter()
    pruned_count = prune_links(graph_store, max_age_days=90, min_weight=0.3)
    elapsed_time = time.perf_counter() - start_time
    
    # Zähle Edges nachher
    edges_after = graph_store.graph.number_of_edges()
    print(f"   📊 Edges nachher: {edges_after}")
    print(f"   ⏱️  Zeit: {elapsed_time*1000:.2f} ms")
    print(f"   ✂️  Entfernte Links: {pruned_count}")
    if pruned_count == 0:
        print(f"   ℹ️  Keine Links entfernt (alle Links sind neu/genug stark)")
    print()
    
    # 2. Benchmark: suggest_relations
    print("🔬 Benchmark: suggest_relations")
    print("-" * 60)
    # Sammle alle Notes für suggest_relations
    from a_mem.models.note import AtomicNote
    notes_dict = {}
    for node_id, attrs in graph_store.graph.nodes(data=True):
        try:
            notes_dict[node_id] = AtomicNote(**attrs)
        except Exception:
            continue
    
    start_time = time.perf_counter()
    suggestions = suggest_relations(
        notes_dict,
        graph_store, 
        llm, 
        threshold=0.75, 
        max_suggestions=10
    )
    elapsed_time = time.perf_counter() - start_time
    results['suggest_relations'] = {
        'time': elapsed_time,
        'suggestions_count': len(suggestions),
        'nodes': node_count,
        'edges': edge_count
    }
    print(f"   ⏱️  Zeit: {elapsed_time*1000:.2f} ms")
    print(f"   💡 Vorschläge: {len(suggestions)}")
    print()
    
    # 3. Benchmark: digest_node (nur für Nodes mit >8 Kindern)
    print("🔬 Benchmark: digest_node")
    print("-" * 60)
    # Finde Nodes mit vielen Kindern
    nodes_to_digest = []
    for node_id in graph_store.graph.nodes():
        children = list(graph_store.graph.successors(node_id))
        if len(children) > 8:
            child_notes = []
            for child_id in children:
                try:
                    child_notes.append(AtomicNote(**graph_store.graph.nodes[child_id]))
                except Exception:
                    continue
            if child_notes:
                nodes_to_digest.append((node_id, child_notes))
    
    print(f"   📊 Nodes mit >8 Kindern gefunden: {len(nodes_to_digest)}")
    if len(nodes_to_digest) == 0:
        print(f"   ℹ️  Keine Nodes zum Verdauen (kein Node hat >8 Kinder)")
        # Zeige max children count
        max_children = max([len(list(graph_store.graph.successors(n))) for n in graph_store.graph.nodes()], default=0)
        print(f"   ℹ️  Max Children pro Node: {max_children}")
    
    start_time = time.perf_counter()
    digested_count = 0
    for node_id, child_notes in nodes_to_digest[:1]:  # Nur ersten testen
        try:
            print(f"   🔄 Verdauen von Node {node_id[:8]}... ({len(child_notes)} Kinder)")
            digest_node(node_id, child_notes, llm)
            digested_count += 1
        except Exception as e:
            print(f"   ⚠️  Error digesting {node_id}: {e}")
    elapsed_time = time.perf_counter() - start_time
    results['digest_node'] = {
        'time': elapsed_time,
        'digested_count': digested_count,
        'nodes': node_count,
        'edges': edge_count
    }
    print(f"   ⏱️  Zeit: {elapsed_time*1000:.2f} ms")
    print(f"   📦 Verdaut: {digested_count} Nodes")
    print()
    
    # 4. Benchmark: run_memory_enzymes (Gesamt)
    print("🔬 Benchmark: run_memory_enzymes (Gesamt)")
    print("-" * 60)
    start_time = time.perf_counter()
    enzyme_results = run_memory_enzymes(
        graph_store,
        llm,
        prune_config={"max_age_days": 90, "min_weight": 0.3},
        suggest_config={"threshold": 0.75, "max_suggestions": 10}
    )
    elapsed_time = time.perf_counter() - start_time
    results['run_memory_enzymes'] = {
        'time': elapsed_time,
        'results': enzyme_results,
        'nodes': node_count,
        'edges': edge_count
    }
    print(f"   ⏱️  Zeit: {elapsed_time*1000:.2f} ms")
    print(f"   ✂️  Pruned: {enzyme_results['pruned_count']}")
    print(f"   💡 Suggested: {enzyme_results['suggestions_count']}")
    print(f"   📦 Digested: {enzyme_results['digested_count']}")
    print()
    
    # Zusammenfassung
    print("="*60)
    print("📊 Benchmark-Zusammenfassung")
    print("="*60)
    print()
    
    total_time = sum(r['time'] for r in results.values() if 'time' in r)
    
    print(f"{'Enzyme':<25} {'Zeit (ms)':<15} {'Zeit (s)':<15} {'Status'}")
    print("-" * 70)
    
    for name, data in results.items():
        time_ms = data['time'] * 1000
        time_s = data['time']
        status = "✅ OK" if time_s < 5.0 else "⚠️  LANG (>5s)" if time_s < 30.0 else "❌ SEHR LANG (>30s)"
        print(f"{name:<25} {time_ms:>12.2f} ms {time_s:>12.3f} s   {status}")
    
    print("-" * 70)
    print(f"{'GESAMT':<25} {total_time*1000:>12.2f} ms {total_time:>12.3f} s")
    print()
    
    # Empfehlung
    print("💡 Empfehlung:")
    if total_time < 1.0:
        print("   ✅ Performance ist ausgezeichnet. ProcessPoolExecutor ist nicht nötig.")
    elif total_time < 5.0:
        print("   ⚠️  Performance ist akzeptabel. ProcessPoolExecutor könnte helfen, ist aber optional.")
    elif total_time < 30.0:
        print("   ⚠️  Performance ist langsam. ProcessPoolExecutor wird empfohlen.")
    else:
        print("   ❌ Performance ist sehr langsam. ProcessPoolExecutor ist dringend empfohlen.")
    
    print()
    print("="*60)
    print("✅ Benchmark abgeschlossen")
    print("="*60 + "\n")
    
    return results

if __name__ == "__main__":
    asyncio.run(benchmark_enzymes())

