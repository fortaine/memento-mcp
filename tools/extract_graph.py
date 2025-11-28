"""
Einfaches Script: Ruft get_graph über HTTP vom laufenden Server auf
"""

import sys
import json
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from a_mem.config import settings

# Output-Datei
OUTPUT_FILE = settings.GRAPH_PATH

async def get_and_save():
    """Verbindet sich mit laufendem Server über HTTP und ruft get_graph auf."""
    from aiohttp import ClientSession
    
    host = settings.TCP_SERVER_HOST
    port = settings.TCP_SERVER_PORT
    url = f"http://{host}:{port}/get_graph"
    
    print(f"🔗 Verbinde mit Server {url}...")
    
    try:
        async with ClientSession() as session:
            print("🛠️  Rufe get_graph auf...")
            async with session.get(url) as response:
                if response.status != 200:
                    print(f"❌ HTTP Fehler: {response.status}")
                    return
                
                graph_data = await response.json()
                
                nodes = graph_data.get("nodes", [])
                edges = graph_data.get("edges", [])
                
                print(f"📊 Graph-Daten erhalten: {len(nodes)} nodes, {len(edges)} edges")
                print()
                
                if len(nodes) == 0:
                    print("⚠️  Graph ist leer. Nichts zu speichern.")
                    return
                
                # Konvertiere zu NetworkX node_link_data Format
                node_link_data = {
                    "directed": True,
                    "multigraph": False,
                    "graph": {},
                    "nodes": nodes,
                    "links": [
                        {
                            "source": edge["source"],
                            "target": edge["target"],
                            "type": edge.get("relation_type", "relates_to"),
                            "reasoning": edge.get("reasoning", ""),
                            "weight": edge.get("weight", 1.0),
                            "created_at": edge.get("created_at")
                        }
                        for edge in edges
                    ]
                }
                
                # Atomic write
                temp_path = OUTPUT_FILE.with_suffix(".tmp")
                
                try:
                    with open(temp_path, "w", encoding="utf-8") as f:
                        json.dump(node_link_data, f, ensure_ascii=False, indent=2)
                    
                    import os
                    os.replace(temp_path, OUTPUT_FILE)
                    
                    print(f"✅ Graph gespeichert: {OUTPUT_FILE}")
                    print(f"   Nodes: {len(nodes)}, Links: {len(edges)}")
                    
                except Exception as e:
                    print(f"❌ Fehler beim Speichern: {e}")
                    if temp_path.exists():
                        temp_path.unlink()
                    raise
                    
    except Exception as e:
        if "ConnectionRefusedError" in str(type(e)) or "Cannot connect" in str(e):
            print(f"❌ Verbindung verweigert!")
            print(f"   Stelle sicher, dass der A-MEM Server läuft und HTTP aktiviert ist:")
            print(f"   Setze TCP_SERVER_ENABLED=true in .env oder als Umgebungsvariable")
            print(f"   Server sollte auf http://{host}:{port}/get_graph laufen")
        else:
            raise

def main():
    """Hauptfunktion."""
    print("\n" + "="*60)
    print("📡 A-MEM Graph Extractor (HTTP)")
    print("="*60 + "\n")
    
    try:
        asyncio.run(get_and_save())
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ Fertig")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
