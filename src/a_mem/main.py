"""
MCP Server for A-MEM: Agentic Memory System

Complete MCP Server with tools for Memory Management.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Sequence
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from .core.logic import MemoryController
from .models.note import NoteInput
from .utils.enzymes import run_memory_enzymes
from .config import settings

# Server initialisieren
server = Server("a-mem")
controller = MemoryController()

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Lists all available tools."""
    return [
        Tool(
            name="create_atomic_note",
            description="Stores a new piece of information in the memory system. Automatically classifies the note type (rule, procedure, concept, tool, reference, integration), extracts metadata, and starts the linking and evolution process in the background.",
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The text of the note/memory to be stored."
                    },
                    "source": {
                        "type": "string",
                        "description": "Source of the information (e.g., 'user_input', 'file', 'api').",
                        "default": "user_input"
                    }
                },
                "required": ["content"]
            }
        ),
        Tool(
            name="retrieve_memories",
            description="Searches for relevant memories based on semantic similarity with priority scoring. Returns the best matches with linked contexts, ranked by combined similarity and priority score.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for the memory search."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5).",
                        "default": 5,
                        "minimum": 1,
                        "maximum": 20
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_memory_stats",
            description="Returns statistics about the memory system (number of nodes, edges, etc.).",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="add_file",
            description="Stores the content of a file (e.g., .md) as a note in the memory system. Supports automatic chunking for large files (>16KB).",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to be stored (relative or absolute)."
                    },
                    "file_content": {
                        "type": "string",
                        "description": "Alternatively: Direct file content as string (when file_path is not provided)."
                    },
                    "chunk_size": {
                        "type": "integer",
                        "description": "Maximum size per chunk in bytes (default: 15000, to stay under 16KB limit).",
                        "default": 15000,
                        "minimum": 1000,
                        "maximum": 16384
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="reset_memory",
            description="Resets the complete memory system (Graph + Vector Store). Deletes all notes, edges, and embeddings. WARNING: This action cannot be undone!",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="list_notes",
            description="Lists all stored notes from the memory graph.",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="get_note",
            description="Returns a single note (metadata + content) by id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "string",
                        "description": "The UUID of the note."
                    }
                },
                "required": ["note_id"]
            }
        ),
        Tool(
            name="update_note",
            description="Updates contextual summary, tags or keywords for an existing note.",
            inputSchema={
                "type": "object",
                "properties": {
                    "note_id": {"type": "string"},
                    "data": {
                        "type": "object",
                        "description": "Fields to update (contextual_summary, tags, keywords)."
                    }
                },
                "required": ["note_id", "data"]
            }
        ),
        Tool(
            name="delete_atomic_note",
            description="Deletes a note from the memory system. Removes the note from Graph and Vector Store as well as all associated connections.",
            inputSchema={
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "string",
                        "description": "The UUID of the note to be deleted."
                    }
                },
                "required": ["note_id"]
            }
        ),
        Tool(
            name="list_relations",
            description="Lists relations in the memory graph, optionally filtered by note id.",
            inputSchema={
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "string",
                        "description": "Optional note id filter"
                    }
                }
            }
        ),
        Tool(
            name="add_relation",
            description="Adds a manual relation between two notes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "relation_type": {"type": "string"},
                    "reasoning": {"type": "string"},
                    "weight": {"type": "number"}
                },
                "required": ["source_id", "target_id"]
            }
        ),
        Tool(
            name="remove_relation",
            description="Removes a relation between two notes.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"}
                },
                "required": ["source_id", "target_id"]
            }
        ),
        Tool(
            name="get_graph",
            description="Returns the full graph snapshot (nodes + edges). Optionally saves the graph to disk for visualization tools.",
            inputSchema={
                "type": "object",
                "properties": {
                    "save": {
                        "type": "boolean",
                        "description": "If true, saves the graph to disk before returning (useful for visualization tools).",
                        "default": False
                    }
                }
            }
        ),
        Tool(
            name="run_memory_enzymes",
            description="Runs memory maintenance enzymes: prunes old/weak links, suggests new relations, and digests overcrowded nodes. Automatically optimizes the graph structure.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prune_max_age_days": {
                        "type": "integer",
                        "description": "Maximum age in days for edges to be pruned (default: 90).",
                        "default": 90
                    },
                    "prune_min_weight": {
                        "type": "number",
                        "description": "Minimum weight for edges to be kept (default: 0.3).",
                        "default": 0.3
                    },
                    "suggest_threshold": {
                        "type": "number",
                        "description": "Minimum similarity threshold for relation suggestions (default: 0.75).",
                        "default": 0.75
                    },
                    "suggest_max": {
                        "type": "integer",
                        "description": "Maximum number of relation suggestions (default: 10).",
                        "default": 10
                    }
                }
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> Sequence[TextContent]:
    """Executes a tool."""
    
    if name == "create_atomic_note":
        content = arguments.get("content", "")
        source = arguments.get("source", "user_input")
        
        if not content:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "content is required"}, indent=2)
            )]
        
        try:
            note_input = NoteInput(content=content, source=source)
            note_id = await controller.create_note(note_input)
            
            result = {
                "status": "success",
                "note_id": note_id,
                "message": f"Note created with ID: {note_id}. Evolution started in background."
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]
    
    elif name == "retrieve_memories":
        query = arguments.get("query", "")
        max_results = arguments.get("max_results", 5)
        
        if not query:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "query is required"}, indent=2)
            )]
        
        try:
            results = await controller.retrieve(query)
            
            # Limit results
            results = results[:max_results]
            
            output = []
            for res in results:
                context_str = ", ".join([f"[{rn.id}] {rn.contextual_summary}" for rn in res.related_notes])
                output.append({
                    "id": res.note.id,
                    "content": res.note.content,
                    "summary": res.note.contextual_summary,
                    "keywords": res.note.keywords,
                    "tags": res.note.tags,
                    "type": res.note.type,  # Node classification
                    "relevance_score": float(res.score),  # Combined similarity × priority
                    "connected_memories": len(res.related_notes),
                    "connected_context": context_str
                })
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "query": query,
                    "results_count": len(output),
                    "results": output
                }, indent=2, ensure_ascii=False)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]
    
    elif name == "get_memory_stats":
        try:
            graph = controller.storage.graph.graph
            stats = {
                "status": "success",
                "graph_nodes": graph.number_of_nodes(),
                "graph_edges": graph.number_of_edges(),
                "memory_count": graph.number_of_nodes(),
                "connection_count": graph.number_of_edges()
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(stats, indent=2)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]
    
    elif name == "add_file":
        file_path = arguments.get("file_path", "")
        file_content = arguments.get("file_content", "")
        chunk_size = arguments.get("chunk_size", 15000)
        
        # Check if file_path or file_content is provided
        if not file_path and not file_content:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Either file_path or file_content is required"}, indent=2)
            )]
        
        try:
            # Read file if file_path is provided
            if file_path:
                path = Path(file_path)
                if not path.exists():
                    return [TextContent(
                        type="text",
                        text=json.dumps({"error": f"File not found: {file_path}"}, indent=2)
                    )]
                
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    source = f"file:{path.name}"
                except UnicodeDecodeError:
                    # Fallback for binary files
                    with open(path, 'rb') as f:
                        content_bytes = f.read()
                    file_content = content_bytes.decode('utf-8', errors='replace')
                    source = f"file:{path.name}"
            else:
                source = "file:direct_content"
            
            # Check size and chunk if necessary
            content_bytes = file_content.encode('utf-8')
            file_size = len(content_bytes)
            
            if file_size <= chunk_size:
                # File fits in one note
                note_input = NoteInput(content=file_content, source=source)
                note_id = await controller.create_note(note_input)
                
                result = {
                    "status": "success",
                    "note_id": note_id,
                    "file_size": file_size,
                    "chunks": 1,
                    "message": f"File stored as single note with ID: {note_id}. Evolution started in background."
                }
            else:
                # Chunking required
                chunks = []
                chunk_count = (file_size + chunk_size - 1) // chunk_size
                
                for i in range(chunk_count):
                    start = i * chunk_size
                    end = min(start + chunk_size, file_size)
                    chunk_content = content_bytes[start:end].decode('utf-8', errors='replace')
                    
                    # Add chunk info
                    chunk_header = f"[Chunk {i+1}/{chunk_count} from {source}]\n\n"
                    chunk_note_content = chunk_header + chunk_content
                    
                    note_input = NoteInput(
                        content=chunk_note_content,
                        source=f"{source}:chunk_{i+1}"
                    )
                    note_id = await controller.create_note(note_input)
                    chunks.append(note_id)
                
                result = {
                    "status": "success",
                    "file_size": file_size,
                    "chunks": chunk_count,
                    "note_ids": chunks,
                    "message": f"File split into {chunk_count} chunks. All notes created. Evolution started in background."
                }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2, ensure_ascii=False)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]
    
    elif name == "reset_memory":
        try:
            success = await controller.reset_memory()
            
            if success:
                result = {
                    "status": "success",
                    "message": "Memory system reset successfully. All notes, edges, and embeddings have been deleted."
                }
            else:
                result = {
                    "status": "error",
                    "message": "Failed to reset memory system. Check logs for details."
                }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]
    
    elif name == "list_notes":
        notes = await controller.list_notes_data()
        return [TextContent(
            type="text",
            text=json.dumps({"notes": notes}, indent=2, ensure_ascii=False)
        )]
    
    elif name == "get_note":
        note_id = arguments.get("note_id", "")
        if not note_id:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "note_id is required"}, indent=2)
            )]
        note = await controller.get_note_data(note_id)
        if not note:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Note '{note_id}' not found"}, indent=2)
            )]
        return [TextContent(
            type="text",
            text=json.dumps({"note": note}, indent=2, ensure_ascii=False)
        )]
    
    elif name == "update_note":
        note_id = arguments.get("note_id", "")
        data = arguments.get("data", {})
        if not note_id:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "note_id is required"}, indent=2)
            )]
        if not isinstance(data, dict):
            return [TextContent(
                type="text",
                text=json.dumps({"error": "data must be an object"}, indent=2)
            )]
        result = await controller.update_note_metadata(note_id, data)
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, ensure_ascii=False)
        )]
    
    elif name == "delete_atomic_note":
        note_id = arguments.get("note_id", "")
        if not note_id:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "note_id is required"}, indent=2)
            )]
        result = await controller.delete_note_data(note_id)
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "list_relations":
        note_id = arguments.get("note_id")
        relations = await controller.list_relations_data(note_id)
        return [TextContent(
            type="text",
            text=json.dumps({"relations": relations}, indent=2, ensure_ascii=False)
        )]
    
    elif name == "add_relation":
        source = arguments.get("source_id", "")
        target = arguments.get("target_id", "")
        relation_type = arguments.get("relation_type", "relates_to")
        reasoning = arguments.get("reasoning", "Manual link")
        weight = float(arguments.get("weight", 1.0))
        if not source or not target:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "source_id and target_id are required"}, indent=2)
            )]
        result = await controller.add_relation(source, target, relation_type, reasoning, weight)
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, ensure_ascii=False)
        )]
    
    elif name == "remove_relation":
        source = arguments.get("source_id", "")
        target = arguments.get("target_id", "")
        if not source or not target:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "source_id and target_id are required"}, indent=2)
            )]
        result = await controller.remove_relation(source, target)
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2)
        )]
    
    elif name == "get_graph":
        save_to_disk = arguments.get("save", False)
        
        # Get graph snapshot first to see what we have
        graph = await controller.get_graph_snapshot()
        node_count = len(graph.get("nodes", []))
        edge_count = len(graph.get("edges", []))
        
        # Log to file
        from pathlib import Path
        from datetime import datetime
        log_file = Path(settings.DATA_DIR) / "graph_save.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        def write_log(msg):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {msg}\n")
            print(msg)
        
        write_log(f"[INFO] [get_graph] Graph snapshot: {node_count} nodes, {edge_count} edges")
        write_log(f"[INFO] [get_graph] Memory graph: {controller.storage.graph.graph.number_of_nodes()} nodes, {controller.storage.graph.graph.number_of_edges()} edges")
        
        # If save is requested, save the graph to disk
        if save_to_disk:
            loop = asyncio.get_running_loop()
            write_log(f"[SAVE] [get_graph] Saving graph to disk...")
            await loop.run_in_executor(None, controller.storage.graph.save_snapshot)
            write_log(f"[SAVE] [get_graph] Save completed")
            
            # Verify after save
            if settings.GRAPH_PATH.exists():
                with open(settings.GRAPH_PATH, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                    saved_nodes = len(saved_data.get("nodes", []))
                    saved_links = len(saved_data.get("links", []))
                    write_log(f"[OK] [get_graph] Verified saved file: {saved_nodes} nodes, {saved_links} links")
                    if saved_nodes == 0 and node_count > 0:
                        write_log(f"[ERROR] [get_graph] ERROR: Graph had {node_count} nodes but saved file has 0 nodes!")
        
        result = {
            "nodes": graph.get("nodes", []),
            "edges": graph.get("edges", []),
            "saved_to_disk": save_to_disk,
            "memory_nodes": controller.storage.graph.graph.number_of_nodes(),
            "memory_edges": controller.storage.graph.graph.number_of_edges()
        }
        
        return [TextContent(
            type="text",
            text=json.dumps(result, indent=2, ensure_ascii=False)
        )]
    
    elif name == "run_memory_enzymes":
        try:
            prune_max_age = arguments.get("prune_max_age_days", 90)
            prune_min_weight = arguments.get("prune_min_weight", 0.3)
            suggest_threshold = arguments.get("suggest_threshold", 0.75)
            suggest_max = arguments.get("suggest_max", 10)
            
            # Run enzymes synchronously (in thread)
            loop = asyncio.get_running_loop()
            
            def _run_enzymes():
                return run_memory_enzymes(
                    controller.storage.graph,
                    controller.llm,
                    prune_config={
                        "max_age_days": prune_max_age,
                        "min_weight": prune_min_weight
                    },
                    suggest_config={
                        "threshold": suggest_threshold,
                        "max_suggestions": suggest_max
                    }
                )
            
            results = await loop.run_in_executor(None, _run_enzymes)
            
            # Log manual enzyme run (use the same log_event from priority module)
            from .utils.priority import log_event
            log_event("ENZYME_MANUAL_RUN", {
                "results": results,
                "prune_max_age_days": prune_max_age,
                "prune_min_weight": prune_min_weight,
                "suggest_threshold": suggest_threshold,
                "suggest_max": suggest_max
            })
            
            # Save graph after enzymes
            await loop.run_in_executor(None, controller.storage.graph.save_snapshot)
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "success",
                    "results": results,
                    "message": f"Enzymes completed: {results['pruned_count']} links pruned, {results.get('zombie_nodes_removed', 0)} zombie nodes removed, {results['suggestions_count']} relations suggested, {results['digested_count']} nodes digested."
                }, indent=2)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)}, indent=2)
            )]
    
    else:
        return [TextContent(
            type="text",
            text=json.dumps({"error": f"Unknown tool: {name}"}, indent=2)
        )]

async def main():
    """Main function for the MCP Server."""
    # Initial save: Save graph to disk on startup so visualizer can access it
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, controller.storage.graph.save_snapshot)
    print("[SAVE] Initial graph snapshot saved to disk")
    
    # Starte automatischen Enzyme-Scheduler (alle 1 Stunde)
    # Auto-Save alle 5 Minuten, damit Visualizer die Daten sehen kann
    controller.start_enzyme_scheduler(interval_hours=1.0, auto_save_interval_minutes=5.0)
    
    # Starte HTTP-Server parallel (wenn enabled) - nutzt GLEICHE controller-Instanz
    http_task = None
    if settings.TCP_SERVER_ENABLED:
        from aiohttp import web
        
        async def get_graph_handler(request):
            """HTTP Handler für get_graph"""
            graph_data = await controller.get_graph_snapshot()
            return web.json_response(graph_data)
        
        async def run_http_server():
            app = web.Application()
            app.router.add_get('/get_graph', get_graph_handler)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, settings.TCP_SERVER_HOST, settings.TCP_SERVER_PORT)
            await site.start()
            print(f"[HTTP] HTTP-Server gestartet auf http://{settings.TCP_SERVER_HOST}:{settings.TCP_SERVER_PORT}/get_graph")
            # Lauf ewig
            await asyncio.Event().wait()
        
        http_task = asyncio.create_task(run_http_server())
    
    # Haupt-Server: stdio für Cursor/IDE
    async with stdio_server() as (read_stream, write_stream):
        try:
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
        finally:
            # Stoppe HTTP-Server wenn aktiv
            if http_task:
                http_task.cancel()
                try:
                    await http_task
                except asyncio.CancelledError:
                    pass
                print("[STOP] HTTP-Server gestoppt")
            
            # Final save: Save graph to disk on shutdown
            await loop.run_in_executor(None, controller.storage.graph.save_snapshot)
            print("[SAVE] Final graph snapshot saved to disk")
            # Stoppe Scheduler beim Shutdown
            controller.stop_enzyme_scheduler()

if __name__ == "__main__":
    asyncio.run(main())
