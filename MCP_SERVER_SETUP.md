# MCP Server Setup for A-MEM

## 📋 Overview

The A-MEM MCP Server provides tools for the Agentic Memory System.

## 🛠️ Available Tools

### 1. `create_atomic_note`
Stores a new piece of information in the memory system.

**Parameters:**
- `content` (string, required): The text of the note/memory
- `source` (string, optional): Source of the information (default: "user_input")

**Example:**
```json
{
  "content": "Python async/await enables non-blocking I/O operations",
  "source": "user_input"
}
```

### 2. `retrieve_memories`
Searches for relevant memories based on semantic similarity.

**Parameters:**
- `query` (string, required): The search query
- `max_results` (integer, optional): Maximum number of results (default: 5, max: 20)

**Example:**
```json
{
  "query": "Python async programming",
  "max_results": 5
}
```

### 3. `get_memory_stats`
Returns statistics about the memory system.

**Parameters:** None

**Example:**
```json
{}
```

### 4. `delete_atomic_note`
Deletes a note from the memory system. Removes the note from Graph and Vector Store as well as all associated connections.

**Parameters:**
- `note_id` (string, required): The UUID of the note to be deleted

**Example:**
```json
{
  "note_id": "732c8c3b-7c71-42a6-9534-a611b4ffe7bf"
}
```

### 5. `add_file`
Stores the content of a file (e.g., .md) as a note in the memory system. Supports automatic chunking for large files (>16KB).

**Parameters:**
- `file_path` (string, optional): Path to the file to be stored (relative or absolute)
- `file_content` (string, optional): Alternatively: Direct file content as string (when file_path is not provided)
- `chunk_size` (integer, optional): Maximum size per chunk in bytes (default: 15000, max: 16384)

**Note:** Either `file_path` OR `file_content` must be provided.

**Example:**
```json
{
  "file_path": "documentation.md",
  "chunk_size": 15000
}
```

Or with direct content:
```json
{
  "file_content": "# Documentation\n\nThis is the content...",
  "chunk_size": 15000
}
```

### 6. `reset_memory`
Resets the complete memory system (Graph + Vector Store). Deletes all notes, edges, and embeddings.

**⚠️ WARNING:** This action cannot be undone!

**Parameters:** None

**Example:**
```json
{}
```

## 🚀 Installation & Start

### 1. Install Dependencies

```bash
pip install mcp
```

### 2. Start MCP Server

```bash
python mcp_server.py
```

Or directly:

```bash
python -m src.a_mem.main
```

## 📝 Cursor/IDE Configuration

### Cursor IDE

Add the following configuration to your MCP configuration file:
- Windows: `%USERPROFILE%\.cursor\mcp.json` (or `C:\Users\<username>\.cursor\mcp.json`)
- macOS: `~/.cursor/mcp.json`
- Linux: `~/.cursor/mcp.json`

```json
{
  "mcpServers": {
    "a-mem": {
      "command": "python",
      "args": ["-m", "src.a_mem.main"],
      "cwd": "/path/to/a-mem-agentic-memory-system"
    }
  }
}
```

**Important:** Adjust `cwd` to the absolute path to your project directory!

### Visual Studio Code

If an MCP Extension is available, use VSCode Settings (JSON) or an `mcp.json` file in the project root:

```json
{
  "mcpServers": {
    "a-mem": {
      "command": "python",
      "args": ["-m", "src.a_mem.main"],
      "cwd": "${workspaceFolder}"
    }
  }
}
```

## 🔧 Configuration

The server uses configuration from `src/a_mem/config.py` and `.env` file.

### Environment Variables (.env)

Copy `.env.example` to `.env` and adjust the values:

**Ollama (default):**
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen3:4b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
```

**OpenRouter:**
```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_LLM_MODEL=openai/gpt-4o-mini
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
```

### Ollama Setup (when LLM_PROVIDER=ollama)

Make sure Ollama is running and both models are installed:
```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text:latest
```

## 📊 Example Usage

### Create Memory:
```python
# Via MCP Tool
create_atomic_note(
    content="Python async/await Patterns",
    source="user_input"
)
```

### Search Memory:
```python
# Via MCP Tool
retrieve_memories(
    query="Python concurrency",
    max_results=5
)
```

### Get Statistics:
```python
# Via MCP Tool
get_memory_stats()
```

### Delete Note:
```python
# Via MCP Tool
delete_atomic_note(
    note_id="732c8c3b-7c71-42a6-9534-a611b4ffe7bf"
)
```

### Import File:
```python
# Via MCP Tool - Automatic chunking for large files
add_file(
    file_path="documentation.md",
    chunk_size=15000
)
```

### Reset Memory System:
```python
# Via MCP Tool - ⚠️ DELETES EVERYTHING!
reset_memory()
```

## ✅ Status

The MCP Server is fully implemented and uses:
- ✅ Local Ollama (qwen3:4b for LLM, nomic-embed-text for embeddings)
- ✅ Async I/O for performance
- ✅ Memory Evolution in background
- ✅ Graph-based linking
