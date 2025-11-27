# A-MEM: Agentic Memory System

An agentic memory system for LLM agents based on the Zettelkasten principle.

> **Based on:** ["A-Mem: Agentic Memory for LLM Agents"](https://arxiv.org/html/2502.12110v11)  
> by Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang  
> Rutgers University, Independent Researcher, AIOS Foundation

## 🚀 Features

- ✅ **Note Construction**: Automatic extraction of keywords, tags, and contextual summary
- ✅ **Link Generation**: Automatic linking of similar memories
- ✅ **Memory Evolution**: Dynamic updating of existing memories
- ✅ **Semantic Retrieval**: Intelligent search with graph traversal
- ✅ **Multi-Provider Support**: Ollama (local) or OpenRouter (cloud)
- ✅ **Environment Variables**: Configuration via `.env` file

## 🔄 Relationship to Original Implementation

This implementation was developed independently based on the research paper ["A-Mem: Agentic Memory for LLM Agents"](https://arxiv.org/html/2502.12110v11). The original authors' production-ready system ([A-mem-sys](https://github.com/WujiangXu/A-mem-sys)) was discovered after this implementation was completed.

**Key Differences:**

This implementation focuses on **MCP Server integration** for IDE environments (Cursor, VSCode), providing:
- Direct IDE integration via MCP protocol
- **Explicit graph-based memory linking** using NetworkX (DiGraph) with typed edges, reasoning, and weights
- File import with automatic chunking
- Memory reset and management tools
- Modern TUI benchmarking tool for Ollama model speed testing

The original [A-mem-sys](https://github.com/WujiangXu/A-mem-sys) repository provides a **pip-installable Python library** with:
- Multiple LLM backend support (OpenAI, Ollama, OpenRouter, SGLang)
- Library-based integration for Python applications
- Comprehensive API for programmatic usage
- **Implicit linking** via ChromaDB embeddings (no explicit graph structure)

**Technical Architecture Difference:**

- **This implementation**: Dual-storage architecture
  - ChromaDB for vector similarity search
  - NetworkX DiGraph for explicit typed relationships (with `relation_type`, `reasoning`, `weight`)
  - Graph traversal for finding directly connected memories
  - Enables complex queries like "find all memories related to X through type Y"

- **Original implementation**: Single-storage architecture
  - ChromaDB as primary storage
  - Implicit linking through embedding similarity
  - Simpler architecture, less overhead

Both implementations are valid approaches to the same research paper, serving different use cases and integration scenarios.

## 🏗️ Framework

The framework of our Agentic Memory system showing the dynamic interaction between LLM agents and memory components, extended with our MCP Server integration and dual-storage architecture:

![Framework](framework_extended.jpg)

*The framework diagram illustrates the core memory system workflow: Note Construction (left), Memory Processing with MCP Server Integration (center), and Memory Retrieval (right). Our implementation extends the original framework with direct IDE integration via MCP protocol, explicit graph-based memory linking using NetworkX DiGraph with typed edges (relation_type, reasoning, weight), file import with automatic chunking, and a dual-storage architecture using ChromaDB for vector similarity search and NetworkX for explicit typed relationships.*

## 📋 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and adjust the values:

```bash
cp .env.example .env
```

**Configuration:**

- **LLM_PROVIDER**: `"ollama"` (local) or `"openrouter"` (cloud)
- **Ollama**: Local models (default)
- **OpenRouter**: Cloud-based LLMs (requires API key)

**Example `.env` for Ollama (default):**
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen3:4b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
```

**Example `.env` for OpenRouter:**
```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_LLM_MODEL=openai/gpt-4o-mini
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
```

### 3. Install Ollama Models (only when LLM_PROVIDER=ollama)

```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text:latest
```

### 4. Start Ollama (only when LLM_PROVIDER=ollama)

Make sure Ollama is running on `http://localhost:11434`.

## 🛠️ MCP Server

### Start

```bash
python mcp_server.py
```

### Available Tools

1. **`create_atomic_note`** - Stores a new piece of information in the memory system
2. **`retrieve_memories`** - Searches for relevant memories based on semantic similarity
3. **`get_memory_stats`** - Returns statistics about the memory system
4. **`delete_atomic_note`** - Deletes a note from the memory system
5. **`add_file`** - Stores the content of a file (e.g., .md) as a note, supports automatic chunking
6. **`reset_memory`** - Resets the complete memory system (⚠️ irreversible)

### IDE Integration

#### Cursor IDE

1. Open the MCP configuration file:
   - Windows: `%USERPROFILE%\.cursor\mcp.json` (or `C:\Users\<username>\.cursor\mcp.json`)
   - macOS: `~/.cursor/mcp.json`
   - Linux: `~/.cursor/mcp.json`

2. Add the following configuration:

```json
{
  "mcpServers": {
    "a-mem": {
      "command": "python",
      "args": [
        "-m",
        "src.a_mem.main"
      ],
      "cwd": "/path/to/a-mem-agentic-memory-system"
    }
  }
}
```

**Important:** Adjust `cwd` to the absolute path to your project directory!

3. Restart Cursor to load the configuration.

#### Visual Studio Code (mit MCP Extension)

1. Install the MCP Extension for VSCode (if available)

2. Open VSCode Settings (JSON):
   - `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (macOS)
   - Type "Preferences: Open User Settings (JSON)"

3. Add the MCP Server configuration:

```json
{
  "mcp.servers": {
    "a-mem": {
      "command": "python",
      "args": ["-m", "src.a_mem.main"],
      "cwd": "/path/to/a-mem-agentic-memory-system"
    }
  }
}
```

**Alternative:** Use the `mcp.json` file in the project root:

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

#### Usage in IDE

After configuration, the MCP tools are directly available in your IDE:

- **Chat/Composer**: Use the tools via natural language
  - "Store this information: ..."
  - "Search for memories about: ..."
  - "Show me the memory statistics"

- **Code**: The tools are automatically available as functions

See `MCP_SERVER_SETUP.md` for detailed information about all available tools.

## 📚 Dokumentation

- `docs/ARCHITECTURE.md` - System Architecture
- `docs/FINAL_COMPLIANCE_CHECK.md` - Paper Compliance
- `docs/TEST_REPORT.md` - Test Results
- `MCP_SERVER_SETUP.md` - MCP Server Setup

## 🧪 Tests

```bash
python tests/test_a_mem.py
python tests/test_code_structure.py
```

## 🧪 Benchmarking

The project includes a modern TUI benchmark tool for testing Ollama model speed and performance:

```bash
python ollama_benchmark.py
```

This tool measures model speed metrics (tokens/sec, latency, first token time) to help you choose the best Ollama model for your use case.

See `BENCHMARK_README.md` for details.

## 📊 Status

✅ **100% Paper-Compliance**  
✅ **All Tests Passed**  
✅ **Modular Structure**  
✅ **Multi-Provider Support** (Ollama + OpenRouter)  
✅ **MCP Server Integration**  
✅ **Memory Reset & Management Tools**

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This implementation is based on the research paper ["A-Mem: Agentic Memory for LLM Agents"](https://arxiv.org/html/2502.12110v11).

## 🙏 Acknowledgments

- Original paper authors: Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang
- Original repositories:
  - [AgenticMemory](https://github.com/WujiangXu/AgenticMemory) - Benchmark Evaluation
  - [A-mem-sys](https://github.com/WujiangXu/A-mem-sys) - Production-ready System

---

**Created by tobi and the CURSOR IDE with the new Composer 1 model for the community ❤️**
