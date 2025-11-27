# A-MEM: Agentic Memory System

Ein agentisches Memory-System für LLM Agents basierend auf dem Zettelkasten-Prinzip.

> **Based on:** ["A-Mem: Agentic Memory for LLM Agents"](https://arxiv.org/html/2502.12110v11)  
> by Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang  
> Rutgers University, Independent Researcher, AIOS Foundation

## 🚀 Features

- ✅ **Note Construction**: Automatische Extraktion von Keywords, Tags und Contextual Summary
- ✅ **Link Generation**: Automatische Verknüpfung ähnlicher Memories
- ✅ **Memory Evolution**: Dynamische Aktualisierung bestehender Memories
- ✅ **Semantic Retrieval**: Intelligente Suche mit Graph-Traversal
- ✅ **Multi-Provider Support**: Ollama (lokal) oder OpenRouter (Cloud)
- ✅ **Environment Variables**: Konfiguration über `.env` Datei

## 📋 Installation

### 1. Dependencies installieren

```bash
pip install -r requirements.txt
```

### 2. Environment Variables konfigurieren

Kopiere `.env.example` zu `.env` und passe die Werte an:

```bash
cp .env.example .env
```

**Konfiguration:**

- **LLM_PROVIDER**: `"ollama"` (lokal) oder `"openrouter"` (Cloud)
- **Ollama**: Lokale Modelle (Standard)
- **OpenRouter**: Cloud-basierte LLMs (benötigt API Key)

**Beispiel `.env` für Ollama (Standard):**
```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen3:4b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest
```

**Beispiel `.env` für OpenRouter:**
```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_LLM_MODEL=openai/gpt-4o-mini
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small
```

### 3. Ollama Modelle installieren (nur bei LLM_PROVIDER=ollama)

```bash
ollama pull qwen3:4b
ollama pull nomic-embed-text:latest
```

### 4. Ollama starten (nur bei LLM_PROVIDER=ollama)

Stelle sicher, dass Ollama auf `http://localhost:11434` läuft.

## 🛠️ MCP Server

### Start

```bash
python mcp_server.py
```

### Verfügbare Tools

1. **`create_atomic_note`** - Speichert eine neue Information im Memory System
2. **`retrieve_memories`** - Sucht nach relevanten Memories basierend auf semantischer Ähnlichkeit
3. **`get_memory_stats`** - Gibt Statistiken über das Memory System zurück
4. **`delete_atomic_note`** - Löscht eine Note aus dem Memory System
5. **`add_file`** - Speichert den Inhalt einer Datei (z.B. .md) als Note, unterstützt automatisches Chunking
6. **`reset_memory`** - Setzt das komplette Memory System zurück (⚠️ nicht rückgängig machbar)

### IDE Integration

#### Cursor IDE

1. Öffne die MCP-Konfigurationsdatei:
   - Windows: `%APPDATA%\Cursor\User\globalStorage\saoudrizwan.claude-dev\settings\cline_mcp_settings.json`
   - macOS: `~/Library/Application Support/Cursor/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
   - Linux: `~/.config/Cursor/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`

2. Füge folgende Konfiguration hinzu:

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

**Wichtig:** Passe `cwd` auf den absoluten Pfad zu deinem Projekt-Verzeichnis an!

3. Starte Cursor neu, damit die Konfiguration geladen wird.

#### Visual Studio Code (mit MCP Extension)

1. Installiere die MCP Extension für VSCode (falls verfügbar)

2. Öffne die VSCode Settings (JSON):
   - `Ctrl+Shift+P` (Windows/Linux) oder `Cmd+Shift+P` (macOS)
   - Tippe "Preferences: Open User Settings (JSON)"

3. Füge die MCP Server Konfiguration hinzu:

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

**Alternative:** Nutze die `mcp.json` Datei im Projekt-Root:

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

#### Verwendung in der IDE

Nach der Konfiguration stehen dir die MCP Tools direkt in der IDE zur Verfügung:

- **Chat/Composer**: Nutze die Tools über natürliche Sprache
  - "Speichere diese Information: ..."
  - "Suche nach Memories über: ..."
  - "Zeige mir die Memory-Statistiken"

- **Code**: Die Tools werden automatisch als Funktionen verfügbar

Siehe `MCP_SERVER_SETUP.md` für detaillierte Informationen zu allen verfügbaren Tools.

## 📚 Dokumentation

- `docs/ARCHITECTURE.md` - System-Architektur
- `docs/FINAL_COMPLIANCE_CHECK.md` - Paper-Compliance
- `docs/TEST_REPORT.md` - Test-Ergebnisse
- `MCP_SERVER_SETUP.md` - MCP Server Setup

## 🧪 Tests

```bash
python tests/test_a_mem.py
python tests/test_code_structure.py
```

## 🧪 Benchmarking

Das Projekt enthält ein modernes TUI-Benchmark-Tool für Ollama-Modelle:

```bash
python ollama_benchmark.py
```

Siehe `BENCHMARK_README.md` für Details.

## 📊 Status

✅ **100% Paper-Compliance**  
✅ **Alle Tests bestanden**  
✅ **Modulare Struktur**  
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

## 🔄 Relationship to Original Implementation

This implementation was developed independently based on the research paper ["A-Mem: Agentic Memory for LLM Agents"](https://arxiv.org/html/2502.12110v11). The original authors' production-ready system ([A-mem-sys](https://github.com/WujiangXu/A-mem-sys)) was discovered after this implementation was completed.

**Key Differences:**

This implementation focuses on **MCP Server integration** for IDE environments (Cursor, VSCode), providing:
- Direct IDE integration via MCP protocol
- **Explicit graph-based memory linking** using NetworkX (DiGraph) with typed edges, reasoning, and weights
- File import with automatic chunking
- Memory reset and management tools
- Modern TUI benchmarking tool

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

---

**Created by tobi and the CURSOR IDE with the new Composer 1 model for the community ❤️**
