# A-MEM Memory Graph Visualizer

Web-basiertes Dashboard für interaktive Visualisierung und Analyse des A-MEM Memory Graphs.

## 🚀 Installation

Installiere die optionalen Dependencies:

```bash
pip install dash plotly pandas
```

Oder installiere alle Dependencies inkl. Visualizer:

```bash
pip install -r requirements.txt
```

## 📊 Starten

```bash
python tools/visualize_memory.py
```

Dann öffne deinen Browser auf: **http://localhost:8050**

## 🎯 Features

### 1. Graph Visualization
- **Interaktive Netzwerk-Visualisierung** des gesamten Memory Graphs
- **Node-Größe** = Priority Score (höhere Priority = größere Nodes)
- **Node-Farbe** = Note Type (rule, procedure, concept, tool, reference, integration)
- **Hover-Info**: Zeigt Node-ID, Type, Summary beim Überfahren
- **Spring-Layout**: Automatische Graph-Layout-Berechnung

### 2. Priority Statistics
- **Box-Plots** zeigen Priority-Verteilung nach Note-Type
- Identifiziert welche Note-Types durchschnittlich höhere Priority haben
- Hilft bei der Optimierung der Priority-Formel

### 3. Relations Analysis
- **Bar-Chart** der Relation-Types (relates_to, supports, extends, etc.)
- Zeigt wie viele Verbindungen jedes Type hat
- Identifiziert häufigste Relation-Patterns

### 4. Event Timeline
- **Timeline-Visualisierung** aller System-Events
- Zeigt NOTE_CREATED, RELATION_CREATED, MEMORY_EVOLVED, etc.
- Gruppiert nach Datum und Event-Type
- Hilft bei der Analyse der System-Aktivität

### 5. Node Details
- **Detaillierte Tabelle** aller Nodes (Top 50)
- Zeigt: ID, Type, Priority, Edge-Count, Summary, Tags
- Sortierbar und filterbar (in zukünftigen Versionen)

## 🔄 Refresh

Klicke auf den **"🔄 Refresh Data"** Button um die neuesten Daten zu laden.

## 🎨 Farb-Codierung

- **🔴 Rule** (rot): `#FF6B6B`
- **🔵 Procedure** (türkis): `#4ECDC4`
- **🟢 Concept** (mint): `#95E1D3`
- **🟠 Tool** (rosa): `#F38181`
- **🟣 Reference** (lila): `#AA96DA`
- **🌸 Integration** (pink): `#FCBAD3`
- **⚪ Unknown/None** (grau): `#C7CEEA`

## 💡 Verwendung

1. **Pattern-Erkennung**: Schaue dir die Graph-Visualisierung an um Cluster und Verbindungen zu erkennen
2. **Priority-Optimierung**: Analysiere Priority-Statistiken um zu sehen welche Types bevorzugt werden
3. **Relation-Analyse**: Prüfe welche Relation-Types am häufigsten vorkommen
4. **Event-Monitoring**: Überwache System-Aktivität über die Event-Timeline
5. **Node-Exploration**: Durchsuche Node-Details um spezifische Notes zu finden

## 🛠️ Technische Details

- **Framework**: Dash (Plotly)
- **Graph-Layout**: NetworkX Spring Layout
- **Datenquelle**: Lädt Graph-Daten aus `data/graph/knowledge_graph.json` (aktualisiert via `extract_graph.py`)
- **Event-Log**: Liest `data/events.jsonl`
- **Port**: 8050 (konfigurierbar im Code)

**Hinweis:** Der Visualizer lädt die Daten direkt aus der JSON-Datei. Um die neuesten Daten zu sehen:
1. Führe `python tools/extract_graph.py` aus (ruft Graph über HTTP vom MCP Server ab)
2. Drücke F5 im Browser, um die Seite neu zu laden

## 🔧 Anpassungen

Du kannst den Visualizer anpassen:
- **Port ändern**: `app.run_server(port=8050)` in `visualize_memory.py`
- **Layout anpassen**: Modifiziere `create_graph_visualization()` für andere Layouts
- **Farben ändern**: Passe `type_colors` Dictionary an
- **Mehr Tabs**: Füge neue Tabs in `dcc.Tabs()` hinzu

## 📝 Beispiel-Output

Nach dem Start siehst du:
- Interaktive Graph-Visualisierung mit allen Nodes und Edges
- Priority-Box-Plots nach Type
- Relation-Type-Verteilung
- Event-Timeline
- Detaillierte Node-Tabelle

**Viel Spaß beim Erkunden deines Memory Graphs! 🧠✨**

