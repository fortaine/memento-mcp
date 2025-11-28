"""
A-MEM Memory Graph Visualizer

Web-basiertes Dashboard für Graph-Visualisierung, Priority-Statistiken und Usage-Analyse.
Startet lokalen Web-Server auf http://localhost:8050
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
import threading
from datetime import datetime
from typing import Dict, List, Any
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import networkx as nx
from a_mem.core.logic import MemoryController
from a_mem.utils.priority import compute_priority
from a_mem.config import settings

# Initialize MemoryController (same as MCP Server uses)
controller = MemoryController()

# Helper function to run async code in a thread
def run_async_in_thread(coro):
    """Runs an async coroutine in a new thread with its own event loop."""
    result = [None]
    exception = [None]
    
    def run_in_thread():
        try:
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result[0] = loop.run_until_complete(coro)
            loop.close()
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=run_in_thread)
    thread.start()
    thread.join()
    
    if exception[0]:
        raise exception[0]
    return result[0]

# Debug: Check if graph has nodes
try:
    graph_data = run_async_in_thread(controller.get_graph_snapshot())
    node_count = len(graph_data.get("nodes", []))
    edge_count = len(graph_data.get("edges", []))
    print(f"🔍 Debug: Graph has {node_count} nodes, {edge_count} edges")
    if node_count == 0:
        print("⚠️  Warning: Graph is empty. Create notes using MCP tools first!")
except Exception as e:
    print(f"⚠️  Warning: Could not check graph: {e}")
    node_count, edge_count = 0, 0

# Dash App
app = dash.Dash(__name__)
app.title = "A-MEM Memory Graph Visualizer"

# Custom CSS for Tabs and overall styling - inject via index_string
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Dash Tabs - Force dark text on all tabs */
            ._dash-undo-redo {
                display: none;
            }
            
            /* Tab container */
            .js-plotly-plot {
                color: #000000;
            }
            
            /* All tab labels - dark text */
            .tab {
                color: #000000 !important;
            }
            
            /* Tab text specifically */
            .tab-label {
                color: #000000 !important;
            }
            
            /* Inactive tabs - still dark but slightly faded */
            .tab:not(.tab--selected) .tab-label {
                color: #333333 !important;
                opacity: 0.8;
            }
            
            /* Active tab - bold and dark */
            .tab.tab--selected .tab-label {
                color: #000000 !important;
                font-weight: bold;
            }
            
            /* Dash specific tab styling */
            div[role="tab"] {
                color: #000000 !important;
            }
            
            div[role="tab"][aria-selected="false"] {
                color: #333333 !important;
            }
            
            div[role="tab"][aria-selected="true"] {
                color: #000000 !important;
                font-weight: bold;
            }
            
            /* Ensure all text is dark */
            body {
                color: #000000;
            }
            
            /* Table styles */
            table td {
                color: #FFFFFF !important;
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
            table th {
                color: #000000 !important;
                background-color: #f2f2f2;
                font-weight: bold;
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# App Layout
app.layout = html.Div([
    html.H1("🧠 A-MEM Memory Graph Visualizer", style={'textAlign': 'center', 'marginBottom': '30px', 'color': '#FFFFFF'}),
    
    # Control Panel
    html.Div([
        html.Button("🔄 Refresh Data", id="refresh-btn", n_clicks=0, 
                   style={'margin': '10px', 'padding': '10px 20px', 'fontSize': '16px', 'color': '#000000', 'backgroundColor': '#f0f0f0'}),
        html.Div(id="last-update", style={'margin': '10px', 'color': '#FFFFFF'}),
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    # Tabs with custom styling
    dcc.Tabs(
        id="main-tabs", 
        value="graph",
        style={
            'color': '#000000',
            'fontColor': '#000000'
        },
        children=[
            dcc.Tab(label="📊 Graph Visualization", value="graph", style={'color': '#000000'}, selected_style={'color': '#000000', 'fontWeight': 'bold'}),
            dcc.Tab(label="📈 Priority Statistics", value="priority", style={'color': '#000000'}, selected_style={'color': '#000000', 'fontWeight': 'bold'}),
            dcc.Tab(label="🔗 Relations Analysis", value="relations", style={'color': '#000000'}, selected_style={'color': '#000000', 'fontWeight': 'bold'}),
            dcc.Tab(label="📅 Event Timeline", value="events", style={'color': '#000000'}, selected_style={'color': '#000000', 'fontWeight': 'bold'}),
            dcc.Tab(label="📋 Node Details", value="nodes", style={'color': '#000000'}, selected_style={'color': '#000000', 'fontWeight': 'bold'}),
        ]
    ),
    
    html.Div(id="tab-content", style={'marginTop': '20px'}),
    
    # Hidden div to store data
    dcc.Store(id="graph-data"),
    dcc.Store(id="events-data"),
])

def load_graph_data() -> Dict[str, Any]:
    """Lädt Graph-Daten direkt aus der JSON-Datei (wird von extract_graph.py aktualisiert)."""
    try:
        # Load directly from JSON file (updated by extract_graph.py via HTTP server)
        graph_path = settings.GRAPH_PATH
        
        if not graph_path.exists():
            print(f"⚠️  Graph file not found: {graph_path}")
            return {"nodes": [], "edges": []}
        
        with open(graph_path, 'r', encoding='utf-8') as f:
            graph_json = json.load(f)
        
        # Convert NetworkX node_link_data format to our format
        nodes = []
        for node_data in graph_json.get("nodes", []):
            # Ensure id is present
            if "id" not in node_data:
                node_data["id"] = node_data.get("id", "")
            nodes.append(node_data)
        
        edges = []
        for link_data in graph_json.get("links", []):
            edges.append({
                "source": link_data.get("source"),
                "target": link_data.get("target"),
                "relation_type": link_data.get("type", "relates_to"),
                "reasoning": link_data.get("reasoning", ""),
                "weight": link_data.get("weight", 1.0),
                "created_at": link_data.get("created_at")
            })
        
        node_count = len(nodes)
        edge_count = len(edges)
        print(f"🔍 Loaded {node_count} nodes, {edge_count} edges from {graph_path}")
        
        return {"nodes": nodes, "edges": edges}
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing graph JSON: {e}")
        return {"nodes": [], "edges": []}
    except Exception as e:
        print(f"❌ Error loading graph data: {e}")
        import traceback
        traceback.print_exc()
        return {"nodes": [], "edges": []}

def load_events_data() -> List[Dict[str, Any]]:
    """Lädt Event-Log Daten."""
    events = []
    event_log_path = settings.DATA_DIR / "events.jsonl"
    
    if event_log_path.exists():
        try:
            with open(event_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))
        except Exception as e:
            print(f"Error loading events: {e}")
    
    return events

def create_graph_visualization(graph_data: Dict[str, Any]) -> go.Figure:
    """Erstellt NetworkX Graph-Visualisierung mit Plotly."""
    G = nx.DiGraph()
    
    # Add nodes
    for node in graph_data.get("nodes", []):
        node_id = node.get("id", "")
        if node_id:
            G.add_node(node_id, **node)
    
    # Add edges
    for edge in graph_data.get("edges", []):
        source = edge.get("source")
        target = edge.get("target")
        if source and target:
            G.add_edge(source, target, **edge)
    
    if len(G.nodes()) == 0:
        # Empty graph
        fig = go.Figure()
        fig.add_annotation(
            text="No nodes in graph yet.<br><br>To create notes, use the MCP tools:<br>• create_atomic_note<br>• Or add notes via the MCP server",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#000000"),
            align="center"
        )
        fig.update_layout(
            title="Graph Visualization - Empty Graph",
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
        )
        return fig
    
    # Layout
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Node positions
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    # Node info
    node_text = []
    node_sizes = []
    node_colors = []
    
    type_colors = {
        "rule": "#FF6B6B",
        "procedure": "#4ECDC4",
        "concept": "#95E1D3",
        "tool": "#F38181",
        "reference": "#AA96DA",
        "integration": "#FCBAD3",
        None: "#C7CEEA"
    }
    
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        node_type = node_data.get("type")
        content = node_data.get("content", "")[:50]
        summary = node_data.get("contextual_summary", "")[:30]
        
        # Calculate priority for size
        try:
            from a_mem.models.note import AtomicNote
            note = AtomicNote(**node_data)
            edge_count = G.degree(node_id)
            priority = compute_priority(note, usage_count=0, edge_count=edge_count)
            node_sizes.append(max(10, min(30, priority * 10)))
        except:
            node_sizes.append(15)
        
        node_colors.append(type_colors.get(node_type, "#C7CEEA"))
        node_text.append(f"{node_id[:8]}...<br>Type: {node_type or 'N/A'}<br>{summary}")
    
    # Edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create figure
    fig = go.Figure()
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        name='Nodes',
        hovertext=node_text,
        hoverinfo='text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        text=[node_id[:6] for node_id in G.nodes()],
        textposition="middle center",
        textfont=dict(size=8, color="#000000")
    ))
    
    fig.update_layout(
        title="Memory Graph Visualization",
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[
            dict(
                text="Node size = Priority | Color = Type",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor="left", yanchor="bottom",
                font=dict(size=10, color="#000000")
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        font=dict(color="#000000")
    )
    
    return fig

def create_priority_stats(graph_data: Dict[str, Any]) -> go.Figure:
    """Erstellt Priority-Statistiken."""
    priorities = []
    types = []
    
    for node in graph_data.get("nodes", []):
        try:
            from a_mem.models.note import AtomicNote
            note = AtomicNote(**node)
            
            # Get edge count from graph
            G = nx.DiGraph()
            for n in graph_data.get("nodes", []):
                G.add_node(n.get("id"), **n)
            for e in graph_data.get("edges", []):
                G.add_edge(e.get("source"), e.get("target"))
            
            edge_count = G.degree(note.id) if note.id in G else 0
            priority = compute_priority(note, usage_count=0, edge_count=edge_count)
            
            priorities.append(priority)
            types.append(note.type or "unknown")
        except:
            continue
    
    if not priorities:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#000000", size=16)
        )
        return fig
    
    df = pd.DataFrame({"Priority": priorities, "Type": types})
    
    fig = px.box(df, x="Type", y="Priority", title="Priority Distribution by Type",
                 color="Type", color_discrete_map={
                     "rule": "#FF6B6B",
                     "procedure": "#4ECDC4",
                     "concept": "#95E1D3",
                     "tool": "#F38181",
                     "reference": "#AA96DA",
                     "integration": "#FCBAD3"
                 })
    
    fig.update_layout(
        showlegend=False,
        font=dict(color="#000000"),
        xaxis=dict(tickfont=dict(color="#000000")),
        yaxis=dict(tickfont=dict(color="#000000"))
    )
    return fig

def create_relations_analysis(graph_data: Dict[str, Any]) -> go.Figure:
    """Analysiert Relations im Graph."""
    relation_types = {}
    edge_weights = []
    
    for edge in graph_data.get("edges", []):
        rel_type = edge.get("relation_type", "relates_to")
        weight = edge.get("weight", 1.0)
        
        relation_types[rel_type] = relation_types.get(rel_type, 0) + 1
        edge_weights.append(weight)
    
    if not relation_types:
        fig = go.Figure()
        fig.add_annotation(
            text="No relations in graph yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#000000", size=16)
        )
        return fig
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(relation_types.keys()),
            y=list(relation_types.values()),
            marker_color='#4ECDC4'
        )
    ])
    
    fig.update_layout(
        title="Relation Types Distribution",
        xaxis_title="Relation Type",
        yaxis_title="Count",
        showlegend=False,
        font=dict(color="#000000"),
        xaxis=dict(tickfont=dict(color="#000000")),
        yaxis=dict(tickfont=dict(color="#000000"))
    )
    
    return fig

def create_event_timeline(events_data: List[Dict[str, Any]]) -> go.Figure:
    """Erstellt Event-Timeline."""
    if not events_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No events logged yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#000000", size=16)
        )
        return fig
    
    # Group by event type and date
    event_counts = {}
    for event in events_data:
        event_type = event.get("event", "unknown")
        timestamp = event.get("timestamp", "")
        if timestamp:
            try:
                date = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
                key = f"{date} - {event_type}"
                event_counts[key] = event_counts.get(key, 0) + 1
            except:
                continue
    
    if not event_counts:
        fig = go.Figure()
        fig.add_annotation(
            text="No valid events found",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#000000", size=16)
        )
        return fig
    
    dates = list(event_counts.keys())
    counts = list(event_counts.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=dates,
            y=counts,
            marker_color='#95E1D3'
        )
    ])
    
    fig.update_layout(
        title="Event Timeline",
        xaxis_title="Date - Event Type",
        yaxis_title="Count",
        xaxis_tickangle=-45,
        showlegend=False,
        font=dict(color="#000000"),
        xaxis=dict(tickfont=dict(color="#000000")),
        yaxis=dict(tickfont=dict(color="#000000"))
    )
    
    return fig

def create_node_details_table(graph_data: Dict[str, Any]) -> html.Div:
    """Erstellt Node-Details Tabelle."""
    nodes = graph_data.get("nodes", [])
    
    if not nodes:
        return html.Div("No nodes in graph yet.")
    
    # Filter out zombie nodes (nodes without content)
    valid_nodes = []
    for node in nodes:
        # Check if node has content
        content = node.get("content", "")
        if content and len(str(content).strip()) > 0:
            valid_nodes.append(node)
    
    if not valid_nodes:
        return html.Div("No valid nodes in graph (all nodes are empty).")
    
    # Create table rows
    rows = []
    for node in valid_nodes[:50]:  # Limit to 50 for performance
        node_id = node.get("id", "")[:8]
        node_type = node.get("type", "N/A")
        summary = node.get("contextual_summary", "")[:50]
        tags = ", ".join(node.get("tags", [])[:3])
        
        # Calculate priority
        try:
            from a_mem.models.note import AtomicNote
            note = AtomicNote(**node)
            G = nx.DiGraph()
            for n in graph_data.get("nodes", []):
                G.add_node(n.get("id"), **n)
            for e in graph_data.get("edges", []):
                G.add_edge(e.get("source"), e.get("target"))
            edge_count = G.degree(note.id) if note.id in G else 0
            priority = compute_priority(note, usage_count=0, edge_count=edge_count)
        except:
            priority = 0.0
            edge_count = 0
        
        rows.append(html.Tr([
            html.Td(node_id, style={'color': '#FFFFFF'}),
            html.Td(node_type, style={'color': '#FFFFFF'}),
            html.Td(f"{priority:.2f}", style={'color': '#FFFFFF'}),
            html.Td(str(edge_count), style={'color': '#FFFFFF'}),
            html.Td(summary, style={'color': '#FFFFFF'}),
            html.Td(tags, style={'color': '#FFFFFF'}),
        ]))
    
    return html.Div([
        html.H3("Node Details (Top 50)", style={'color': '#FFFFFF'}),
        html.Table([
            html.Thead([
                html.Tr([
                    html.Th("ID", style={'color': '#000000'}),
                    html.Th("Type", style={'color': '#000000'}),
                    html.Th("Priority", style={'color': '#000000'}),
                    html.Th("Edges", style={'color': '#000000'}),
                    html.Th("Summary", style={'color': '#000000'}),
                    html.Th("Tags", style={'color': '#000000'}),
                ])
            ]),
            html.Tbody(rows)
        ], style={
            'width': '100%', 
            'borderCollapse': 'collapse',
            'color': '#FFFFFF'
        })
    ])

@app.callback(
    [Output("graph-data", "data"),
     Output("events-data", "data"),
     Output("last-update", "children")],
    [Input("refresh-btn", "n_clicks")]
)
def update_data(n_clicks):
    """Aktualisiert Daten beim Refresh."""
    graph_data = load_graph_data()
    events_data = load_events_data()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return graph_data, events_data, f"Last updated: {timestamp}"

@app.callback(
    Output("tab-content", "children"),
    [Input("main-tabs", "value"),
     Input("graph-data", "data"),
     Input("events-data", "data")]
)
def update_tab_content(tab, graph_data, events_data):
    """Aktualisiert Tab-Content."""
    if not graph_data:
        return html.Div("Loading...", style={'textAlign': 'center', 'padding': '50px'})
    
    if tab == "graph":
        fig = create_graph_visualization(graph_data)
        return dcc.Graph(figure=fig, style={'height': '800px'})
    
    elif tab == "priority":
        fig = create_priority_stats(graph_data)
        return dcc.Graph(figure=fig, style={'height': '600px'})
    
    elif tab == "relations":
        fig = create_relations_analysis(graph_data)
        return dcc.Graph(figure=fig, style={'height': '600px'})
    
    elif tab == "events":
        fig = create_event_timeline(events_data or [])
        return dcc.Graph(figure=fig, style={'height': '600px'})
    
    elif tab == "nodes":
        return create_node_details_table(graph_data)
    
    return html.Div("Unknown tab")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧠 A-MEM Memory Graph Visualizer")
    print("="*60)
    print("\n🌐 Starting web server on http://localhost:8050")
    print("📊 Open your browser to view the dashboard")
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='127.0.0.1', port=8050, use_reloader=False)

