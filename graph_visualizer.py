"""
Graph Visualization Module for Medical Knowledge Graph
Provides interactive visualizations using pyvis and matplotlib
"""

import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Set, Optional, Tuple
import random
from collections import defaultdict


class KnowledgeGraphVisualizer:
    """Visualize medical knowledge graph"""
    
    def __init__(self, graph: nx.MultiDiGraph):
        """
        Initialize visualizer
        
        Args:
            graph: NetworkX MultiDiGraph containing the knowledge graph
        """
        self.graph = graph
        
        # Color scheme for different entity types
        self.entity_colors = {
            'Disease': '#ff6b6b',           # Red
            'Clinical_Manifestation': '#4ecdc4',  # Teal
            'Treatment': '#45b7d1',         # Blue
            'Drug': '#96ceb4',              # Green
            'Diagnostic_Procedure': '#ffeaa7',  # Yellow
            'Risk_Factor': '#dfe6e9',       # Gray
            'Complication': '#fd79a8',      # Pink
            'Pathophysiology': '#a29bfe',   # Purple
        }
        
        # Default color for unknown types
        self.default_color = '#95a5a6'
        
        # Color scheme for different relation types (for better edge visibility)
        self.relation_colors = {
            'PRESENTS_WITH': '#e74c3c',        # Red
            'TREATED_WITH': '#3498db',         # Blue
            'CAUSES': '#e67e22',               # Orange
            'LEADS_TO': '#f39c12',             # Yellow-orange
            'SUGGESTS': '#9b59b6',             # Purple
            'CONFIRMED_BY': '#1abc9c',         # Turquoise
            'INCREASES_RISK': '#c0392b',       # Dark red
            'DECREASES_RISK': '#27ae60',       # Green
            'CONTRAINDICATED_WITH': '#e74c3c', # Red
            'SYNERGISTIC_WITH': '#2ecc71',     # Light green
            'HAS_MECHANISM': '#8e44ad',        # Dark purple
        }
        self.default_edge_color = '#95a5a6'  # Gray for unknown relations
    
    def get_node_color(self, entity_type: str) -> str:
        """Get color for entity type"""
        return self.entity_colors.get(entity_type, self.default_color)
    
    def _get_edge_color(self, relation_type: str) -> str:
        """Get color for relation type"""
        return self.relation_colors.get(relation_type, self.default_edge_color)
    
    def extract_subgraph_by_entities(self, entity_names: List[str], 
                                    max_depth: int = 1) -> nx.MultiDiGraph:
        """
        Extract subgraph centered around specific entities
        
        Args:
            entity_names: List of entity names to center the subgraph around
            max_depth: Maximum distance from center entities
            
        Returns:
            Subgraph containing the entities and their neighbors
        """
        # Find nodes matching the entity names
        center_nodes = set()
        for entity_name in entity_names:
            matching = [n for n, d in self.graph.nodes(data=True) 
                       if entity_name.lower() in d.get('name', '').lower()]
            center_nodes.update(matching)
        
        if not center_nodes:
            print(f"Warning: No nodes found matching {entity_names}")
            return nx.MultiDiGraph()
        
        # Get neighbors up to max_depth
        nodes_to_include = set(center_nodes)
        current_layer = center_nodes
        
        for depth in range(max_depth):
            next_layer = set()
            for node in current_layer:
                # Add predecessors and successors
                next_layer.update(self.graph.predecessors(node))
                next_layer.update(self.graph.successors(node))
            
            nodes_to_include.update(next_layer)
            current_layer = next_layer
        
        # Create subgraph
        subgraph = self.graph.subgraph(nodes_to_include).copy()
        
        print(f"Extracted subgraph with {subgraph.number_of_nodes()} nodes "
              f"and {subgraph.number_of_edges()} edges")
        
        return subgraph
    
    def extract_subgraph_by_relation_type(self, relation_types: List[str],
                                         max_nodes: int = 100) -> nx.MultiDiGraph:
        """
        Extract subgraph containing only specific relation types
        
        Args:
            relation_types: List of relation types to include
            max_nodes: Maximum number of nodes to include
            
        Returns:
            Filtered subgraph
        """
        # Find edges with matching relation types
        edges_to_include = []
        nodes_to_include = set()
        
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            if data.get('relation_type') in relation_types:
                edges_to_include.append((u, v, key))
                nodes_to_include.add(u)
                nodes_to_include.add(v)
                
                if len(nodes_to_include) >= max_nodes:
                    break
        
        # Create subgraph
        subgraph = nx.MultiDiGraph()
        
        # Add nodes with attributes
        for node in nodes_to_include:
            subgraph.add_node(node, **self.graph.nodes[node])
        
        # Add edges with attributes
        for u, v, key in edges_to_include:
            subgraph.add_edge(u, v, key=key, **self.graph[u][v][key])
        
        print(f"Extracted subgraph with {subgraph.number_of_nodes()} nodes "
              f"and {subgraph.number_of_edges()} edges")
        
        return subgraph
    
    def visualize_with_matplotlib(self, subgraph: Optional[nx.MultiDiGraph] = None,
                                 max_nodes: int = 50,
                                 output_file: str = "graph_visualization.png",
                                 figsize: Tuple[int, int] = (20, 16),
                                 show_labels: bool = True):
        """
        Create static visualization using matplotlib
        
        Args:
            subgraph: Subgraph to visualize (if None, samples from main graph)
            max_nodes: Maximum number of nodes to visualize
            output_file: Path to save the visualization
            figsize: Figure size (width, height)
            show_labels: Whether to show node labels
        """
        # Use subgraph or sample from main graph
        if subgraph is None:
            if self.graph.number_of_nodes() > max_nodes:
                # Sample nodes
                nodes_sample = random.sample(list(self.graph.nodes()), max_nodes)
                subgraph = self.graph.subgraph(nodes_sample).copy()
            else:
                subgraph = self.graph
        
        print(f"Visualizing {subgraph.number_of_nodes()} nodes "
              f"and {subgraph.number_of_edges()} edges...")
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Use spring layout for positioning
        pos = nx.spring_layout(subgraph, k=2, iterations=50, seed=42)
        
        # Prepare node colors based on entity type
        node_colors = []
        for node in subgraph.nodes():
            entity_type = subgraph.nodes[node].get('entity_type', 'Unknown')
            node_colors.append(self.get_node_color(entity_type))
        
        # Draw nodes
        nx.draw_networkx_nodes(subgraph, pos,
                              node_color=node_colors,
                              node_size=300,
                              alpha=0.9)
        
        # Draw edges with arrows
        # Convert multigraph to simple graph for visualization
        simple_graph = nx.DiGraph()
        edge_labels = {}
        for u, v, data in subgraph.edges(data=True):
            if not simple_graph.has_edge(u, v):
                simple_graph.add_edge(u, v, **data)
                # Collect edge label (relation type)
                rel_type = data.get('relation_type', '')
                if rel_type:
                    edge_labels[(u, v)] = rel_type
        
        nx.draw_networkx_edges(simple_graph, pos,
                              edge_color='#666666',
                              arrows=True,
                              arrowsize=15,
                              width=2.0,
                              alpha=0.7,
                              connectionstyle='arc3,rad=0.1')
        
        # Draw edge labels (relation types)
        if show_labels and edge_labels:
            nx.draw_networkx_edge_labels(simple_graph, pos,
                                        edge_labels,
                                        font_size=7,
                                        font_color='red',
                                        bbox=dict(boxstyle='round,pad=0.3', 
                                                facecolor='white', 
                                                edgecolor='none',
                                                alpha=0.7))
        
        # Draw labels
        if show_labels:
            labels = {}
            for node in subgraph.nodes():
                name = subgraph.nodes[node].get('name', '')
                # Truncate long names
                if len(name) > 30:
                    name = name[:27] + '...'
                labels[node] = name
            
            nx.draw_networkx_labels(subgraph, pos,
                                   labels,
                                   font_size=8,
                                   font_weight='bold')
        
        # Create legend
        legend_elements = []
        for entity_type, color in self.entity_colors.items():
            # Count nodes of this type in subgraph
            count = sum(1 for n, d in subgraph.nodes(data=True) 
                       if d.get('entity_type') == entity_type)
            if count > 0:
                legend_elements.append(
                    plt.scatter([], [], c=color, s=100, label=f'{entity_type} ({count})')
                )
        
        plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        plt.title("Medical Knowledge Graph Visualization", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
        
        plt.close()
    
    def visualize_interactive_pyvis(self, subgraph: Optional[nx.MultiDiGraph] = None,
                                   max_nodes: int = 100,
                                   output_file: str = "interactive_graph.html",
                                   notebook: bool = False):
        """
        Create interactive visualization using pyvis
        
        Args:
            subgraph: Subgraph to visualize (if None, samples from main graph)
            max_nodes: Maximum number of nodes to visualize
            output_file: Path to save the HTML file
            notebook: Whether to display in Jupyter notebook
        """
        try:
            from pyvis.network import Network
        except ImportError:
            print("Error: pyvis not installed. Install it with: pip install pyvis")
            return
        
        # Use subgraph or sample from main graph
        if subgraph is None:
            if self.graph.number_of_nodes() > max_nodes:
                # Sample nodes
                nodes_sample = random.sample(list(self.graph.nodes()), max_nodes)
                subgraph = self.graph.subgraph(nodes_sample).copy()
            else:
                subgraph = self.graph
        
        print(f"Creating interactive visualization with {subgraph.number_of_nodes()} nodes "
              f"and {subgraph.number_of_edges()} edges...")
        
        # Create network
        net = Network(height="900px", width="100%", 
                     bgcolor="#ffffff", font_color="black",
                     notebook=notebook, directed=True)
        
        # Configure physics
        net.barnes_hut(gravity=-80000, central_gravity=0.3,
                      spring_length=100, spring_strength=0.001,
                      damping=0.09, overlap=0)
        
        # Add nodes
        for node in subgraph.nodes():
            node_data = subgraph.nodes[node]
            entity_type = node_data.get('entity_type', 'Unknown')
            name = node_data.get('name', 'Unknown')
            confidence = node_data.get('confidence', 0)
            
            # Prepare hover info
            title = f"<b>{name}</b><br>"
            title += f"Type: {entity_type}<br>"
            title += f"Confidence: {confidence:.2f}<br>"
            title += f"Node ID: {node_data.get('entity_id', 'N/A')}<br>"
            title += f"PMID: {node_data.get('pmid', 'N/A')}"
            
            # Add node
            color = self.get_node_color(entity_type)
            net.add_node(node, 
                        label=name[:30],  # Truncate long labels
                        title=title,
                        color=color,
                        size=20 + confidence * 10)
        
        # Add edges
        edge_count = defaultdict(int)
        for u, v, key, data in subgraph.edges(keys=True, data=True):
            edge_count[(u, v)] += 1
        
        for u, v, key, data in subgraph.edges(keys=True, data=True):
            relation_type = data.get('relation_type', 'RELATED_TO')
            weight = data.get('weight', 0.5)
            evidence = data.get('evidence', '')[:100] + '...'  # Truncate
            
            # Prepare hover info
            title = f"<b>{relation_type}</b><br>"
            title += f"Weight: {weight:.2f}<br>"
            title += f"Evidence: {evidence}"
            
            # Make edge labels more visible
            # Show label for all edges (not just single edges)
            edge_label = relation_type
            
            # Vary edge color by relation type for better visibility
            edge_color = self._get_edge_color(relation_type)
            
            # Add edge
            net.add_edge(u, v,
                        title=title,
                        label=edge_label,  # Always show label
                        arrows='to',
                        color=edge_color,
                        width=1.5 + weight * 3,  # Thicker edges
                        font={'size': 10, 'color': '#333333', 'strokeWidth': 0})  # Better label visibility
        
        # Add options
        net.set_options("""
        {
          "nodes": {
            "font": {
              "size": 12,
              "face": "Arial"
            },
            "borderWidth": 2,
            "borderWidthSelected": 4
          },
          "edges": {
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.5
              }
            },
            "smooth": {
              "enabled": true,
              "type": "continuous"
            }
          },
          "physics": {
            "enabled": true,
            "stabilization": {
              "enabled": true,
              "iterations": 100
            }
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """)
        
        # Save and show
        net.save_graph(output_file)
        print(f"Interactive visualization saved to {output_file}")
        print(f"Open {output_file} in a web browser to view the interactive graph")
        
        if notebook:
            return net
    
    def visualize_entity_neighborhood(self, entity_name: str,
                                     max_depth: int = 2,
                                     output_html: str = "entity_neighborhood.html",
                                     output_png: Optional[str] = None):
        """
        Visualize the neighborhood of a specific entity
        
        Args:
            entity_name: Name of the entity to center on
            max_depth: Maximum depth of neighborhood
            output_html: Path to save interactive HTML
            output_png: Optional path to save static PNG
        """
        print(f"Extracting neighborhood for '{entity_name}'...")
        
        # Extract subgraph
        subgraph = self.extract_subgraph_by_entities([entity_name], max_depth=max_depth)
        
        if subgraph.number_of_nodes() == 0:
            print(f"Error: Entity '{entity_name}' not found in graph")
            return
        
        # Create interactive visualization
        self.visualize_interactive_pyvis(subgraph, 
                                        max_nodes=subgraph.number_of_nodes(),
                                        output_file=output_html)
        
        # Optionally create static visualization
        if output_png:
            self.visualize_with_matplotlib(subgraph,
                                          max_nodes=subgraph.number_of_nodes(),
                                          output_file=output_png,
                                          show_labels=True)
    
    def visualize_relation_network(self, relation_types: List[str],
                                  max_nodes: int = 100,
                                  output_html: str = "relation_network.html"):
        """
        Visualize network of specific relation types
        
        Args:
            relation_types: List of relation types to visualize
            max_nodes: Maximum number of nodes
            output_html: Path to save interactive HTML
        """
        print(f"Extracting network for relations: {', '.join(relation_types)}")
        
        # Extract subgraph
        subgraph = self.extract_subgraph_by_relation_type(relation_types, 
                                                         max_nodes=max_nodes)
        
        if subgraph.number_of_nodes() == 0:
            print(f"Error: No edges found with relation types {relation_types}")
            return
        
        # Create interactive visualization
        self.visualize_interactive_pyvis(subgraph,
                                        max_nodes=subgraph.number_of_nodes(),
                                        output_file=output_html)
    
    def create_summary_statistics_plot(self, output_file: str = "graph_statistics.png"):
        """
        Create summary statistics visualization
        
        Args:
            output_file: Path to save the plot
        """
        # Collect statistics
        entity_counts = defaultdict(int)
        for node, data in self.graph.nodes(data=True):
            entity_type = data.get('entity_type', 'Unknown')
            entity_counts[entity_type] += 1
        
        relation_counts = defaultdict(int)
        for u, v, data in self.graph.edges(data=True):
            rel_type = data.get('relation_type', 'Unknown')
            relation_counts[rel_type] += 1
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Entity type distribution
        entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        entity_names = [e[0] for e in entities]
        entity_values = [e[1] for e in entities]
        colors1 = [self.get_node_color(e) for e in entity_names]
        
        ax1.barh(entity_names, entity_values, color=colors1, alpha=0.8)
        ax1.set_xlabel('Count', fontsize=12)
        ax1.set_title('Entity Type Distribution', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Relation type distribution (top 15)
        relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        relation_names = [r[0] for r in relations]
        relation_values = [r[1] for r in relations]
        
        ax2.barh(relation_names, relation_values, color='#3498db', alpha=0.8)
        ax2.set_xlabel('Count', fontsize=12)
        ax2.set_title('Top 15 Relation Types', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Statistics plot saved to {output_file}")
        plt.close()


def demo_visualizations():
    """Demonstrate visualization functions"""
    from json_to_network import NetworkXKnowledgeGraph
    
    print("=" * 70)
    print("Knowledge Graph Visualization Demo")
    print("=" * 70)
    
    # Load graph
    print("\nLoading graph...")
    kg = NetworkXKnowledgeGraph()
    if not kg.load_graph():
        print("Error: Could not load graph. Run json_to_networkx.py first.")
        return
    
    # Initialize visualizer
    viz = KnowledgeGraphVisualizer(kg.graph)
    
    # Create summary statistics
    print("\n1. Creating summary statistics plot...")
    viz.create_summary_statistics_plot("graph_statistics.png")
    
    # Create sample subgraph visualization
    print("\n2. Creating sample subgraph visualization...")
    # Pick a well-connected node and extract its neighborhood
    # This ensures the subgraph has edges!
    degrees = dict(kg.graph.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    
    if top_nodes:
        # Start from a highly connected node
        center_node = top_nodes[0][0]
        center_name = kg.graph.nodes[center_node].get('name', 'Unknown')
        print(f"   Centering on: {center_name} (degree: {top_nodes[0][1]})")
        
        # Extract 2-hop neighborhood
        import networkx as nx
        subgraph_nodes = {center_node}
        # Add 1-hop neighbors
        for neighbor in kg.graph.neighbors(center_node):
            subgraph_nodes.add(neighbor)
        # Add 2-hop neighbors (limited)
        for node in list(subgraph_nodes)[:10]:  # Limit to avoid too many nodes
            neighbors = list(kg.graph.neighbors(node))[:3]  # Max 3 neighbors each
            subgraph_nodes.update(neighbors)
        
        # Create subgraph
        subgraph = kg.graph.subgraph(list(subgraph_nodes)[:50])  # Limit to 50 nodes
    else:
        # Fallback: random sample
        sample_nodes = random.sample(list(kg.graph.nodes()), 
                                    min(50, kg.graph.number_of_nodes()))
        subgraph = kg.graph.subgraph(sample_nodes)
    
    print(f"   Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
    
    viz.visualize_with_matplotlib(subgraph, 
                                 output_file="sample_subgraph.png",
                                 show_labels=True)
    
    # Create interactive visualization
    print("\n3. Creating interactive visualization...")
    viz.visualize_interactive_pyvis(subgraph,
                                   output_file="interactive_sample.html")
    
    print("\n" + "=" * 70)
    print("Visualization demo completed!")
    print("\nGenerated files:")
    print("  - graph_statistics.png (summary statistics)")
    print("  - sample_subgraph.png (static network visualization)")
    print("  - interactive_sample.html (interactive visualization)")
    print("\nOpen the HTML file in a web browser to explore the interactive graph!")
    print("=" * 70)


if __name__ == "__main__":
    demo_visualizations()