"""
Convert JSON Knowledge Graph to NetworkX Graph
A lightweight replacement for Neo4j-based knowledge graph storage
"""

import json
import pickle
import networkx as nx
from tqdm import tqdm
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import os

# Try to import config, if not available use defaults
try:
    import config
    ENTITIES_OUTPUT = config.ENTITIES_OUTPUT
    RELATIONS_OUTPUT = config.RELATIONS_OUTPUT
    ENTITY_TYPES = config.ENTITY_TYPES
except ImportError:
    print("Warning: config.py not found, using default values")
    ENTITIES_OUTPUT = "entities_output.json"
    RELATIONS_OUTPUT = "relations_output.json"
    ENTITY_TYPES = {
        'Disease': 'disease',
        'Clinical_Manifestation': 'clinical_manifestation',
        'Treatment': 'treatment',
        'Drug': 'drug',
        'Diagnostic_Procedure': 'diagnostic_procedure',
        'Risk_Factor': 'risk_factor',
        'Complication': 'complication',
        'Pathophysiology': 'pathophysiology',
    }


class NetworkXKnowledgeGraph:
    """Knowledge Graph using NetworkX"""
    
    def __init__(self, graph_file: str = "knowledge_graph.gpickle"):
        """
        Initialize NetworkX knowledge graph
        
        Args:
            graph_file: Path to save/load graph pickle file
        """
        # Use MultiDiGraph to allow multiple edges between same nodes
        self.graph = nx.MultiDiGraph()
        self.graph_file = graph_file
        
        # Statistics
        self.stats = {
            'total_nodes': 0,
            'total_edges': 0,
            'nodes_by_type': {},
            'edges_by_type': {},
            'pmids': set()
        }
        
    def clear_graph(self):
        """Clear all nodes and edges"""
        print("Clearing existing graph...")
        self.graph.clear()
        self.stats = {
            'total_nodes': 0,
            'total_edges': 0,
            'nodes_by_type': {},
            'edges_by_type': {},
            'pmids': set()
        }
        print("Graph cleared.")
    
    def create_entity_node(self, entity: Dict[str, Any], pmid: str) -> str:
        """
        Create an entity node in the graph
        
        Args:
            entity: Entity dictionary
            pmid: PubMed ID
            
        Returns:
            Unique entity ID
        """
        entity_id = entity.get('id')
        entity_type = entity.get('type')
        name = entity.get('name', '')
        properties = entity.get('properties', {})
        context = entity.get('context', '')
        confidence = entity.get('confidence', 0.0)
        
        # Create unique entity_id combining PMID and local ID
        unique_entity_id = f"{pmid}_{entity_id}"
        
        # Prepare node attributes
        node_attributes = {
            'entity_id': unique_entity_id,
            'name': name,
            'entity_type': entity_type,
            'pmid': pmid,
            'context': context,
            'confidence': confidence,
        }
        
        # Add additional properties
        for key, value in properties.items():
            if value is not None and value != "":
                node_attributes[f'prop_{key}'] = value
        
        # Add node to graph
        self.graph.add_node(unique_entity_id, **node_attributes)
        
        # Update statistics
        self.stats['nodes_by_type'][entity_type] = \
            self.stats['nodes_by_type'].get(entity_type, 0) + 1
        self.stats['pmids'].add(pmid)
        
        return unique_entity_id
    
    def create_relationship(self, relation: Dict[str, Any], pmid: str, 
                          entity_id_map: Dict[str, str]):
        """
        Create a relationship (edge) in the graph
        
        Args:
            relation: Relation dictionary
            pmid: PubMed ID
            entity_id_map: Mapping from local entity IDs to unique entity IDs
        """
        source_local_id = relation.get('source_id')
        target_local_id = relation.get('target_id')
        relation_type = relation.get('relation_type')
        properties = relation.get('properties', {})
        evidence = relation.get('evidence', '')
        weight = relation.get('weight', 0.5)
        confidence = relation.get('confidence', 0.0)
        
        # Get unique entity IDs
        source_id = entity_id_map.get(source_local_id)
        target_id = entity_id_map.get(target_local_id)
        
        if not source_id or not target_id:
            return
        
        # Check if nodes exist
        if source_id not in self.graph.nodes or target_id not in self.graph.nodes:
            return
        
        # Prepare edge attributes
        edge_attributes = {
            'relation_type': relation_type,
            'pmid': pmid,
            'evidence': evidence,
            'weight': weight,
            'confidence': confidence,
        }
        
        # Add additional properties
        for key, value in properties.items():
            if value is not None and value != "":
                edge_attributes[f'prop_{key}'] = value
        
        # Add edge to graph
        self.graph.add_edge(source_id, target_id, **edge_attributes)
        
        # Update statistics
        self.stats['edges_by_type'][relation_type] = \
            self.stats['edges_by_type'].get(relation_type, 0) + 1
    
    def load_knowledge_graph(self, entities_file: str, relations_file: str):
        """
        Load knowledge graph from JSON files
        
        Args:
            entities_file: Path to entities JSON file
            relations_file: Path to relations JSON file
        """
        print("=" * 70)
        print("Loading Knowledge Graph into NetworkX")
        print("=" * 70)
        
        # Load entities
        print(f"\nLoading entities from {entities_file}...")
        with open(entities_file, 'r', encoding='utf-8') as f:
            entities_data = json.load(f)
        
        # Load relations
        print(f"Loading relations from {relations_file}...")
        with open(relations_file, 'r', encoding='utf-8') as f:
            relations_data = json.load(f)
        
        # Handle both formats: with/without metadata wrapper
        if isinstance(entities_data, dict) and 'entities' in entities_data:
            entities_docs = entities_data['entities']
            print(f"Metadata: {entities_data.get('metadata', {})}")
        else:
            entities_docs = entities_data
            
        if isinstance(relations_data, dict) and 'relations' in relations_data:
            relations_docs = relations_data['relations']
        else:
            relations_docs = relations_data
        
        print(f"\nTotal entity documents: {len(entities_docs)}")
        print(f"Total relation documents: {len(relations_docs)}")
        
        # Create entity ID mapping for relations
        entity_id_mapping = {}  # Maps PMID -> {local_id -> unique_id}
        
        # Process entities
        print("\nCreating entity nodes...")
        total_entities = sum(len(doc.get('entities', [])) for doc in entities_docs)
        
        with tqdm(total=total_entities, desc="Creating entities") as pbar:
            for doc in entities_docs:
                pmid = doc.get('pmid', 'unknown')
                entities = doc.get('entities', [])
                
                if pmid not in entity_id_mapping:
                    entity_id_mapping[pmid] = {}
                
                for entity in entities:
                    local_id = entity.get('id')
                    unique_id = self.create_entity_node(entity, pmid)
                    entity_id_mapping[pmid][local_id] = unique_id
                    pbar.update(1)
        
        # Process relations
        print("\nCreating relationships...")
        total_relations = sum(len(doc.get('relations', [])) for doc in relations_docs)
        
        with tqdm(total=total_relations, desc="Creating relations") as pbar:
            for doc in relations_docs:
                pmid = doc.get('pmid', 'unknown')
                relations = doc.get('relations', [])
                
                # Get entity mapping for this PMID
                id_map = entity_id_mapping.get(pmid, {})
                
                for relation in relations:
                    self.create_relationship(relation, pmid, id_map)
                    pbar.update(1)
        
        # Update final statistics
        self.stats['total_nodes'] = self.graph.number_of_nodes()
        self.stats['total_edges'] = self.graph.number_of_edges()
        
        print("\n" + "=" * 70)
        print("Knowledge Graph Loaded Successfully!")
        print("=" * 70)
        
        # Print statistics
        self.print_statistics()
    
    def print_statistics(self):
        """Print graph statistics"""
        print("\nGraph Statistics:")
        print("-" * 70)
        
        print(f"Total nodes: {self.stats['total_nodes']}")
        print(f"Total edges: {self.stats['total_edges']}")
        print(f"Total PMIDs: {len(self.stats['pmids'])}")
        
        # Nodes by type
        print("\nNodes by type:")
        for entity_type, count in sorted(self.stats['nodes_by_type'].items(), 
                                        key=lambda x: x[1], reverse=True):
            print(f"  - {entity_type}: {count}")
        
        # Edges by type (top 10)
        print("\nTop 10 edge types:")
        sorted_edges = sorted(self.stats['edges_by_type'].items(), 
                            key=lambda x: x[1], reverse=True)[:10]
        for rel_type, count in sorted_edges:
            print(f"  - {rel_type}: {count}")
        
        # Graph properties
        print("\nGraph properties:")
        print(f"  - Is directed: {self.graph.is_directed()}")
        print(f"  - Is multigraph: {self.graph.is_multigraph()}")
        
        # Connectivity
        if self.graph.number_of_nodes() > 0:
            if nx.is_weakly_connected(self.graph):
                print(f"  - Graph is weakly connected")
            else:
                num_components = nx.number_weakly_connected_components(self.graph)
                print(f"  - Number of weakly connected components: {num_components}")
                largest_component_size = len(max(nx.weakly_connected_components(self.graph), 
                                                key=len))
                print(f"  - Largest component size: {largest_component_size}")
        
        print("-" * 70)
    
    def save_graph(self, filepath: Optional[str] = None):
        """
        Save graph to disk using pickle
        
        Args:
            filepath: Path to save file (default: self.graph_file)
        """
        if filepath is None:
            filepath = self.graph_file
        
        print(f"\nSaving graph to {filepath}...")
        
        # Save graph
        with open(filepath, 'wb') as f:
            pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save statistics separately for quick access
        stats_file = filepath.replace('.gpickle', '_stats.json')
        stats_to_save = {
            'total_nodes': self.stats['total_nodes'],
            'total_edges': self.stats['total_edges'],
            'nodes_by_type': self.stats['nodes_by_type'],
            'edges_by_type': self.stats['edges_by_type'],
            'num_pmids': len(self.stats['pmids']),
            'saved_at': datetime.now().isoformat()
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_to_save, f, indent=2)
        
        # Get file size
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"Graph saved successfully! (Size: {file_size:.2f} MB)")
        print(f"Statistics saved to {stats_file}")
    
    def load_graph(self, filepath: Optional[str] = None):
        """
        Load graph from disk
        
        Args:
            filepath: Path to load file (default: self.graph_file)
        """
        if filepath is None:
            filepath = self.graph_file
        
        if not os.path.exists(filepath):
            print(f"Error: Graph file {filepath} not found!")
            return False
        
        print(f"\nLoading graph from {filepath}...")
        
        with open(filepath, 'rb') as f:
            self.graph = pickle.load(f)
        
        # Rebuild statistics
        self.stats['total_nodes'] = self.graph.number_of_nodes()
        self.stats['total_edges'] = self.graph.number_of_edges()
        
        # Count nodes by type
        self.stats['nodes_by_type'] = {}
        for node, data in self.graph.nodes(data=True):
            entity_type = data.get('entity_type', 'Unknown')
            self.stats['nodes_by_type'][entity_type] = \
                self.stats['nodes_by_type'].get(entity_type, 0) + 1
        
        # Count edges by type
        self.stats['edges_by_type'] = {}
        for u, v, data in self.graph.edges(data=True):
            rel_type = data.get('relation_type', 'Unknown')
            self.stats['edges_by_type'][rel_type] = \
                self.stats['edges_by_type'].get(rel_type, 0) + 1
        
        # Collect PMIDs
        self.stats['pmids'] = set()
        for node, data in self.graph.nodes(data=True):
            pmid = data.get('pmid')
            if pmid:
                self.stats['pmids'].add(pmid)
        
        print(f"Graph loaded successfully!")
        print(f"  - Nodes: {self.stats['total_nodes']}")
        print(f"  - Edges: {self.stats['total_edges']}")
        
        return True
    
    def export_to_formats(self, base_filename: str = "knowledge_graph"):
        """
        Export graph to various formats for analysis
        
        Args:
            base_filename: Base name for output files
        """
        print("\nExporting graph to various formats...")
        
        # 1. GraphML (for Gephi, Cytoscape)
        graphml_file = f"{base_filename}.graphml"
        nx.write_graphml(self.graph, graphml_file)
        print(f"  - GraphML exported to {graphml_file}")
        
        # 2. GML (for network analysis tools)
        gml_file = f"{base_filename}.gml"
        nx.write_gml(self.graph, gml_file)
        print(f"  - GML exported to {gml_file}")
        
        # 3. Edge list (simple format)
        edgelist_file = f"{base_filename}_edges.txt"
        nx.write_edgelist(self.graph, edgelist_file, data=True)
        print(f"  - Edge list exported to {edgelist_file}")
        
        # 4. JSON (node-link format)
        json_file = f"{base_filename}.json"
        from networkx.readwrite import json_graph
        data = json_graph.node_link_data(self.graph)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"  - JSON exported to {json_file}")
        
        print("Export completed!")


def print_sample_queries():
    """Print sample query patterns that can be implemented"""
    print("\n" + "=" * 70)
    print("Sample Query Patterns (Python/NetworkX Implementation)")
    print("=" * 70)
    
    queries = [
        {
            "description": "Find all diseases and their symptoms",
            "code": """
# Find disease nodes connected to clinical manifestations
for disease_node in [n for n, d in kg.graph.nodes(data=True) 
                     if d.get('entity_type') == 'Disease']:
    disease_name = kg.graph.nodes[disease_node]['name']
    
    # Find outgoing edges with PRESENTS_WITH relation
    for neighbor in kg.graph.neighbors(disease_node):
        for edge_key in kg.graph[disease_node][neighbor]:
            edge_data = kg.graph[disease_node][neighbor][edge_key]
            if edge_data.get('relation_type') == 'PRESENTS_WITH':
                neighbor_data = kg.graph.nodes[neighbor]
                if neighbor_data.get('entity_type') == 'Clinical_Manifestation':
                    symptom = neighbor_data['name']
                    weight = edge_data.get('weight', 0)
                    print(f"{disease_name} -> {symptom} (strength: {weight})")
            """
        },
        {
            "description": "Find treatment options for a specific disease",
            "code": """
# Search for disease by name
disease_name = "diabetes mellitus"
disease_nodes = [n for n, d in kg.graph.nodes(data=True) 
                 if d.get('entity_type') == 'Disease' 
                 and disease_name.lower() in d.get('name', '').lower()]

for disease_node in disease_nodes:
    for neighbor in kg.graph.neighbors(disease_node):
        for edge_key in kg.graph[disease_node][neighbor]:
            edge_data = kg.graph[disease_node][neighbor][edge_key]
            rel_type = edge_data.get('relation_type')
            
            if 'TREATMENT' in rel_type:
                neighbor_data = kg.graph.nodes[neighbor]
                if neighbor_data.get('entity_type') == 'Treatment':
                    treatment = neighbor_data['name']
                    print(f"{disease_name} -{rel_type}-> {treatment}")
            """
        },
        {
            "description": "Find multi-hop paths (symptom -> disease -> test)",
            "code": """
# Find all paths of length 2
for symptom_node in [n for n, d in kg.graph.nodes(data=True) 
                     if d.get('entity_type') == 'Clinical_Manifestation']:
    
    # Find diseases suggested by this symptom
    for disease_node in kg.graph.neighbors(symptom_node):
        for edge1_key in kg.graph[symptom_node][disease_node]:
            edge1_data = kg.graph[symptom_node][disease_node][edge1_key]
            
            if edge1_data.get('relation_type') == 'SUGGESTS':
                # Find diagnostic tests for this disease
                for test_node in kg.graph.neighbors(disease_node):
                    for edge2_key in kg.graph[disease_node][test_node]:
                        edge2_data = kg.graph[disease_node][test_node][edge2_key]
                        
                        if edge2_data.get('relation_type') == 'CONFIRMED_BY':
                            symptom = kg.graph.nodes[symptom_node]['name']
                            disease = kg.graph.nodes[disease_node]['name']
                            test = kg.graph.nodes[test_node]['name']
                            print(f"{symptom} -> {disease} -> {test}")
            """
        },
        {
            "description": "Find diseases with shared risk factors",
            "code": """
# Build a mapping: risk_factor -> list of diseases
risk_factor_to_diseases = {}

for rf_node in [n for n, d in kg.graph.nodes(data=True) 
                if d.get('entity_type') == 'Risk_Factor']:
    rf_name = kg.graph.nodes[rf_node]['name']
    diseases = []
    
    for disease_node in kg.graph.neighbors(rf_node):
        for edge_key in kg.graph[rf_node][disease_node]:
            edge_data = kg.graph[rf_node][disease_node][edge_key]
            if edge_data.get('relation_type') == 'INCREASES_RISK':
                disease_data = kg.graph.nodes[disease_node]
                if disease_data.get('entity_type') == 'Disease':
                    diseases.append(disease_data['name'])
    
    if len(diseases) >= 2:
        risk_factor_to_diseases[rf_name] = diseases

# Print shared risk factors
for rf, diseases in risk_factor_to_diseases.items():
    print(f"Risk factor '{rf}' is shared by: {', '.join(diseases)}")
            """
        },
        {
            "description": "Calculate node centrality (most important entities)",
            "code": """
# Calculate degree centrality
degree_centrality = nx.degree_centrality(kg.graph)

# Get top 10 most central nodes
top_nodes = sorted(degree_centrality.items(), 
                  key=lambda x: x[1], reverse=True)[:10]

print("Top 10 most connected entities:")
for node_id, centrality in top_nodes:
    node_data = kg.graph.nodes[node_id]
    name = node_data.get('name', 'Unknown')
    entity_type = node_data.get('entity_type', 'Unknown')
    print(f"  {name} ({entity_type}): {centrality:.4f}")
            """
        }
    ]
    
    for i, q in enumerate(queries, 1):
        print(f"\n{i}. {q['description']}")
        print(f"```python")
        print(q['code'].strip())
        print(f"```")
    
    print("\n" + "=" * 70)
    print("Note: Use graph_queries.py for ready-to-use query functions!")
    print("=" * 70)


def main():
    """Main function to load knowledge graph"""
    print("NetworkX Knowledge Graph Loader")
    print("=" * 70)
    
    # Initialize graph
    kg = NetworkXKnowledgeGraph(graph_file="medical_knowledge_graph.gpickle")
    
    # Ask want to load existing graph or create new one
    if os.path.exists(kg.graph_file):
        response = input(f"\nExisting graph found at {kg.graph_file}. Load it? (yes/no): ").strip().lower()
        if response == 'yes':
            if kg.load_graph():
                kg.print_statistics()
                
                kg.export_to_formats()
                
                print("\nGraph loaded successfully.")
                print_sample_queries()
                return
    
    # Create new graph from JSON files
    try:
        # Load knowledge graph
        kg.load_knowledge_graph(
            entities_file=ENTITIES_OUTPUT,
            relations_file=RELATIONS_OUTPUT
        )
        
        # Save graph
        kg.save_graph()
        
        kg.export_to_formats()
        
        # Print sample queries
        print_sample_queries()
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure the JSON files exist:")
        print(f"  - {ENTITIES_OUTPUT}")
        print(f"  - {RELATIONS_OUTPUT}")
        print("\nYou can modify paths in config.py or pass them as arguments.")
    except Exception as e:
        print(f"\nError loading knowledge graph: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()