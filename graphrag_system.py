"""
GraphRAG System for Medical Question Answering
Uses Knowledge Graph + Mixtral-8x7B via HuggingFace Inference API
"""

import json
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import re
from tqdm import tqdm
import time


@dataclass
class RetrievedKnowledge:
    """Structure for retrieved knowledge from graph"""
    entities: List[Dict[str, Any]]  # Retrieved entity nodes
    relations: List[Dict[str, Any]]  # Retrieved relations/edges
    paths: List[List[str]]  # Multi-hop paths
    subgraph_summary: str  # Summary of retrieved subgraph
    
    def to_context_string(self) -> str:
        """Convert retrieved knowledge to context string for LLM"""
        context_parts = []
        
        # Add entities
        if self.entities:
            context_parts.append("### Relevant Medical Entities:")
            for ent in self.entities[:10]:  # Limit to top 10
                entity_info = f"- {ent['name']} ({ent['type']})"
                if ent.get('context'):
                    entity_info += f": {ent['context'][:150]}..."
                context_parts.append(entity_info)
        
        # Add relations
        if self.relations:
            context_parts.append("\n### Relevant Medical Relations:")
            for rel in self.relations[:15]:  # Limit to top 15
                rel_info = f"- {rel['source']} -> {rel['relation']} -> {rel['target']}"
                if rel.get('evidence'):
                    rel_info += f" (Evidence: {rel['evidence'][:100]}...)"
                context_parts.append(rel_info)
        
        # Add paths
        if self.paths:
            context_parts.append("\n### Relevant Knowledge Paths:")
            for path in self.paths[:5]:  # Limit to top 5 paths
                context_parts.append(f"- {' -> '.join(path)}")
        
        return "\n".join(context_parts)


class MedicalEntityExtractor:
    """Extract medical entities from questions using simple pattern matching"""
    
    def __init__(self):
        # Common medical term patterns
        self.medical_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:syndrome|disease|disorder|cancer|tumor|carcinoma)\b',
            r'\b(?:diabetes|hypertension|asthma|pneumonia|infection|anemia|failure)\b',
            r'\b[A-Z]{2,}\b',  # Acronyms (e.g., COPD, HIV)
        ]
        
        # Medical stopwords to filter out
        self.stopwords = {
            'Which', 'What', 'Who', 'When', 'Where', 'Why', 'How',
            'The', 'A', 'An', 'In', 'On', 'At', 'To', 'For', 'Of',
            'ECG', 'CBC', 'MRI', 'CT'  # Keep imaging but don't use as search terms
        }
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract potential medical entities from text"""
        entities = set()
        
        # Extract using patterns
        for pattern in self.medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.update(matches)
        
        # Extract capitalized multi-word terms
        words = text.split()
        for i in range(len(words)):
            # Single capitalized words
            if words[i] and words[i][0].isupper() and len(words[i]) > 3:
                if words[i] not in self.stopwords:
                    entities.add(words[i].lower())
            
            # Two-word medical terms
            if i < len(words) - 1:
                bigram = f"{words[i]} {words[i+1]}"
                if any(term in bigram.lower() for term in ['blood', 'heart', 'kidney', 'liver', 'lung']):
                    entities.add(bigram.lower())
        
        # Clean up entities
        cleaned_entities = []
        for ent in entities:
            ent_clean = ent.strip().lower()
            if len(ent_clean) > 2 and ent_clean not in self.stopwords:
                cleaned_entities.append(ent_clean)
        
        return list(set(cleaned_entities))


class KnowledgeGraphRetriever:
    """Retrieve relevant knowledge from the medical knowledge graph"""
    
    def __init__(self, graph: nx.MultiDiGraph):
        """
        Initialize retriever with knowledge graph
        
        Args:
            graph: NetworkX MultiDiGraph containing medical knowledge
        """
        self.graph = graph
        self.entity_extractor = MedicalEntityExtractor()
        
        # Build search index for fast entity lookup
        self._build_search_index()
    
    def _build_search_index(self):
        """Build search index mapping entity names to node IDs"""
        print("Building search index...")
        self.entity_name_to_ids = defaultdict(list)
        self.entity_name_lower_to_ids = defaultdict(list)
        
        for node_id, data in self.graph.nodes(data=True):
            entity_name = data.get('name', '').strip()
            if entity_name:
                # Exact match index
                self.entity_name_to_ids[entity_name].append(node_id)
                # Case-insensitive index
                self.entity_name_lower_to_ids[entity_name.lower()].append(node_id)
        
        print(f"Indexed {len(self.entity_name_lower_to_ids)} unique entities")
    
    def _find_matching_nodes(self, query_entities: List[str], top_k: int = 10) -> List[str]:
        """Find graph nodes matching query entities"""
        matching_nodes = []
        
        for query_entity in query_entities:
            query_lower = query_entity.lower()
            
            # Exact match
            if query_lower in self.entity_name_lower_to_ids:
                matching_nodes.extend(self.entity_name_lower_to_ids[query_lower])
            else:
                # Partial match
                for entity_name, node_ids in self.entity_name_lower_to_ids.items():
                    if query_lower in entity_name or entity_name in query_lower:
                        matching_nodes.extend(node_ids)
        
        # Remove duplicates and limit
        return list(set(matching_nodes))[:top_k]
    
    def _extract_node_info(self, node_id: str) -> Dict[str, Any]:
        """Extract information from a graph node"""
        data = self.graph.nodes[node_id]
        return {
            'id': node_id,
            'name': data.get('name', 'Unknown'),
            'type': data.get('entity_type', 'Unknown'),
            'context': data.get('context', ''),
            'confidence': data.get('confidence', 0.0),
            'pmid': data.get('pmid', '')
        }
    
    def _extract_relation_info(self, source_id: str, target_id: str, 
                               edge_data: Dict) -> Dict[str, Any]:
        """Extract information from a graph edge"""
        source_name = self.graph.nodes[source_id].get('name', 'Unknown')
        target_name = self.graph.nodes[target_id].get('name', 'Unknown')
        
        return {
            'source': source_name,
            'target': target_name,
            'relation': edge_data.get('relation_type', 'RELATED_TO'),
            'evidence': edge_data.get('evidence', ''),
            'weight': edge_data.get('weight', 0.5),
            'confidence': edge_data.get('confidence', 0.0),
            'pmid': edge_data.get('pmid', '')
        }
    
    def _find_paths(self, source_nodes: List[str], max_depth: int = 2) -> List[List[str]]:
        """Find multi-hop paths between entities"""
        paths = []
        
        if len(source_nodes) < 2:
            return paths
        
        # Try to find paths between pairs of source nodes
        for i in range(len(source_nodes)):
            for j in range(i + 1, min(i + 3, len(source_nodes))):  # Limit pairs
                source = source_nodes[i]
                target = source_nodes[j]
                
                try:
                    # Find shortest path
                    path = nx.shortest_path(self.graph, source, target)
                    if len(path) <= max_depth + 1:  # +1 because path includes both endpoints
                        # Convert to entity names
                        path_names = [self.graph.nodes[n].get('name', 'Unknown') 
                                     for n in path]
                        paths.append(path_names)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue
                
                if len(paths) >= 5:  # Limit number of paths
                    break
            if len(paths) >= 5:
                break
        
        return paths
    
    def retrieve(self, question: str, max_entities: int = 10, 
                max_relations: int = 20) -> RetrievedKnowledge:
        """
        Retrieve relevant knowledge for a question
        
        Args:
            question: The medical question
            max_entities: Maximum number of entities to retrieve
            max_relations: Maximum number of relations to retrieve
            
        Returns:
            RetrievedKnowledge object with retrieved information
        """
        # Extract entities from question
        query_entities = self.entity_extractor.extract_entities(question)
        
        # Find matching nodes in graph
        matching_nodes = self._find_matching_nodes(query_entities, top_k=max_entities)
        
        if not matching_nodes:
            # No matches found - return empty knowledge
            return RetrievedKnowledge(
                entities=[],
                relations=[],
                paths=[],
                subgraph_summary="No relevant entities found in knowledge graph."
            )
        
        # Extract entity information
        entities = [self._extract_node_info(node_id) for node_id in matching_nodes]
        
        # Get relations involving these entities
        relations = []
        for node_id in matching_nodes:
            # Outgoing edges
            for neighbor in self.graph.neighbors(node_id):
                for edge_key in self.graph[node_id][neighbor]:
                    edge_data = self.graph[node_id][neighbor][edge_key]
                    rel_info = self._extract_relation_info(node_id, neighbor, edge_data)
                    relations.append(rel_info)
                    
                    if len(relations) >= max_relations:
                        break
                if len(relations) >= max_relations:
                    break
            
            if len(relations) >= max_relations:
                break
        
        # Find paths between entities
        paths = self._find_paths(matching_nodes, max_depth=2)
        
        # Create summary
        summary = f"Found {len(entities)} relevant entities and {len(relations)} relations."
        
        return RetrievedKnowledge(
            entities=entities,
            relations=relations,
            paths=paths,
            subgraph_summary=summary
        )


class MixtralLLM:
    """Interface to Mixtral-8x7B via HuggingFace Inference API"""
    
    def __init__(self, api_token: str, model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):
        """
        Initialize Mixtral LLM
        
        Args:
            api_token: HuggingFace API token
            model_name: Model identifier on HuggingFace
        """
        self.api_token = api_token
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        # Import requests here to avoid dependency issues
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("Please install requests: pip install requests")
    
    def generate(self, prompt: str, max_tokens: int = 512, 
                temperature: float = 0.1) -> str:
        """
        Generate text using Mixtral
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        headers = {
            "Authorization": f"Bearer {self.api_token}"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.95,
                "do_sample": True if temperature > 0 else False,
                "return_full_text": False
            }
        }
        
        # Retry logic for API calls
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self.requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get('generated_text', '').strip()
                    else:
                        return str(result)
                elif response.status_code == 503:
                    # Model loading, wait and retry
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        return f"Error: Model loading failed after {max_retries} attempts"
                else:
                    return f"Error: API returned status {response.status_code}: {response.text}"
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    return f"Error: {str(e)}"
        
        return "Error: Max retries exceeded"


class GraphRAGSystem:
    """Complete GraphRAG system for medical QA"""
    
    def __init__(self, graph: nx.MultiDiGraph, llm: MixtralLLM):
        """
        Initialize GraphRAG system
        
        Args:
            graph: Medical knowledge graph
            llm: Mixtral LLM interface
        """
        self.retriever = KnowledgeGraphRetriever(graph)
        self.llm = llm
    
    def create_prompt(self, question: str, options: Dict[str, str], 
                     knowledge: RetrievedKnowledge) -> str:
        """
        Create prompt with retrieved knowledge
        
        Args:
            question: Medical question
            options: Answer options
            knowledge: Retrieved knowledge from graph
            
        Returns:
            Complete prompt for LLM
        """
        # Convert knowledge to context string
        context = knowledge.to_context_string()
        
        # Format options
        options_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
        
        # Create prompt
        prompt = f"""[INST] You are a medical expert assistant. Use the following knowledge from a medical knowledge graph to help answer the question.

{context}

Question: {question}

Options:
{options_text}

Based on the provided medical knowledge and your expertise, which option is correct? Respond with ONLY the letter (A, B, C, D, or E) of the correct answer. [/INST]

Answer: """
        
        return prompt
    
    def answer_question(self, question: str, options: Dict[str, str]) -> Tuple[str, RetrievedKnowledge]:
        """
        Answer a medical question using GraphRAG
        
        Args:
            question: Medical question
            options: Answer options dict
            
        Returns:
            Tuple of (predicted_answer, retrieved_knowledge)
        """
        # Retrieve relevant knowledge
        knowledge = self.retriever.retrieve(question, max_entities=10, max_relations=20)
        
        # Create prompt
        prompt = self.create_prompt(question, options, knowledge)
        
        # Generate answer
        response = self.llm.generate(prompt, max_tokens=10, temperature=0.1)
        
        # Extract answer letter
        predicted_answer = self._extract_answer_letter(response)
        
        return predicted_answer, knowledge
    
    def _extract_answer_letter(self, response: str) -> str:
        """Extract answer letter from model response"""
        # Look for A, B, C, D, or E
        match = re.search(r'\b([A-E])\b', response.upper())
        if match:
            return match.group(1)
        
        # If no match, return first letter of response
        if response and response[0].upper() in 'ABCDE':
            return response[0].upper()
        
        return "A"  # Default fallback
    
    def evaluate_dataset(self, test_file: str, output_file: str, 
                        limit: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate on test dataset
        
        Args:
            test_file: Path to test JSONL file
            output_file: Path to save results
            limit: Optional limit on number of samples to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        print("=" * 70)
        print("GraphRAG Evaluation")
        print("=" * 70)
        
        # Load test data
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line))
        
        if limit:
            test_data = test_data[:limit]
        
        print(f"\nEvaluating on {len(test_data)} questions...")
        
        # Evaluate
        results = []
        correct = 0
        total = 0
        
        for i, item in enumerate(tqdm(test_data, desc="Evaluating")):
            question = item['question']
            options = item['options']
            correct_answer = item['answer_idx']
            
            # Get prediction
            predicted_answer, knowledge = self.answer_question(question, options)
            
            # Check if correct
            is_correct = (predicted_answer == correct_answer)
            if is_correct:
                correct += 1
            total += 1
            
            # Save result
            result = {
                'question_id': i,
                'question': question,
                'options': options,
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'meta_info': item.get('meta_info', ''),
                'num_entities_retrieved': len(knowledge.entities),
                'num_relations_retrieved': len(knowledge.relations),
                'num_paths_found': len(knowledge.paths)
            }
            results.append(result)
            
            # Print progress every 50 questions
            if (i + 1) % 50 == 0:
                current_accuracy = correct / total
                print(f"\n[{i+1}/{len(test_data)}] Current Accuracy: {current_accuracy:.4f}")
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
        
        # Save results
        output_data = {
            'metrics': metrics,
            'results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 70)
        print("Evaluation Results")
        print("=" * 70)
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"Results saved to: {output_file}")
        print("=" * 70)
        
        return metrics


def main():
    """Main function to run GraphRAG evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='GraphRAG Medical QA System')
    parser.add_argument('--graph_file', type=str, 
                       default='knowledge_graph.gpickle',
                       help='Path to knowledge graph pickle file')
    parser.add_argument('--test_file', type=str,
                       default='test.jsonl',
                       help='Path to test JSONL file')
    parser.add_argument('--output_file', type=str,
                       default='graphrag_results.json',
                       help='Path to save results')
    parser.add_argument('--api_token', type=str, required=True,
                       help='HuggingFace API token')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of test samples')
    parser.add_argument('--model_name', type=str,
                       default='mistralai/Mixtral-8x7B-Instruct-v0.1',
                       help='HuggingFace model name')
    
    args = parser.parse_args()
    
    # Load knowledge graph
    print("Loading knowledge graph...")
    import pickle
    with open(args.graph_file, 'rb') as f:
        graph = pickle.load(f)
    print(f"Loaded graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Initialize LLM
    print("\nInitializing Mixtral LLM...")
    llm = MixtralLLM(api_token=args.api_token, model_name=args.model_name)
    
    # Initialize GraphRAG system
    print("Initializing GraphRAG system...")
    graphrag = GraphRAGSystem(graph=graph, llm=llm)
    
    # Evaluate
    metrics = graphrag.evaluate_dataset(
        test_file=args.test_file,
        output_file=args.output_file,
        limit=args.limit
    )
    
    print("\nEvaluation complete!")
    return metrics


if __name__ == "__main__":
    main()