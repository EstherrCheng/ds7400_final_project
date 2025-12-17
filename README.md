# PubMed Knowledge Graph Extraction System

A comprehensive system for extracting disease-related knowledge from PubMed abstracts and building a Neo4j knowledge graph for Graph-RAG applications.

## Overview

This system performs **two-stage extraction**:
1. **Stage 1**: Entity Recognition - Extract medical entities (diseases, symptoms, treatments, etc.)
2. **Stage 2**: Relation Extraction - Identify relationships between entities with confidence weights

**Target Dataset**: [MedRAG/pubmed](https://huggingface.co/datasets/MedRAG/pubmed)  
**Benchmark**: [MedQA-USMLE](https://www.kaggle.com/datasets/moaaztameer/medqa-usmle)  
**Model**: Qwen2.5-7b/Mixtral-8x7b-v0.1

---

## Quick Start

### Installation Steps

```pip install -r requirements.txt
```

## Usage

### Step 1: Extract Knowledge from PubMed

Run the extraction script:

```
python pubmed_knowledge_extraction.py --api_token 'your_hf_token'
```

**Output Files:**
```
pubmed_entities.json           # Extracted entities
pubmed_relations.json          # Extracted relations
pubmed_knowledge_graph.json    # Combined knowledge graph
extraction_checkpoint.json     # Resume checkpoint
```

The extraction is resumeable; you can rename the output files to save merged extraction outputs.

### Step 2: Create Knowledge graph

```
python json_to_network.py
```

### Step 3: Evaluation

```
test.jsonl   # MedQA test dataset
```

Run
```
python graph_rag/graphrag_system.py    --api_token 'your_hf_token' 
```
for graphrag results.

Run 
```
baseline.py
```
for baseline results.


## Configuration

Edit `config.py` and `graphrag_config.py`  to customize the parameters.


## File Structure

```
.
├── config.py                          # Configuration file
├── knowledge_extraction.py     # Main extraction script
├── json_to_network.py                   # Network loader script
├── llmgpt_huggingface.py                   # LLM wrapper
├── baseline.py                   # Querying baseline for medqa
├── knowledge_graph.py                   # Constructed knowledge graph
├── baseline_results.json                   # Baseline results
├── graghrag_results.py                   # Baseline+GraphRAG results
│── gragh_rag
	├── graph_visualizer.py               # Sample a subgraph and visualization
	├── graphrag_analyis.py               # Analysis script for graphrag results
	├── graphrag_config.py               # Configuration for graphrag
	├── graphrag_system.py               # Main graphrag system file
│── gragh_outputs
	├── pubmed_entities_merged.json               # Output: Extracted entities
	├── pubmed_relations_merged.json              # Output: Extracted relations
	├── pubmed_knowledge_graph_merged.json        # Output: Combined graph
	└── extraction_checkpoint.json         # Checkpoint for resuming
```

---
