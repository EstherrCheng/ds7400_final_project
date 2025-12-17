"""
PubMed Knowledge Extraction System - Two-Stage Extraction (Continue from 1001)
Stage 1: Entity Recognition
Stage 2: Relation Extraction

Modified to:
- Start from dataset index 1000 (item 1001)
- Extract 4000 more items (1001-5000)
- Merge with existing data
"""

import json
import time
import os
from tqdm import tqdm
import sys
from typing import List, Dict, Any, Optional
import re
from datasets import load_dataset

# Import LLM library
sys.path.append('.')
from llm_huggingface import llm_api, set_llm_provider
import config

# ==================== Configuration Override ====================
# Start from dataset index 1000 (the 1001st item)
CUSTOM_START_INDEX = 7200
# Extract 4000 items (from 1001 to 5000)
CUSTOM_EXTRACT_COUNT = 2800
# Total target index
CUSTOM_END_INDEX = CUSTOM_START_INDEX + CUSTOM_EXTRACT_COUNT  # 10000

# Modified output files to preserve originals
ENTITIES_OUTPUT_NEW = config.ENTITIES_OUTPUT.replace('.json', '_continued2.json')
RELATIONS_OUTPUT_NEW = config.RELATIONS_OUTPUT.replace('.json', '_continued2.json')
KNOWLEDGE_GRAPH_OUTPUT_NEW = config.KNOWLEDGE_GRAPH_OUTPUT.replace('.json', '_continued2.json')
CHECKPOINT_FILE_NEW = config.CHECKPOINT_FILE.replace('.json', '_continued2.json')

# ==================== Stage 1: Entity Extraction Prompt ====================
ENTITY_EXTRACTION_PROMPT = """You are a professional biomedical knowledge extraction expert. Extract disease-related entities from the following PubMed abstract.

**Abstract:**
{text}

**PMID:** {pmid}

**Instructions:**
Extract ALL entities that fall into these categories:
1. Disease - disease names, conditions, disorders
2. Clinical_Manifestation - symptoms, signs, clinical findings (specify type: symptom/sign/lab_finding)
3. Diagnostic_Procedure - tests, examinations, diagnostic methods (specify modality: lab/imaging/physical_exam/biopsy)
4. Treatment - therapeutic interventions (specify type: pharmacologic/surgical/supportive/preventive)
5. Drug - medications, pharmaceutical agents
6. Risk_Factor - risk factors, predisposing conditions (specify type: modifiable/non-modifiable)
7. Pathophysiology - disease mechanisms, pathological processes
8. Complication - complications, adverse outcomes
9. Differential_Diagnosis - diseases that need to be differentiated

**Output Format (JSON only, no explanation):**
{{
  "entities": [
    {{
      "id": "E1",
      "type": "Disease",
      "name": "diabetes mellitus",
      "properties": {{
        "category": "endocrine disorder"
      }},
      "context": "the sentence or phrase where this entity appears",
      "confidence": 0.95
    }},
    {{
      "id": "E2",
      "type": "Clinical_Manifestation",
      "name": "hyperglycemia",
      "properties": {{
        "type": "lab_finding",
        "specificity": "typical"
      }},
      "context": "context sentence",
      "confidence": 0.90
    }}
  ]
}}

**Important:**
- Assign unique IDs (E1, E2, E3, ...) to each entity
- Include ALL required properties for each entity type
- Set confidence score (0.0-1.0) based on clarity in the text
- Include surrounding context (1-2 sentences)
- If no entities found, return {{"entities": []}}
- Output ONLY valid JSON, no additional text
"""

# ==================== Stage 2: Relation Extraction Prompt ====================
RELATION_EXTRACTION_PROMPT = """You are a professional biomedical knowledge extraction expert. Extract relationships between the previously identified entities.

**Abstract:**
{text}

**Identified Entities:**
{entities_json}

**Instructions:**
For each pair of entities, determine if there is a meaningful relationship. Use these relation types:

**Diagnostic Reasoning:**
- SUGGESTS: Clinical manifestation suggests a disease
- PATHOGNOMONIC_FOR: Clinical manifestation is pathognomonic for disease
- PRESENTS_WITH: Disease presents with clinical manifestation
- CONFIRMED_BY: Disease confirmed by diagnostic procedure
- DIAGNOSTIC_FOR: Diagnostic procedure is diagnostic for disease
- REQUIRES_WORKUP: Disease requires diagnostic workup

**Treatment:**
- FIRST_LINE_TREATMENT: Primary treatment for disease
- ALTERNATIVE_TREATMENT: Alternative treatment option
- TREATED_WITH: Disease treated with drug/treatment
- ALLEVIATES: Drug/treatment alleviates manifestation
- CAUSES: Drug causes side effect (manifestation)

**Etiology:**
- INCREASES_RISK: Risk factor increases disease risk
- EXPLAINS: Pathophysiology explains disease
- HAS_MECHANISM: Disease has pathophysiological mechanism

**Disease Associations:**
- LEADS_TO: Disease leads to complication
- DIFFERENTIAL_DIAGNOSIS: Diseases need differentiation
- COMMONLY_COEXISTS_WITH: Diseases commonly coexist

**Drug Interactions:**
- CONTRAINDICATED_WITH: Drug contraindicated with another drug
- CONTRAINDICATED_IN: Drug contraindicated in disease
- SYNERGISTIC_WITH: Drugs have synergistic effect

**Temporal:**
- PRECEDES: One manifestation precedes another
- PREVENTS: Treatment prevents complication

**Output Format (JSON only):**
{{
  "relations": [
    {{
      "source_id": "E1",
      "target_id": "E2",
      "relation_type": "PRESENTS_WITH",
      "properties": {{
        "frequency": "common",
        "strength": "strong"
      }},
      "evidence": "exact sentence from text supporting this relation",
      "weight": 0.85,
      "confidence": 0.90
    }}
  ]
}}

**Important:**
- weight: Strength of relationship (0.0-1.0), consider specificity and frequency
- confidence: Confidence in extraction (0.0-1.0)
- Only extract relations explicitly or implicitly stated in the text
- If no relations found, return {{"relations": []}}
- Output ONLY valid JSON, no additional text
"""

# ==================== Helper Functions ====================

def is_valid_json(json_str: str) -> bool:
    """Check if string is valid JSON"""
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False

def clean_json_text(text: str) -> str:
    """Clean LLM output to extract JSON"""
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Find JSON object
    json_match = re.search(r'(\{[\s\S]*\})', text)
    if json_match:
        json_text = json_match.group(1)
        # Fix common issues
        json_text = json_text.replace('\n', ' ')
        json_text = re.sub(r'\s+', ' ', json_text)
        return json_text
    return text

def extract_entities_from_abstract(abstract: Dict[str, Any], model: str) -> Optional[Dict]:
    """
    Stage 1: Extract entities from PubMed abstract
    
    Args:
        abstract: Dictionary containing 'text' and 'pmid'
        model: Model name to use
        
    Returns:
        Dictionary with extracted entities or None if failed
    """
    text = abstract.get('text', '')
    pmid = abstract.get('pmid', 'unknown')
    
    # Skip if text too long or too short
    if len(text) > config.MAX_TEXT_LENGTH or len(text) < 100:
        return None
    
    prompt = ENTITY_EXTRACTION_PROMPT.format(text=text, pmid=pmid)
    messages = [
        {"role": "system", "content": "You are a biomedical knowledge extraction expert. Output only valid JSON."},
        {"role": "user", "content": prompt}
    ]
    
    for attempt in range(config.MAX_RETRIES):
        try:
            response = llm_api(messages, model=model)
            cleaned_response = clean_json_text(response)
            
            if is_valid_json(cleaned_response):
                result = json.loads(cleaned_response)
                
                # Validate structure
                if "entities" in result and isinstance(result["entities"], list):
                    # Filter by confidence
                    result["entities"] = [
                        e for e in result["entities"] 
                        if e.get("confidence", 0) >= config.MIN_CONFIDENCE
                    ]
                    
                    if len(result["entities"]) > 0:
                        result["pmid"] = pmid
                        result["abstract"] = text
                        print(f"Entities from abstract: {result}")
                        return result
            
            return None
            
        except Exception as e:
            if attempt == config.MAX_RETRIES - 1:
                print(f"\nEntity extraction failed for PMID {pmid}: {str(e)}")
                return None
            time.sleep(config.RETRY_DELAY)
    
    return None

def extract_relations_from_entities(abstract: Dict[str, Any], entities_result: Dict, model: str) -> Optional[Dict]:
    """
    Stage 2: Extract relations between identified entities
    
    Args:
        abstract: Dictionary containing abstract text
        entities_result: Result from Stage 1 entity extraction
        model: Model name to use
        
    Returns:
        Dictionary with extracted relations or None if failed
    """
    text = abstract.get('text', '')
    pmid = entities_result.get('pmid', 'unknown')
    entities = entities_result.get('entities', [])
    
    # Skip if no entities
    if len(entities) == 0:
        return None
    
    # Format entities for prompt
    entities_json = json.dumps(entities, indent=2, ensure_ascii=False)
    
    prompt = RELATION_EXTRACTION_PROMPT.format(text=text, entities_json=entities_json)
    messages = [
        {"role": "system", "content": "You are a biomedical knowledge extraction expert. Output only valid JSON."},
        {"role": "user", "content": prompt}
    ]
    
    for attempt in range(config.MAX_RETRIES):
        try:
            response = llm_api(messages, model=model)
            cleaned_response = clean_json_text(response)
            
            if is_valid_json(cleaned_response):
                result = json.loads(cleaned_response)
                
                # Validate structure
                if "relations" in result and isinstance(result["relations"], list):
                    # Filter by confidence
                    result["relations"] = [
                        r for r in result["relations"] 
                        if r.get("confidence", 0) >= config.MIN_CONFIDENCE
                    ]
                    
                    result["pmid"] = pmid
                    print(f"Relations from entities: {result}")
                    return result
            
            return None
            
        except Exception as e:
            if attempt == config.MAX_RETRIES - 1:
                print(f"\nRelation extraction failed for PMID {pmid}: {str(e)}")
                return None
            time.sleep(config.RETRY_DELAY)
    
    return None

def load_existing_data() -> tuple:
    """Load existing extracted data from previous run"""
    existing_entities = []
    existing_relations = []
    
    # Try to load existing data
    if os.path.exists(config.ENTITIES_OUTPUT):
        try:
            with open(config.ENTITIES_OUTPUT, 'r', encoding='utf-8') as f:
                existing_entities = json.load(f)
            print(f"Loaded {len(existing_entities)} existing entity records")
        except Exception as e:
            print(f"Warning: Could not load existing entities: {e}")
    
    if os.path.exists(config.RELATIONS_OUTPUT):
        try:
            with open(config.RELATIONS_OUTPUT, 'r', encoding='utf-8') as f:
                existing_relations = json.load(f)
            print(f"Loaded {len(existing_relations)} existing relation records")
        except Exception as e:
            print(f"Warning: Could not load existing relations: {e}")
    
    return existing_entities, existing_relations

def save_checkpoint(current_dataset_index: int, entities_data: List, relations_data: List):
    """Save checkpoint to resume interrupted runs"""
    checkpoint = {
        "dataset_start_index": CUSTOM_START_INDEX,
        "current_dataset_index": current_dataset_index,
        "processed_in_this_run": current_dataset_index - CUSTOM_START_INDEX,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "entities_count": len(entities_data),
        "relations_count": len(relations_data)
    }
    
    with open(CHECKPOINT_FILE_NEW, 'w', encoding='utf-8') as f:
        json.dump(checkpoint, f, indent=2)
    
    # Save current data
    with open(ENTITIES_OUTPUT_NEW, 'w', encoding='utf-8') as f:
        json.dump(entities_data, f, ensure_ascii=False, indent=2)
    
    with open(RELATIONS_OUTPUT_NEW, 'w', encoding='utf-8') as f:
        json.dump(relations_data, f, ensure_ascii=False, indent=2)

def load_checkpoint() -> tuple:
    """Load checkpoint if exists"""
    if os.path.exists(CHECKPOINT_FILE_NEW):
        with open(CHECKPOINT_FILE_NEW, 'r', encoding='utf-8') as f:
            checkpoint = json.load(f)
        
        # Load existing data from this run
        entities_data = []
        relations_data = []
        
        if os.path.exists(ENTITIES_OUTPUT_NEW):
            with open(ENTITIES_OUTPUT_NEW, 'r', encoding='utf-8') as f:
                entities_data = json.load(f)
        
        if os.path.exists(RELATIONS_OUTPUT_NEW):
            with open(RELATIONS_OUTPUT_NEW, 'r', encoding='utf-8') as f:
                relations_data = json.load(f)
        
        resume_index = checkpoint['current_dataset_index']
        print(f"Resuming from checkpoint: dataset index {resume_index}")
        print(f"Processed in this run: {checkpoint['processed_in_this_run']} abstracts")
        return resume_index, entities_data, relations_data
    
    return CUSTOM_START_INDEX, [], []

# ==================== Main Extraction Function ====================

def extract_knowledge_from_pubmed():
    """
    Main function to extract knowledge from PubMed dataset
    Starting from index 1000, extracting 4000 items
    """
    print("=" * 70)
    print("PubMed Knowledge Extraction System - Continue Extraction")
    print("=" * 70)
    print(f"Model: {config.HF_MODEL_NAME}")
    print(f"Dataset start index: {CUSTOM_START_INDEX} (item #{CUSTOM_START_INDEX + 1})")
    print(f"Dataset end index: {CUSTOM_END_INDEX - 1} (item #{CUSTOM_END_INDEX})")
    print(f"Items to extract: {CUSTOM_EXTRACT_COUNT}")
    print(f"New output files:")
    print(f"  - Entities: {ENTITIES_OUTPUT_NEW}")
    print(f"  - Relations: {RELATIONS_OUTPUT_NEW}")
    print(f"  - Knowledge Graph: {KNOWLEDGE_GRAPH_OUTPUT_NEW}")
    print("=" * 70)
    
    # Set LLM provider
    set_llm_provider(config.LLM_PROVIDER)
    
    # Load existing data from previous run (items 1-1000)
    print("\nLoading existing data from previous run...")
    existing_entities, existing_relations = load_existing_data()
    
    # Load checkpoint for this run if exists
    print("\nChecking for checkpoint...")
    resume_index, new_entities_data, new_relations_data = load_checkpoint()
    
    # Load PubMed dataset with streaming
    print("\nLoading PubMed dataset...")
    try:
        dataset = load_dataset(
            config.DATASET_NAME,
            split=config.DATASET_SPLIT,
            streaming=config.DATASET_STREAMING,
            trust_remote_code=True
        )
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please install: pip install datasets")
        return
    
    # Process abstracts
    current_index = resume_index
    successful_extractions = len(new_entities_data)
    
    print(f"\nStarting extraction from dataset index {current_index}...")
    print("=" * 70)
    
    with tqdm(total=CUSTOM_EXTRACT_COUNT, 
              initial=current_index - CUSTOM_START_INDEX,
              desc="Processing abstracts") as pbar:
        
        for idx, item in enumerate(dataset):
            # Skip items before our start index
            if idx < current_index:
                continue
            
            # Stop when we reach the end index
            if idx >= CUSTOM_END_INDEX:
                break
            
            # Extract abstract text and PMID
            abstract = {
                'text': item.get('contents', item.get('text', '')),
                'pmid': item.get('pmid', f'PMID_{idx}')
            }
            
            # Stage 1: Entity Extraction
            entities_result = extract_entities_from_abstract(abstract, config.HF_MODEL_NAME)
            
            if entities_result is not None:
                new_entities_data.append(entities_result)
                
                # Stage 2: Relation Extraction
                relations_result = extract_relations_from_entities(abstract, entities_result, config.HF_MODEL_NAME)
                
                if relations_result is not None:
                    new_relations_data.append(relations_result)
                
                successful_extractions += 1
            
            current_index = idx + 1
            pbar.update(1)
            pbar.set_postfix({
                'dataset_idx': idx,
                'new_entities': len(new_entities_data),
                'new_relations': len(new_relations_data),
                'success_rate': f'{successful_extractions/(idx - CUSTOM_START_INDEX + 1)*100:.1f}%'
            })
            
            # Save checkpoint periodically
            if (idx - CUSTOM_START_INDEX + 1) % config.CHECKPOINT_FREQUENCY == 0:
                save_checkpoint(current_index, new_entities_data, new_relations_data)
            
            # Sleep between batches
            if (idx - CUSTOM_START_INDEX + 1) % config.BATCH_SIZE == 0:
                time.sleep(config.SLEEP_BETWEEN_BATCHES)
    
    # Final save of new data
    print("\n\nSaving new extraction results...")
    save_checkpoint(current_index, new_entities_data, new_relations_data)
    
    # Merge with existing data
    print("Merging with existing data...")
    merged_entities = existing_entities + new_entities_data
    merged_relations = existing_relations + new_relations_data
    
    # Generate combined knowledge graph
    print("Generating combined knowledge graph...")
    
    total_processed = len(existing_entities) + successful_extractions
    total_entities_count = sum(len(e.get("entities", [])) for e in merged_entities)
    total_relations_count = sum(len(r.get("relations", [])) for r in merged_relations)
    
    knowledge_graph = {
        "metadata": {
            "total_abstracts_processed": len(merged_entities),
            "previous_run_abstracts": len(existing_entities),
            "new_run_abstracts": len(new_entities_data),
            "successful_extractions_new_run": successful_extractions,
            "total_entities": total_entities_count,
            "total_relations": total_relations_count,
            "model": config.HF_MODEL_NAME,
            "extraction_range": f"Items 1-1900 (previous) + Items {CUSTOM_START_INDEX + 1}-{current_index} (new)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "entities": merged_entities,
        "relations": merged_relations
    }
    
    # Save merged knowledge graph
    with open(KNOWLEDGE_GRAPH_OUTPUT_NEW, 'w', encoding='utf-8') as f:
        json.dump(knowledge_graph, f, ensure_ascii=False, indent=2)
    
    # Also save merged entities and relations separately
    merged_entities_file = config.ENTITIES_OUTPUT.replace('.json', '_merged2.json')
    merged_relations_file = config.RELATIONS_OUTPUT.replace('.json', '_merged2.json')
    
    with open(merged_entities_file, 'w', encoding='utf-8') as f:
        json.dump(merged_entities, f, ensure_ascii=False, indent=2)
    
    with open(merged_relations_file, 'w', encoding='utf-8') as f:
        json.dump(merged_relations, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Extraction Complete!")
    print("=" * 70)
    print(f"Dataset range processed: Index {CUSTOM_START_INDEX} - {current_index - 1}")
    print(f"Items processed in this run: {current_index - CUSTOM_START_INDEX}")
    print(f"Successful extractions in this run: {successful_extractions} ({successful_extractions/(current_index - CUSTOM_START_INDEX)*100:.1f}%)")
    print(f"\nNew extraction results:")
    print(f"  - New entities extracted: {sum(len(e.get('entities', [])) for e in new_entities_data)}")
    print(f"  - New relations extracted: {sum(len(r.get('relations', [])) for r in new_relations_data)}")
    print(f"\nMerged results (1-1000 + 1001-5000):")
    print(f"  - Total entity records: {len(merged_entities)}")
    print(f"  - Total relation records: {len(merged_relations)}")
    print(f"  - Total entities: {total_entities_count}")
    print(f"  - Total relations: {total_relations_count}")
    print(f"\nOutput files:")
    print(f"  - New entities only: {ENTITIES_OUTPUT_NEW}")
    print(f"  - New relations only: {RELATIONS_OUTPUT_NEW}")
    print(f"  - Merged entities: {merged_entities_file}")
    print(f"  - Merged relations: {merged_relations_file}")
    print(f"  - Merged knowledge graph: {KNOWLEDGE_GRAPH_OUTPUT_NEW}")
    print("=" * 70)

if __name__ == "__main__":
    extract_knowledge_from_pubmed()