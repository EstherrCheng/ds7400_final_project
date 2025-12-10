"""
Configuration file for PubMed Knowledge Extraction System
"""

import os

# ==================== LLM Configuration ====================
LLM_PROVIDER = "huggingface"  # Use HuggingFace Transformers for local processing

# HuggingFace Model Configuration
HF_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_CACHE_DIR = os.environ.get("HF_HOME", "./hf_models")  # Model cache directory
HF_DEVICE = "cuda"  # Use GPU (cuda) or CPU (cpu)
HF_TORCH_DTYPE = "bfloat16"  # Options: "float16", "bfloat16", "float32" (bfloat16 recommended for A100)
HF_MAX_NEW_TOKENS = 2048  # Maximum tokens to generate
HF_TEMPERATURE = 0.1  # Lower = more deterministic, higher = more creative
HF_TOP_P = 0.9  # Nucleus sampling parameter
HF_DO_SAMPLE = True  # Use sampling for generation

# Memory optimization
HF_USE_FLASH_ATTENTION = False
HF_LOAD_IN_4BIT = False


# ==================== Data Configuration ====================
# Number of PubMed abstracts to process
MAX_ABSTRACTS = 1000

# HuggingFace dataset configuration
DATASET_NAME = "MedRAG/pubmed"
DATASET_SPLIT = "train"
DATASET_STREAMING = True  # Use streaming to avoid downloading entire dataset

# ==================== Output Configuration ====================
# Output files, original newest stopped file 
ENTITIES_OUTPUT = "pubmed_entities_merged_merged2.json"
RELATIONS_OUTPUT = "pubmed_relations_merged_merged2.json"
KNOWLEDGE_GRAPH_OUTPUT = "pubmed_knowledge_graph_merged.json"

# Checkpoint files (for resuming interrupted runs)
CHECKPOINT_FILE = "extraction_checkpoint.json"
CHECKPOINT_FREQUENCY = 100  # Save checkpoint every N abstracts

# ==================== Processing Configuration ====================
# Batch processing
BATCH_SIZE = 10  # Process N abstracts before saving
SLEEP_BETWEEN_BATCHES = 2  # Seconds to sleep between batches

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 5  # Seconds

# ==================== Neo4j Configuration ====================
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "cyycyycyy"  # Change this to your Neo4j password
NEO4J_DATABASE = "med_diseases_papers"

# ==================== Entity Types ====================
ENTITY_TYPES = {
    "Disease": {
        "properties": ["name", "icd_code", "category", "prevalence"],
        "required": ["name"]
    },
    "Clinical_Manifestation": {
        "properties": ["name", "type", "specificity", "timing"],
        "required": ["name", "type"]
    },
    "Diagnostic_Procedure": {
        "properties": ["name", "modality", "gold_standard", "sensitivity", "specificity"],
        "required": ["name", "modality"]
    },
    "Treatment": {
        "properties": ["name", "type", "line_of_therapy", "mechanism"],
        "required": ["name", "type"]
    },
    "Drug": {
        "properties": ["name", "class", "mechanism", "contraindication"],
        "required": ["name"]
    },
    "Risk_Factor": {
        "properties": ["name", "type", "strength"],
        "required": ["name", "type"]
    },
    "Pathophysiology": {
        "properties": ["process", "level"],
        "required": ["process"]
    },
    "Complication": {
        "properties": ["name", "severity", "frequency"],
        "required": ["name"]
    },
    "Differential_Diagnosis": {
        "properties": ["disease_name", "key_difference"],
        "required": ["disease_name"]
    }
}

# ==================== Relation Types ====================
RELATION_TYPES = [
    # Diagnostic reasoning chain
    "SUGGESTS",
    "PATHOGNOMONIC_FOR",
    "PRESENTS_WITH",
    
    # Diagnostic confirmation
    "CONFIRMED_BY",
    "DIAGNOSTIC_FOR",
    "REQUIRES_WORKUP",
    
    # Treatment decisions
    "FIRST_LINE_TREATMENT",
    "ALTERNATIVE_TREATMENT",
    "TREATED_WITH",
    "ALLEVIATES",
    "CAUSES",
    
    # Etiology
    "INCREASES_RISK",
    "EXPLAINS",
    "HAS_MECHANISM",
    
    # Disease associations
    "LEADS_TO",
    "DIFFERENTIAL_DIAGNOSIS",
    "COMMONLY_COEXISTS_WITH",
    
    # Drug-related
    "CONTRAINDICATED_WITH",
    "CONTRAINDICATED_IN",
    "SYNERGISTIC_WITH",
    
    # Temporal relations
    "PRECEDES",
    "PREVENTS"
]

# ==================== Extraction Configuration ====================
# Minimum confidence score for extracted entities/relations
MIN_CONFIDENCE = 0.6

# Maximum text length to process (in characters)
MAX_TEXT_LENGTH = 5000

# Language filtering
EXTRACT_ENGLISH_ONLY = True