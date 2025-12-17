"""
HuggingFace Transformers LLM Wrapper for Qwen2.5-14B-Instruct
Replaces Ollama framework with direct model loading
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Union, Generator
import time
import config

class HuggingFaceLLM:
    """Wrapper for HuggingFace Transformers models"""
    
    def __init__(self):
        """Initialize HuggingFace model and tokenizer"""
        self.model = None
        self.tokenizer = None
        self.device = config.HF_DEVICE
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer with optimizations"""
        print("=" * 70)
        print(f"Loading model: {config.HF_MODEL_NAME}")
        print(f"Device: {self.device}")
        print(f"Data type: {config.HF_TORCH_DTYPE}")
        print(f"Flash Attention: {config.HF_USE_FLASH_ATTENTION}")
        print("=" * 70)
        
        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        torch_dtype = dtype_map.get(config.HF_TORCH_DTYPE, torch.bfloat16)
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.HF_MODEL_NAME,
            cache_dir=config.HF_MODEL_CACHE_DIR,
            trust_remote_code=True
        )
        print("✓ Tokenizer loaded")
        
        # Prepare model loading arguments
        model_kwargs = {
            "cache_dir": config.HF_MODEL_CACHE_DIR,
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
        }
        
        # Add device map for multi-GPU or single GPU
        if self.device == "cuda":
            model_kwargs["device_map"] = "auto"
        
        # Add quantization if specified
        if config.HF_LOAD_IN_8BIT:
            print("Loading in 8-bit mode...")
            model_kwargs["load_in_8bit"] = True
        elif config.HF_LOAD_IN_4BIT:
            print("Loading in 4-bit mode...")
            model_kwargs["load_in_4bit"] = True
        
        # Add Flash Attention if enabled
        if config.HF_USE_FLASH_ATTENTION:
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                print("Flash Attention 2 enabled")
            except Exception as e:
                print(f"Flash Attention not available: {e}")
                print("Falling back to standard attention")
        
        # Load model
        print("Loading model (this may take a few minutes)...")
        start_time = time.time()
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.HF_MODEL_NAME,
            **model_kwargs
        )
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f} seconds")
        
        # Set to evaluation mode
        self.model.eval()
        
        # Print memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        print("=" * 70)
        print("Model ready for inference!")
        print("=" * 70 + "\n")
    
    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages using Qwen's chat template"""
        # Qwen2.5-Instruct uses a specific chat format
        # The tokenizer's chat template will handle this
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted
    
    def chat(self, 
             messages: List[Dict[str, str]], 
             max_tokens: Optional[int] = None,
             temperature: Optional[float] = None,
             top_p: Optional[float] = None,
             stream: bool = False,
             max_retries: int = 3,
             retry_delay: int = 5) -> Union[str, Generator]:
        """
        Generate response from messages
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stream: Whether to stream the response
            max_retries: Number of retries on failure
            retry_delay: Delay between retries
            
        Returns:
            Generated text or generator for streaming
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")
        
        # Use config defaults if not specified
        max_tokens = max_tokens or config.HF_MAX_NEW_TOKENS
        temperature = temperature if temperature is not None else config.HF_TEMPERATURE
        top_p = top_p if top_p is not None else config.HF_TOP_P
        
        for attempt in range(max_retries):
            try:
                # Format messages
                formatted_prompt = self.format_messages(messages)
                
                # Tokenize
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    add_special_tokens=False
                )
                
                # Move to device
                if self.device == "cuda":
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generation parameters
                gen_kwargs = {
                    "max_new_tokens": max_tokens,
                    "do_sample": config.HF_DO_SAMPLE,
                    "temperature": temperature,
                    "top_p": top_p,
                    "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                }
                
                # Generate
                with torch.no_grad():
                    if stream:
                        # Streaming not implemented for now, return full response
                        outputs = self.model.generate(**inputs, **gen_kwargs)
                    else:
                        outputs = self.model.generate(**inputs, **gen_kwargs)
                
                # Decode only the generated part (exclude input)
                generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
                response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
                return response
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Final chat attempt failed: {str(e)}")
                    return f"Model request failed: {str(e)}"
                print(f"Attempt {attempt + 1} failed: {str(e)}, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
        
        return "Model request failed after all retries"
    
    def __del__(self):
        """Cleanup GPU memory when object is destroyed"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Global model instance (singleton pattern for efficiency)
_global_model = None

def get_model() -> HuggingFaceLLM:
    """Get or create global model instance"""
    global _global_model
    if _global_model is None:
        _global_model = HuggingFaceLLM()
    return _global_model


# ==================== Compatible API with llmgpt_ollama.py ====================

def llm_api(messages: List[Dict], stream: bool = False, model: str = None) -> str:
    """
    Unified LLM API compatible with original llmgpt_ollama.py interface
    
    Args:
        messages: List of message dictionaries
        stream: Whether to stream (not implemented, returns full response)
        model: Model name (ignored, uses config.HF_MODEL_NAME)
        
    Returns:
        Generated response text
    """
    llm = get_model()
    return llm.chat(messages, stream=stream)


def set_llm_provider(provider: str):
    """
    Set LLM provider (for compatibility with original code)
    
    Args:
        provider: Provider name (only "huggingface" is supported)
    """
    if provider != "huggingface":
        print(f"Warning: Provider '{provider}' not supported. Using HuggingFace.")
    print(f"LLM Provider set to: HuggingFace ({config.HF_MODEL_NAME})")


# Test function
def test_model():
    """Test the model with a simple query"""
    print("Testing model...")
    
    messages = [
        {"role": "system", "content": "You are a helpful medical knowledge extraction expert."},
        {"role": "user", "content": "Extract the disease name from this text: 'Patient has diabetes mellitus.' Output only JSON: {\"disease\": \"...\"}"}
    ]
    
    response = llm_api(messages)
    print("\nTest Query:")
    print(messages[1]["content"])
    print("\nModel Response:")
    print(response)
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Test when run directly
    test_model()