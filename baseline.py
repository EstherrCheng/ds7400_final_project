"""
Baseline Evaluation: Pure LLM without Knowledge Graph
Direct Mixtral-8x7B inference for comparison with GraphRAG
"""

import json
import time
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class BaselineResult:
    """Structure for baseline evaluation result"""
    question_id: int
    question: str
    options: Dict[str, str]
    correct_answer: str
    predicted_answer: str
    is_correct: bool
    meta_info: str
    response_time: float


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
        
        # Import requests
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("Please install requests: pip install requests")
    
    def generate(self, prompt: str, max_tokens: int = 512, 
                temperature: float = 0.1) -> Tuple[str, float]:
        """
        Generate text using Mixtral
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Tuple of (generated_text, response_time)
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
        
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                response = self.requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
                
                elapsed_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get('generated_text', '').strip(), elapsed_time
                    else:
                        return str(result), elapsed_time
                elif response.status_code == 503:
                    # Model loading, wait and retry
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    else:
                        return f"Error: Model loading failed after {max_retries} attempts", elapsed_time
                else:
                    return f"Error: API returned status {response.status_code}", elapsed_time
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    elapsed_time = time.time() - start_time
                    return f"Error: {str(e)}", elapsed_time
        
        elapsed_time = time.time() - start_time
        return "Error: Max retries exceeded", elapsed_time


class BaselineEvaluator:
    """Baseline evaluator using pure LLM without knowledge graph"""
    
    def __init__(self, llm: MixtralLLM):
        """
        Initialize baseline evaluator
        
        Args:
            llm: Mixtral LLM interface
        """
        self.llm = llm
    
    def create_prompt(self, question: str, options: Dict[str, str]) -> str:
        """
        Create prompt without knowledge graph augmentation
        
        Args:
            question: Medical question
            options: Answer options
            
        Returns:
            Prompt for LLM
        """
        # Format options
        options_text = "\n".join([f"{key}. {value}" for key, value in options.items()])
        
        # Create simple prompt without knowledge
        prompt = f"""[INST] You are a medical expert. Answer the following medical question by selecting the correct option.

Question: {question}

Options:
{options_text}

Based on your medical knowledge, which option is correct? Respond with ONLY the letter (A, B, C, D, or E) of the correct answer. [/INST]

Answer: """
        
        return prompt
    
    def answer_question(self, question: str, options: Dict[str, str]) -> Tuple[str, float]:
        """
        Answer a medical question using pure LLM
        
        Args:
            question: Medical question
            options: Answer options dict
            
        Returns:
            Tuple of (predicted_answer, response_time)
        """
        # Create prompt
        prompt = self.create_prompt(question, options)
        
        # Generate answer
        response, response_time = self.llm.generate(prompt, max_tokens=10, temperature=0.1)
        
        # Extract answer letter
        predicted_answer = self._extract_answer_letter(response)
        
        return predicted_answer, response_time
    
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
        print("Baseline Evaluation (Pure LLM)")
        print("=" * 70)
        
        # Load test data
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                test_data.append(json.loads(line))
        
        if limit:
            test_data = test_data[:limit]
        
        print(f"\nEvaluating on {len(test_data)} questions...")
        print("Mode: Pure Mixtral (NO knowledge graph augmentation)")
        print()
        
        # Evaluate
        results = []
        correct = 0
        total = 0
        total_time = 0
        
        for i, item in enumerate(tqdm(test_data, desc="Evaluating")):
            question = item['question']
            options = item['options']
            correct_answer = item['answer_idx']
            
            # Get prediction
            predicted_answer, response_time = self.answer_question(question, options)
            total_time += response_time
            
            # Check if correct
            is_correct = (predicted_answer == correct_answer)
            if is_correct:
                correct += 1
            total += 1
            
            # Save result
            result = BaselineResult(
                question_id=i,
                question=question,
                options=options,
                correct_answer=correct_answer,
                predicted_answer=predicted_answer,
                is_correct=is_correct,
                meta_info=item.get('meta_info', ''),
                response_time=response_time
            )
            results.append(result.__dict__)
            
            # Print progress every 50 questions
            if (i + 1) % 50 == 0:
                current_accuracy = correct / total
                avg_time = total_time / total
                print(f"\n[{i+1}/{len(test_data)}] Current Accuracy: {current_accuracy:.4f} | "
                      f"Avg Response Time: {avg_time:.2f}s")
        
        # Calculate metrics
        accuracy = correct / total if total > 0 else 0
        avg_response_time = total_time / total if total > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_response_time': avg_response_time,
            'total_time': total_time
        }
        
        # Save results
        output_data = {
            'model': self.llm.model_name,
            'method': 'baseline_pure_llm',
            'metrics': metrics,
            'results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 70)
        print("Baseline Evaluation Results")
        print("=" * 70)
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"Average Response Time: {avg_response_time:.2f}s")
        print(f"Total Time: {total_time/60:.2f} minutes")
        print(f"Results saved to: {output_file}")
        print("=" * 70)
        
        return metrics


def main():
    """Main function to run baseline evaluation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Baseline Medical QA System (Pure LLM)')
    parser.add_argument('--test_file', type=str,
                       default='test.jsonl',
                       help='Path to test JSONL file')
    parser.add_argument('--output_file', type=str,
                       default='baseline_results.json',
                       help='Path to save results')
    parser.add_argument('--api_token', type=str, required=True,
                       help='HuggingFace API token')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of test samples')
    parser.add_argument('--model_name', type=str,
                       default='mistralai/Mixtral-8x7B-Instruct-v0.1',
                       help='HuggingFace model name')
    
    args = parser.parse_args()
    
    # Initialize LLM
    print("Initializing Mixtral LLM...")
    llm = MixtralLLM(api_token=args.api_token, model_name=args.model_name)
    
    # Initialize baseline evaluator
    print("Initializing Baseline Evaluator...")
    evaluator = BaselineEvaluator(llm=llm)
    
    # Evaluate
    metrics = evaluator.evaluate_dataset(
        test_file=args.test_file,
        output_file=args.output_file,
        limit=args.limit
    )
    
    print("\nEvaluation complete!")
    return metrics


if __name__ == "__main__":
    main()