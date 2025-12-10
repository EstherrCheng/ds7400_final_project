"""
Analysis Script for GraphRAG Results
Provides detailed analysis and visualization of evaluation results
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from typing import Dict, List, Any
import numpy as np


class GraphRAGAnalyzer:
    """Analyze GraphRAG evaluation results"""
    
    def __init__(self, results_file: str):
        """
        Initialize analyzer
        
        Args:
            results_file: Path to results JSON file
        """
        with open(results_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.metrics = self.data['metrics']
        self.results = self.data['results']
        
        # Convert to DataFrame for easier analysis
        self.df = pd.DataFrame(self.results)
    
    def print_summary(self):
        """Print summary statistics"""
        print("=" * 70)
        print("GraphRAG Evaluation Summary")
        print("=" * 70)
        
        print(f"\nOverall Accuracy: {self.metrics['accuracy']:.4f}")
        print(f"Correct: {self.metrics['correct']} / {self.metrics['total']}")
        
        # Accuracy by meta_info (step1, step2&3)
        if 'meta_info' in self.df.columns:
            print("\n--- Accuracy by Question Type ---")
            meta_accuracy = self.df.groupby('meta_info')['is_correct'].agg(['mean', 'count'])
            for meta_type, row in meta_accuracy.iterrows():
                print(f"{meta_type}: {row['mean']:.4f} ({int(row['mean'] * row['count'])}/{int(row['count'])})")
        
        # Retrieval statistics
        print("\n--- Retrieval Statistics ---")
        print(f"Avg Entities Retrieved: {self.df['num_entities_retrieved'].mean():.2f}")
        print(f"Avg Relations Retrieved: {self.df['num_relations_retrieved'].mean():.2f}")
        print(f"Avg Paths Found: {self.df['num_paths_found'].mean():.2f}")
        
        # Questions with no retrieval
        no_retrieval = (self.df['num_entities_retrieved'] == 0).sum()
        print(f"Questions with No Retrieved Entities: {no_retrieval} ({no_retrieval/len(self.df)*100:.1f}%)")
        
        print("=" * 70)
    
    def analyze_by_retrieval(self):
        """Analyze accuracy based on retrieval success"""
        print("\n--- Accuracy by Retrieval Success ---")
        
        # Categorize by retrieval
        self.df['has_entities'] = self.df['num_entities_retrieved'] > 0
        self.df['has_relations'] = self.df['num_relations_retrieved'] > 0
        self.df['has_paths'] = self.df['num_paths_found'] > 0
        
        # Accuracy with/without entities
        acc_with_entities = self.df[self.df['has_entities']]['is_correct'].mean()
        acc_without_entities = self.df[~self.df['has_entities']]['is_correct'].mean()
        
        print(f"With Retrieved Entities: {acc_with_entities:.4f} (n={self.df['has_entities'].sum()})")
        print(f"Without Retrieved Entities: {acc_without_entities:.4f} (n={(~self.df['has_entities']).sum()})")
        
        # Accuracy with/without relations
        acc_with_relations = self.df[self.df['has_relations']]['is_correct'].mean()
        acc_without_relations = self.df[~self.df['has_relations']]['is_correct'].mean()
        
        print(f"\nWith Retrieved Relations: {acc_with_relations:.4f} (n={self.df['has_relations'].sum()})")
        print(f"Without Retrieved Relations: {acc_without_relations:.4f} (n={(~self.df['has_relations']).sum()})")
    
    def show_error_analysis(self, n: int = 10):
        """Show examples of incorrect predictions"""
        print("\n" + "=" * 70)
        print(f"Error Analysis - Showing {n} Examples")
        print("=" * 70)
        
        incorrect = self.df[~self.df['is_correct']]
        
        print(f"\nTotal Incorrect: {len(incorrect)}")
        
        for i, (idx, row) in enumerate(incorrect.head(n).iterrows()):
            print(f"\n--- Error {i+1} ---")
            print(f"Question: {row['question'][:200]}...")
            print(f"Correct Answer: {row['correct_answer']}")
            print(f"Predicted Answer: {row['predicted_answer']}")
            print(f"Entities Retrieved: {row['num_entities_retrieved']}")
            print(f"Relations Retrieved: {row['num_relations_retrieved']}")
    
    def show_success_examples(self, n: int = 5):
        """Show examples of correct predictions with retrieval"""
        print("\n" + "=" * 70)
        print(f"Success Analysis - Showing {n} Examples")
        print("=" * 70)
        
        # Correct answers with good retrieval
        correct_with_retrieval = self.df[
            (self.df['is_correct']) & 
            (self.df['num_entities_retrieved'] > 0)
        ]
        
        for i, (idx, row) in enumerate(correct_with_retrieval.head(n).iterrows()):
            print(f"\n--- Success {i+1} ---")
            print(f"Question: {row['question'][:200]}...")
            print(f"Correct Answer: {row['correct_answer']} ✓")
            print(f"Entities Retrieved: {row['num_entities_retrieved']}")
            print(f"Relations Retrieved: {row['num_relations_retrieved']}")
            print(f"Paths Found: {row['num_paths_found']}")
    
    def create_visualizations(self, output_prefix: str = "graphrag_analysis"):
        """Create visualization plots"""
        print("\nCreating visualizations...")
        
        # Figure 1: Accuracy by question type
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Subplot 1: Overall accuracy
        ax = axes[0, 0]
        categories = ['Correct', 'Incorrect']
        values = [self.metrics['correct'], self.metrics['total'] - self.metrics['correct']]
        colors = ['#2ecc71', '#e74c3c']
        ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_ylabel('Count')
        ax.set_title(f'Overall Accuracy: {self.metrics["accuracy"]:.2%}')
        ax.grid(axis='y', alpha=0.3)
        
        # Subplot 2: Accuracy by meta_info
        ax = axes[0, 1]
        if 'meta_info' in self.df.columns:
            meta_groups = self.df.groupby('meta_info')['is_correct'].mean()
            ax.bar(meta_groups.index, meta_groups.values, color='#3498db', alpha=0.7)
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy by Question Type')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
        
        # Subplot 3: Retrieval statistics
        ax = axes[1, 0]
        retrieval_stats = [
            self.df['num_entities_retrieved'].mean(),
            self.df['num_relations_retrieved'].mean(),
            self.df['num_paths_found'].mean()
        ]
        labels = ['Entities', 'Relations', 'Paths']
        ax.bar(labels, retrieval_stats, color=['#9b59b6', '#e67e22', '#1abc9c'], alpha=0.7)
        ax.set_ylabel('Average Count')
        ax.set_title('Average Retrieval Statistics')
        ax.grid(axis='y', alpha=0.3)
        
        # Subplot 4: Accuracy vs retrieval
        ax = axes[1, 1]
        categories = ['With\nEntities', 'Without\nEntities', 'With\nRelations', 'Without\nRelations']
        accuracies = [
            self.df[self.df['num_entities_retrieved'] > 0]['is_correct'].mean(),
            self.df[self.df['num_entities_retrieved'] == 0]['is_correct'].mean(),
            self.df[self.df['num_relations_retrieved'] > 0]['is_correct'].mean(),
            self.df[self.df['num_relations_retrieved'] == 0]['is_correct'].mean()
        ]
        ax.bar(categories, accuracies, color='#34495e', alpha=0.7)
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Retrieval Success')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_overview.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_prefix}_overview.png")
        plt.close()
        
        # Figure 2: Distribution of retrieval counts
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Entities distribution
        ax = axes[0]
        ax.hist(self.df['num_entities_retrieved'], bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Entities')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Retrieved Entities')
        ax.grid(axis='y', alpha=0.3)
        
        # Relations distribution
        ax = axes[1]
        ax.hist(self.df['num_relations_retrieved'], bins=20, color='#e67e22', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Relations')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Retrieved Relations')
        ax.grid(axis='y', alpha=0.3)
        
        # Paths distribution
        ax = axes[2]
        ax.hist(self.df['num_paths_found'], bins=20, color='#1abc9c', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Paths')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Found Paths')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_distributions.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_prefix}_distributions.png")
        plt.close()
    
    def export_detailed_report(self, output_file: str = "graphrag_detailed_report.txt"):
        """Export detailed text report"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("GraphRAG Detailed Evaluation Report\n")
            f.write("=" * 70 + "\n\n")
            
            # Overall metrics
            f.write("OVERALL METRICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Accuracy: {self.metrics['accuracy']:.4f}\n")
            f.write(f"Correct: {self.metrics['correct']}\n")
            f.write(f"Total: {self.metrics['total']}\n\n")
            
            # By question type
            if 'meta_info' in self.df.columns:
                f.write("ACCURACY BY QUESTION TYPE\n")
                f.write("-" * 70 + "\n")
                meta_accuracy = self.df.groupby('meta_info')['is_correct'].agg(['mean', 'count', 'sum'])
                for meta_type, row in meta_accuracy.iterrows():
                    f.write(f"{meta_type}:\n")
                    f.write(f"  Accuracy: {row['mean']:.4f}\n")
                    f.write(f"  Correct: {int(row['sum'])} / {int(row['count'])}\n\n")
            
            # Retrieval statistics
            f.write("RETRIEVAL STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Entities Retrieved:\n")
            f.write(f"  Mean: {self.df['num_entities_retrieved'].mean():.2f}\n")
            f.write(f"  Median: {self.df['num_entities_retrieved'].median():.2f}\n")
            f.write(f"  Max: {self.df['num_entities_retrieved'].max()}\n\n")
            
            f.write(f"Relations Retrieved:\n")
            f.write(f"  Mean: {self.df['num_relations_retrieved'].mean():.2f}\n")
            f.write(f"  Median: {self.df['num_relations_retrieved'].median():.2f}\n")
            f.write(f"  Max: {self.df['num_relations_retrieved'].max()}\n\n")
            
            f.write(f"Paths Found:\n")
            f.write(f"  Mean: {self.df['num_paths_found'].mean():.2f}\n")
            f.write(f"  Median: {self.df['num_paths_found'].median():.2f}\n")
            f.write(f"  Max: {self.df['num_paths_found'].max()}\n\n")
            
            # Error analysis
            f.write("ERROR ANALYSIS\n")
            f.write("-" * 70 + "\n")
            incorrect = self.df[~self.df['is_correct']]
            f.write(f"Total Errors: {len(incorrect)}\n")
            f.write(f"Errors with No Retrieval: {(incorrect['num_entities_retrieved'] == 0).sum()}\n")
            f.write(f"Errors with Retrieval: {(incorrect['num_entities_retrieved'] > 0).sum()}\n\n")
            
        print(f"Detailed report saved to: {output_file}")


def main():
    """Main function for analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze GraphRAG Results')
    parser.add_argument('--results_file', type=str, 
                       default='graphrag_results.json',
                       help='Path to results JSON file')
    parser.add_argument('--output_prefix', type=str,
                       default='graphrag_analysis',
                       help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Load and analyze results
    analyzer = GraphRAGAnalyzer(args.results_file)
    
    # Print summary
    analyzer.print_summary()
    
    # Analyze by retrieval
    analyzer.analyze_by_retrieval()
    
    # Show examples
    analyzer.show_error_analysis(n=5)
    analyzer.show_success_examples(n=5)
    
    # Create visualizations
    analyzer.create_visualizations(output_prefix=args.output_prefix)
    
    # Export detailed report
    analyzer.export_detailed_report(f"{args.output_prefix}_report.txt")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()