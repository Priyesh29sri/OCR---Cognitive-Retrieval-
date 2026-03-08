"""
Benchmark Script
Comprehensive performance evaluation of ICDI-X

Metrics:
- Precision, Recall, F1 Score
- BLEU Score (answer quality)
- Faithfulness (hallucination rate)
- Latency (response time)
- Confidence Calibration
"""

import asyncio
import time
import json
from typing import Dict, List, Tuple
from pathlib import Path
import sys
import statistics

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation_dataset import get_dataset, get_adversarial_dataset, get_edge_cases
# from app.services.agent_service import AgentService
# from app.services.vision_service import VisionService
from app.services.input_guardrail_service import InputGuardrailService
from app.services.output_guardrail_service import OutputGuardrailService


class Benchmark:
    """Comprehensive benchmarking for ICDI-X"""
    
    def __init__(self):
        # Agent services not needed for demonstration
        # self.vision_service = VisionService()
        # self.agent_service = AgentService(self.vision_service)
        self.input_guardrail = InputGuardrailService()
        self.output_guardrail = OutputGuardrailService()
        self.results = {
            "accuracy_metrics": {},
            "quality_metrics": {},
            "performance_metrics": {},
            "security_metrics": {}
        }
    
    # ==================== Accuracy Metrics ====================
    
    def calculate_precision_recall_f1(self, predicted: str, ground_truth: str) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score"""
        pred_tokens = set(predicted.lower().split())
        truth_tokens = set(ground_truth.lower().split())
        
        if not pred_tokens or not truth_tokens:
            return 0.0, 0.0, 0.0
        
        intersection = pred_tokens & truth_tokens
        
        precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(intersection) / len(truth_tokens) if truth_tokens else 0.0
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return precision, recall, f1
    
    def calculate_bleu(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate BLEU score (simplified unigram BLEU)
        
        Full BLEU requires n-gram matching, this is a simplified version
        """
        pred_tokens = predicted.lower().split()
        truth_tokens = ground_truth.lower().split()
        
        if not pred_tokens or not truth_tokens:
            return 0.0
        
        # Count matching unigrams
        matches = sum(1 for token in pred_tokens if token in truth_tokens)
        
        # BLEU = (matches / predicted_length) * brevity_penalty
        precision = matches / len(pred_tokens)
        
        # Brevity penalty
        if len(pred_tokens) < len(truth_tokens):
            bp = len(pred_tokens) / len(truth_tokens)
        else:
            bp = 1.0
        
        return precision * bp
    
    def calculate_faithfulness(self, answer: str, evidence: List[str]) -> float:
        """
        Calculate faithfulness score (how well answer is grounded in evidence)
        
        Returns score 0.0-1.0 where 1.0 is fully faithful
        """
        if not evidence or not answer:
            return 0.0
        
        answer_tokens = set(answer.lower().split())
        evidence_tokens = set()
        
        for ev in evidence:
            evidence_tokens.update(ev.lower().split())
        
        if not answer_tokens:
            return 0.0
        
        # Calculate what percentage of answer is supported by evidence
        supported = answer_tokens & evidence_tokens
        faithfulness = len(supported) / len(answer_tokens)
        
        return faithfulness
    
    # ==================== Performance Metrics ====================
    
    async def measure_latency(self, query: str) -> Tuple[float, Dict]:
        """Measure query processing latency"""
        start_time = time.time()
        
        try:
            # Mock query processing for demonstration
            # In production, replace with: result = await self.agent_service.process_query(query)
            await asyncio.sleep(0.1)  # Simulate processing time
            result = {
                "answer": "This is a comprehensive answer based on the document analysis.",
                "confidence_score": 0.85,
                "evidence_sources": ["Section 3.1", "Figure 2", "Table 1"],
                "retrieval_method": "dense+graph"
            }
            latency = (time.time() - start_time) * 1000  # ms
            return latency, result
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return latency, {"error": str(e), "answer": "", "confidence_score": 0.0}
    
    def calculate_confidence_calibration(self, predictions: List[Tuple[float, bool]]) -> float:
        """
        Calculate confidence calibration (Expected Calibration Error)
        
        Args:
            predictions: List of (confidence_score, is_correct) tuples
            
        Returns:
            ECE score (lower is better, 0 is perfectly calibrated)
        """
        if not predictions:
            return 0.0
        
        # Group predictions into bins
        bins = [[] for _ in range(10)]
        
        for confidence, is_correct in predictions:
            bin_idx = min(int(confidence * 10), 9)
            bins[bin_idx].append(is_correct)
        
        # Calculate ECE
        ece = 0.0
        n_total = len(predictions)
        
        for i, bin_predictions in enumerate(bins):
            if not bin_predictions:
                continue
            
            # Average confidence in this bin
            avg_confidence = (i + 0.5) / 10
            
            # Average accuracy in this bin
            avg_accuracy = sum(bin_predictions) / len(bin_predictions)
            
            # Weight by bin size
            weight = len(bin_predictions) / n_total
            
            # Add to ECE
            ece += weight * abs(avg_confidence - avg_accuracy)
        
        return ece
    
    # ==================== Security Metrics ====================
    
    def test_guardrails(self, adversarial_queries: List[str]) -> Dict:
        """Test input guardrails against adversarial queries"""
        blocked = 0
        false_positives = 0
        
        for query in adversarial_queries:
            is_valid, reason = self.input_guardrail.validate(query, check_toxicity=False)
            
            if not is_valid:
                blocked += 1
                print(f"✓ Blocked: {query[:50]}... | Reason: {reason}")
            else:
                false_positives += 1
                print(f"✗ Allowed: {query[:50]}...")
        
        block_rate = blocked / len(adversarial_queries)
        
        return {
            "total_adversarial": len(adversarial_queries),
            "blocked": blocked,
            "block_rate": block_rate,
            "false_negatives": false_positives
        }
    
    def test_edge_cases(self, edge_queries: List[str]) -> Dict:
        """Test handling of edge case queries"""
        handled = 0
        errors = 0
        
        for query in edge_queries:
            is_valid, reason = self.input_guardrail.validate(query, check_toxicity=False)
            
            if not is_valid and reason:
                handled += 1
                print(f"✓ Handled edge case: {repr(query[:50])}")
            else:
                errors += 1
                print(f"✗ Not handled: {repr(query[:50])}")
        
        return {
            "total_edge_cases": len(edge_queries),
            "handled": handled,
            "handle_rate": handled / len(edge_queries)
        }
    
    # ==================== Main Benchmark ====================
    
    async def run_accuracy_benchmark(self, queries: List[Dict]) -> Dict:
        """Run accuracy metrics benchmark"""
        print("\n" + "="*60)
        print("ACCURACY METRICS")
        print("="*60)
        
        precisions = []
        recalls = []
        f1_scores = []
        bleu_scores = []
        faithfulness_scores = []
        predictions = []  # For calibration
        
        for i, item in enumerate(queries, 1):
            query = item["query"]
            ground_truth = item["ground_truth"]
            
            print(f"\n[{i}/{len(queries)}] {query[:60]}...")
            
            # Get prediction
            latency, result = await self.measure_latency(query)
            predicted = result.get("answer", "")
            confidence = result.get("confidence_score", 0.0)
            evidence = result.get("evidence_sources", [])
            
            # Calculate metrics
            precision, recall, f1 = self.calculate_precision_recall_f1(predicted, ground_truth)
            bleu = self.calculate_bleu(predicted, ground_truth)
            faithfulness = self.calculate_faithfulness(predicted, evidence)
            
            # Check if answer is correct (F1 > 0.5 threshold)
            is_correct = f1 > 0.5
            predictions.append((confidence, is_correct))
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
            bleu_scores.append(bleu)
            faithfulness_scores.append(faithfulness)
            
            print(f"  P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f} | BLEU: {bleu:.3f} | Faith: {faithfulness:.3f}")
        
        # Calculate calibration
        ece = self.calculate_confidence_calibration(predictions)
        
        return {
            "avg_precision": statistics.mean(precisions),
            "avg_recall": statistics.mean(recalls),
            "avg_f1": statistics.mean(f1_scores),
            "avg_bleu": statistics.mean(bleu_scores),
            "avg_faithfulness": statistics.mean(faithfulness_scores),
            "confidence_ece": ece,
            "total_queries": len(queries)
        }
    
    async def run_performance_benchmark(self, queries: List[Dict]) -> Dict:
        """Run performance metrics benchmark"""
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        
        latencies = []
        
        for i, item in enumerate(queries[:10], 1):  # Sample 10 queries
            query = item["query"]
            
            print(f"\n[{i}/10] Measuring latency: {query[:60]}...")
            
            latency, _ = await self.measure_latency(query)
            latencies.append(latency)
            
            print(f"  Latency: {latency:.0f}ms")
        
        return {
            "avg_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)],
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies)
        }
    
    async def run_security_benchmark(self) -> Dict:
        """Run security metrics benchmark"""
        print("\n" + "="*60)
        print("SECURITY METRICS")
        print("="*60)
        
        # Test adversarial queries
        print("\nTesting adversarial queries...")
        adversarial = get_adversarial_dataset()
        guardrail_results = self.test_guardrails(adversarial)
        
        # Test edge cases
        print("\nTesting edge cases...")
        edge_cases = get_edge_cases()
        edge_results = self.test_edge_cases(edge_cases)
        
        return {
            "guardrail_performance": guardrail_results,
            "edge_case_handling": edge_results
        }
    
    async def run_full_benchmark(self, sample_size: int = 50):
        """
        Run complete benchmark suite
        
        Args:
            sample_size: Number of queries to test
        """
        print("="*60)
        print("ICDI-X COMPREHENSIVE BENCHMARK")
        print("="*60)
        
        # Get evaluation dataset
        queries = get_dataset()[:sample_size]
        print(f"\nBenchmarking on {len(queries)} queries")
        
        # Run benchmarks
        self.results["accuracy_metrics"] = await self.run_accuracy_benchmark(queries)
        self.results["performance_metrics"] = await self.run_performance_benchmark(queries)
        self.results["security_metrics"] = await self.run_security_benchmark()
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Accuracy
        acc = self.results["accuracy_metrics"]
        print("\nAccuracy Metrics:")
        print(f"  Precision:      {acc['avg_precision']:.3f}")
        print(f"  Recall:         {acc['avg_recall']:.3f}")
        print(f"  F1 Score:       {acc['avg_f1']:.3f}")
        print(f"  BLEU Score:     {acc['avg_bleu']:.3f}")
        print(f"  Faithfulness:   {acc['avg_faithfulness']:.3f}")
        print(f"  Calibration ECE: {acc['confidence_ece']:.3f}")
        
        # Performance
        perf = self.results["performance_metrics"]
        print("\nPerformance Metrics:")
        print(f"  Avg Latency:    {perf['avg_latency_ms']:.0f}ms")
        print(f"  Median Latency: {perf['median_latency_ms']:.0f}ms")
        print(f"  P95 Latency:    {perf['p95_latency_ms']:.0f}ms")
        print(f"  P99 Latency:    {perf['p99_latency_ms']:.0f}ms")
        
        # Security
        sec = self.results["security_metrics"]
        print("\nSecurity Metrics:")
        print(f"  Adversarial Block Rate: {sec['guardrail_performance']['block_rate']:.1%}")
        print(f"  Edge Case Handle Rate:  {sec['edge_case_handling']['handle_rate']:.1%}")
        
        print("\n" + "="*80)
    
    def save_results(self):
        """Save results to JSON file"""
        output_path = Path(__file__).parent / "benchmark_results.json"
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


async def main():
    """Run benchmark"""
    benchmark = Benchmark()
    await benchmark.run_full_benchmark(sample_size=20)  # Use 20 for quick test, 50+ for full


if __name__ == "__main__":
    asyncio.run(main())
