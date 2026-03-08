"""
Real Document Testing Script
Upload actual PDFs and measure real performance with ground truth answers
"""

import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

# Server configuration
BASE_URL = "http://127.0.0.1:8000"

# Test documents with ground truth Q&A pairs
TEST_DOCUMENTS = [
    {
        "name": "Test Document 1",
        "file_path": "test_docs/sample_paper.pdf",  # User needs to provide
        "questions": [
            {
                "query": "What is the main contribution of this paper?",
                "ground_truth": "novel intrusion detection architecture",
                "expected_keywords": ["architecture", "detection", "novel", "system"]
            },
            {
                "query": "What datasets were used for evaluation?",
                "ground_truth": "NSL-KDD and CICIDS2017",
                "expected_keywords": ["NSL-KDD", "CICIDS", "dataset"]
            },
            {
                "query": "What is the reported accuracy?",
                "ground_truth": "95.3% accuracy",
                "expected_keywords": ["95", "accuracy", "percent"]
            }
        ]
    }
]


class RealDocumentTester:
    """Test ICDI-X with real documents and measure actual performance"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.auth_token = None
        self.document_ids = []
        self.results = {
            "upload_success": 0,
            "upload_failures": 0,
            "queries_tested": 0,
            "precision_scores": [],
            "recall_scores": [],
            "f1_scores": [],
            "latencies": [],
            "confidence_scores": []
        }
    
    def register_and_login(self) -> bool:
        """Create test user and get auth token"""
        print("\n" + "="*60)
        print("AUTHENTICATION")
        print("="*60)
        
        # Register
        try:
            response = requests.post(
                f"{self.base_url}/auth/register",
                json={
                    "email": f"test_user_{int(time.time())}@example.com",
                    "password": "TestPass123!",
                    "full_name": "Test User"
                }
            )
            if response.status_code == 201:
                print("✓ Registration successful")
            elif response.status_code == 400 and "already exists" in response.text:
                print("⚠ User exists, proceeding to login")
            else:
                print(f"✗ Registration failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Registration error: {e}")
            return False
        
        # Login
        try:
            response = requests.post(
                f"{self.base_url}/auth/login",
                json={
                    "email": f"test_user_{int(time.time())}@example.com",
                    "password": "TestPass123!"
                }
            )
            
            # Try alternative email if first fails
            if response.status_code != 200:
                response = requests.post(
                    f"{self.base_url}/auth/login",
                    json={
                        "email": "test@example.com",
                        "password": "TestPass123!"
                    }
                )
            
            if response.status_code == 200:
                self.auth_token = response.json()["access_token"]
                print(f"✓ Login successful")
                return True
            else:
                print(f"✗ Login failed: {response.status_code}")
                print(f"  Response: {response.text}")
                return False
        except Exception as e:
            print(f"✗ Login error: {e}")
            return False
    
    def upload_document(self, file_path: str, doc_name: str) -> Tuple[bool, str]:
        """Upload a real PDF document"""
        print(f"\nUploading: {doc_name}")
        print(f"  File: {file_path}")
        
        # Check if file exists
        if not Path(file_path).exists():
            print(f"  ✗ File not found: {file_path}")
            print(f"  → Please place test PDFs in test_docs/ directory")
            return False, None
        
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            
            with open(file_path, "rb") as f:
                files = {"file": (Path(file_path).name, f, "application/pdf")}
                start_time = time.time()
                
                response = requests.post(
                    f"{self.base_url}/upload",
                    headers=headers,
                    files=files
                )
                
                upload_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                doc_id = result.get("document_id")
                print(f"  ✓ Upload successful ({upload_time:.2f}s)")
                print(f"    Document ID: {doc_id}")
                print(f"    Pages: {result.get('total_pages', 'N/A')}")
                print(f"    Elements: {result.get('total_elements_detected', 'N/A')}")
                self.results["upload_success"] += 1
                return True, doc_id
            else:
                print(f"  ✗ Upload failed: {response.status_code}")
                print(f"    Response: {response.text[:200]}")
                self.results["upload_failures"] += 1
                return False, None
                
        except Exception as e:
            print(f"  ✗ Upload error: {e}")
            self.results["upload_failures"] += 1
            return False, None
    
    def calculate_metrics(self, predicted: str, ground_truth: str, keywords: List[str]) -> Tuple[float, float, float]:
        """Calculate precision, recall, F1 based on keyword matching"""
        predicted_lower = predicted.lower()
        ground_truth_lower = ground_truth.lower()
        
        # Token-level metrics
        pred_tokens = set(predicted_lower.split())
        truth_tokens = set(ground_truth_lower.split())
        
        if not pred_tokens:
            return 0.0, 0.0, 0.0
        
        intersection = pred_tokens & truth_tokens
        precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(intersection) / len(truth_tokens) if truth_tokens else 0.0
        
        # Keyword-based boost
        keyword_matches = sum(1 for kw in keywords if kw.lower() in predicted_lower)
        keyword_score = keyword_matches / len(keywords) if keywords else 0.0
        
        # Combined score
        precision = (precision + keyword_score) / 2
        recall = (recall + keyword_score) / 2
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        return precision, recall, f1
    
    def test_query(self, query: str, ground_truth: str, keywords: List[str]) -> Dict:
        """Test a single query and measure performance"""
        print(f"\n  Query: {query}")
        
        try:
            headers = {"Authorization": f"Bearer {self.auth_token}"}
            
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/query",
                headers=headers,
                json={"query": query}
            )
            latency = (time.time() - start_time) * 1000  # ms
            
            if response.status_code != 200:
                print(f"    ✗ Query failed: {response.status_code}")
                return None
            
            result = response.json()
            answer = result.get("answer", "")
            confidence = result.get("confidence_score", 0.0)
            
            # Calculate metrics
            precision, recall, f1 = self.calculate_metrics(answer, ground_truth, keywords)
            
            # Store results
            self.results["queries_tested"] += 1
            self.results["precision_scores"].append(precision)
            self.results["recall_scores"].append(recall)
            self.results["f1_scores"].append(f1)
            self.results["latencies"].append(latency)
            self.results["confidence_scores"].append(confidence)
            
            print(f"    ✓ Answer received")
            print(f"      P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}")
            print(f"      Latency: {latency:.0f}ms | Confidence: {confidence:.3f}")
            print(f"      Answer: {answer[:100]}...")
            
            return {
                "query": query,
                "answer": answer,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "latency": latency,
                "confidence": confidence
            }
            
        except Exception as e:
            print(f"    ✗ Query error: {e}")
            return None
    
    def run_tests(self):
        """Run complete test suite"""
        print("\n" + "="*60)
        print("ICDI-X REAL DOCUMENT TESTING")
        print("="*60)
        
        # Authenticate
        if not self.register_and_login():
            print("\n✗ Authentication failed. Cannot proceed.")
            return
        
        # Test each document
        for doc_info in TEST_DOCUMENTS:
            print(f"\n{'='*60}")
            print(f"Testing: {doc_info['name']}")
            print(f"{'='*60}")
            
            # Upload document
            success, doc_id = self.upload_document(
                doc_info["file_path"],
                doc_info["name"]
            )
            
            if not success:
                print(f"⚠ Skipping queries for {doc_info['name']} (upload failed)")
                continue
            
            self.document_ids.append(doc_id)
            
            # Wait for processing
            print("  Waiting for document processing...")
            time.sleep(3)
            
            # Test queries
            print(f"\n  Testing {len(doc_info['questions'])} queries:")
            for q in doc_info["questions"]:
                self.test_query(
                    q["query"],
                    q["ground_truth"],
                    q["expected_keywords"]
                )
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print test results summary"""
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        
        print(f"\nDocument Upload:")
        print(f"  Successful: {self.results['upload_success']}")
        print(f"  Failed: {self.results['upload_failures']}")
        
        print(f"\nQuery Performance:")
        print(f"  Total Queries: {self.results['queries_tested']}")
        
        if self.results['f1_scores']:
            avg_p = statistics.mean(self.results['precision_scores'])
            avg_r = statistics.mean(self.results['recall_scores'])
            avg_f1 = statistics.mean(self.results['f1_scores'])
            avg_latency = statistics.mean(self.results['latencies'])
            avg_confidence = statistics.mean(self.results['confidence_scores'])
            
            print(f"\nAccuracy Metrics:")
            print(f"  Avg Precision: {avg_p:.3f}")
            print(f"  Avg Recall: {avg_r:.3f}")
            print(f"  Avg F1 Score: {avg_f1:.3f}")
            
            print(f"\nPerformance Metrics:")
            print(f"  Avg Latency: {avg_latency:.0f}ms")
            print(f"  Avg Confidence: {avg_confidence:.3f}")
            
            print(f"\n{'='*60}")
            if avg_f1 >= 0.80:
                print("🎯 TARGET ACHIEVED: F1 >= 0.80")
            elif avg_f1 >= 0.60:
                print("🟡 GOOD: F1 >= 0.60 (competitive)")
            elif avg_f1 >= 0.40:
                print("🟠 ACCEPTABLE: F1 >= 0.40 (needs improvement)")
            else:
                print("🔴 CRITICAL: F1 < 0.40 (major improvements needed)")
            print(f"{'='*60}")
        else:
            print("\n⚠ No successful queries to analyze")
    
    def save_results(self, output_file: str = "scripts/real_test_results.json"):
        """Save results to JSON file"""
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


def main():
    """Run real document tests"""
    tester = RealDocumentTester()
    
    print("\n📋 SETUP INSTRUCTIONS:")
    print("1. Place test PDF files in test_docs/ directory")
    print("2. Update TEST_DOCUMENTS with your file paths and Q&A pairs")
    print("3. Ensure server is running on http://127.0.0.1:8000")
    print("\nPress Enter to start testing...")
    input()
    
    tester.run_tests()
    tester.save_results()


if __name__ == "__main__":
    main()
