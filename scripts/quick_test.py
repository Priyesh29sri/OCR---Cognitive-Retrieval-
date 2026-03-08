#!/usr/bin/env python3
"""
Quick Manual Test - Upload any PDF and test queries
Usage: python scripts/quick_test.py <path_to_pdf>
"""

import sys
import requests
import json
from pathlib import Path
import time

BASE_URL = "http://127.0.0.1:8000"

def test_with_pdf(pdf_path):
    """Test ICDI-X with any PDF file"""
    
    print("\n" + "="*70)
    print("ICDI-X REAL DATA TEST")
    print("="*70)
    
    if not Path(pdf_path).exists():
        print(f"\n❌ File not found: {pdf_path}")
        print("\nTip: You can use any PDF file you have:")
        print("  python scripts/quick_test.py ~/Downloads/paper.pdf")
        print("  python scripts/quick_test.py ~/Documents/report.pdf")
        return
    
    print(f"\n📄 Testing with: {pdf_path}")
    print(f"   File size: {Path(pdf_path).stat().st_size / 1024:.1f} KB")
    
    # Step 1: Register/Login
    print("\n" + "-"*70)
    print("STEP 1: Authentication")
    print("-"*70)
    
    email = f"test_{int(time.time())}@example.com"
    password = "TestPass123!"
    
    # Try to register
    response = requests.post(
        f"{BASE_URL}/auth/register",
        json={"email": email, "password": password, "full_name": "Test User"}
    )
    
    if response.status_code == 201:
        print("✅ User registered")
    elif "already exists" in response.text:
        print("ℹ️  User exists, will login")
    else:
        print(f"⚠️  Registration response: {response.status_code}")
    
    # Login
    response = requests.post(
        f"{BASE_URL}/auth/login",
        json={"email": email, "password": password}
    )
    
    if response.status_code != 200:
        # Try with default test user
        response = requests.post(
            f"{BASE_URL}/auth/login",
            json={"email": "test@example.com", "password": "TestPass123!"}
        )
    
    if response.status_code == 200:
        token = response.json()["access_token"]
        print(f"✅ Authenticated successfully")
    else:
        print(f"❌ Login failed: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
        return
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # Step 2: Upload PDF
    print("\n" + "-"*70)
    print("STEP 2: Upload Document")
    print("-"*70)
    
    with open(pdf_path, "rb") as f:
        files = {"file": (Path(pdf_path).name, f, "application/pdf")}
        
        print(f"⏳ Uploading {Path(pdf_path).name}...")
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/upload",
            headers=headers,
            files=files,
            timeout=120
        )
        
        upload_time = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        doc_id = result.get("document_id")
        
        print(f"✅ Upload successful! ({upload_time:.1f}s)")
        print(f"   Document ID: {doc_id}")
        print(f"   Pages processed: {result.get('total_pages', 'N/A')}")
        print(f"   Elements detected: {result.get('total_elements_detected', 'N/A')}")
        print(f"   Text extracted: {result.get('total_text_length', 'N/A')} chars")
    else:
        print(f"❌ Upload failed: {response.status_code}")
        print(f"   Response: {response.text[:500]}")
        return
    
    # Wait for processing
    print("\n⏳ Waiting for document processing (5 seconds)...")
    time.sleep(5)
    
    # Step 3: Test Queries
    print("\n" + "-"*70)
    print("STEP 3: Test Queries")
    print("-"*70)
    
    # Generic questions that work for any document
    test_queries = [
        "What is this document about? Summarize the main topic.",
        "What are the key points or findings in this document?",
        "List any important numbers, statistics, or metrics mentioned.",
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}: {query}")
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/query",
            headers=headers,
            json={"query": query},
            timeout=60
        )
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("answer", "")
            confidence = result.get("confidence_score", 0.0)
            method = result.get("retrieval_method", "unknown")
            
            print(f"✅ Response received ({latency:.0f}ms)")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Method: {method}")
            print(f"   Answer: {answer[:200]}...")
            if len(answer) > 200:
                print(f"           {answer[200:400]}...")
            
            results.append({
                "query": query,
                "answer": answer,
                "confidence": confidence,
                "latency_ms": latency
            })
        else:
            print(f"❌ Query failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    if results:
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        avg_latency = sum(r["latency_ms"] for r in results) / len(results)
        
        print(f"\n📊 Performance Metrics:")
        print(f"   Queries tested: {len(results)}")
        print(f"   Average confidence: {avg_confidence:.2f}")
        print(f"   Average latency: {avg_latency:.0f}ms")
        
        print(f"\n✅ SYSTEM IS WORKING WITH REAL DATA!")
        print(f"\nNext steps:")
        print(f"1. Try your own questions specific to this document")
        print(f"2. Compare answers to what's actually in the PDF")
        print(f"3. Calculate F1 score by comparing to ground truth")
        
        # Save results
        output_file = "scripts/last_test_results.json"
        with open(output_file, "w") as f:
            json.dump({
                "document": pdf_path,
                "upload_time": upload_time,
                "results": results
            }, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
    else:
        print(f"\n⚠️  No successful queries")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python scripts/quick_test.py <path_to_pdf>")
        print("\nExample:")
        print("  python scripts/quick_test.py ~/Downloads/research_paper.pdf")
        print("  python scripts/quick_test.py ~/Documents/report.pdf")
        print("\nOr test with any PDF you have on your computer!")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    test_with_pdf(pdf_path)
