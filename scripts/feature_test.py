import requests, json, time, sys

API = "http://127.0.0.1:8000"

ts = int(time.time())
email = f"fulltest{ts}@evaltest.com"
pw = "Test123!"

r = requests.post(f"{API}/auth/register", json={"email": email, "password": pw, "full_name": "FullTest"})
print("Reg:", r.status_code)

r = requests.post(f"{API}/auth/login", json={"email": email, "password": pw})
if r.status_code != 200:
    print("Login failed:", r.text)
    sys.exit(1)
token = r.json()["access_token"]
hdrs = {"Authorization": f"Bearer {token}"}
print("Logged in. Token len:", len(token))

print("Uploading PDF...")
with open("Untitled document-9.pdf", "rb") as f:
    r = requests.post(f"{API}/upload", headers=hdrs,
                      files={"file": ("test.pdf", f, "application/pdf")}, timeout=180)
print("Upload:", r.status_code)
if r.status_code != 200:
    print(r.text[:300])
    sys.exit(1)

doc_id = r.json().get("document_id")
print("doc_id:", doc_id)
print("Waiting 8s for indexing...")
time.sleep(8)

results = {}

# insights
t0 = time.time()
r2 = requests.get(f"{API}/insights/{doc_id}", headers=hdrs, timeout=60)
lat = (time.time() - t0) * 1000
print(f"insights: {r2.status_code} {lat:.0f}ms")
if r2.status_code == 200:
    d = r2.json()
    results["insights"] = {
        "latency_ms": round(lat, 1),
        "n_insights": len(d.get("insights", [])),
        "n_questions": len(d.get("suggested_questions", [])),
        "n_entities": len(d.get("key_entities", [])),
        "chunks_selected": d.get("chunks_selected_by_ib", 0),
        "chunks_analyzed": d.get("chunks_analyzed", 0),
        "doc_type": d.get("doc_type", ""),
        "complexity": d.get("complexity", ""),
    }

# study guide
t0 = time.time()
r2 = requests.get(f"{API}/studyguide/{doc_id}", headers=hdrs, timeout=60)
lat = (time.time() - t0) * 1000
print(f"studyguide: {r2.status_code} {lat:.0f}ms")
if r2.status_code == 200:
    d = r2.json()
    bq = d.get("blooms_questions", {})
    results["study_guide"] = {
        "latency_ms": round(lat, 1),
        "title": d.get("title", ""),
        "n_concepts": len(d.get("key_concepts", [])),
        "study_time_min": d.get("estimated_study_time_minutes", 0),
        "blooms_levels": sum(1 for v in bq.values() if v),
    }
    print("  StudyGuide title:", d.get("title", "")[:60])

# pipeline summary
t0 = time.time()
r2 = requests.get(f"{API}/pipeline/summary", headers=hdrs, timeout=30)
lat = (time.time() - t0) * 1000
print(f"pipeline: {r2.status_code} {lat:.0f}ms")
if r2.status_code == 200:
    d = r2.json()
    results["pipeline_summary"] = {"latency_ms": round(lat, 1), "summary": d}
    print("  Pipeline keys:", list(d.keys()))

# KG export
t0 = time.time()
r2 = requests.get(f"{API}/knowledge-graph/export", headers=hdrs, timeout=30)
lat = (time.time() - t0) * 1000
print(f"kg export: {r2.status_code} {lat:.0f}ms")
if r2.status_code == 200:
    d = r2.json()
    kg = d.get("knowledge_graph", d)
    results["kg_export"] = {
        "latency_ms": round(lat, 1),
        "n_entities": len(kg.get("entities", [])),
        "n_relations": len(kg.get("relations", [])),
    }
    print("  KG entities:", results["kg_export"]["n_entities"])

# contradictions
t0 = time.time()
r2 = requests.post(f"{API}/contradictions", headers=hdrs,
                   params={"doc_a_id": doc_id, "doc_b_id": doc_id}, timeout=60)
lat = (time.time() - t0) * 1000
print(f"contradictions: {r2.status_code} {lat:.0f}ms")
if r2.status_code == 200:
    d = r2.json()
    results["contradictions"] = {"latency_ms": round(lat, 1), "result": d}
else:
    print("  body:", r2.text[:200])

# query_multi
t0 = time.time()
r2 = requests.post(f"{API}/query_multi", headers=hdrs,
                   json={"queries": ["What is this document about?", "Who is the author?"]},
                   timeout=60)
lat = (time.time() - t0) * 1000
print(f"query_multi: {r2.status_code} {lat:.0f}ms")
if r2.status_code == 200:
    d = r2.json()
    results["query_multi"] = {"latency_ms": round(lat, 1), "n_answers": len(d) if isinstance(d, list) else 1}
else:
    print("  body:", r2.text[:300])

# streaming
t0 = time.time()
r2 = requests.post(f"{API}/query/stream", headers={**hdrs, "Accept": "text/event-stream"},
                   json={"query": "Summarize this document"}, stream=True, timeout=30)
lat = (time.time() - t0) * 1000
chunks = 0
if r2.status_code == 200:
    for c in r2.iter_content(128):
        if c:
            chunks += 1
        if chunks >= 10:
            break
    r2.close()
print(f"streaming: {r2.status_code} {lat:.0f}ms chunks={chunks}")
results["streaming"] = {"status": r2.status_code, "latency_ms": round(lat, 1), "chunks_received": chunks}

out = "scripts/feature_test_results.json"
with open(out, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved -> {out}")
print(json.dumps(results, indent=2, default=str))
