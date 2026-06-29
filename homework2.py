import numpy as np
from tqdm import tqdm

# ============================================================
# SETUP
# ============================================================

from embedder import Embedder
embedder = Embedder()

# ============================================================
# Q1. Embedding a query
# ============================================================

query = "How does approximate nearest neighbor search work?"
v_query = embedder.encode(query)

print("=" * 60)
print("Q1. Embedding a query")
print("=" * 60)
print(f"Query: {query}")
print(f"Vector shape: {v_query.shape}")
print(f"First value (v[0]): {v_query[0]:.4f}")
print()

# ============================================================
# Loading the data
# ============================================================

from gitsource import GithubRepositoryDataReader

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)

documents = [file.parse() for file in reader.read()]
print(f"Loaded {len(documents)} documents")
print()

# ============================================================
# Q2. Cosine similarity
# ============================================================

target_file = "02-vector-search/lessons/07-sqlitesearch-vector.md"
target_doc = None
for doc in documents:
    if doc["filename"] == target_file:
        target_doc = doc
        break

v_doc = embedder.encode(target_doc["content"])
cosine_sim = v_query.dot(v_doc)

print("=" * 60)
print("Q2. Cosine similarity")
print("=" * 60)
print(f"File: {target_file}")
print(f"Cosine similarity: {cosine_sim:.4f}")
print()

# ============================================================
# Q3. Chunking and search by hand
# ============================================================

from gitsource import chunk_documents

chunks = chunk_documents(documents, size=2000, step=1000)
print(f"Total chunks: {len(chunks)}")

chunk_texts = [chunk["content"] for chunk in chunks]
chunk_vectors = embedder.encode_batch(chunk_texts)
X = np.array(chunk_vectors)

scores = X.dot(v_query)
best_idx = np.argmax(scores)
best_chunk = chunks[best_idx]

print("=" * 60)
print("Q3. Chunking and search by hand")
print("=" * 60)
print(f"Best score: {scores[best_idx]:.4f}")
print(f"Best chunk filename: {best_chunk['filename']}")
print(f"Best chunk start: {best_chunk['start']}")
print()

# ============================================================
# Q4. Vector search (manual numpy - no minsearch classes)
# ============================================================

for i, chunk in enumerate(chunks):
    chunk["content_vector"] = X[i].tolist()

query_q4 = "What metric do we use to evaluate a search engine?"
v_q4 = embedder.encode(query_q4)

scores_q4 = X.dot(v_q4)
top5_idx = np.argsort(scores_q4)[-5:][::-1]
results_vector = [chunks[i] for i in top5_idx]

print("=" * 60)
print("Q4. Vector search")
print("=" * 60)
print(f"Query: {query_q4}")
print(f"First result filename: {results_vector[0]['filename']}")
print()

# ============================================================
# Q5. Text search vs vector search
# ============================================================

def manual_text_search(query_text, documents, num_results=5):
    query_words = set(query_text.lower().split())
    scored = []
    for doc in documents:
        content_words = set(doc["content"].lower().split())
        score = len(query_words & content_words)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:num_results]]

query_q5 = "How do I store vectors in PostgreSQL?"
v_q5 = embedder.encode(query_q5)

scores_q5_vec = X.dot(v_q5)
top5_vec_idx = np.argsort(scores_q5_vec)[-5:][::-1]
vector_results_q5 = [chunks[i] for i in top5_vec_idx]

text_results_q5 = manual_text_search(query_q5, chunks, num_results=5)

vector_files_q5 = {r["filename"] for r in vector_results_q5}
text_files_q5 = {r["filename"] for r in text_results_q5}

only_vector = vector_files_q5 - text_files_q5

print("=" * 60)
print("Q5. Text search vs vector search")
print("=" * 60)
print(f"Query: {query_q5}")
print(f"Vector results: {vector_files_q5}")
print(f"Text results: {text_files_q5}")
print(f"In vector but NOT in text: {only_vector}")
print()

# ============================================================
# Q6. Hybrid search with RRF
# ============================================================

def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc.get("start", 0))
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]

query_q6 = "How do I give the model access to tools?"
v_q6 = embedder.encode(query_q6)

scores_q6_vec = X.dot(v_q6)
top10_vec_idx = np.argsort(scores_q6_vec)[-10:][::-1]
vector_results_q6 = [chunks[i] for i in top10_vec_idx]

text_results_q6 = manual_text_search(query_q6, chunks, num_results=10)

hybrid_results = rrf([vector_results_q6, text_results_q6])

print("=" * 60)
print("Q6. Hybrid search with RRF")
print("=" * 60)
print(f"Query: {query_q6}")
print(f"Top RRF result filename: {hybrid_results[0]['filename']}")
print()

# ============================================================
# SUMMARY
# ============================================================

print("=" * 60)
print("ANSWERS SUMMARY")
print("=" * 60)
print(f"Q1. First value v[0]: {v_query[0]:.4f}")
print(f"Q2. Cosine similarity: {cosine_sim:.4f}")
print(f"Q3. Highest-scoring chunk file: {best_chunk['filename']}")
print(f"Q4. First vector search result: {results_vector[0]['filename']}")
print(f"Q5. In vector but not text: {only_vector}")
print(f"Q6. Top RRF result: {hybrid_results[0]['filename']}")
print("=" * 60)