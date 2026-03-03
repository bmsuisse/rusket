# Working with Vector Databases

Export trained recommendation embeddings to production vector databases for real-time retrieval, similarity search, and hybrid fusion.

`rusket` supports **11 vector database backends** through two APIs:

1. **Functional API** — `export_vectors()` / `export_multi_vectors()` with auto-detection
2. **Class API** — `VectorStore` subclasses with `upload()` / `upload_multi()`

---

## Quick Start

### Functional API (auto-detect)

```python
import rusket

model = rusket.ALS(factors=64, iterations=15).fit(interactions)

# Auto-detect backend from client object
from qdrant_client import QdrantClient
client = QdrantClient(":memory:")

rusket.export_vectors(model.item_factors, client=client, collection_name="items")
```

### Class API (VectorStore)

```python
from rusket import QdrantVectorStore
from qdrant_client import QdrantClient

store = QdrantVectorStore(QdrantClient(":memory:"))
store.upload(model.item_factors, collection_name="items")
```

---

## Supported Backends

| VectorStore class | Multi-vector? | Client library | Install |
|---|:---:|---|---|
| `QdrantVectorStore` | ✅ | `qdrant-client` | `pip install qdrant-client` |
| `MeilisearchVectorStore` | ✅ | `meilisearch` | `pip install meilisearch` |
| `WeaviateVectorStore` | ✅ | `weaviate-client` | `pip install weaviate-client` |
| `PgVectorStore` | — | `psycopg2` | `pip install psycopg2-binary` |
| `ChromaVectorStore` | — | `chromadb` | `pip install chromadb` |
| `PineconeVectorStore` | — | `pinecone-client` | `pip install pinecone-client` |
| `MilvusVectorStore` | — | `pymilvus` | `pip install pymilvus` |
| `ElasticsearchVectorStore` | — | `elasticsearch` | `pip install elasticsearch` |
| `MongoDBVectorStore` | — | `pymongo` | `pip install pymongo` |
| `LanceDBVectorStore` | — | `lancedb` | `pip install lancedb` |
| `TypesenseVectorStore` | — | `typesense` | `pip install typesense` |

All backends are **optional** — install only the ones you need.

---

## Real-World Examples

### 1. E-Commerce Product Recommendations → Qdrant

Serve personalised "For You" recommendations from an ALS model via Qdrant's high-performance vector search.

```python
import rusket
import pandas as pd
from qdrant_client import QdrantClient
from rusket import QdrantVectorStore

# ── Train the recommender ──────────────────────────────────────────
purchases = pd.DataFrame({
    "user_id":  [1, 1, 2, 2, 3, 3, 3, 4, 4],
    "item_id":  [101, 102, 101, 103, 102, 103, 104, 101, 104],
})

als = rusket.ALS(factors=64, iterations=15).from_transactions(
    purchases, user_col="user_id", item_col="item_id"
).fit()

# ── Export item embeddings to Qdrant ───────────────────────────────
store = QdrantVectorStore(QdrantClient("localhost", port=6333))
store.upload(
    als.item_factors,
    collection_name="product_embeddings",
    ids=[101, 102, 103, 104],           # use real product IDs
    payloads=[                            # attach metadata
        {"name": "Laptop",      "category": "Electronics"},
        {"name": "Mouse",       "category": "Accessories"},
        {"name": "Keyboard",    "category": "Accessories"},
        {"name": "Monitor",     "category": "Electronics"},
    ],
    distance="Cosine",
)

# ── Query at serving time ──────────────────────────────────────────
# Get the user's embedding and search for similar items
user_vector = als.user_factors[0].tolist()
results = store.client.query_points(
    collection_name="product_embeddings",
    query=user_vector,
    limit=5,
)
print(results)
```

### 2. Hybrid Fusion → Meilisearch (Multi-Vector)

Let Meilisearch handle the fusion of collaborative filtering and semantic embeddings at query time. Each document stores separate `"cf"` and `"semantic"` embedders.

```python
import rusket
import meilisearch
from rusket import MeilisearchVectorStore

# ── Train ALS + get semantic embeddings ────────────────────────────
als = rusket.ALS(factors=64, iterations=15).fit(interactions)

from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("all-MiniLM-L6-v2")
text_vectors = encoder.encode(product_descriptions)  # (n_items, 384)

# ── Fuse into a HybridEmbeddingIndex ──────────────────────────────
hybrid = rusket.HybridEmbeddingIndex(
    cf_embeddings=als.item_factors,
    semantic_embeddings=text_vectors,
)

# ── Export as separate named vectors to Meilisearch ───────────────
client = meilisearch.Client("http://localhost:7700", "masterKey")
store = MeilisearchVectorStore(client)

# Multi-vector upload: each document gets _vectors.cf and _vectors.semantic
store.upload_multi(
    hybrid.named_embeddings,
    collection_name="products",
    ids=product_ids,
    payloads=[{"name": n, "price": p} for n, p in zip(names, prices)],
)

# Or equivalently via the HybridEmbeddingIndex directly:
hybrid.export_vectors(client, mode="multi", collection_name="products")
```

!!! tip "Meilisearch handles the fusion"
    With `mode="multi"`, Meilisearch stores CF and semantic vectors separately.
    At query time, you can weight them differently or use Meilisearch's built-in
    hybrid search to combine both signals automatically.

### 3. Multi-Vector Hybrid Search → Qdrant Named Vectors

Qdrant's named vectors let you store multiple embedding spaces per point and query them independently or together.

```python
import rusket
from qdrant_client import QdrantClient
from rusket import QdrantVectorStore

# ── Build hybrid embeddings ────────────────────────────────────────
als = rusket.ALS(factors=64).fit(interactions)
hybrid = rusket.HybridEmbeddingIndex(
    cf_embeddings=als.item_factors,
    semantic_embeddings=text_vectors,
    strategy="weighted_concat",
    alpha=0.6,
)

# ── Export as named vectors ────────────────────────────────────────
store = QdrantVectorStore(QdrantClient("localhost"))

# Each point gets two vectors: "cf" (64-d) and "semantic" (384-d)
store.upload_multi(
    hybrid.named_embeddings,
    collection_name="hybrid_products",
    distance="Cosine",
)

# ── Query with different strategies ───────────────────────────────
from qdrant_client.models import NamedVector

# Query CF space only (behavioural similarity)
cf_results = store.client.query_points(
    collection_name="hybrid_products",
    query=NamedVector(name="cf", vector=user_cf_vector),
    limit=10,
)

# Query semantic space only (content similarity)
sem_results = store.client.query_points(
    collection_name="hybrid_products",
    query=NamedVector(name="semantic", vector=query_text_vector),
    limit=10,
)
```

### 4. Production Pipeline → PostgreSQL (pgvector)

For teams already running PostgreSQL, `pgvector` lets you add vector search without a separate service.

```python
import rusket
import psycopg2
from rusket import PgVectorStore

# ── Connect to your existing PostgreSQL database ──────────────────
conn = psycopg2.connect(
    host="localhost", dbname="myapp", user="api_user", password="secret"
)

# ── Train and export ──────────────────────────────────────────────
model = rusket.ALS(factors=32).fit(interactions)

store = PgVectorStore(conn)
store.upload(
    model.item_factors,
    collection_name="item_embeddings",
    ids=item_ids,
)

# ── Query with SQL! ───────────────────────────────────────────────
cursor = conn.cursor()
user_vec = model.user_factors[42].tolist()
cursor.execute("""
    SELECT id, payload
    FROM item_embeddings
    ORDER BY embedding <=> %s::vector
    LIMIT 10
""", (user_vec,))
results = cursor.fetchall()
```

### 5. Serverless Search → ChromaDB (Local Dev / Prototyping)

ChromaDB is perfect for local development and prototyping — zero infrastructure needed.

```python
import rusket
import chromadb
from rusket import ChromaVectorStore

# ── In-memory for development ─────────────────────────────────────
client = chromadb.Client()
store = ChromaVectorStore(client)

model = rusket.BPR(factors=64).fit(interactions)
store.upload(
    model.item_factors,
    collection_name="recommendations",
    ids=[str(i) for i in range(model.item_factors.shape[0])],
)

# ── Persistent for staging ────────────────────────────────────────
client = chromadb.PersistentClient(path="./chroma_db")
store = ChromaVectorStore(client)
store.upload(model.item_factors, collection_name="recommendations")
```

### 6. Real-Time Weaviate with Named Vectors (v4)

Weaviate v4's named vector support enables multi-modal search across different embedding spaces.

```python
import rusket
import weaviate
from rusket import WeaviateVectorStore

# ── Connect to Weaviate ───────────────────────────────────────────
client = weaviate.connect_to_local()  # or weaviate.connect_to_wcs(...)

# ── Multi-vector export ───────────────────────────────────────────
hybrid = rusket.HybridEmbeddingIndex(
    cf_embeddings=als.item_factors,
    semantic_embeddings=text_vectors,
)

store = WeaviateVectorStore(client)
store.upload_multi(
    hybrid.named_embeddings,
    collection_name="Products",
)

# Check capabilities at runtime
assert store.supports_multi_vector  # True for v4 clients
```

### 7. Batch Export with Metadata

All stores support `payloads` for attaching rich metadata to each vector:

```python
import rusket
import pandas as pd
from rusket import QdrantVectorStore
from qdrant_client import QdrantClient

# ── Prepare metadata from your product catalog ───────────────────
catalog = pd.DataFrame({
    "item_id": [1, 2, 3, 4, 5],
    "name":    ["Laptop", "Mouse", "Keyboard", "Monitor", "Webcam"],
    "price":   [999.99, 29.99, 79.99, 449.99, 89.99],
    "category": ["Electronics", "Accessories", "Accessories", "Electronics", "Accessories"],
    "in_stock": [True, True, False, True, True],
})

# ── Train and export with metadata ────────────────────────────────
model = rusket.ALS(factors=64).fit(interactions)

store = QdrantVectorStore(QdrantClient("localhost"))
store.upload(
    model.item_factors,
    collection_name="products",
    ids=catalog["item_id"].tolist(),
    payloads=catalog.drop(columns="item_id").to_dict("records"),
    batch_size=500,   # control upload batch size
)

# ── Filter at query time using metadata ───────────────────────────
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = store.client.query_points(
    collection_name="products",
    query=user_vector,
    query_filter=Filter(
        must=[
            FieldCondition(key="in_stock", match=MatchValue(value=True)),
            FieldCondition(key="category", match=MatchValue(value="Electronics")),
        ]
    ),
    limit=5,
)
```

### 8. End-to-End: Train → Fuse → Export → Serve

A complete production pipeline from training to serving:

```python
import rusket
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# ═══════════════════════════════════════════════════════════════════
# STEP 1: Train the recommender
# ═══════════════════════════════════════════════════════════════════
als = rusket.ALS(factors=64, iterations=15, alpha=40.0).from_transactions(
    purchase_log, user_col="user_id", item_col="item_id", rating_col="quantity"
).fit()

# ═══════════════════════════════════════════════════════════════════
# STEP 2: Generate semantic embeddings
# ═══════════════════════════════════════════════════════════════════
encoder = SentenceTransformer("all-MiniLM-L6-v2")
text_vectors = encoder.encode(
    [f"{row['name']} {row['description']}" for _, row in products.iterrows()]
)

# ═══════════════════════════════════════════════════════════════════
# STEP 3: Fuse into hybrid space
# ═══════════════════════════════════════════════════════════════════
hybrid = rusket.HybridEmbeddingIndex(
    cf_embeddings=als.item_factors,
    semantic_embeddings=text_vectors,
    strategy="weighted_concat",
    alpha=0.6,  # 60% collaborative, 40% semantic
)

# ═══════════════════════════════════════════════════════════════════
# STEP 4: Export to vector DB
# ═══════════════════════════════════════════════════════════════════
client = QdrantClient("qdrant.mycompany.com", port=6333, api_key="...")

# Option A: Single fused vector (simpler, works everywhere)
hybrid.export_vectors(client, collection_name="product_recs")

# Option B: Separate named vectors (DB handles fusion at query time)
hybrid.export_vectors(client, mode="multi", collection_name="product_recs_multi")

# ═══════════════════════════════════════════════════════════════════
# STEP 5: Serve recommendations via your API
# ═══════════════════════════════════════════════════════════════════
def get_recommendations(user_id: int, n: int = 10) -> list[dict]:
    """FastAPI endpoint handler."""
    user_vector = als.user_factors[user_id].tolist()
    results = client.query_points(
        collection_name="product_recs",
        query=user_vector,
        limit=n,
    )
    return [{"id": r.id, "score": r.score, **r.payload} for r in results.points]
```

---

## Multi-Vector Export

Databases like Qdrant, Meilisearch, and Weaviate support **multiple vectors per document**. This lets the database handle fusion at query time instead of pre-computing a single fused vector.

### When to use multi-vector

| Approach | Pros | Cons |
|---|---|---|
| **Single fused vector** | Simple, works with all backends | Must re-export when changing fusion weights |
| **Multi-vector (DB-side)** | Change weights at query time, A/B test fusion strategies | Requires Qdrant/Meilisearch/Weaviate |

### API

**Via `HybridEmbeddingIndex`:**

```python
# Fused (default) — single vector per item
hybrid.export_vectors(client, collection_name="items")

# Multi — separate named CF + semantic vectors
hybrid.export_vectors(client, mode="multi", collection_name="items")
```

**Via `export_multi_vectors()` function:**

```python
rusket.export_multi_vectors(
    {"cf": als.item_factors, "semantic": text_vectors},
    client=qdrant_client,
    collection_name="items",
)
```

**Via `VectorStore.upload_multi()`:**

```python
store = QdrantVectorStore(client)
store.upload_multi(
    {"cf": als.item_factors, "semantic": text_vectors},
    collection_name="items",
)
```

---

## VectorStore ABC

All concrete stores inherit from `VectorStore`:

```python
from rusket import VectorStore

class VectorStore(ABC):
    def __init__(self, client: Any) -> None: ...

    @property
    def client(self) -> Any: ...

    @property
    def supports_multi_vector(self) -> bool: ...

    @abstractmethod
    def upload(self, vectors, collection_name, *, ids, payloads, ...) -> int: ...

    def upload_multi(self, named_vectors, collection_name, ...) -> int: ...
```

You can subclass `VectorStore` to add your own backend:

```python
from rusket import VectorStore

class MyCustomStore(VectorStore):
    def upload(self, vectors, collection_name="items", **kwargs) -> int:
        # Your custom upload logic
        for i, vec in enumerate(vectors):
            self.client.insert(collection_name, id=i, vector=vec.tolist())
        return len(vectors)
```

---

## Tips & Best Practices

!!! tip "Normalise for cosine similarity"
    Most vector DBs default to cosine distance. Use `distance="Cosine"` in Qdrant
    or ensure your vectors are L2-normalised. `HybridEmbeddingIndex.named_embeddings`
    returns normalised vectors automatically.

!!! tip "Batch size tuning"
    The default `batch_size=1000` is good for most cases. For very large exports
    (millions of vectors), increase to 5000–10000 to reduce HTTP round-trips.

!!! tip "Runtime capability check"
    ```python
    store = QdrantVectorStore(client)
    if store.supports_multi_vector:
        store.upload_multi(hybrid.named_embeddings, ...)
    else:
        store.upload(hybrid.fused_embeddings, ...)
    ```

!!! warning "Multi-vector compatibility"
    Only Qdrant, Meilisearch, and Weaviate (v4) support multi-vector storage.
    Other backends will raise `NotImplementedError` if you call `upload_multi()`.
