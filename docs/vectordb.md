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

## Serving Recommendations from Vector Databases

Once you've exported embeddings, you need to **retrieve recommendations at serving time**. This section shows production-ready patterns using the **native SDKs** of Qdrant, Meilisearch, and PostgreSQL (pgvector).

### Qdrant — Full Recommendation Serving

#### Personalised "For You" Recommendations

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
import rusket

# ── Setup (one-time) ──────────────────────────────────────────────
als = rusket.ALS(factors=64, iterations=15).from_transactions(
    purchases, user_col="user_id", item_col="item_id"
).fit()

client = QdrantClient("localhost", port=6333)
rusket.export_vectors(als.item_factors, client=client, collection_name="items")

# ── Recommend for a user ──────────────────────────────────────────
def recommend_for_user(
    user_id: int, n: int = 10, category: str | None = None
) -> list[dict]:
    """Get top-N recommendations for a user, optionally filtered by category."""
    user_vector = als.user_factors[user_id].tolist()

    # Build optional filters
    filters = []
    if category:
        filters.append(
            FieldCondition(key="category", match=MatchValue(value=category))
        )

    results = client.query_points(
        collection_name="items",
        query=user_vector,
        query_filter=Filter(must=filters) if filters else None,
        limit=n,
        with_payload=True,
    )
    return [
        {"id": r.id, "score": r.score, **r.payload}
        for r in results.points
    ]

# Usage
recs = recommend_for_user(user_id=42, n=10, category="Electronics")
```

#### "Similar Items" (Item-to-Item)

```python
def similar_items(item_id: int, n: int = 10) -> list[dict]:
    """Find items similar to a given item using its embedding."""
    item_vector = als.item_factors[item_id].tolist()

    results = client.query_points(
        collection_name="items",
        query=item_vector,
        limit=n + 1,  # +1 to exclude the item itself
        with_payload=True,
    )
    # Filter out the query item
    return [
        {"id": r.id, "score": r.score, **r.payload}
        for r in results.points if r.id != item_id
    ][:n]

similar = similar_items(item_id=101, n=5)
# → [{"id": 104, "score": 0.95, "name": "Monitor", ...}, ...]
```

#### Hybrid Named Vector Search (CF + Semantic)

```python
from qdrant_client.models import Prefetch, FusionQuery, Fusion

def hybrid_recommend(
    user_id: int, query_text: str, n: int = 10
) -> list[dict]:
    """Combine CF and semantic signals at query time using Qdrant's fusion."""
    user_cf_vector = als.user_factors[user_id].tolist()
    text_vector = encoder.encode(query_text).tolist()

    # Reciprocal Rank Fusion across both vector spaces
    results = client.query_points(
        collection_name="hybrid_items",
        prefetch=[
            Prefetch(query=user_cf_vector, using="cf", limit=50),
            Prefetch(query=text_vector, using="semantic", limit=50),
        ],
        query=FusionQuery(fusion=Fusion.RRF),  # Reciprocal Rank Fusion
        limit=n,
        with_payload=True,
    )
    return [{"id": r.id, "score": r.score, **r.payload} for r in results.points]

# "Show me laptops this user would like"
recs = hybrid_recommend(user_id=42, query_text="lightweight laptop for travel")
```

#### FastAPI Integration

```python
from fastapi import FastAPI, Query
from qdrant_client import QdrantClient
import rusket

app = FastAPI()
client = QdrantClient("localhost")
model = rusket.load_model("trained_als.pkl")

@app.get("/recommendations/{user_id}")
async def get_recommendations(
    user_id: int,
    n: int = Query(default=10, le=100),
    category: str | None = None,
):
    user_vector = model.user_factors[user_id].tolist()

    filters = []
    if category:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        filters.append(FieldCondition(key="category", match=MatchValue(value=category)))

    results = client.query_points(
        collection_name="product_embeddings",
        query=user_vector,
        query_filter=Filter(must=filters) if filters else None,
        limit=n,
        with_payload=True,
    )
    return {
        "user_id": user_id,
        "recommendations": [
            {"item_id": r.id, "score": round(r.score, 4), **r.payload}
            for r in results.points
        ],
    }

@app.get("/similar/{item_id}")
async def get_similar(item_id: int, n: int = Query(default=5, le=50)):
    item_vector = model.item_factors[item_id].tolist()
    results = client.query_points(
        collection_name="product_embeddings",
        query=item_vector,
        limit=n + 1,
        with_payload=True,
    )
    return {
        "item_id": item_id,
        "similar": [
            {"item_id": r.id, "score": round(r.score, 4), **r.payload}
            for r in results.points if r.id != item_id
        ][:n],
    }
```

---

### Meilisearch — Hybrid Search Recommendations

Meilisearch excels at combining **keyword search with vector similarity** in a single query — perfect for e-commerce where users type product queries and you want to boost results with collaborative filtering signals.

#### Setup: Export with Product Metadata

```python
import rusket
import meilisearch
from rusket import MeilisearchVectorStore

als = rusket.ALS(factors=64).fit(interactions)
client = meilisearch.Client("http://localhost:7700", "masterKey")

# Configure searchable and filterable attributes
index = client.index("products")
index.update_settings({
    "searchableAttributes": ["name", "description", "category"],
    "filterableAttributes": ["category", "price", "in_stock"],
    "sortableAttributes": ["price", "popularity"],
})

# Export with rich metadata
store = MeilisearchVectorStore(client)
store.upload(
    als.item_factors,
    collection_name="products",
    ids=product_ids,
    payloads=[
        {
            "name": row["name"],
            "description": row["description"],
            "category": row["category"],
            "price": row["price"],
            "in_stock": row["in_stock"],
            "popularity": row["sales_count"],
        }
        for _, row in catalog.iterrows()
    ],
)
```

#### Personalised Recommendations (Pure Vector)

```python
def recommend_for_user(user_id: int, n: int = 10, category: str | None = None):
    """Pure vector search — find items closest to user's taste."""
    user_vector = als.user_factors[user_id].tolist()

    index = client.index("products")
    results = index.search(
        "",  # empty query = pure vector search
        opt_params={
            "vector": user_vector,
            "limit": n,
            "filter": f'category = "{category}"' if category else None,
            "attributesToRetrieve": ["name", "price", "category"],
        },
    )
    return results["hits"]

recs = recommend_for_user(42, category="Electronics")
# → [{"name": "Laptop Pro", "price": 1299, "_rankingScore": 0.95}, ...]
```

#### Hybrid Search: Keyword + CF Embeddings

```python
def hybrid_search(
    user_id: int, query: str, n: int = 10, category: str | None = None
):
    """Combine text search with CF vector for relevance + personalisation."""
    user_vector = als.user_factors[user_id].tolist()

    index = client.index("products")
    results = index.search(
        query,  # keyword query ("wireless mouse", "laptop bag", etc.)
        opt_params={
            "vector": user_vector,  # personalisation signal
            "hybrid": {
                "semanticRatio": 0.5,  # 50% keyword, 50% vector
            },
            "limit": n,
            "filter": f'in_stock = true AND category = "{category}"' if category else "in_stock = true",
            "attributesToRetrieve": ["name", "price", "category", "description"],
        },
    )
    return results["hits"]

# User 42 searches "wireless keyboard" → results personalised to their taste
hits = hybrid_search(42, "wireless keyboard", category="Accessories")
```

#### Multi-Embedder Search (CF + Semantic)

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")

# ── Export multi-vector (done once) ───────────────────────────────
hybrid = rusket.HybridEmbeddingIndex(
    cf_embeddings=als.item_factors,
    semantic_embeddings=text_vectors,
)
hybrid.export_vectors(client, mode="multi", collection_name="products_multi")

# ── Query with specific embedder ─────────────────────────────────
def search_by_description(query_text: str, n: int = 10):
    """Search using semantic similarity to product descriptions."""
    query_vector = encoder.encode(query_text).tolist()

    index = client.index("products_multi")
    results = index.search(
        "",
        opt_params={
            "vector": query_vector,
            "hybridEmbedder": "semantic",  # use the semantic embedder
            "limit": n,
        },
    )
    return results["hits"]

def recommend_collaborative(user_id: int, n: int = 10):
    """Recommend using pure CF signal."""
    user_vector = als.user_factors[user_id].tolist()

    index = client.index("products_multi")
    results = index.search(
        "",
        opt_params={
            "vector": user_vector,
            "hybridEmbedder": "cf",  # use the CF embedder
            "limit": n,
        },
    )
    return results["hits"]
```

#### Faceted Recommendations with Meilisearch

```python
def recommend_with_facets(user_id: int, n: int = 20):
    """Get recommendations grouped by category for a discovery page."""
    user_vector = als.user_factors[user_id].tolist()

    index = client.index("products")
    results = index.search(
        "",
        opt_params={
            "vector": user_vector,
            "limit": n,
            "facets": ["category"],
            "filter": "in_stock = true",
            "attributesToRetrieve": ["name", "price", "category"],
        },
    )

    # Group by category
    from collections import defaultdict
    by_category = defaultdict(list)
    for hit in results["hits"]:
        by_category[hit["category"]].append(hit)

    return {
        "facets": results.get("facetDistribution", {}),
        "by_category": dict(by_category),
    }

# Returns: {"facets": {"category": {"Electronics": 8, "Accessories": 12}},
#           "by_category": {"Electronics": [...], "Accessories": [...]}}
```

---

### PostgreSQL (pgvector) — SQL-Native Recommendations

pgvector lets you serve recommendations using plain SQL — no extra infrastructure, no client libraries, just your existing PostgreSQL database.

#### Setup: Create the Embeddings Table

```python
import rusket
import psycopg2

als = rusket.ALS(factors=64).fit(interactions)

conn = psycopg2.connect(
    host="localhost", dbname="myapp", user="api_user", password="secret"
)

# rusket's export handles this automatically, but here's the manual SQL:
cursor = conn.cursor()
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS item_embeddings (
        id          INTEGER PRIMARY KEY,
        embedding   vector(64),
        name        TEXT,
        category    TEXT,
        price       NUMERIC(10,2),
        in_stock    BOOLEAN DEFAULT TRUE
    )
""")

# Create an IVFFlat index for fast approximate search
cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_items_embedding
    ON item_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100)
""")
conn.commit()

# Export embeddings
from rusket import PgVectorStore
store = PgVectorStore(conn)
store.upload(als.item_factors, collection_name="item_embeddings", ids=item_ids)
```

#### Personalised Recommendations (SQL)

```python
def recommend_for_user(
    conn, user_id: int, n: int = 10, category: str | None = None
) -> list[dict]:
    """Get top-N recommendations using cosine distance."""
    user_vec = als.user_factors[user_id].tolist()
    cursor = conn.cursor()

    if category:
        cursor.execute("""
            SELECT id, name, category, price,
                   1 - (embedding <=> %s::vector) AS score
            FROM item_embeddings
            WHERE in_stock = TRUE AND category = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (user_vec, category, user_vec, n))
    else:
        cursor.execute("""
            SELECT id, name, category, price,
                   1 - (embedding <=> %s::vector) AS score
            FROM item_embeddings
            WHERE in_stock = TRUE
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (user_vec, user_vec, n))

    columns = ["id", "name", "category", "price", "score"]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]

recs = recommend_for_user(conn, user_id=42, category="Electronics")
```

#### Similar Items (SQL)

```python
def similar_items(conn, item_id: int, n: int = 10) -> list[dict]:
    """Find similar items using item embedding similarity."""
    cursor = conn.cursor()
    cursor.execute("""
        WITH target AS (
            SELECT embedding FROM item_embeddings WHERE id = %s
        )
        SELECT ie.id, ie.name, ie.category, ie.price,
               1 - (ie.embedding <=> t.embedding) AS similarity
        FROM item_embeddings ie, target t
        WHERE ie.id != %s AND ie.in_stock = TRUE
        ORDER BY ie.embedding <=> t.embedding
        LIMIT %s
    """, (item_id, item_id, n))

    columns = ["id", "name", "category", "price", "similarity"]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]

similar = similar_items(conn, item_id=101, n=5)
# → [{"id": 104, "name": "Monitor", "similarity": 0.93, ...}]
```

#### "Customers Who Bought X Also Bought Y" (SQL)

```python
def also_bought(conn, cart_item_ids: list[int], n: int = 5) -> list[dict]:
    """Average the embeddings of items in cart, find nearest neighbours."""
    cursor = conn.cursor()

    # Average the embeddings of cart items to create a "session vector"
    cursor.execute("""
        WITH cart_avg AS (
            SELECT AVG(embedding) AS centroid
            FROM item_embeddings
            WHERE id = ANY(%s)
        )
        SELECT ie.id, ie.name, ie.category, ie.price,
               1 - (ie.embedding <=> ca.centroid) AS relevance
        FROM item_embeddings ie, cart_avg ca
        WHERE ie.id != ALL(%s) AND ie.in_stock = TRUE
        ORDER BY ie.embedding <=> ca.centroid
        LIMIT %s
    """, (cart_item_ids, cart_item_ids, n))

    columns = ["id", "name", "category", "price", "relevance"]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]

# User has Laptop + Mouse in cart
cross_sells = also_bought(conn, cart_item_ids=[101, 102], n=3)
# → [{"id": 103, "name": "Keyboard", "relevance": 0.87}, ...]
```

#### Category-Scoped Diversity (SQL)

```python
def diverse_recommendations(
    conn, user_id: int, per_category: int = 3, max_categories: int = 5
) -> list[dict]:
    """Get top-N per category for a discovery feed — avoids category bubbles."""
    user_vec = als.user_factors[user_id].tolist()
    cursor = conn.cursor()

    cursor.execute("""
        WITH ranked AS (
            SELECT id, name, category, price,
                   1 - (embedding <=> %s::vector) AS score,
                   ROW_NUMBER() OVER (
                       PARTITION BY category
                       ORDER BY embedding <=> %s::vector
                   ) AS rank_in_cat
            FROM item_embeddings
            WHERE in_stock = TRUE
        )
        SELECT id, name, category, price, score
        FROM ranked
        WHERE rank_in_cat <= %s
        ORDER BY score DESC
        LIMIT %s
    """, (user_vec, user_vec, per_category, per_category * max_categories))

    columns = ["id", "name", "category", "price", "score"]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]

# Returns top-3 items from each of the 5 best-matching categories
diverse = diverse_recommendations(conn, user_id=42)
```

#### FastAPI + pgvector

```python
from fastapi import FastAPI, Depends, Query
import psycopg2
from psycopg2.pool import ThreadedConnectionPool

app = FastAPI()
pool = ThreadedConnectionPool(
    minconn=2, maxconn=10,
    host="localhost", dbname="myapp", user="api_user", password="secret"
)
model = rusket.load_model("trained_als.pkl")

def get_db():
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)

@app.get("/api/recommendations/{user_id}")
async def recommendations(
    user_id: int,
    n: int = Query(default=10, le=100),
    category: str | None = None,
    conn=Depends(get_db),
):
    user_vec = model.user_factors[user_id].tolist()
    cursor = conn.cursor()

    sql = """
        SELECT id, name, category, price,
               1 - (embedding <=> %s::vector) AS score
        FROM item_embeddings
        WHERE in_stock = TRUE
    """
    params = [user_vec]

    if category:
        sql += " AND category = %s"
        params.append(category)

    sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
    params.extend([user_vec, n])

    cursor.execute(sql, params)
    columns = ["id", "name", "category", "price", "score"]

    return {
        "user_id": user_id,
        "recommendations": [dict(zip(columns, row)) for row in cursor.fetchall()],
    }

@app.get("/api/similar/{item_id}")
async def similar(
    item_id: int,
    n: int = Query(default=5, le=50),
    conn=Depends(get_db),
):
    cursor = conn.cursor()
    cursor.execute("""
        WITH target AS (SELECT embedding FROM item_embeddings WHERE id = %s)
        SELECT ie.id, ie.name, ie.category, ie.price,
               1 - (ie.embedding <=> t.embedding) AS similarity
        FROM item_embeddings ie, target t
        WHERE ie.id != %s AND ie.in_stock = TRUE
        ORDER BY ie.embedding <=> t.embedding
        LIMIT %s
    """, (item_id, item_id, n))

    columns = ["id", "name", "category", "price", "similarity"]
    return {
        "item_id": item_id,
        "similar": [dict(zip(columns, row)) for row in cursor.fetchall()],
    }
```

!!! tip "pgvector distance operators"
    | Operator | Distance | Best for |
    |---|---|---|
    | `<=>` | Cosine distance | L2-normalised embeddings (default) |
    | `<->` | L2 (Euclidean) distance | Raw embeddings |
    | `<#>` | Inner product (negative) | When magnitude matters |

    Use `<=>` (cosine) for rusket embeddings — `HybridEmbeddingIndex` returns L2-normalised vectors automatically.

!!! tip "pgvector index types"
    | Index | Speed | Recall | Build time | Best for |
    |---|---|---|---|---|
    | **IVFFlat** | Fast | ~95% | Minutes | < 1M vectors |
    | **HNSW** | Fastest | ~99% | Slower | Any scale, production |

    ```sql
    -- IVFFlat (faster to build, good for smaller datasets)
    CREATE INDEX ON item_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

    -- HNSW (better recall, recommended for production)
    CREATE INDEX ON item_embeddings USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200);
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
