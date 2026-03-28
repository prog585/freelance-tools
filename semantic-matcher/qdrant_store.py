"""Qdrant operations for storing and querying embeddings."""

import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from embedder import VECTOR_SIZE

COLLECTION = "job_listings"
PROFILE_ID = "00000000-0000-0000-0000-000000000001"
HOST = "localhost"
PORT = 6333


def get_client() -> QdrantClient:
    return QdrantClient(host=HOST, port=PORT)


def ensure_collection(client: QdrantClient) -> None:
    """Create the collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION not in collections:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def upsert_profile(client: QdrantClient, vector: list[float], text: str) -> None:
    """Store the profile vector with a fixed ID."""
    ensure_collection(client)
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=PROFILE_ID,
                vector=vector,
                payload={"type": "profile", "text": text[:500]},
            )
        ],
    )


def add_job(
    client: QdrantClient,
    vector: list[float],
    title: str,
    description: str,
    url: str = "",
    budget: str = "",
    posted_date: str = "",
    skills: str = "",
) -> str:
    """Store a job vector. Returns the generated point ID."""
    ensure_collection(client)
    point_id = str(uuid.uuid4())
    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "type": "job",
                    "title": title,
                    "description": description[:500],
                    "url": url,
                    "budget": budget,
                    "posted_date": posted_date,
                    "skills": skills,
                },
            )
        ],
    )
    return point_id


def search_jobs(
    client: QdrantClient, profile_vector: list[float], top: int = 10
) -> list[dict]:
    """Find top-N jobs most similar to the profile vector."""
    results = client.query_points(
        collection_name=COLLECTION,
        query=profile_vector,
        query_filter=Filter(
            must=[FieldCondition(key="type", match=MatchValue(value="job"))]
        ),
        limit=top,
        with_payload=True,
    )
    return [
        {"score": hit.score, **hit.payload}
        for hit in results.points
    ]


def get_profile_vector(client: QdrantClient) -> Optional[list]:
    """Retrieve the stored profile vector."""
    results = client.retrieve(
        collection_name=COLLECTION,
        ids=[PROFILE_ID],
        with_vectors=True,
    )
    if results:
        return results[0].vector
    return None


def clear_jobs(client: QdrantClient) -> int:
    """Delete all job points (keep profile). Returns count deleted."""
    # Scroll all job points
    deleted = 0
    offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="type", match=MatchValue(value="job"))]
            ),
            limit=100,
            offset=offset,
            with_payload=False,
        )
        if not points:
            break
        ids = [p.id for p in points]
        client.delete(
            collection_name=COLLECTION,
            points_selector=ids,
        )
        deleted += len(ids)
        offset = next_offset
        if offset is None:
            break
    return deleted
