import asyncio
from engine.heuristic_engine.similarity import compute_cosine_similarity
from engine.interceptor.embeddings import EmbeddingService

async def main():
    service = EmbeddingService()
    responses = [
        "The corpus does not contain the specific quantitative data needed to answer this query. After thorough retrieval, the available documents lack the exact figures and granular statistics requested. The corpus cannot provide this information.",
        "The corpus does not contain the specific quantitative data required to answer this query. After thorough retrieval, the available documents lack the exact figures and granular statistics requested. The corpus cannot supply this information.",
        "The corpus does not contain the specific quantitative data necessary to answer this query. After thorough retrieval, the available documents lack the exact figures and granular statistics requested. The corpus cannot provide these details."
    ]
    v1 = await asyncio.to_thread(service.embed, responses[0])
    v2 = await asyncio.to_thread(service.embed, responses[1])
    v3 = await asyncio.to_thread(service.embed, responses[2])
    print("sim(1, 2) =", compute_cosine_similarity(v2, v1))
    print("sim(2, 3) =", compute_cosine_similarity(v3, v2))
    print("sim(1, 3) =", compute_cosine_similarity(v3, v1))

asyncio.run(main())
