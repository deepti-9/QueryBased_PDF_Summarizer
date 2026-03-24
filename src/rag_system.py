import time
from typing import List, Dict, Any, Optional


class RAGSystem:
    def __init__(self, vector_store, embedding_manager, llm):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager
        self.llm = llm

    # Correct distance → similarity conversion
    def _convert_distance_to_score(self, distance: float) -> float:
        return max(0, 1 - (distance / 2))  # for cosine distance [0,2]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.35,
        filter_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:

        query_emb = self.embedding_manager.generate_embeddings([query])[0]

        where_clause = {"type": filter_type.lower()} if filter_type else None

        results = self.vector_store.collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=top_k * 3,
            where=where_clause
        )

        retrieved_docs = []

        if results.get("documents") and results["documents"][0]:

            for i in range(len(results["ids"][0])):

                distance = results["distances"][0][i]
                score = self._convert_distance_to_score(distance)

                # ✅ REAL filtering (no bypass)
                if score >= score_threshold:
                    retrieved_docs.append({
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "score": round(score, 4)
                    })

        # ✅ Proper sorting
        retrieved_docs = sorted(
            retrieved_docs,
            key=lambda x: x["score"],
            reverse=True
        )

        return retrieved_docs[:top_k]

    def ask(self, query: str, top_k: int = 5, score_threshold: float = 0.35):

        start_time = time.time()

        docs = self.retrieve(query, top_k, score_threshold)
        fetch_time = time.time() - start_time

        if not docs:
            return {
                "answer": "Not found in document",
                "sources": [],
                "metrics": {}
            }

        #  Controlled context (avoid overload)
        context = "\n\n".join(
            f"[Source {i+1} | Page {d['metadata'].get('page','?')} | File: {d['metadata'].get('source_file','?')}]\n{d['content'][:600]}"
            for i, d in enumerate(docs)
        )

        # Strong prompt (LLM guidance)
        prompt = f"""
You are a precise document question-answering system.

Instructions:
- Answer ONLY using the provided context
- Do NOT use external knowledge
-Mention page numbers when referencing information
- If answer is not present, say: "Not found in document"
- If multiple points exist,  cite like (Page X)
- Be concise but complete
-- If answer not found, say: "Not found in document"

Context:
{context}

Question: {query}

Answer:
"""

        response = self.llm.invoke([prompt])
        answer = response.content.strip()

        if len(answer) < 10:
            answer = "Not found in document"

        avg_score = sum(d["score"] for d in docs) / len(docs)

        return {
            "answer": answer,
            "sources": [d["metadata"] for d in docs],
            "metrics": {
                "fetch_time": f"{fetch_time:.2f}s",
                "relevance_score": f"{avg_score * 100:.1f}%"
            }
        }