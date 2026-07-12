"""Phase 7 — multi-path retrieval with reciprocal-rank fusion.

Up to four ranked lists are produced per query and fused:

  1. query text  -> bge            -> text index
  2. query text  -> CLIP text tower -> image index
  3. query image -> CLIP image tower -> image index   (similar-image match)
  4. query image -> VLM caption -> bge -> text index   (second chance via text)

Fused results are hydrated from the docstore into their raw payloads, plus one
same-page text neighbour per retrieved image for extra context.
"""
from __future__ import annotations

import json
from collections import defaultdict

import config
from src.models import embedders, vlm


class Retriever:
    def __init__(self):
        import chromadb

        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.text_col = client.get_collection(config.TEXT_COLLECTION)
        self.image_col = client.get_collection(config.IMAGE_COLLECTION)
        self.docstore = json.loads(config.DOCSTORE_PATH.read_text(encoding="utf-8"))

        # page number -> text doc_ids, for same-page neighbour hydration
        self._page_text: dict[int, list[str]] = defaultdict(list)
        for did, payload in self.docstore.items():
            if payload.get("type") == "text":
                self._page_text[payload["page"]].append(did)

        # the document's opening chunk (title / authors / abstract). Always added
        # as anchor context so meta questions ("what's the title/authors?") work —
        # those phrases don't semantically match the title text, so plain
        # retrieval misses them. docstore preserves chunk reading order.
        self._opening_id = next(
            (did for did, v in self.docstore.items() if v.get("type") == "text"), None
        )

    # -- one path: query a collection, return an ordered list of doc_ids --
    def _query(self, collection, embedding, n, sim_floor=None) -> list[str]:
        res = collection.query(query_embeddings=[embedding], n_results=n)
        ids = res["ids"][0]
        dists = res["distances"][0]
        out = []
        for did, dist in zip(ids, dists):
            similarity = 1.0 - dist          # chroma cosine distance -> similarity
            if sim_floor is not None and similarity < sim_floor:
                continue
            out.append(did)
        return out

    def _rrf(self, ranked_lists) -> list[tuple[str, float]]:
        scores: dict[str, float] = defaultdict(float)
        for lst in ranked_lists:
            for rank, did in enumerate(lst):
                scores[did] += 1.0 / (config.RRF_K + rank)
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    def retrieve(self, text_query: str = "", image_query=None, k: int | None = None):
        """Return a list of hydrated context dicts, best first."""
        k = k or config.FINAL_K
        ranked_lists: list[list[str]] = []

        if text_query:
            ranked_lists.append(
                self._query(self.text_col, embedders.embed_query(text_query),
                            config.TOP_N_BGE_TEXT)
            )
            ranked_lists.append(
                self._query(self.image_col, embedders.clip_embed_text(text_query),
                            config.TOP_N_CLIP_TEXT)
            )

        if image_query is not None:
            ranked_lists.append(
                self._query(self.image_col, embedders.clip_embed_image(image_query),
                            config.TOP_N_CLIP_IMAGE, sim_floor=config.CLIP_SIM_FLOOR)
            )
            caption = vlm.ask_about_image(
                image_query, "Describe this image briefly in one sentence.",
                max_new_tokens=config.CAPTION_MAX_NEW_TOKENS,
            )
            ranked_lists.append(
                self._query(self.text_col, embedders.embed_query(caption),
                            config.TOP_N_CAPTION)
            )

        fused = self._rrf(ranked_lists)
        top_ids = [did for did, _ in fused[:k]]

        # add one same-page text neighbour for each retrieved image
        extra: list[str] = []
        for did in top_ids:
            payload = self.docstore.get(did, {})
            if payload.get("type") == "image":
                for nid in self._page_text.get(payload["page"], []):
                    if nid not in top_ids and nid not in extra:
                        extra.append(nid)
                        break

        all_ids = top_ids + extra
        # always anchor with the document opening (title/abstract), low priority
        if self._opening_id and self._opening_id not in all_ids:
            all_ids.append(self._opening_id)

        return [
            {"doc_id": did, **self.docstore[did]}
            for did in all_ids
            if did in self.docstore
        ]
