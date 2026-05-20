"""kg_state_encoder.py — KG-enriched state representation for Deep RL News Recommendation.

Augments the existing 3,840-dim sentence-embedding state with 3 KG-derived
features per candidate article, producing a (3,840 + 3)-dim enriched state
that gives the DQN agent structural, graph-based signals alongside semantic ones.

KG Features (per candidate article × user history):
    1. entity_overlap   — fraction of candidate's entities that appeared in
                          the user's click history (0–1). Measures topical
                          alignment between candidate and user interest profile.

    2. entity_centrality — mean PageRank score of the candidate's entities in
                           the global entity co-occurrence graph (0–1 normalised).
                           High score → article covers "hot" entities dominating
                           the current news cycle.

    3. entity_novelty    — 1 - entity_overlap. Encourages the agent to also
                           recommend articles on entities the user hasn't seen yet.
                           Complements the ICM intrinsic reward in DQN+ICM.

Usage — drop-in enrichment:
    from src.kg_state_encoder import KGStateEncoder

    encoder = KGStateEncoder(kg_dir="kg/")

    # During environment step:
    base_state = build_state_embedding(click_history, title_embeddings)  # (3840,)
    enriched   = encoder.enrich(base_state, candidate_id, click_history)  # (3843,)

The enriched state is a simple numpy concatenation — no neural architecture
changes required. Just update STATE_DIM from 3840 to 3843 in your DQN config.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("kg-state-encoder")

# New total state dimension after KG enrichment
KG_FEATURE_DIM = 3
BASE_STATE_DIM  = 3840
ENRICHED_DIM    = BASE_STATE_DIM + KG_FEATURE_DIM   # 3843


class KGStateEncoder:
    """
    Lightweight KG feature extractor for RL state enrichment.

    Loads pre-built artefacts from news_kg_builder.py:
        kg/article_entities.json  — news_id → [entity_name, ...]
        kg/entity_pagerank.json   — entity_name → pagerank_score

    All lookups are O(1) dict operations — negligible overhead per env step.
    """

    def __init__(
        self,
        kg_dir: str | Path = "kg",
        max_pr_score: float | None = None,
    ) -> None:
        kg_dir = Path(kg_dir)

        # ── Article → entity names ─────────────────────────────
        ae_path = kg_dir / "article_entities.json"
        if not ae_path.exists():
            log.warning(
                "article_entities.json not found at %s. "
                "Run src/news_kg_builder.py first. "
                "KG features will be zero-filled.", ae_path
            )
            self.article_entities: dict[str, list[str]] = {}
        else:
            with open(ae_path) as f:
                self.article_entities = json.load(f)
            log.info("Loaded entity map: %d articles", len(self.article_entities))

        # ── Entity PageRank scores ─────────────────────────────
        pr_path = kg_dir / "entity_pagerank.json"
        if not pr_path.exists():
            log.warning("entity_pagerank.json not found. Centrality features will be 0.")
            self.pagerank: dict[str, float] = {}
            self._max_pr = 1.0
        else:
            with open(pr_path) as f:
                self.pagerank = json.load(f)
            self._max_pr = max(self.pagerank.values()) if self.pagerank else 1.0
            if max_pr_score is not None:
                self._max_pr = max_pr_score
            log.info("Loaded PageRank: %d entities, max PR=%.6f", len(self.pagerank), self._max_pr)

    # ──────────────────────────────────────────────────────────
    #  Core feature computation
    # ──────────────────────────────────────────────────────────

    def _get_entities(self, news_id: str) -> set[str]:
        """Return the set of entity names for a given news_id."""
        return set(self.article_entities.get(news_id, []))

    def _history_entity_pool(self, click_history: list[str]) -> set[str]:
        """Union of all entities across all articles in click history."""
        pool: set[str] = set()
        for nid in click_history:
            pool |= self._get_entities(nid)
        return pool

    def kg_features(
        self,
        candidate_id: str,
        click_history: list[str],
    ) -> np.ndarray:
        """
        Compute the 3-dim KG feature vector for a candidate article
        given the user's click history.

        Parameters
        ----------
        candidate_id  : news_id of the article being considered
        click_history : list of news_ids the user has already clicked

        Returns
        -------
        np.ndarray of shape (3,), dtype float32
            [entity_overlap, entity_centrality, entity_novelty]
        """
        cand_entities = self._get_entities(candidate_id)

        # ── Feature 1: Entity overlap ──────────────────────────
        if not cand_entities or not click_history:
            entity_overlap = 0.0
        else:
            history_pool   = self._history_entity_pool(click_history)
            shared         = cand_entities & history_pool
            entity_overlap = len(shared) / len(cand_entities)

        # ── Feature 2: Entity centrality (mean PageRank) ───────
        if not cand_entities or not self.pagerank:
            entity_centrality = 0.0
        else:
            scores = [self.pagerank.get(e, 0.0) for e in cand_entities]
            raw_centrality    = sum(scores) / len(scores)
            entity_centrality = min(raw_centrality / self._max_pr, 1.0)  # normalise 0–1

        # ── Feature 3: Entity novelty ──────────────────────────
        entity_novelty = 1.0 - entity_overlap

        return np.array(
            [entity_overlap, entity_centrality, entity_novelty],
            dtype=np.float32,
        )

    # ──────────────────────────────────────────────────────────
    #  State enrichment API
    # ──────────────────────────────────────────────────────────

    def enrich(
        self,
        base_state: np.ndarray,
        candidate_id: str,
        click_history: list[str],
    ) -> np.ndarray:
        """
        Concatenate KG features onto the existing embedding-based state.

        Parameters
        ----------
        base_state    : (3840,) float32 — original sentence-embedding state
        candidate_id  : news_id of candidate article
        click_history : user click history (list of news_ids)

        Returns
        -------
        np.ndarray of shape (3843,), dtype float32
        """
        kg_feat = self.kg_features(candidate_id, click_history)
        return np.concatenate([base_state, kg_feat]).astype(np.float32)

    def enrich_batch(
        self,
        base_states: np.ndarray,
        candidate_ids: list[str],
        click_histories: list[list[str]],
    ) -> np.ndarray:
        """
        Batch version of enrich() — vectorised over impression list.

        Parameters
        ----------
        base_states     : (N, 3840) array — one state per candidate
        candidate_ids   : list of N news_ids
        click_histories : list of N click history lists

        Returns
        -------
        np.ndarray of shape (N, 3843)
        """
        kg_feats = np.stack(
            [self.kg_features(cid, hist)
             for cid, hist in zip(candidate_ids, click_histories)],
            axis=0,
        )  # (N, 3)
        return np.concatenate([base_states, kg_feats], axis=1).astype(np.float32)

    # ──────────────────────────────────────────────────────────
    #  Diagnostics
    # ──────────────────────────────────────────────────────────

    def explain(
        self,
        candidate_id: str,
        click_history: list[str],
    ) -> dict:
        """
        Human-readable breakdown of the KG features for a given
        candidate + history pair. Useful for debugging and analysis.
        """
        cand_entities  = self._get_entities(candidate_id)
        history_pool   = self._history_entity_pool(click_history)
        shared         = cand_entities & history_pool
        feats          = self.kg_features(candidate_id, click_history)

        top_entities = sorted(
            cand_entities,
            key=lambda e: self.pagerank.get(e, 0.0),
            reverse=True,
        )[:5]

        return {
            "candidate_id":       candidate_id,
            "candidate_entities": sorted(cand_entities),
            "history_pool_size":  len(history_pool),
            "shared_entities":    sorted(shared),
            "top_central_entities": top_entities,
            "features": {
                "entity_overlap":     float(feats[0]),
                "entity_centrality":  float(feats[1]),
                "entity_novelty":     float(feats[2]),
            },
        }


# ──────────────────────────────────────────────────────────
#  Integration guide (printed when run directly)
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("""
KGStateEncoder — Integration Guide
====================================

1. Build the KG first (one-time):
       python src/news_kg_builder.py --news data/news.tsv

2. In your DQN / DQN+ICM training loop, replace:

       # BEFORE (original)
       state = build_state_embedding(click_history, title_embeddings)  # (3840,)

       # AFTER (KG-enriched)
       from src.kg_state_encoder import KGStateEncoder, ENRICHED_DIM
       encoder = KGStateEncoder(kg_dir="kg/")
       state = encoder.enrich(
           build_state_embedding(click_history, title_embeddings),
           candidate_id,
           click_history,
       )  # (3843,)

3. Update your DQN network input dimension:
       STATE_DIM = ENRICHED_DIM   # 3843 instead of 3840

4. Everything else (replay buffer, Tianshou trainer, reward) stays identical.

Feature interpretation:
    entity_overlap    [0–1]  High → user has seen this topic before (exploitation)
    entity_centrality [0–1]  High → article covers trending entities (relevance)
    entity_novelty    [0–1]  High → new topic for user (exploration)
""")
