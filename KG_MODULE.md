## Knowledge Graph Module — Entity-Enriched State Representation

This module extends the base RL state with **3 graph-derived features** per candidate article, giving the DQN agent structural knowledge about entities alongside semantic embeddings.

### Why KG enrichment?

The original state is a 3,840-dim embedding concatenation — purely semantic, with no knowledge of *which real-world entities* a candidate article covers or how those entities relate to what the user has been reading. The KG layer adds that relational signal.

### Architecture

```
news.tsv (51K articles)
    │
    ▼  spaCy NER (en_core_web_sm)
Named Entities per article
(Person, Org, Location, Event, Topic)
    │
    ├──► RDFLib → OWL triples (news_kg_populated.ttl)
    │              Article –[mentionsEntity]→ Entity
    │              Article –[hasCategory]→ Topic
    │              Entity  –[coOccursWith]→ Entity
    │
    └──► NetworkX entity co-occurrence graph
              → PageRank centrality scores
              → article_entities.json (fast lookup)

During RL training (per env step):
    base_state  (3840-dim)  ← sentence embeddings of click history
         +
    kg_features  (3-dim)   ← KGStateEncoder.enrich(candidate, history)
         ║
         ▼
    enriched_state  (3843-dim)  → DQN Q-network input
```

### The 3 KG Features

| Feature | Range | Description |
|---------|-------|-------------|
| `entity_overlap` | 0–1 | Fraction of candidate's entities seen in user's click history. Measures topical alignment (exploitation signal). |
| `entity_centrality` | 0–1 | Mean PageRank of candidate's entities in the global co-occurrence graph. High = article covers trending entities. |
| `entity_novelty` | 0–1 | `1 - entity_overlap`. New entities for this user (exploration signal — complements DQN+ICM intrinsic reward). |

### Files

```
ontology/news_kg.ttl          OWL ontology — 8 classes, 10 object properties,
                               10 seed topic individuals (MIND categories)
src/news_kg_builder.py        Build KG from news.tsv → TTL + GraphML + JSON
src/kg_state_encoder.py       KGStateEncoder class for state enrichment
kg/                           Generated outputs (gitignored)
  ├── news_kg_populated.ttl
  ├── news_entity_graph.graphml
  ├── entity_pagerank.json
  └── article_entities.json
```

### Setup

```bash
pip install rdflib networkx spacy
python -m spacy download en_core_web_sm

# Build the KG (one-time, ~5 min on MINDsmall)
python src/news_kg_builder.py --news data/news.tsv

# Quick test on 1K articles
python src/news_kg_builder.py --news data/news.tsv --max-articles 1000
```

### Integration (2-line change)

```python
from src.kg_state_encoder import KGStateEncoder, ENRICHED_DIM

encoder = KGStateEncoder(kg_dir="kg/")

# Replace this in your env step:
state = encoder.enrich(
    build_state_embedding(click_history, title_embeddings),  # (3840,)
    candidate_id,
    click_history,
)  # → (3843,)

# Update DQN input dim:
STATE_DIM = ENRICHED_DIM  # 3843
```

### Ontology Design (OWL/Turtle)

The `ontology/news_kg.ttl` defines a formal semantic schema:

- **Classes**: `Article`, `Entity` (+ subclasses `Person`, `Organization`, `Location`, `Event`), `Topic`, `User`, `Interaction`
- **Object Properties**: `mentionsEntity`, `coOccursWith` (symmetric), `hasCategory`, `relatedTopic` (symmetric), `clickedBy`
- **Datatype Properties**: `pageRankScore`, `coOccurrenceCount`, `entityType`
- **Seed individuals**: 10 MIND category topics pre-wired with `relatedTopic` edges

This layer demonstrates semantic architecture design on top of a real recommendation system.
