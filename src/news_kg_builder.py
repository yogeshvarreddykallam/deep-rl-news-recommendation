"""news_kg_builder.py — Build a News Entity Knowledge Graph from MINDsmall.

Reads the MIND news.tsv file, extracts named entities from article titles
using spaCy NER, builds RDF triples against the OWL ontology, and persists:

    kg/news_kg_populated.ttl    — full RDF graph (Turtle)
    kg/news_entity_graph.graphml — entity co-occurrence graph (NetworkX)
    kg/entity_pagerank.json      — pre-computed PageRank centrality scores
    kg/article_entities.json     — article_id → [entity names] mapping

These outputs are consumed by kg_state_encoder.py during RL training.

Usage:
    python src/news_kg_builder.py
    python src/news_kg_builder.py --news data/news.tsv --out-dir kg/
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd
import spacy
from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL, URIRef, XSD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("news-kg-builder")

# ─────────────────────────────────────────────────────────────
#  Namespace
# ─────────────────────────────────────────────────────────────

NKG = Namespace("http://yogeshvarreddykallam.github.io/news-kg#")

# spaCy NER label → OWL class mapping
LABEL_TO_CLASS = {
    "PERSON":  NKG.Person,
    "ORG":     NKG.Organization,
    "GPE":     NKG.Location,
    "LOC":     NKG.Location,
    "FAC":     NKG.Location,
    "EVENT":   NKG.Event,
    "NORP":    NKG.Organization,   # nationalities / political groups
    "WORK_OF_ART": NKG.Entity,
    "PRODUCT": NKG.Entity,
}

# MIND category label → seed topic URI
CATEGORY_TO_URI = {
    "news":          NKG.TopicNews,
    "sports":        NKG.TopicSports,
    "finance":       NKG.TopicFinance,
    "entertainment": NKG.TopicEntertainment,
    "health":        NKG.TopicHealth,
    "technology":    NKG.TopicTechnology,
    "travel":        NKG.TopicTravel,
    "lifestyle":     NKG.TopicLifestyle,
    "politics":      NKG.TopicPolitics,
    "weather":       NKG.TopicWeather,
}


# ─────────────────────────────────────────────────────────────
#  Step 1 — Load MIND news.tsv
# ─────────────────────────────────────────────────────────────

def load_news(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        tsv_path, sep="\t", header=None,
        names=["news_id", "category", "subcategory", "title",
               "abstract", "url", "title_entities", "abstract_entities"],
        usecols=["news_id", "category", "title", "abstract"],
    )
    df["title"]    = df["title"].fillna("")
    df["abstract"] = df["abstract"].fillna("")
    df["category"] = df["category"].str.lower().fillna("unknown")
    log.info("Loaded %d articles from %s", len(df), tsv_path)
    return df


# ─────────────────────────────────────────────────────────────
#  Step 2 — Extract entities with spaCy NER
# ─────────────────────────────────────────────────────────────

def extract_entities(
    df: pd.DataFrame,
    nlp,
    batch_size: int = 256,
) -> dict[str, list[tuple[str, str]]]:
    """
    Returns {news_id: [(entity_text, spacy_label), ...]}
    Combines title + abstract for richer coverage.
    """
    news_ids  = df["news_id"].tolist()
    texts     = (df["title"] + ". " + df["abstract"]).tolist()

    article_entities: dict[str, list[tuple[str, str]]] = {}

    for i, (doc, nid) in enumerate(
        zip(nlp.pipe(texts, batch_size=batch_size), news_ids)
    ):
        ents = [
            (ent.text.strip(), ent.label_)
            for ent in doc.ents
            if ent.label_ in LABEL_TO_CLASS and len(ent.text.strip()) > 1
        ]
        # Deduplicate within article
        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for text, label in ents:
            key = text.lower()
            if key not in seen:
                seen.add(key)
                unique.append((text, label))
        article_entities[nid] = unique

        if (i + 1) % 5000 == 0:
            log.info("  NER processed %d / %d articles", i + 1, len(news_ids))

    total_ents = sum(len(v) for v in article_entities.values())
    log.info("Entity extraction complete: %d total entity mentions across %d articles",
             total_ents, len(article_entities))
    return article_entities


# ─────────────────────────────────────────────────────────────
#  Step 3 — Build RDF graph
# ─────────────────────────────────────────────────────────────

def build_rdf_graph(
    rdf: Graph,
    df: pd.DataFrame,
    article_entities: dict[str, list[tuple[str, str]]],
) -> None:
    """Populate the RDF graph with Article and Entity individuals + triples."""

    entity_uris: dict[str, URIRef] = {}   # entity_text_lower → URI

    def get_entity_uri(text: str, label: str) -> URIRef:
        key = text.lower().replace(" ", "_").replace("/", "_")
        if key not in entity_uris:
            uri = NKG[f"entity_{key}"]
            entity_uris[key] = uri
            cls = LABEL_TO_CLASS.get(label, NKG.Entity)
            rdf.add((uri, RDF.type, cls))
            rdf.add((uri, NKG.entityName, Literal(text, datatype=XSD.string)))
            rdf.add((uri, NKG.entityType, Literal(label, datatype=XSD.string)))
        return entity_uris[key]

    for _, row in df.iterrows():
        nid = row["news_id"]
        article_uri = NKG[f"article_{nid}"]
        rdf.add((article_uri, RDF.type, NKG.Article))
        rdf.add((article_uri, NKG.newsId,  Literal(nid, datatype=XSD.string)))
        rdf.add((article_uri, NKG.title,   Literal(row["title"], datatype=XSD.string)))
        rdf.add((article_uri, NKG.categoryLabel, Literal(row["category"], datatype=XSD.string)))

        # hasCategory → seed topic
        cat_uri = CATEGORY_TO_URI.get(row["category"])
        if cat_uri:
            rdf.add((article_uri, NKG.hasCategory, cat_uri))

        # mentionsEntity for each extracted entity
        for text, label in article_entities.get(nid, []):
            ent_uri = get_entity_uri(text, label)
            rdf.add((article_uri, NKG.mentionsEntity, ent_uri))

    log.info("RDF graph: %d triples, %d unique entities",
             len(rdf), len(entity_uris))


# ─────────────────────────────────────────────────────────────
#  Step 4 — Build entity co-occurrence graph (NetworkX)
# ─────────────────────────────────────────────────────────────

def build_entity_graph(
    article_entities: dict[str, list[tuple[str, str]]],
) -> nx.Graph:
    """
    Undirected weighted graph where:
      Nodes = unique entity names (lowercased)
      Edges = co-occurrence in same article
      Weight = number of articles both entities appear in together
    """
    G = nx.Graph()
    co_count: dict[tuple[str, str], int] = defaultdict(int)

    for nid, ents in article_entities.items():
        entity_names = list({e[0].lower() for e in ents})  # unique per article
        for i in range(len(entity_names)):
            G.add_node(entity_names[i])
            for j in range(i + 1, len(entity_names)):
                key = (min(entity_names[i], entity_names[j]),
                       max(entity_names[i], entity_names[j]))
                co_count[key] += 1

    for (a, b), w in co_count.items():
        G.add_edge(a, b, weight=w)

    log.info("Entity co-occurrence graph: %d nodes, %d edges",
             G.number_of_nodes(), G.number_of_edges())
    return G


# ─────────────────────────────────────────────────────────────
#  Step 5 — Compute PageRank centrality
# ─────────────────────────────────────────────────────────────

def compute_pagerank(G: nx.Graph) -> dict[str, float]:
    """PageRank over the entity co-occurrence graph.
    High-centrality entities are 'hot topics' in the news corpus."""
    if G.number_of_nodes() == 0:
        return {}
    pr = nx.pagerank(G, weight="weight", alpha=0.85)
    log.info("PageRank computed. Top-5 entities: %s",
             sorted(pr, key=pr.get, reverse=True)[:5])
    return pr


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build News Entity Knowledge Graph.")
    parser.add_argument("--news",      type=Path, default=Path("data/news.tsv"))
    parser.add_argument("--ontology",  type=Path, default=Path("ontology/news_kg.ttl"))
    parser.add_argument("--out-dir",   type=Path, default=Path("kg"))
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--max-articles", type=int, default=None,
                        help="Limit articles for quick testing (default: all)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load news ──────────────────────────────────────────────
    if not args.news.exists():
        log.error("news.tsv not found at %s. Download MINDsmall from https://msnews.github.io", args.news)
        return
    df = load_news(args.news)
    if args.max_articles:
        df = df.head(args.max_articles)
        log.info("Limiting to %d articles for testing", args.max_articles)

    # ── Load OWL ontology as starting graph ────────────────────
    rdf = Graph()
    rdf.parse(str(args.ontology), format="turtle")
    log.info("Base ontology loaded: %d triples", len(rdf))

    # ── spaCy NER ──────────────────────────────────────────────
    try:
        nlp = spacy.load(args.spacy_model, disable=["parser", "lemmatizer"])
    except OSError:
        log.error("spaCy model '%s' not found. Run: python -m spacy download %s",
                  args.spacy_model, args.spacy_model)
        return

    # ── Extract entities ───────────────────────────────────────
    article_entities = extract_entities(df, nlp)

    # ── Populate RDF ───────────────────────────────────────────
    build_rdf_graph(rdf, df, article_entities)

    ttl_out = args.out_dir / "news_kg_populated.ttl"
    rdf.serialize(destination=str(ttl_out), format="turtle")
    log.info("Saved RDF graph → %s", ttl_out)

    # ── Build entity co-occurrence graph ───────────────────────
    G = build_entity_graph(article_entities)
    gml_out = args.out_dir / "news_entity_graph.graphml"
    nx.write_graphml(G, str(gml_out))
    log.info("Saved entity graph → %s", gml_out)

    # ── PageRank ───────────────────────────────────────────────
    pr = compute_pagerank(G)
    pr_out = args.out_dir / "entity_pagerank.json"
    with open(pr_out, "w") as f:
        json.dump(pr, f)
    log.info("Saved PageRank scores → %s", pr_out)

    # ── Article→entities mapping (plain JSON for fast lookup) ──
    ae_out = args.out_dir / "article_entities.json"
    with open(ae_out, "w") as f:
        # Store as {news_id: [entity_name, ...]} (names only, lowercased)
        json.dump(
            {nid: [e[0].lower() for e in ents]
             for nid, ents in article_entities.items()},
            f
        )
    log.info("Saved article→entities map → %s", ae_out)
    log.info("Knowledge Graph build complete.")


if __name__ == "__main__":
    main()
