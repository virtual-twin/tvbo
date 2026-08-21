#!/usr/bin/env python3
"""Write a Turtle graph so that regenerating it from an unchanged schema gives the same bytes.

rdflib names blank nodes randomly and emits a subject's objects in store order, so two
`gen-owl` / `gen-shacl` runs over one schema differ by thousands of lines. A generated
artefact that cannot be diffed cannot be reviewed, and CI cannot gate on
"regenerated == committed". Relabelling every blank node by a hash of its own
neighbourhood — refined until the labels stop splitting — settles both the names and the
order, because the serialiser sorts on the labels it is given.

Nodes that stay indistinguishable after refinement are automorphic: exchanging them maps
the graph to itself, so which of them takes which label does not reach the output.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys
from collections import defaultdict

from rdflib import BNode, Graph
from rdflib.namespace import RDF

MAX_ROUNDS = 32


def _digest(parts: list[str]) -> str:
    return hashlib.sha256("\x00".join(sorted(parts)).encode()).hexdigest()


def canonical_bnode_labels(graph: Graph) -> dict[BNode, BNode]:
    """A content-derived name for every blank node in *graph*.

    Refines a Weisfeiler-Lehman colouring: each round re-hashes a node from its own previous
    label and its incident edges, so a node one hop further from a named neighbour is
    distinguished one round later. Carrying the node's own label forward is what makes each
    round a *refinement* of the last, so an unchanged count of distinct labels is a fixpoint
    and not an oscillation — without it a round could re-merge nodes an earlier one had
    separated, and the loop would stop on a coarser colouring than it had already reached.
    """
    labels = {n: "" for n in set(graph.subjects()) | set(graph.objects()) if isinstance(n, BNode)}
    distinct = 0
    for _ in range(MAX_ROUNDS):
        nxt = {}
        for node in labels:
            parts = [
                f">{p}\x01{labels[o] if isinstance(o, BNode) else o.n3()}"
                for p, o in graph.predicate_objects(node)
            ]
            parts += [
                f"<{p}\x01{labels[s] if isinstance(s, BNode) else s.n3()}"
                for s, p in graph.subject_predicates(node)
            ]
            nxt[node] = _digest(parts + [f"={labels[node]}"])
        settled = len(set(nxt.values()))
        labels = nxt
        if settled == distinct:
            break
        distinct = settled

    groups: dict[str, list[BNode]] = defaultdict(list)
    for node, digest in labels.items():
        groups[digest].append(node)
    mapping: dict[BNode, BNode] = {}
    for digest, nodes in groups.items():
        for index, node in enumerate(sorted(nodes, key=str)):
            suffix = "" if len(nodes) == 1 else f"a{index}"
            mapping[node] = BNode(f"n{digest[:24]}{suffix}")
    return mapping


def canonicalize(graph: Graph) -> Graph:
    """*graph* with every blank node renamed to its canonical label."""
    mapping = canonical_bnode_labels(graph)
    out = Graph()
    for prefix, namespace in graph.namespaces():
        out.bind(prefix, namespace)
    for s, p, o in graph:
        out.add((mapping.get(s, s), p, mapping.get(o, o)))
    return out


def write_canonical(graph: Graph, destination) -> None:
    """Serialise *graph* to *destination* as Turtle, reproducibly."""
    canonicalize(graph).serialize(destination=str(destination), format="turtle")


def sort_rdf_list(graph: Graph, predicate) -> int:
    """Order the members of every ``rdf:List`` reached by *predicate*, in place.

    LinkML builds set-valued list objects (`sh:ignoredProperties`) from a Python set, so
    their member order is a fresh accident on every run even though the list denotes a
    set. Returns how many lists were reordered.
    """
    reordered = 0
    for head in set(graph.objects(None, predicate)):
        cells, cell = [], head
        while cell != RDF.nil:
            cells.append(cell)
            cell = next(graph.objects(cell, RDF.rest), RDF.nil)
        if not cells:
            continue
        members = sorted((next(graph.objects(c, RDF.first)) for c in cells), key=lambda t: t.n3())
        for cell, member in zip(cells, members):
            graph.set((cell, RDF.first, member))
        reordered += 1
    return reordered


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=pathlib.Path, help="Turtle file to rewrite in place")
    ap.add_argument(
        "--sort-list",
        action="append",
        default=[],
        metavar="CURIE",
        help="Order the rdf:List reached by this predicate (repeatable)",
    )
    args = ap.parse_args()

    graph = Graph()
    graph.parse(args.path, format="turtle")
    sorted_lists = sum(
        sort_rdf_list(graph, graph.namespace_manager.expand_curie(curie))
        for curie in args.sort_list
    )
    write_canonical(graph, args.path)
    print(f"✓ Canonical Turtle written to {args.path} ({sorted_lists} lists ordered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
