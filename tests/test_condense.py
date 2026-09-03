"""``condense`` against an independent verifier, and against the fixed graphs.

Why this file exists
--------------------
The condensation had no Python original before 0.4.0: it lived in the
JavaScript package and in the R package, and the R version is a translation
of the JavaScript one. Two copies of one algorithm cannot disagree, so they
cannot check each other. ``prerelation.scan.condense`` is written from the
specification rather than translated, and it uses Kosaraju's two-pass search
where both ports use an iterative Tarjan.

That still leaves the new implementation checked only against the thing it
was built to arbitrate. So the classes, the quotient and the Hasse edges are
also recomputed here by a **third** route that is not a component algorithm
at all: the reachability closure. Two nodes are in one class exactly when
each reaches the other, and an edge of the quotient is redundant exactly
when some third class sits between its endpoints. The verifier is quadratic
and would be a poor implementation; that is not what it is for.

The verifier is deliberately in the repository rather than in a session
notebook. A check that exists only where it was invented is evidence, not a
check.
"""

import itertools
import random

import pytest

from prerelation.scan import condense, find_cycles, transitive_reduction

# The six graphs of the cross-language parity set, plus the two graphs the
# R package's own tests pin. Held here as literals so this file depends on
# nothing outside the repository.
GRAPHS = {
    "chain_isolate": (["a1", "a2", "a3", "a4"],
                      [("a1", "a2"), ("a1", "a3"), ("a2", "a3")]),
    "two_cycle": (["a", "b", "c"],
                  [("a", "b"), ("b", "a"), ("b", "c")]),
    "shared_node": (["a1", "a2", "a3", "a4", "a5"],
                    [("a1", "a2"), ("a2", "a1"), ("a2", "a3"),
                     ("a3", "a4"), ("a4", "a2"), ("a4", "a5")]),
    "no_edges": (["p", "q", "r"], []),
    "diamond_redundant": (["A", "B", "C", "D"],
                          [("A", "B"), ("B", "D"), ("A", "C"),
                           ("C", "D"), ("A", "D")]),
    "two_cycles_cross": (["a", "b", "c", "d"],
                         [("a", "b"), ("b", "a"), ("c", "d"), ("d", "c"),
                          ("b", "c"), ("a", "d")]),
}


def reachability(nodes, edges):
    """``reach[(u, v)]`` iff v is reachable from u by a path of length >= 1.

    Warshall's closure over the whole node set: no depth-first search, no
    component stack, no low-link.
    """
    reach = {(u, v): False for u in nodes for v in nodes}
    for u, v in edges:
        reach[(u, v)] = True
    for k in nodes:
        for i in nodes:
            if not reach[(i, k)]:
                continue
            for j in nodes:
                if reach[(k, j)]:
                    reach[(i, j)] = True
    return reach


def verify(nodes, edges):
    """Classes, quotient and Hasse edges from the closure alone."""
    reach = reachability(nodes, edges)
    position = {u: i for i, u in enumerate(nodes)}

    def same(u, v):
        return u == v or (reach[(u, v)] and reach[(v, u)])

    classes = []
    placed = set()
    for u in nodes:
        if u in placed:
            continue
        cls = [v for v in nodes if same(u, v)]
        placed.update(cls)
        classes.append(cls)
    classes.sort(key=lambda c: position[c[0]])
    class_of = {u: ci for ci, c in enumerate(classes) for u in c}

    quotient = []
    for u, v in edges:
        cu, cv = class_of[u], class_of[v]
        if cu != cv and (cu, cv) not in quotient:
            quotient.append((cu, cv))

    # An edge of the quotient is redundant exactly when some other class
    # lies strictly between its endpoints. Reachability on the quotient is
    # inherited from reachability on the nodes.
    def q_reach(a, b):
        return any(reach[(u, v)] for u in classes[a] for v in classes[b])

    hasse = [
        (a, b) for a, b in quotient
        if not any(c not in (a, b) and q_reach(a, c) and q_reach(c, b)
                   for c in range(len(classes)))
    ]
    return classes, class_of, quotient, hasse


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_condense_agrees_with_the_independent_verifier(name):
    nodes, edges = GRAPHS[name]
    got = condense(nodes, edges)
    classes, class_of, quotient, hasse = verify(nodes, edges)
    assert got.classes == classes
    assert got.class_of == class_of
    assert sorted(got.quotient_edges) == sorted(quotient)
    assert sorted(got.hasse_edges) == sorted(hasse)


def random_graph(rng, n_nodes, p_edge):
    nodes = [f"n{i}" for i in range(n_nodes)]
    edges = [(u, v) for u in nodes for v in nodes
             if u != v and rng.random() < p_edge]
    return nodes, edges


@pytest.mark.parametrize("seed", range(40))
def test_condense_agrees_with_the_verifier_on_random_graphs(seed):
    """The fixed six are the graphs the manuscript leans on; these are not.

    A check that only ever sees the fixtures it was written against tests
    the fixtures. Densities are chosen to straddle the interesting regime:
    sparse graphs are mostly singletons, dense ones collapse to one class.
    """
    rng = random.Random(seed)
    nodes, edges = random_graph(rng, rng.randint(2, 9), rng.choice([0.1, 0.25, 0.5]))
    got = condense(nodes, edges)
    classes, class_of, quotient, hasse = verify(nodes, edges)
    assert got.classes == classes
    assert got.class_of == class_of
    assert sorted(got.quotient_edges) == sorted(quotient)
    assert sorted(got.hasse_edges) == sorted(hasse)


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_classes_partition_the_node_set(name):
    nodes, edges = GRAPHS[name]
    got = condense(nodes, edges)
    flat = [u for c in got.classes for u in c]
    assert sorted(flat) == sorted(nodes)
    assert len(flat) == len(set(flat))
    assert set(got.class_of) == set(nodes)


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_the_quotient_is_acyclic_and_the_reduction_therefore_exists(name):
    """This is the property that makes condensation worth having."""
    nodes, edges = GRAPHS[name]
    got = condense(nodes, edges)
    ids = list(range(len(got.classes)))
    assert find_cycles(ids, got.quotient_edges) == []
    assert got.hasse_edges == transitive_reduction(ids, got.quotient_edges)


@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_labelling_is_a_function_of_the_node_order_alone(name):
    """Shuffling the edge list must not move a class label."""
    nodes, edges = GRAPHS[name]
    base = condense(nodes, edges)
    rng = random.Random(7)
    for _ in range(5):
        shuffled = list(edges)
        rng.shuffle(shuffled)
        other = condense(nodes, shuffled)
        assert other.classes == base.classes
        assert other.class_of == base.class_of
        assert sorted(other.quotient_edges) == sorted(base.quotient_edges)
        assert sorted(other.hasse_edges) == sorted(base.hasse_edges)


def test_on_an_acyclic_graph_condense_reproduces_the_transitive_reduction():
    """The two functions must not disagree where both are defined."""
    for name in ("chain_isolate", "diamond_redundant", "no_edges"):
        nodes, edges = GRAPHS[name]
        got = condense(nodes, edges)
        assert all(len(c) == 1 for c in got.classes)
        named = [(got.classes[a][0], got.classes[b][0]) for a, b in got.hasse_edges]
        assert sorted(named) == sorted(transitive_reduction(nodes, edges))


def test_r_package_recorded_assertions():
    """The five condensation facts pinned by the R package's own tests.

    R cannot be run in every environment this suite runs in, so the R
    expectations are carried here as literals rather than as a live
    comparison. They are the only recorded R-side condensation answers.
    """
    nodes = ["a", "b", "c", "d"]
    cond = condense(nodes, [("a", "b"), ("b", "a"), ("b", "c")])
    assert cond.classes[0] == ["a", "b"]
    assert cond.classes[1] == ["c"] and cond.classes[2] == ["d"]
    assert cond.quotient_edges == [(0, 1)]
    assert len(cond.hasse_edges) == 1
    cross = [("a", "b"), ("b", "a"), ("c", "d"), ("d", "c"), ("b", "c"), ("a", "d")]
    assert len(condense(nodes, cross).quotient_edges) == 1


def test_transitive_reduction_still_refuses_a_cyclic_graph():
    """0.4.0 adds a function; it does not soften an existing one."""
    with pytest.raises(ValueError, match="acyclic"):
        transitive_reduction(["a", "b", "c"], [("a", "b"), ("b", "a"), ("b", "c")])


def test_condense_rejects_an_edge_on_an_unknown_node():
    with pytest.raises(ValueError, match="not in the node set"):
        condense(["a", "b"], [("a", "z")])


def test_condense_rejects_repeated_nodes():
    with pytest.raises(ValueError, match="must not repeat"):
        condense(["a", "a"], [])


def test_a_complete_digraph_is_one_class():
    nodes = ["a", "b", "c", "d"]
    edges = [(u, v) for u, v in itertools.permutations(nodes, 2)]
    got = condense(nodes, edges)
    assert got.classes == [nodes]
    assert got.quotient_edges == []
    assert got.hasse_edges == []
