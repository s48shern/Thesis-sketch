import itertools
import math
from collections import defaultdict


###############################################################################
# DATA MODEL
###############################################################################

class Gadget:
    def __init__(self):
        self.vertices = set()
        self.special_edges = []   # (u,v,w)
        self.forced_edges = []    # (u,v,w)
        self.nonforced_edges = [] # (u,v,w)

    def add_special(self, u, v, w=0.5):
        self.vertices.update([u, v])
        self.special_edges.append((u, v, w))

    def add_forced(self, u, v, w=1):
        self.vertices.update([u, v])
        self.forced_edges.append((u, v, w))

    def add_nonforced(self, u, v, w=1):
        self.vertices.update([u, v])
        self.nonforced_edges.append((u, v, w))

    def all_edges(self):
        return (
            [("special",) + e for e in self.special_edges]
            + [("forced",) + e for e in self.forced_edges]
            + [("nonforced",) + e for e in self.nonforced_edges]
        )


###############################################################################
# EULERIAN CHECKS
###############################################################################




def spanning(vertices, used_edges):
    used_vertices = set()

    for (_, u, v, _, mult) in used_edges:
        if mult > 0:
            used_vertices.add(u)
            used_vertices.add(v)

    return used_vertices == set(vertices)


def eulerian(vertices, used_edges):
    deg = defaultdict(int)

    for (_, u, v, _, mult) in used_edges:
        deg[u] += mult
        deg[v] += mult

    for v in vertices:
        if deg[v] % 2 != 0:
            return False

    return connected_used_graph(vertices, used_edges)


###############################################################################
# ENUMERATE TOURS
###############################################################################

def enumerate_tours(gadget):
    """
    Enumerates all feasible spanning Eulerian tours
    respecting:
      - special/nonforced edges used at most once
      - forced edges used once or twice
    """

    all_edges = gadget.all_edges()

    usage_options = []

    for etype, u, v, w in all_edges:

        if etype == "forced":
            usage_options.append([1, 2])
        else:
            usage_options.append([0, 1])

    for mults in itertools.product(*usage_options):

        used = []

        for edge, mult in zip(all_edges, mults):
            etype, u, v, w = edge
            used.append((etype, u, v, w, mult))

        if not spanning(gadget.vertices, used):
            continue

        if not eulerian(gadget.vertices, used):
            continue

        yield used


###############################################################################
# OBJECTIVE FUNCTIONS
###############################################################################

def total_weight(tour):
    return sum(w * mult for (_, _, _, w, mult) in tour)


def special_usage_pattern(tour):
    """
    Returns:
      singles = number special edges used once
      doubles = number special edges used twice
    """

    singles = 0
    doubles = 0

    for etype, u, v, w, mult in tour:
        if etype != "special":
            continue

        if mult == 1:
            singles += 1
        elif mult == 2:
            doubles += 1

    return singles, doubles


def completeness(gadget):

    best_one_double = math.inf
    best_all_double = math.inf

    for T in enumerate_tours(gadget):

        singles, doubles = special_usage_pattern(T)
        cost = total_weight(T)

        # exactly one special edge doubled
        if doubles == 1 and singles == 0:
            best_one_double = min(best_one_double, cost)

        # all three doubled
        if doubles == 3:
            best_all_double = min(best_all_double, cost)

    return max(best_one_double, best_all_double)

###############################################################################
# CONNECTIVITY WITHOUT NETWORKX
###############################################################################

def adjacency_from_edges(vertices, used_edges):

    adj = {v: set() for v in vertices}

    for (_, u, v, _, mult) in used_edges:

        if mult > 0:
            adj[u].add(v)
            adj[v].add(u)

    return adj


def connected_used_graph(vertices, used_edges):

    adj = adjacency_from_edges(vertices, used_edges)

    nonisolated = [
        v for v in vertices
        if len(adj[v]) > 0
    ]

    if not nonisolated:
        return False

    start = nonisolated[0]

    stack = [start]
    seen = set()

    while stack:

        u = stack.pop()

        if u in seen:
            continue

        seen.add(u)

        for w in adj[u]:
            if w not in seen:
                stack.append(w)

    return all(v in seen for v in nonisolated)
def soundness(gadget):

    best = math.inf

    for T in enumerate_tours(gadget):

        singles, doubles = special_usage_pattern(T)

        correction = singles

        if doubles == 2:
            correction += 1

        val = total_weight(T) - correction

        best = min(best, val)

    return best


###############################################################################
# GRAPH GENERATION
###############################################################################

def generate_candidate_graphs():

    contacts = ["a", "b", "c"]
    aux = ["x"]

    base_vertices = ["s"] + contacts + aux

    possible_edges = []

    # all possible edges among contacts+aux
    core = contacts + aux

    for u, v in itertools.combinations(core, 2):
        possible_edges.append((u, v))

    # small weight set
    weights = [0.5, 1]
    # brute force subsets
    for forced_subset_bits in range(1 << len(possible_edges)):

        forced_edges = []
        remaining = []

        for i, e in enumerate(possible_edges):
            if (forced_subset_bits >> i) & 1:
                forced_edges.append(e)
            else:
                remaining.append(e)

        # choose some nonforced edges
        for k in range(len(remaining) + 1):

            for nonforced_subset in itertools.combinations(remaining, k):

                # assign weights
                forced_weight_choices = itertools.product(
                    weights, repeat=len(forced_edges)
                )

                nonforced_weight_choices = itertools.product(
                    weights, repeat=len(nonforced_subset)
                )

                for fw in forced_weight_choices:

                    for nw in nonforced_weight_choices:

                        G = Gadget()

                        # special edges
                        G.add_special("s", "a", 0.5)
                        G.add_special("s", "b", 0.5)
                        G.add_special("s", "c", 0.5)

                        for (u, v), w in zip(forced_edges, fw):
                            G.add_forced(u, v, w)

                        for (u, v), w in zip(nonforced_subset, nw):
                            G.add_nonforced(u, v, w)

                        core_vertices = contacts + aux

                        test_edges = []

                        for u, v, w in G.forced_edges:
                            test_edges.append(("x", u, v, w, 1))

                        for u, v, w in G.nonforced_edges:
                            test_edges.append(("x", u, v, w, 1))
                        if not connected_used_graph(core_vertices, test_edges):
                            continue

                        yield G


###############################################################################
# SEARCH
###############################################################################

def search(target=9):

    best = []

    count = 0

    for G in generate_candidate_graphs():

        count += 1

        try:
            c = completeness(G)
            s = soundness(G)

            if c == s and c <= target:

                print("\nFOUND CANDIDATE")
                print("====================")
                print("completeness =", c)
                print("soundness    =", s)

                print("\nFORCED")
                for e in G.forced_edges:
                    print(e)

                print("\nNONFORCED")
                for e in G.nonforced_edges:
                    print(e)

                best.append((c, s, G))

        except Exception:
            pass

    return best


###############################################################################
# RUN
###############################################################################

if __name__ == "__main__":

    results = search(target=9)

    print("\nTOTAL FOUND:", len(results))