from rxn_network.network.entry import NetworkEntry, NetworkEntryType


def get_rxn_nodes_and_edges(rxns):
    nodes, edges = [], []

    for rxn in rxns:
        reactant_node = NetworkEntry(rxn.reactant_entries,
                                     NetworkEntryType.Reactants)
        product_node = NetworkEntry(rxn.product_entries, NetworkEntryType.Products)

        if reactant_node not in nodes:
            nodes.append(reactant_node)
            reactant_idx = len(nodes) - 1
        else:
            reactant_idx = nodes.index(reactant_node)

        if product_node not in nodes:
            nodes.append(product_node)
            product_idx = len(nodes) - 1
        else:
            product_idx = nodes.index(product_node)

        edges.append((reactant_idx, product_idx))

    return nodes, edges


def get_loopback_edges(g, nodes):
    edges = []
    for idx1, p in enumerate(nodes):
        if p.description != NetworkEntryType.Products:
            continue
        for idx2, r in enumerate(nodes):
            if r.description != NetworkEntryType.Reactants:
                continue
            if p.entries == r.entries:
                edges.append([g.vertex(idx1), g.vertex(idx2), 0.0, None])

    return edges