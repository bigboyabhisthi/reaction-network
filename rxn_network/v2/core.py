import logging
from itertools import combinations, chain, groupby, compress
from tqdm import tqdm

import numpy as np
from numba import njit, prange
from numba.typed import List

import graph_tool.all as gt
import queue

from pymatgen.entries.computed_entries import ComputedEntry, GibbsComputedStructureEntry
from pymatgen.analysis.reaction_calculator import ComputedReaction, ReactionError
from pymatgen.entries.entry_tools import EntrySet

from rxn_network.v2.helpers import *


__author__ = "Matthew McDermott"
__copyright__ = "Copyright 2020, Matthew McDermott"
__version__ = "0.0"
__email__ = "mcdermott@lbl.gov"
__date__ = "October 20, 2020"


DEFAULT_TEMPS = list(np.arange(300, 1100, 100))


class InterfaceReactionNetwork:
    def __init__(self, entries, temps=None, include_metastable=False):
        self.logger = logging.getLogger("InterfaceReactionNetwork")
        self.logger.setLevel("INFO")

        self.original_entries = EntrySet(entries)
        self.include_metastable = include_metastable

        if temps is None:
            temps = DEFAULT_TEMPS

        self.elements = {
            elem for entry in self.original_entries for elem in
            entry.composition.elements
        }

        low_temp_entries = GibbsComputedStructureEntry.from_entries(
            self.original_entries, temps[0])
        _, filtered_entries = filter_entries(low_temp_entries, include_metastable)

        entry_dict = dict()
        entry_dict[temps[0]] = EntrySet(filtered_entries)

        for temp in temps[1:]:
            gibbs_entries = GibbsComputedStructureEntry.from_entries(
                filtered_entries, temp)
            entry_dict[temp] = EntrySet(gibbs_entries)


    def generate_rxn_network(
            self,
            precursors=None,
            cost_function="softplus",
    ):

        precursors_entries = Phases(precursors, "s")

        g = gt.Graph()

        g.vp["entries"] = g.new_vertex_property("object")
        g.vp["type"] = g.new_vertex_property(
            "int"
        )  # 0: precursors, 1: reactants, 2: products, 3: target
        g.vp["bool"] = g.new_vertex_property("bool")
        g.vp["path"] = g.new_vertex_property("bool")
        g.vp["chemsys"] = g.new_vertex_property("string")

        g.ep["weight"] = g.new_edge_property("double")
        g.ep["rxn"] = g.new_edge_property("object")
        g.ep["bool"] = g.new_edge_property("bool")
        g.ep["path"] = g.new_edge_property("bool")

        precursors_v = g.add_vertex()
        self._update_vertex_properties(
            g,
            precursors_v,
            {
                "entries": precursors_entries,
                "type": 0,
                "bool": True,
                "path": True,
                "chemsys": precursors_entries.chemsys,
            },
        )

    @staticmethod
    def _update_vertex_properties(g, v, prop_dict):
        """
        Helper method for updating several vertex properties at once in a graph-tool
        graph.

        Args:
            g (gt.Graph): a graph-tool Graph object.
            v (gt.Vertex or int): a graph-tool Vertex object (or its index) for a vertex
                in the provided graph.
            prop_dict (dict): a dictionary of the form {"prop": val}, where prop is the
                name of a VertexPropertyMap of the graph and val is the new updated
                value for that vertex's property.

        Returns:
            None
        """
        for prop, val in prop_dict.items():
            g.vp[prop][v] = val
        return None




