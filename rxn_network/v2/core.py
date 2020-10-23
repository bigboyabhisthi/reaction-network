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
__date__ = "October 23, 2020"


DEFAULT_TEMPS = list(np.arange(300, 1100, 100))


class InterfaceReactionNetwork:
    def __init__(self, entries, temps=None, include_metastable=False):
        self.logger = logging.getLogger("InterfaceReactionNetwork")
        self.logger.setLevel("INFO")

        self.original_entries = EntrySet(entries)
        self.include_metastable = include_metastable
        self.g = None

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

        self.entry_dict = entry_dict

        filtered_entries_str = ", ".join(
            [entry.composition.reduced_formula for entry in filtered_entries]
        )
        self.logger.info(
            f"Initializing network with {len(filtered_entries)} "
            f"entries: \n{filtered_entries_str}"
        )

    def generate_rxn_network(self, precursors=None):
        g = gt.Graph()

        g.vp["phases"] = g.new_vertex_property("object")
        g.vp["chemsys"] = g.new_vertex_property("string")
        g.vp["type"] = g.new_vertex_property("int")

        g.ep["weight"] = g.new_edge_property("double")
        g.ep["rxn"] = g.new_edge_property("object")
        g.ep["bool"] = g.new_edge_property("bool")
        g.ep["path"] = g.new_edge_property("bool")

        precursors = Phases(precursors)
        precursors_v = g.add_vertex()
        self._update_vertex_properties(
            g,
            precursors_v,
            {
                "phases": precursors,
                "type": 0,
                "bool": True,
                "path": True,
                "chemsys": precursors.chemsys,
            },
        )

        interfaces = generate_all_combos(precursors.entries)

        self.g = g

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




