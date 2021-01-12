import logging
import inspect
from itertools import combinations, chain, groupby, compress, product
from tqdm import tqdm
from pprint import pprint

import numpy as np
from numba import njit, prange
from functools import partial
import pandas

import graph_tool.all as gt
import queue
from time import time

from pymatgen.entries.entry_tools import EntrySet
from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram
from scipy.special import comb
from dask import delayed, compute, bag

from rxn_network.helpers import *
from rxn_network.reaction import *
from rxn_network.entries import *

from matterials.solids import ChempotMap


class RxnEnumerator:

    def __init__(self, entries, target, temp, open_comps=None, include_metastable=False,
                 include_polymorphs=False):
        self.entries = entries
        self.target = Composition(target).reduced_composition
        self.temp = temp
        self.open_comps = [Composition(c).reduced_composition for c in open_comps]
        self.include_metastable = include_metastable
        self.include_polymorphs = include_polymorphs
        self.target_elems = set(self.target.chemical_system.split("-"))

        self._pd_dict, self._filtered_entries = filter_entries(
            entries, include_metastable, temp, include_polymorphs
        )

        self.target_entry = min(list(filter(lambda e:
                                           e.composition.reduced_composition ==
                                             self.target, self._filtered_entries)),
                                key=lambda e: e.energy_per_atom)
        self.open_entries = [min(list(filter(lambda e:
                                           e.composition.reduced_composition ==
                                             comp, self._filtered_entries)),
                                key=lambda e: e.energy_per_atom) for comp in
                             self.open_comps]

        self.pd = PhaseDiagram(self._filtered_entries)
        if self.target_entry not in self.pd.stable_entries:
            self.stabilized_pd = self.stabilize_entries(self.pd, [self.target_entry])
            self.cmap = ChempotMap(self.stabilized_pd, default_limit=-20)
        else:
            self.cmap = ChempotMap(self.pd, default_limit=-20)

    def enumerate(self, n=2):
        combos = list(filter(lambda combo: {e for c in combo for e in
                         c.composition.chemical_system.split("-")}.issuperset(self.target_elems),
                             generate_all_combos(self._filtered_entries, n)))
        products = list(filter(lambda c: self.target_entry in c, combos))

        all_rxn_combos = product(combos, products)

        db = bag.from_sequence(all_rxn_combos, partition_size=10000)
        edges = db.map_partitions(self.calculate_rxns, open_entries=self.open_entries,
                                  cmap=self.cmap,
                                  temp=self.temp).compute()
        cols = ["rxn", "energy", "max_distance", "avg_distance", "cost"]

        df = pandas.DataFrame(edges, columns=cols).drop_duplicates().sort_values("cost")
        return df

    @staticmethod
    def calculate_rxns(combos, open_entries, cmap, temp):
        edges = []
        for r, p in combos:
            print(r)
            print("\n")
            print(p)
            print("\n")
            r_open = [r] + [r+(o,) for o in open_entries]
            p_open = [p] + [p+(o,) for o in open_entries]
            for r_o, p_o in product(r_open, p_open):
                if r_o == p_o:
                    continue
                rxn = ComputedReaction(r_o, p_o)

                if not rxn._balanced:
                    continue

                if rxn._lowest_num_errors != 0:
                    continue

                total_num_atoms = sum(
                    [rxn.get_el_amount(elem) for elem in rxn.elements])
                energy = rxn.calculated_reaction_energy / total_num_atoms

                if all(elem in cmap.pd.stable_entries for elem in rxn.all_entries):
                    distances = [cmap.shortest_domain_distance(combo[0], combo[1]) for
                                 combo in product(rxn._reactant_entries,
                                                  rxn._product_entries)]
                    max_distance = max(distances)
                    avg_distance = sum(distances) / len(distances)
                else:
                    max_distance = 100

                cost = softplus([energy, max_distance], [1, 1], temp)

                edges.append([rxn, energy, max_distance, avg_distance, cost])

        return edges

    @staticmethod
    def stabilize_entries(original_pd, entries_to_adjust, tol=1e-6):
        indices = [original_pd.all_entries.index(entry) for entry in entries_to_adjust]
        all_entries_new = original_pd.all_entries.copy()
        for idx, entry in zip(indices, entries_to_adjust):
            e_above_hull = original_pd.get_e_above_hull(entry)
            new_entry = ComputedStructureEntry(entry.structure,
                                               entry.uncorrected_energy -
                                               e_above_hull *
                                               entry.composition.num_atoms - tol,
                                               energy_adjustments=entry.energy_adjustments,
                                               parameters=entry.parameters,
                                               data=entry.data,
                                               entry_id=entry.entry_id)
            all_entries_new[idx] = new_entry
        return PhaseDiagram(all_entries_new)

