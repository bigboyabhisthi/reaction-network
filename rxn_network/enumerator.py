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

        self._pd_dict, self.filtered_entries = filter_entries(
            entries, include_metastable, temp, include_polymorphs
        )
        self.elements = sorted({elem.value for entry in self.filtered_entries
                      for elem in entry.composition.elements})
        self.dim = len(self.elements)

        target_pd = None
        for chemsys in self._pd_dict:
            if self.target_elems.issubset(chemsys.split("-")):
                target_pd = self._pd_dict[chemsys]

        self.target_entry = min(list(filter(lambda e:
                                           e.composition.reduced_composition ==
                                             self.target, target_pd.all_entries)),
                                key=lambda e: e.energy_per_atom)
        self.open_entries = [min(list(filter(lambda e:
                                           e.composition.reduced_composition ==
                                             comp, self.filtered_entries)),
                                 key=lambda e: e.energy_per_atom) for comp in
                             self.open_comps]

        if self.target_entry not in target_pd.stable_entries:
            self.target_entry = self.stabilize_entries(target_pd,
                                                       [self.target_entry])[0]
        target_idx = None
        for idx, e in enumerate(self.filtered_entries):
            if e.entry_id == self.target_entry.entry_id:
                target_idx = idx
                self.filtered_entries[target_idx] = self.target_entry
                break
        else:
            self.filtered_entries.append(self.target_entry)

    def enumerate(self, n=2):
        combos = list(filter(lambda combo: {e for c in combo for e in
                         c.composition.chemical_system.split("-")}.issuperset(self.target_elems),
                             generate_all_combos(self.filtered_entries, n)))
        products = list(filter(lambda c: self.target_entry in c, combos))

        all_rxn_combos = product(combos, products)

        edges = []
        cmaps = []
        current_idx = 0
        chemsys_arr = np.zeros((500, self.dim))

        open_entries = tuple(self.open_entries)

        for r, p in all_rxn_combos:
            r_open = [r] + [r + (o,) for o in open_entries]
            p_open = [p] + [p + (o,) for o in open_entries]

            chemsys = {elem.value for e in r+p+open_entries for elem in
                       e.composition.elements}
            chemsys_vec = np.array([1 if elem in chemsys else 0 for elem in
                                    self.elements])

            matches = np.argwhere(((chemsys_arr - chemsys_vec) >= 0).all(axis=1))
            if matches.size == 0:
                entries = list(filter(lambda e: chemsys.issuperset(
                    e.composition.chemical_system.split("-")),
                                      self.filtered_entries))
                cmap = ChempotMap(PhaseDiagram(entries), default_limit=-50)
                chemsys_arr[current_idx] = chemsys_vec
                current_idx = current_idx + 1
                cmaps.append(cmap)
            else:
                cmap = cmaps[matches[0][0]]

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
                    distances = [cmap.shortest_domain_distance(
                        combo[0].composition.reduced_formula,
                        combo[1].composition.reduced_formula)
                                 for combo in product(rxn._reactant_entries,
                                                      rxn._product_entries)]

                    elem_distances = [cmap.shortest_elemental_domain_distances(combo[0].composition.reduced_formula,
                                                                 combo[
                                                                     1].composition.reduced_formula).max()
                                 for combo in product(rxn._reactant_entries,
                                                      rxn._product_entries)]

                    max_distance = max(distances)
                    avg_distance = sum(distances) / len(distances)
                    max_elem_distance = max(elem_distances)
                else:
                    max_distance = 100
                    avg_distance = 100
                    max_elem_distance = 100

                cost = softplus([energy, max_distance], [1, 1], self.temp)

                edges.append([rxn, energy, max_distance, avg_distance,
                              max_elem_distance, cost])

        self.cmaps = cmaps
        cols = ["rxn", "energy", "max_distance", "avg_distance", "max_elem_distance","cost"]
        df = pandas.DataFrame(edges, columns=cols).drop_duplicates().sort_values("cost")
        return df


    @staticmethod
    def stabilize_entries(original_pd, entries_to_adjust, tol=1e-6):
        indices = [original_pd.all_entries.index(entry) for entry in entries_to_adjust]
        new_entries = []
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
            new_entries.append(new_entry)
        return new_entries

