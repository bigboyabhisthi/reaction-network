import os
from itertools import combinations, chain, groupby, compress
import queue
import json

import numpy as np

import graph_tool.all as gt

from pymatgen import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.reaction_calculator import ComputedReaction
from pymatgen.analysis.interface_reactions import InterfacialReactivity
from pymatgen.entries.computed_entries import ComputedStructureEntry

from typing import List

import numpy as np
from scipy.interpolate import interp1d
from monty.json import MontyDecoder, MontyEncoder, MSONable

from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from pymatgen.entries.entry_tools import EntrySet


__author__ = "Matthew McDermott"
__email__ = "mcdermott@lbl.gov"
__date__ = "October 23, 2020"


with open(os.path.join(os.path.dirname(__file__), "g_els.json")) as f:
    G_ELEMS = json.load(f)
with open(os.path.join(os.path.dirname(__file__), "nist_gas_gf.json")) as f:
    G_GASES = json.load(f)
with open(os.path.join(os.path.dirname(__file__), "compounds.json")) as f:
    G_COMPOUNDS = json.load(f)


class Node(MSONable):
    def __init__(self, entries):
        """
        Args:
            entries [ComputedEntry]: list of ComputedEntry-like objects
        """
        self._entries = set(entries)
        self.formulas = [e.composition.reduced_composition.alphabetical_formula.replace(
            " ","") for e in self._entries]
        self.temp = getattr(list(entries)[0], "temp", 0)
        self.chemsys = (
            "-".join(
                sorted(
                    {
                        str(el)
                        for entry in self._entries
                        for el in entry.composition.elements
                    }
                )
            )
            if entries
            else None
        )

    @property
    def entries(self):
        return self._entries

    def __repr__(self):
        formulas = self.formulas.copy()
        formulas.sort()
        return f"{','.join(formulas)}"

    def __eq__(self, other):
        equal = False
        if isinstance(other, self.__class__):
            if self.chemsys == other.chemsys:
                equal = self.entries == other.entries
        return equal

    def __hash__(self):
        return hash(frozenset(self._entries))


class Phases(Node):
    """
    """
    def __init__(self, entries, intermediates=None):
        self.intermediates = intermediates if intermediates else []
        super().__init__(entries)


class Interface(Node):
    """
    """
    def __init__(self, entries):
        self.n = len(entries)
        self.r1 = entries[0]

        if self.n == 1:
            self.r2 = entries[0]
        elif self.n == 2:
            self.r2 = entries[1]
        else:
            raise ValueError("Can't have an interface that is not 1 to 2 entries!")

        super().__init__(entries)

    def react(self, pd):
        ir = InterfacialReactivity(self.r1, self.r2, pd, norm=False,
                                   include_no_mixing_energy=False,
                                   pd_non_grand=None,
                                   use_hull_energy=False,
        )

        rxns = [
            {"fraction": round(ratio, 3),
             "rxn": rxn,
             "E_per_mol": round(rxn_energy, 1),
             "E_per_atom": round(reactivity, 3),
             } for _, ratio, reactivity, rxn, rxn_energy in ir.get_kinks()
        ]


    def __repr__(self):
        formulas = self.formulas.copy()
        formulas.sort()
        return f"{'|'.join(formulas)}"

class RxnPathway(MSONable):
    """
    Helper class for storing multiple ComputedReaction objects which form a single
    reaction pathway as identified via pathfinding methods. Includes cost of each
    reaction.
    """

    def __init__(self, rxns, costs):
        """
        Args:
            rxns ([ComputedReaction]): list of ComputedReaction objects in pymatgen
                which occur along path.
            costs ([float]): list of corresponding costs for each reaction.
        """
        self._rxns = list(rxns)
        self._costs = list(costs)

        self.total_cost = sum(self._costs)
        self._dg_per_atom = [
            rxn.calculated_reaction_energy
            / sum([rxn.get_el_amount(elem) for elem in rxn.elements])
            for rxn in self._rxns
        ]

    @property
    def rxns(self):
        return self._rxns

    @property
    def costs(self):
        return self._costs

    @property
    def dg_per_atom(self):
        return self._dg_per_atom

    def __repr__(self):
        path_info = ""
        for rxn, dg in zip(self._rxns, self._dg_per_atom):
            path_info += f"{rxn} (dG = {round(dg, 3)} eV/atom) \n"

        path_info += f"Total Cost: {round(self.total_cost,3)}"

        return path_info

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.as_dict() == other.as_dict()
        else:
            return False

    def __hash__(self):
        return hash(tuple(self._rxns))


class BalancedPathway(MSONable):
    """
    Helper class for combining multiple reactions which stoichiometrically balance to
    form a net reaction.
    """

    def __init__(self, rxn_dict, net_rxn, balance=True):
        """
        Args:
            rxn_dict (dict): dictionary of ComputedReaction objects (keys) and their
                associated costs (values).
            net_rxn (ComputedReaction): net reaction to use for stoichiometric
                constraints.
            balance (bool): whether to solve for multiplicities on initialization.
                You might want this to be False if you're balancing the pathways first
                and then initializing the object later, as is done in the pathfinding
                methods.
        """
        self.rxn_dict = rxn_dict
        self.all_rxns = list(self.rxn_dict.keys())
        self.net_rxn = net_rxn
        self.all_reactants = set()
        self.all_products = set()
        self.is_balanced = False
        self.multiplicities = None
        self.total_cost = None
        self.average_cost = None

        for rxn in self.rxn_dict.keys():
            self.all_reactants.update(rxn.reactants)
            self.all_products.update(rxn.products)

        self.all_comp = list(
            self.all_reactants | self.all_products | set(self.net_rxn.all_comp)
        )
        self.net_coeffs = self._get_net_coeffs(net_rxn, self.all_comp)
        self.comp_matrix = self._get_comp_matrix(self.all_comp, self.all_rxns)

        if balance:
            self.is_balanced, multiplicities = self._balance_rxns(
                self.comp_matrix, self.net_coeffs
            )
            self.set_multiplicities(multiplicities)

        if self.is_balanced:
            self.calculate_costs()

    def set_multiplicities(self, multiplicities):
        """
        Stores the provided multiplicities (e.g. if solved for outside of object
        initialization).

        Args:
            multiplicities ([float]): list of multiplicities in same order as list of
                all rxns (see self.all_rxns).
        """
        self.multiplicities = {
            rxn: multiplicity
            for (rxn, multiplicity) in zip(self.all_rxns, multiplicities)
        }

    def calculate_costs(self):
        """
        Calculates and sets total and average cost of all pathways using the reaction
        dict.
        """
        self.total_cost = sum(
            [mult * self.rxn_dict[rxn] for (rxn, mult) in self.multiplicities.items()]
        )
        self.average_cost = self.total_cost / len(self.rxn_dict)

    @staticmethod
    def _balance_rxns(comp_matrix, net_coeffs, tol=1e-6):
        """
        Internal method for balancing a set of reactions to achieve the same
        stoichiometry as a net reaction. Solves for multiplicities of reactions by
        using matrix psuedoinverse and checks to see if solution works.

        Args:
            comp_matrix (np.array): Matrix of stoichiometric coeffs for each reaction.
            net_coeffs (np.array): Vector of stoichiometric coeffs for net reaction.
            tol (float): Numerical tolerance for checking solution.

        Returns:

        """
        comp_pseudo_inverse = np.linalg.pinv(comp_matrix).T
        multiplicities = comp_pseudo_inverse @ net_coeffs

        is_balanced = False

        if (multiplicities < tol).any():
            is_balanced = False
        elif np.allclose(comp_matrix.T @ multiplicities, net_coeffs):
            is_balanced = True

        return is_balanced, multiplicities

    @staticmethod
    def _get_net_coeffs(net_rxn, all_comp):
        """
        Internal method for getting the net reaction coefficients vector.

        Args:
            net_rxn (ComputedReaction): net reaction object.
            all_comp ([Composition]): list of compositions in system of reactions.

        Returns:
            Numpy array which is a vector of the stoichiometric coeffs of net
            reaction and zeros for all intermediate phases.
        """
        return np.array(
            [
                net_rxn.get_coeff(comp) if comp in net_rxn.all_comp else 0
                for comp in all_comp
            ]
        )

    @staticmethod
    def _get_comp_matrix(all_comp, all_rxns):
        """
        Internal method for getting the composition matrix used in the balancing
        procedure.

        Args:
            all_comp ([Composition]): list of compositions in system of reactions.
            all_rxns ([ComputedReaction]): list of all reaction objects.

        Returns:
            Numpy array which is a matrix of the stoichiometric coeffs of each
            reaction in the system of reactions.
        """
        return np.array(
            [
                [
                    rxn.get_coeff(comp) if comp in rxn.all_comp else 0
                    for comp in all_comp
                ]
                for rxn in all_rxns
            ]
        )

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return set(self.all_rxns) == set(other.all_rxns)
        else:
            return False

    def __repr__(self):
        rxn_info = ""
        for rxn, cost in self.rxn_dict.items():
            dg_per_atom = rxn.calculated_reaction_energy / sum(
                [rxn.get_el_amount(elem) for elem in rxn.elements]
            )
            rxn_info += f"{rxn} (dG = {round(dg_per_atom,3)} eV/atom) \n"
        rxn_info += f"\nAverage Cost: {round(self.average_cost,3)} \n" \
                    f"Total Cost: {round(self.total_cost,3)}"

        return rxn_info

    def __hash__(self):
        return hash(frozenset(self.all_rxns))


class GibbsComputedStructureEntry(ComputedStructureEntry):
    """
    An extension to ComputedStructureEntry which includes the estimated Gibbs
    free energy of formation via a machine-learned model.
    """

    def __init__(
        self,
        structure: Structure,
        formation_enthalpy: float,
        temp: float = 300,
        gibbs_model: str = "SISSO",
        correction: float = 0.0,
        energy_adjustments: list = None,
        parameters: dict = None,
        data: dict = None,
        entry_id: object = None,
    ):
        """
        Args:
            structure (Structure): The pymatgen Structure object of an entry.
            formation_enthalpy (float): Formation enthalpy of the entry, calculated
                using phase diagram construction (eV)
            temp (float): Temperature in Kelvin. If temperature is not selected from
                one of [300, 400, 500, ... 2000 K], then free energies will
                be interpolated. Defaults to 300 K.
            gibbs_model (str): Model for Gibbs Free energy. Currently the default (and
                only supported) option is "SISSO", the descriptor created by Bartel et
                al. (2018).
            correction (float): A correction to be applied to the energy. Defaults to 0
            parameters (dict): An optional dict of parameters associated with
                the entry. Defaults to None.
            data (dict): An optional dict of any additional data associated
                with the entry. Defaults to None.
            entry_id: An optional id to uniquely identify the entry.
        """
        self.structure = structure
        self.formation_enthalpy = formation_enthalpy
        self.temp = temp
        self.interpolated = False

        if self.temp < 300 or self.temp > 2000:
            raise ValueError("Temperature must be selected from range: [300, 2000] K.")

        if self.temp % 100:
            self.interpolated = True

        if gibbs_model.lower() == "sisso":
            gibbs_energy = self.gf_sisso()
        else:
            raise ValueError(
                f"{gibbs_model} not a valid model. Please select from [" f"'SISSO']"
            )

        self.gibbs_model = gibbs_model

        super().__init__(
            structure,
            energy=gibbs_energy,
            correction=correction,
            energy_adjustments=energy_adjustments,
            parameters=parameters,
            data=data,
            entry_id=entry_id,
        )

    def gf_sisso(self) -> float:
        """
        Gibbs Free Energy of formation as calculated by SISSO descriptor from Bartel
        et al. (2018). Units: eV (not normalized)

        WARNING: This descriptor only applies to solids. The implementation here
        attempts to detect and use downloaded NIST-JANAF data for common gases (e.g.
        CO2) where possible.

        Reference: Bartel, C. J., Millican, S. L., Deml, A. M., Rumptz, J. R.,
        Tumas, W., Weimer, A. W., … Holder, A. M. (2018). Physical descriptor for
        the Gibbs energy of inorganic crystalline solids and
        temperature-dependent materials chemistry. Nature Communications, 9(1),
        4168. https://doi.org/10.1038/s41467-018-06682-4

        Returns:
            float: Gibbs free energy of formation (eV)
        """
        comp = self.structure.composition

        if comp.is_element:
            return self.formation_enthalpy

        exp_data = False
        if comp.reduced_formula in G_GASES.keys():
            exp_data = True
            data = G_GASES[comp.reduced_formula]
            factor = comp.get_reduced_formula_and_factor()[1]
        elif comp.reduced_formula in G_COMPOUNDS.keys():
            exp_data = True
            data = G_COMPOUNDS[comp.reduced_formula]
            factor = comp.get_reduced_formula_and_factor()[1]

        if exp_data:
            if self.interpolated:
                g_interp = interp1d([int(t) for t in data.keys()], list(data.values()))
                return g_interp(self.temp) * factor
            else:
                return data[str(self.temp)] * factor

        num_atoms = self.structure.num_sites
        vol_per_atom = self.structure.volume / num_atoms
        reduced_mass = self._reduced_mass()

        return (
            self.formation_enthalpy
            + num_atoms * self._g_delta_sisso(vol_per_atom, reduced_mass, self.temp)
            - self._sum_g_i()
        )

    def _sum_g_i(self) -> float:
        """
        Sum of the stoichiometrically weighted chemical potentials of the elements
        at specified temperature, as acquired from "g_els.json".

        Returns:
             float: sum of weighted chemical potentials [eV]
        """
        elems = self.structure.composition.get_el_amt_dict()

        if self.interpolated:
            sum_g_i = 0
            for elem, amt in elems.items():
                g_interp = interp1d(
                    [float(t) for t in G_ELEMS.keys()],
                    [g_dict[elem] for g_dict in G_ELEMS.values()],
                )
                sum_g_i += amt * g_interp(self.temp)
        else:
            sum_g_i = sum(
                [amt * G_ELEMS[str(self.temp)][elem] for elem, amt in elems.items()]
            )

        return sum_g_i

    def _reduced_mass(self) -> float:
        """
        Reduced mass as calculated via Eq. 6 in Bartel et al. (2018)

        Returns:
            float: reduced mass (amu)
        """
        reduced_comp = self.structure.composition.reduced_composition
        num_elems = len(reduced_comp.elements)
        elem_dict = reduced_comp.get_el_amt_dict()

        denominator = (num_elems - 1) * reduced_comp.num_atoms

        all_pairs = combinations(elem_dict.items(), 2)
        mass_sum = 0

        for pair in all_pairs:
            m_i = Composition(pair[0][0]).weight
            m_j = Composition(pair[1][0]).weight
            alpha_i = pair[0][1]
            alpha_j = pair[1][1]

            mass_sum += (alpha_i + alpha_j) * (m_i * m_j) / (m_i + m_j)

        reduced_mass = (1 / denominator) * mass_sum

        return reduced_mass

    @staticmethod
    def _g_delta_sisso(vol_per_atom, reduced_mass, temp) -> float:
        """
        G^delta as predicted by SISSO-learned descriptor from Eq. (4) in
        Bartel et al. (2018).

        Args:
            vol_per_atom (float): volume per atom [Å^3/atom]
            reduced_mass (float) - reduced mass as calculated with pair-wise sum formula
                [amu]
            temp (float) - Temperature [K]

        Returns:
            float: G^delta
        """

        return (
            (-2.48e-4 * np.log(vol_per_atom) - 8.94e-5 * reduced_mass / vol_per_atom)
            * temp
            + 0.181 * np.log(temp)
            - 0.882
        )

    @classmethod
    def from_pd(
        cls, pd, temp=300, gibbs_model="SISSO"
    ) -> List["GibbsComputedStructureEntry"]:
        """
        Constructor method for initializing a list of GibbsComputedStructureEntry
        objects from an existing T = 0 K phase diagram composed of
        ComputedStructureEntry objects, as acquired from a thermochemical database;
        e.g. The Materials Project.

        Args:
            pd (PhaseDiagram): T = 0 K phase diagram as created in pymatgen. Must
                contain ComputedStructureEntry objects.
            temp (int): Temperature [K] for estimating Gibbs free energy of formation.
            gibbs_model (str): Gibbs model to use; currently the only option is "SISSO".

        Returns:
            [GibbsComputedStructureEntry]: list of new entries which replace the orig.
                entries with inclusion of Gibbs free energy of formation at the
                specified temperature.
        """
        gibbs_entries = []
        for entry in pd.all_entries:
            if (
                entry in pd.el_refs.values()
                or not entry.structure.composition.is_element
            ):
                gibbs_entries.append(
                    cls(
                        entry.structure,
                        formation_enthalpy=pd.get_form_energy(entry),
                        temp=temp,
                        correction=0,
                        gibbs_model=gibbs_model,
                        data=entry.data,
                        entry_id=entry.entry_id,
                    )
                )
        return gibbs_entries

    @classmethod
    def from_entries(
        cls, entries, temp=300, gibbs_model="SISSO"
    ) -> List["GibbsComputedStructureEntry"]:
        """
        Constructor method for initializing GibbsComputedStructureEntry objects from
        T = 0 K ComputedStructureEntry objects, as acquired from a thermochemical
        database e.g. The Materials Project.

        Args:
            entries ([ComputedStructureEntry]): List of ComputedStructureEntry objects,
                as downloaded from The Materials Project API.
            temp (int): Temperature [K] for estimating Gibbs free energy of formation.
            gibbs_model (str): Gibbs model to use; currently the only option is "SISSO".

        Returns:
            [GibbsComputedStructureEntry]: list of new entries which replace the orig.
                entries with inclusion of Gibbs free energy of formation at the
                specified temperature.
        """
        pd_dict = expand_pd(entries)
        gibbs_entries = set()
        for entry in entries:
            for chemsys, phase_diag in pd_dict.items():
                if set(entry.composition.chemical_system.split("-")).issubset(
                    chemsys.split("-")
                ):
                    if (
                        entry in phase_diag.el_refs.values()
                        or not entry.structure.composition.is_element
                    ):
                        gibbs_entries.add(
                            cls(
                                entry.structure,
                                formation_enthalpy=phase_diag.get_form_energy(entry),
                                temp=temp,
                                correction=0,
                                gibbs_model=gibbs_model,
                                data=entry.data,
                                entry_id=entry.entry_id,
                            )
                        )
                    break

        return list(gibbs_entries)

    def as_dict(self) -> dict:
        """
        :return: MSONAble dict.
        """
        d = super().as_dict()
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        d["formation_enthalpy"] = self.formation_enthalpy
        d["temp"] = self.temp
        d["gibbs_model"] = self.gibbs_model
        d["interpolated"] = self.interpolated
        return d

    @classmethod
    def from_dict(cls, d) -> "GibbsComputedStructureEntry":
        """
        :param d: Dict representation.
        :return: GibbsComputedStructureEntry
        """
        dec = MontyDecoder()
        return cls(
            dec.process_decoded(d["structure"]),
            d["formation_enthalpy"],
            d["temp"],
            d["gibbs_model"],
            correction=d["correction"],
            energy_adjustments=[
                dec.process_decoded(e) for e in d.get("energy_adjustments", {})
            ],
            parameters={
                k: dec.process_decoded(v) for k, v in d.get("parameters", {}).items()
            },
            data={k: dec.process_decoded(v) for k, v in d.get("data", {}).items()},
            entry_id=d.get("entry_id", None),
        )

    def __repr__(self):
        output = [
            "GibbsComputedStructureEntry {} - {}".format(
                self.entry_id, self.composition.formula
            ),
            "Gibbs Free Energy (Formation) = {:.4f}".format(self.energy),
        ]
        return "\n".join(output)


def yens_ksp(
        g, num_k, precursors_v, target_v, edge_prop="bool", weight_prop="weight"
):
    """
    Yen's Algorithm for k-shortest paths. Inspired by igraph implementation by
    Antonin Lenfant. Ref: Jin Y. Yen, "Finding the K Shortest Loopless Paths
    in a Network", Management Science, Vol. 17, No. 11, Theory Series (Jul.,
    1971), pp. 712-716.

    Args:
        g (gt.Graph): the graph-tool graph object.
        num_k (int): number of k shortest paths that should be found.
        precursors_v (gt.Vertex): graph-tool vertex object containing precursors.
        target_v (gt.Vertex): graph-tool vertex object containing target.
        edge_prop (str): name of edge property map which allows for filtering edges.
            Defaults to the word "bool".
        weight_prop (str): name of edge property map that stores edge weights/costs.
            Defaults to the word "weight".

    Returns:
        List of lists of graph vertices corresponding to each shortest path
            (sorted in increasing order by cost).
    """

    def path_cost(vertices):
        """Calculates path cost given a list of vertices."""
        cost = 0
        for j in range(len(vertices) - 1):
            cost += g.ep[weight_prop][g.edge(vertices[j], vertices[j + 1])]
        return cost

    path = gt.shortest_path(g, precursors_v, target_v, weights=g.ep[weight_prop])[0]

    if not path:
        return []
    a = [path]
    a_costs = [path_cost(path)]

    b = queue.PriorityQueue()  # automatically sorts by path cost (priority)

    for k in range(1, num_k):
        try:
            prev_path = a[k - 1]
        except IndexError:
            print(f"Identified only k={k} paths before exiting. \n")
            break

        for i in range(len(prev_path) - 1):
            spur_v = prev_path[i]
            root_path = prev_path[:i]

            filtered_edges = []

            for path in a:
                if len(path) - 1 > i and root_path == path[:i]:
                    e = g.edge(path[i], path[i + 1])
                    if not e:
                        continue
                    g.ep[edge_prop][e] = False
                    filtered_edges.append(e)

            gv = gt.GraphView(g, efilt=g.ep[edge_prop])
            spur_path = gt.shortest_path(
                gv, spur_v, target_v, weights=g.ep[weight_prop]
            )[0]

            for e in filtered_edges:
                g.ep[edge_prop][e] = True

            if spur_path:
                total_path = root_path + spur_path
                total_path_cost = path_cost(total_path)
                b.put((total_path_cost, total_path))

        while True:
            try:
                cost_, path_ = b.get(block=False)
            except queue.Empty:
                break
            if path_ not in a:
                a.append(path_)
                a_costs.append(cost_)
                break

    return a


def expand_pd(entries):
    """
    Helper method for expanding a single PhaseDiagram into a set of smaller phase
    diagrams, indexed by chemical subsystem. This is an absolutely necessary
    approach when considering chemical systems which contain > ~10 elements,
    due to limitations of the ConvexHull algorithm.

    Args:
        entries ([ComputedEntry]): list of ComputedEntry-like objects for building
            phase diagram.

    Returns:
        Dictionary of PhaseDiagram objects indexed by chemical subsystem string;
        e.g. {"Li-Mn-O": <PhaseDiagram object>, "C-Y": <PhaseDiagram object>, ...}
    """

    pd_dict = dict()

    for e in sorted(entries, key=lambda x: len(x.composition.elements), reverse=True):
        for chemsys in pd_dict.keys():
            if set(e.composition.chemical_system.split("-")).issubset(
                chemsys.split("-")
            ):
                break
        else:
            pd_dict[e.composition.chemical_system] = PhaseDiagram(
                list(
                    filter(
                        lambda x: set(x.composition.elements).issubset(
                            e.composition.elements
                        ),
                        entries,
                    )
                )
            )

    return pd_dict


def filter_entries(all_entries, e_above_hull, include_polymorphs=False):
    """
    Helper method for filtering entries by specified energy above hull

    Args:
        all_entries ([ComputedEntry]): List of ComputedEntry-like objects to be
            filtered
        e_above_hull (float): Thermodynamic stability threshold (energy above hull)
            [eV/atom]
        include_polymorphs (bool): whether to include higher energy polymorphs of
            existing structures

    Returns:
        [ComputedEntry]: list of all entries with energies above hull equal to or
            less than the specified e_above_hull.
    """
    pd_dict = expand_pd(all_entries)
    energies_above_hull = dict()

    for entry in all_entries:
        for chemsys, phase_diag in pd_dict.items():
            if set(entry.composition.chemical_system.split("-")).issubset(
                    chemsys.split("-")
            ):
                energies_above_hull[entry] = phase_diag.get_e_above_hull(entry)
                break

    if e_above_hull == 0:
        filtered_entries = [e[0] for e in energies_above_hull.items() if e[1] == 0]
    else:
        filtered_entries = [
            e[0] for e in energies_above_hull.items() if e[1] <= e_above_hull
        ]

        if not include_polymorphs:
            filtered_entries_no_polymorphs = []
            all_comp = {
                entry.composition.reduced_composition for entry in filtered_entries
            }
            for comp in all_comp:
                polymorphs = [
                    entry
                    for entry in filtered_entries
                    if entry.composition.reduced_composition == comp
                ]
                min_entry = min(polymorphs, key=lambda x: x.energy_per_atom)
                filtered_entries_no_polymorphs.append(min_entry)

            filtered_entries = filtered_entries_no_polymorphs

    return pd_dict, EntrySet(filtered_entries)


def get_alphabetical_reduced_formula(entry):
    return entry.composition.reduced_composition.alphabetical_formula.replace(" ", "")


def generate_all_combos(entries, max_num_combos):
    """
    Helper static method for generating combination sets ranging from singular
    length to maximum length specified by max_num_combos.

    Args:
        entries (list/set): list/set of all entry objects to combine
        max_num_combos (int): upper limit for size of combinations of entries

    Returns:
        list: all combination sets
    """
    return chain.from_iterable(
        [
            combinations(entries, num_combos)
            for num_combos in range(1, max_num_combos + 1)
        ]
    )
