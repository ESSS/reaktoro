# Reaktoro is a unified framework for modeling chemically reactive systems.
#
# Copyright (C) 2014-2018 Allan Leal
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library. If not, see <http://www.gnu.org/licenses/>.

from typing import List, Optional
import numpy as np
import pytest
from pytest import approx

import reaktoro as rkt


def test_euniquac_longrange_model_options():
    """
    Test if the option for long range model is working properly
    """

    euniquac_params = rkt.EUNIQUACParams()
    euniquac_params.setLongRangeModelType(rkt.LongRangeModelType.DH_Phreeqc)
    long_range_model_type = euniquac_params.longRangeModelType()
    assert long_range_model_type == rkt.LongRangeModelType.DH_Phreeqc


def test_euniquac_fallback_all_species_are_known_match_previous_results():
    """
    Test if the E-UNIQUAC with fallback modifications, but with all species
    with known q, r and bips, match the previous results.
    """

    list_aqueous_species = [
        "H2O(l)",
        "H+",
        "OH-",
        "Na+",
        'Cl-',
    ]

    mineral_name = 'Halite'

    system_euniquac = _create_euniquac_system_for_testing(mineral_name, list_aqueous_species)
    problem = rkt.EquilibriumProblem(system_euniquac)

    problem.setTemperature(25.0, "celsius")
    problem.setPressure(1.0, "atm")

    problem.add("H2O", 1, "kg")
    problem.add(mineral_name, 100, "mol")  # excess quantity

    state = rkt.equilibrate(problem)

    solubility = state.elementAmountInPhase('Na', 'Aqueous') #mol/kgW
    pH = rkt.ChemicalProperty.pH(system_euniquac)(state.properties()).val
    mol_species = state.speciesAmounts() #mols

    assert solubility == approx(6.189990)
    assert pH == approx(8.001948)
    expected_mols = [5.55084350e+01, 5.20522014e-08, 5.20522014e-08, 6.18999020e+00, 6.18999020e+00, 9.38100098e+01]
    assert mol_species == approx(expected_mols)


def test_euniquac_fallback_all_species_are_known_match_previous_results_bdot():
    """
    Test if the E-UNIQUAC with fallback modifications, but with all species
    with known q, r and bips, match the previous results.
    """

    list_aqueous_species = [
        "H2O(l)",
        "H+",
        "OH-",
        "Na+",
        'Cl-',
    ]

    mineral_name = 'Halite'

    euniquac_params = rkt.EUNIQUACParams()
    # euniquac_params.setVillafafilaGarcia2006()  # very important!
    calc_type_from_enum = getattr(rkt.LongRangeModelType, 'DH_Phreeqc')
    # euniquac_params.setLongRangeModelType(calc_type_from_enum)

    system_euniquac = _create_euniquac_system_for_testing(mineral_name, list_aqueous_species, 
        euniquac_params=euniquac_params
    )

    problem = rkt.EquilibriumProblem(system_euniquac)

    problem.setTemperature(25.0, "celsius")
    problem.setPressure(1.0, "atm")

    problem.add("H2O", 1, "kg")
    problem.add(mineral_name, 100, "mol")  # excess quantity

    state = rkt.equilibrate(problem)

    solubility = state.elementAmountInPhase('Na', 'Aqueous') #mol/kgW
    pH = rkt.ChemicalProperty.pH(system_euniquac)(state.properties()).val
    mol_species = state.speciesAmounts() #mols

    assert solubility == approx(6.189990)
    assert pH == approx(8.001948)
    expected_mols = [5.55084350e+01, 5.20522014e-08, 5.20522014e-08, 6.18999020e+00, 6.18999020e+00, 9.38100098e+01]
    assert mol_species == approx(expected_mols)

def test_euniquac_fallback_missing_species_fallback():
    """
    Test the E-UNIQUAC with fallback modifications with missing species, Quartz case.
    """

    list_aqueous_species = [
        "H2O(l)",
        "H+",
        "OH-",
        "Na+",
        'Cl-',
        'HSiO3-',
        'SiO2(aq)',
    ]

    mineral_name = 'Quartz'

    system_euniquac = _create_euniquac_system_for_testing(mineral_name, list_aqueous_species)

    problem = rkt.EquilibriumProblem(system_euniquac)
    problem.setTemperature(25.0, "celsius")
    problem.setPressure(1.0, "atm")

    problem.add("H2O", 1, "kg")
    problem.add("NaCl", 0.1, "mol")
    problem.add(mineral_name, 10, "mol")  # excess quantity

    state = rkt.equilibrate(problem)

    solubility = state.elementAmountInPhase('Si', 'Aqueous') * 1e3
    pH = rkt.ChemicalProperty.pH(system_euniquac)(state.properties()).val
    mol_species = state.speciesAmounts()

    assert np.isclose(solubility, 0.10278223633476395, 1e-1)
    assert np.isclose(pH, 6.726597777002442, 1e-3)
    expected_mols = [
        5.55084348e+01, 2.48292415e-07, 6.64015320e-08, 1.00000000e-01,
        1.00000000e-01, 1.81890883e-07, 1.02600345e-04, 9.99989722e+00
    ]
    np.testing.assert_array_almost_equal(mol_species, expected_mols, decimal=2)


def test_euniquac_fallback_add_aqueous_phase_with_elements():
    """
    Test if the E-UNIQUAC system converges with the fallback modifications when adding
    many species using the `addAqueousPhaseWithElementsOf` funcion
    """

    mineral_name = 'Barite'
    string_of_elements = "H O Na Ca K Sr Mg Ba Cl S"

    euniquac_params = rkt.EUNIQUACParams()
    euniquac_params.setVillafafilaGarcia2006()  # very important!

    system_euniquac = _create_euniquac_system_for_testing(
        mineral_name, 
        string_of_elements=string_of_elements, 
        euniquac_params=euniquac_params
    )

    problem = rkt.EquilibriumProblem(system_euniquac)
    problem.setTemperature(25.0, "celsius")
    problem.setPressure(1.0, "atm")

    problem.add("H2O", 1, "kg")
    problem.add("NaCl", 0.1, "mol")
    problem.add(mineral_name, 10, "mol")

    state = rkt.equilibrate(problem)

    equilibrium_result = rkt.equilibrate(state)
    succeeded = equilibrium_result.optimum.succeeded

    assert succeeded


# @pytest.mark.parametrize(
#     'calculation_type,solubility_expected',
#     [
#         ['DH_Phreeqc', 0.10016713283821646],
#         ['HKF', 0.10016713283821646],
#     ],
# )
@pytest.mark.parametrize(
    'calculation_type',
    [
        'DH_Phreeqc',
        'HKF',
    ],
)
def test_euniquac_with_longrange_as_bdot_equals_bdot_when_only_unknown_species(
    calculation_type
):
    """
    (1) Test if output of euniquac with long range as B-dot is equal to the
    original B-dot (DebyeHuckel) output when the species are all unkowns for e-uniquac

    (2) Test if output of euniquac with long range as HKF is equal to the
    HKF output when the species are all unkowns for e-uniquac

    Obs: HKF and B-dot results are equal
    """

    list_aqueous_species = [
        "H2O(l)",
        'HSiO3-',
        'SiO2(aq)',
    ]

    mineral_name = 'Quartz'

    euniquac_params = rkt.EUNIQUACParams()
    calc_type_from_enum = getattr(rkt.LongRangeModelType, calculation_type)
    euniquac_params.setLongRangeModelType(calc_type_from_enum)

    system_euniquac = _create_euniquac_system_for_testing(
        mineral_name,
        list_aqueous_species,
        euniquac_params=euniquac_params
    )

    problem = rkt.EquilibriumProblem(system_euniquac)
    problem.setTemperature(25.0, "celsius")
    problem.setPressure(1.0, "atm")

    problem.add("H2O", 1, "kg")
    problem.add('Quartz', 10, "mol")  # excess quantity

    state = rkt.equilibrate(problem)

    solubility = state.elementAmountInPhase('Si', 'Aqueous') * 1e3
    mol_species = state.speciesAmounts()

    assert solubility == approx(0.10016713283821646)
    expected_mols = [5.55084351e+01, 1.00000000e-20, 1.00167133e-04, 9.99989983e+00]
    assert mol_species == approx(expected_mols)


def _create_euniquac_system_for_testing(
    mineral_name: str, 
    list_aqueous_species: List[str]=None, 
    gas_species: List[str]=[], 
    euniquac_params: Optional[rkt.EUNIQUACParams]=None,
    string_of_elements: str=None,
) -> rkt.ChemicalSystem:
    editor = rkt.ChemicalEditor()
    editor.addMineralPhase(mineral_name)
    if string_of_elements is None:
        aqueous_phase = editor.addAqueousPhase(list_aqueous_species)
    else:
        aqueous_phase = editor.addAqueousPhaseWithElementsOf(string_of_elements)

    if len(gas_species):
        gas_phase = editor.addGaseousPhase(gas_species)
        gas_phase.setChemicalModelSoaveRedlichKwong()

    if euniquac_params is None:
        euniquac_params = rkt.EUNIQUACParams()
    editor.aqueousPhase().setChemicalModelEUNIQUAC(euniquac_params)

    system_euniquac = rkt.ChemicalSystem(editor)

    return system_euniquac