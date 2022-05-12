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

import numpy as np
import pytest

import reaktoro as rkt


def test_euniquac_longrange_model_options():
    """
    Test if the option for long range model is working properly

    Test will pass if no error occurs.
    """

    euniquac_params = rkt.EUNIQUACParams()
    euniquac_params.setLongRangeModelType(rkt.LongRangeModelType.DH_Phreeqc)
    long_range_model_type = euniquac_params.longRangeModelType()
    assert long_range_model_type == rkt.LongRangeModelType.DH_Phreeqc


def test_euniquac_fallback_all_species_are_known_match_previous_results():
    """
    Test if the E-UNIQUAC with fallback modifications but with all species
    with known q, r and bips match the previous results
    """

    list_aqueous_species = [
        "H2O(l)",
        "H+",
        "OH-",
        "Na+",
        'Cl-',
    ]
    gas_species = []

    mineral_name = 'Halite'

    # Create Euniquac
    editor = rkt.ChemicalEditor()
    editor.addMineralPhase(mineral_name)
    aqueous_phase = editor.addAqueousPhase(list_aqueous_species)

    if len(gas_species):
        gas_phase = editor.addGaseousPhase(gas_species)
        gas_phase.setChemicalModelSoaveRedlichKwong()

    euniquac_params = rkt.EUNIQUACParams()
    editor.aqueousPhase().setChemicalModelEUNIQUAC(euniquac_params)
    system_euniquac = rkt.ChemicalSystem(editor)

    problem = rkt.EquilibriumProblem(system_euniquac)
    problem.setTemperature(25.0, "celsius")
    problem.setPressure(1.0, "atm")

    problem.add("H2O", 1, "kg")
    problem.add(mineral_name, 100, "mol")  # excess quantity

    state = rkt.equilibrate(problem)

    solubility = state.elementAmountInPhase('Na', 'Aqueous') * 1e3
    pH = rkt.ChemicalProperty.pH(system_euniquac)(state.properties()).val
    mol_species = state.speciesAmounts()

    assert np.isclose(solubility, 6189.990199851658)
    assert np.isclose(pH, 8.001947880191114)
    expected_mols = [5.55084350e+01, 5.20522014e-08, 5.20522014e-08, 6.18999020e+00, 6.18999020e+00, 9.38100098e+01]
    np.testing.assert_array_almost_equal(mol_species, expected_mols)


def test_euniquac_fallback_missing_species_fallback():
    """
    Test if the E-UNIQUAC with fallback modifications with missing species case Quartz
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
    gas_species = []

    mineral_name = 'Quartz'

    # Create Euniquac
    editor = rkt.ChemicalEditor()
    editor.addMineralPhase(mineral_name)
    aqueous_phase = editor.addAqueousPhase(list_aqueous_species)

    if len(gas_species):
        gas_phase = editor.addGaseousPhase(gas_species)
        gas_phase.setChemicalModelSoaveRedlichKwong()

    euniquac_params = rkt.EUNIQUACParams()
    editor.aqueousPhase().setChemicalModelEUNIQUAC(euniquac_params)
    system_euniquac = rkt.ChemicalSystem(editor)

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
    Test if the E-UNIQUAC with fallback modifications with missing species case Barite
    with many species
    """

    mineral_name = 'Barite'

    # Create Euniquac
    editor = rkt.ChemicalEditor()
    editor.addMineralPhase(mineral_name)
    aqueous_phase = editor.addAqueousPhaseWithElementsOf("H O Na Ca K Sr Mg Ba Cl S")

    euniquac_params = rkt.EUNIQUACParams()
    euniquac_params.setVillafafilaGarcia2006()  # very important!
    editor.aqueousPhase().setChemicalModelEUNIQUAC(euniquac_params)
    system_euniquac = rkt.ChemicalSystem(editor)

    problem = rkt.EquilibriumProblem(system_euniquac)
    problem.setTemperature(25.0, "celsius")
    problem.setPressure(1.0, "atm")

    problem.add("H2O", 1, "kg")
    problem.add("NaCl", 0.1, "mol")
    problem.add(mineral_name, 10, "mol")

    state = rkt.equilibrate(problem)

    solubility = state.elementAmountInPhase('Ba', 'Aqueous') * 1e3
    pH = rkt.ChemicalProperty.pH(system_euniquac)(state.properties()).val
    mol_species = state.speciesAmounts()
    # Question: No assert here is needed, just want to run without raising exception (due to convergence)


@pytest.mark.parametrize(
    'calculation_type,solubility_expected',
    [
        ['DH_Phreeqc', 0.10016713283821646],
        ['HKF', 0.10016713283821646],
    ],
)
def test_euniquac_with_longrange_as_bdot_equals_bdot_when_only_unknown_species(
    calculation_type, solubility_expected
):
    """
    (1) Test if output of euniquac with long range as B-dot is equal to the
    original B-dot (DebyeHuckel) output when the species are all unkowns for e-uniquac

    (2) Test if output of euniquac with long range as B-dot is equal to the
    original B-dot (DebyeHuckel) output when the species are all unkowns for e-uniquac
    """

    list_aqueous_species = [
        "H2O(l)",
        'HSiO3-',
        'SiO2(aq)',
    ]

    mineral_name = 'Quartz'

    # Create Euniquac
    editor = rkt.ChemicalEditor()
    editor.addMineralPhase(mineral_name)
    aqueous_phase = editor.addAqueousPhase(list_aqueous_species)

    euniquac_params = rkt.EUNIQUACParams()
    calc_type_from_enum = getattr(rkt.LongRangeModelType, calculation_type)
    euniquac_params.setLongRangeModelType(calc_type_from_enum)

    editor.aqueousPhase().setChemicalModelEUNIQUAC(euniquac_params)
    system_euniquac = rkt.ChemicalSystem(editor)

    problem = rkt.EquilibriumProblem(system_euniquac)
    problem.setTemperature(25.0, "celsius")
    problem.setPressure(1.0, "atm")

    problem.add("H2O", 1, "kg")
    problem.add('Quartz', 10, "mol")  # excess quantity

    state = rkt.equilibrate(problem)

    solubility = state.elementAmountInPhase('Si', 'Aqueous') * 1e3
    mol_species = state.speciesAmounts()

    assert(np.isclose(solubility, solubility_expected))
    expected_mols = [5.55084351e+01, 1.00000000e-20, 1.00167133e-04, 9.99989983e+00]
    np.testing.assert_array_almost_equal(mol_species, expected_mols, decimal=6)