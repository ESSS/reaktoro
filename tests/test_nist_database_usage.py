import numpy as np
import pytest
from reaktoro import (
    ChemicalEditor,
    ChemicalSystem,
    ChemicalProperties,
    XmlDatabaseType,
    ChemicalState,
    EquilibriumSolver,
)

KJ_TO_K = 1.0e3


def _calculate_cp(T, a, b, c):
    """
    Convenient function to calculate Cp following Kaj Thomsen's approach at given temperatures.
    """
    return a + b * T + c / (T - 200.0)


def _get_species_cp_parameters(nist_database, species_name):
    species_in_database = nist_database.aqueousSpecies(species_name)
    species_nist_thermodata = species_in_database.thermoData().nist
    species_cp_a = species_nist_thermodata.cp_a
    species_cp_b = species_nist_thermodata.cp_b
    species_cp_c = species_nist_thermodata.cp_c
    return species_cp_a, species_cp_b, species_cp_c


@pytest.mark.parametrize(
    "species_name", ["Na+", "Cl-"]
)
def test_reaktoro_setup_with_nist_database(species_name, nist_database, chemical_editor_nacl_nist):
    editor = chemical_editor_nacl_nist

    system = ChemicalSystem(editor)
    species_index = system.indexSpecies(species_name)

    properties = ChemicalProperties(system)
    P_std = 1e5
    T_std = 298.15
    properties.update(T_std, P_std)
    species_partial_molar_gibbs_energy = properties.standardPartialMolarGibbsEnergies().val
    species_g0_from_properties = species_partial_molar_gibbs_energy[species_index]

    species_in_database = nist_database.aqueousSpecies(species_name)
    species_nist_thermodata = species_in_database.thermoData().nist
    species_g0_from_thermodata = species_nist_thermodata.G0
    assert species_g0_from_properties == pytest.approx(species_g0_from_thermodata * KJ_TO_K)


def test_nist_database_thermodata_type(nist_database):
    database = nist_database
    assert database.databaseType() == XmlDatabaseType.NIST


def test_equilibriumsolver_with_nist(nist_database, chemical_editor_nacl_nist):
    editor = chemical_editor_nacl_nist
    system = ChemicalSystem(editor)
    state = ChemicalState(system)

    # 1 molal NaCl solution
    state.setTemperature(25.0, "celsius")
    state.setPressure(1, "bar")
    state.setSpeciesAmount("Na+", 1, "mol")
    state.setSpeciesAmount("Cl-", 1, "mol")
    state.setSpeciesMass("H2O(l)", 1, "kg")

    solver = EquilibriumSolver(system)
    solver.solve(state)

    # All NaCl should dissociate in the 1 molal solution
    assert state.speciesAmount("Na+") == pytest.approx(1.0)
    assert state.speciesAmount("Cl-") == pytest.approx(1.0)

    # Check if ChemicalState has the correct G0 value in properties
    na_in_database = nist_database.aqueousSpecies("Na+")
    na_nist_thermodata = na_in_database.thermoData().nist
    na_g0_from_thermodata = na_nist_thermodata.G0

    na_index = system.indexSpecies("Na+")
    properties = state.properties()
    na_g0_from_state = properties.standardPartialMolarGibbsEnergies().val[na_index]
    assert na_g0_from_state == pytest.approx(na_g0_from_thermodata * KJ_TO_K)


def test_heat_capacity_calculation(euniquac_nist_database):
    # Reaktoro basic setup
    editor = ChemicalEditor(euniquac_nist_database)
    num_interpolation_points = 100
    T_min = 25.0  # degC
    T_max = 200.0  # degC
    interpolation_temperatures = np.linspace(T_min, T_max, num_interpolation_points)
    editor.setTemperatures(interpolation_temperatures, "celsius")
    
    solute_species = ("Na+", "Cl-", "Ca++", "Mg++", "Sr", "CO3--", "HCO3-", "CO2(aq)")
    aqueous_species = ("H2O(l)", "H+", "OH-") + solute_species
    editor.addAqueousPhase(aqueous_species)
    
    system = ChemicalSystem(editor)
    state = ChemicalState(system)
    P_reference = 1.0  # bar 
    state.setPressure(P_reference, "bar")
    state.setTemperature(25.0, "celsius")
    state.setSpeciesMass("H2O(l)", 1, "kg")
    for solute in solute_species:
        state.setSpeciesAmount(solute, 1, "mol")
        
    # Collecting solutes' Cp parameters
    solute_cp_params = {}
    for solute in solute_species:
        solute_params = _get_species_cp_parameters(euniquac_nist_database, solute)
        solute_cp_params[solute] = solute_params
        
    # Calculating Cp values for reference
    solute_cp_values_reference = {}
    solute_cp_values_reference['T (degC)'] = interpolation_temperatures
    for solute in solute_species:
        solute_cp_a, solute_cp_b, solute_cp_c = _get_species_cp_parameters(
            euniquac_nist_database, solute
        )
        solute_cp_values = _calculate_cp(
            interpolation_temperatures, solute_cp_a, solute_cp_b, solute_cp_c
        )
        solute_cp_values_reference[solute] = solute_cp_values
        
    # Calculating Cp values with Reaktoro thermo properties
    properties = state.properties()
    solute_cp_values = {}
    solute_cp_values['T (degC)'] = interpolation_temperatures
    for T in interpolation_temperatures:
        T_in_K = T + 273.15
        bar_to_Pa = 1e5
        P_in_Pa = P_reference * bar_to_Pa
        properties.update(T_in_K, P_in_Pa)
    assert False
