import numpy as np
import pandas as pd
import pytest
from reaktoro import (
    ChemicalEditor,
    ChemicalSystem,
    ChemicalProperties,
    XmlDatabaseType,
    ChemicalState,
    EquilibriumSolver,
)

KJ_PER_MOL_TO_J_PER_MOL = 1.0e3


def _calculate_cp(T, a, b, c):
    """
    Convenient function to calculate Cp following Kaj Thomsen's approach at given temperatures (
    in Kelvin).
    """
    return a + b * T + c / (T - 200.0)


def _get_species_cp_parameters(nist_database, species_name):
    species_in_database = nist_database.aqueousSpecies(species_name)
    species_nist_thermodata = species_in_database.thermoData().nist
    species_cp_constant = species_nist_thermodata.Cp

    cp_a_value = species_nist_thermodata.Cp_a
    species_cp_a = cp_a_value if cp_a_value != np.inf else species_cp_constant

    cp_b_value = species_nist_thermodata.Cp_b
    species_cp_b = cp_b_value if cp_b_value != np.inf else 0.0

    cp_c_value = species_nist_thermodata.Cp_c
    species_cp_c = cp_c_value if cp_c_value != np.inf else 0.0

    return species_cp_a, species_cp_b, species_cp_c


def _get_heat_capacity_value(
    T_in_K: float,
    P_in_Pa: float,
    chemical_state: ChemicalState,
    species_name: str
) -> float:
    system = chemical_state.system()
    species_index = system.indexSpecies(species_name)
    properties = chemical_state.properties()
    properties.update(T_in_K, P_in_Pa)
    cp_value = properties.standardPartialMolarHeatCapacitiesConstP().val[species_index]
    return cp_value


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
    assert species_g0_from_properties == pytest.approx(species_g0_from_thermodata * KJ_PER_MOL_TO_J_PER_MOL)


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
    assert na_g0_from_state == pytest.approx(
        na_g0_from_thermodata * KJ_PER_MOL_TO_J_PER_MOL
    )


def test_heat_capacity_calculation(nist_database, dataframe_regression):
    # Reaktoro basic setup
    editor = ChemicalEditor(nist_database)
    num_interpolation_points = 100
    T_min = 25.0  # degC
    T_max = 200.0  # degC
    interpolation_temperatures = np.linspace(T_min, T_max, num_interpolation_points)
    editor.setTemperatures(interpolation_temperatures, "celsius")

    aqueous_species = nist_database.aqueousSpecies()
    aqueous_species = [species.name() for species in aqueous_species]
    editor.addAqueousPhase(aqueous_species)
    
    system = ChemicalSystem(editor)
    state = ChemicalState(system)
    P_reference = 1.0  # bar 
    state.setPressure(P_reference, "bar")
    state.setTemperature(25.0, "celsius")
    for species in aqueous_species:
        state.setSpeciesAmount(species, 1, "mol")
        
    # Collecting solutes' Cp parameters
    species_cp_params = {}
    for species in aqueous_species:
        cp_params = _get_species_cp_parameters(nist_database, species)
        species_cp_params[species] = cp_params
        
    # Calculating Cp values for reference
    interpolation_temperatures_in_K = interpolation_temperatures + 273.15
    species_cp_values_reference = {}
    species_cp_values_reference['T (degC)'] = interpolation_temperatures
    for species in aqueous_species:
        solute_cp_a, solute_cp_b, solute_cp_c = species_cp_params[species]
        species_cp_values_analytic = _calculate_cp(
            interpolation_temperatures_in_K, solute_cp_a, solute_cp_b, solute_cp_c
        )
        species_cp_values_reference[species] = species_cp_values_analytic
        
    # Calculating Cp values with Reaktoro thermo properties
    species_cp_values = {species_name: [] for species_name in aqueous_species}
    species_cp_values['T (degC)'] = interpolation_temperatures
    for species in aqueous_species:
        for T_in_K in interpolation_temperatures_in_K:
            bar_to_Pa = 1e5
            P_in_Pa = P_reference * bar_to_Pa
            species_cp_value = _get_heat_capacity_value(T_in_K, P_in_Pa, state, species)
            species_cp_values[species].append(species_cp_value)

        species_cp_values[species] = np.array(species_cp_values[species])
        assert species_cp_values[species] == pytest.approx(species_cp_values_reference[species])

    df_species_cp_values = pd.DataFrame.from_dict(species_cp_values)
    dataframe_regression.check(df_species_cp_values)
