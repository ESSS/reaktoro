import pytest
from reaktoro import (
    ChemicalSystem,
    ChemicalProperties,
    XmlDatabaseType,
    ChemicalState,
    EquilibriumSolver,
)

KJ_TO_K = 1.0e3


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
