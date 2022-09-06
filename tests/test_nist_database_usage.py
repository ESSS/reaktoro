import pytest
from reaktoro import (
    ChemicalSystem,
    ChemicalProperties,
    XmlDatabaseType
)


@pytest.mark.xfail(reason="Unable to set NIST DB as std properties.")
@pytest.mark.parametrize(
    "species_name", ["Na+", "Cl-", "H2O(l)"]
)
def test_reaktoro_setup_with_nist_database(species_name, nist_database, chemical_editor_nacl_nist):
    editor = chemical_editor_nacl_nist

    system = ChemicalSystem(editor)
    species_index = system.indexSpecies(species_name)

    properties = ChemicalProperties(system)
    species_partial_molar_gibbs_energy = properties.standardPartialMolarGibbsEnergies().val
    species_g0_from_properties = species_partial_molar_gibbs_energy[species_index]

    species_in_database = nist_database.aqueousSpecies(species_name)
    species_nist_thermodata = species_in_database.thermoData().nist
    species_g0_from_thermodata = species_nist_thermodata.G0
    assert species_g0_from_properties == species_g0_from_thermodata


def test_nist_database_thermodata_type(nist_database):
    database = nist_database
    assert database.databaseType() == XmlDatabaseType.NIST
