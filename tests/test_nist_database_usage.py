from reaktoro import (
    ChemicalEditor,
    ChemicalSystem,
)


def test_reaktoro_setup_with_nist_database(nist_database):
    editor = ChemicalEditor(nist_database)
    system = ChemicalSystem(editor)
