import numpy as np
import pytest
import reaktoro


def get_test_data_dir():
    from pathlib import Path
    import os
    return Path(os.path.abspath(__file__)).parents[0] / "data"


def _add_hydrocarbons_to_database(db):
    hydrocarbons_db = reaktoro.Database(str(get_test_data_dir() / 'hydrocarbons.xml'))

    element_names_in_db = set(e.name() for e in db.elements())
    for element in hydrocarbons_db.elements():
        if element.name() not in element_names_in_db:
            db.addElement(element)

    for species in hydrocarbons_db.gaseousSpecies():
        db.addGaseousSpecies(species)

    for species in hydrocarbons_db.liquidSpecies():
        db.addLiquidSpecies(species)


@pytest.mark.parametrize(
    "P, T, F_expected, x_expected, y_expected",
    [
        # psi, degF, [mol / mol], [mol / mol], [mol / mol]
        (500, 280, [0.853401, 1 - 0.853401], [0.08588, 0.46349, 0.45064], [0.57114, 0.41253, 0.01633]),
        # (1500, 280, [0.566844, 1 - 0.566844], [0.330082, 0.513307, 0.156611], [0.629843, 0.348699, 0.021457]),
    ]
)
def test_ternary_c1_c4_c10_mixture(P, T, F_expected, x_expected, y_expected):
    """
    This is a ternary example from Whitson monograph. Retrieved from Problem 18 in
    Appendix B.

    .. reference:
        Whitson, C. H., & Brulé, M. R. (2000). Phase behavior (Vol. 20).
        Richardson, TX: Henry L. Doherty Memorial Fund of AIME, Society of Petroleum Engineers.
    """
    temperature = T  # degF
    pressure = P  # psi

    molar_base = 1
    composition = molar_base * np.array([0.5, 0.42, 0.08])

    db = reaktoro.Database('supcrt07.xml')
    _add_hydrocarbons_to_database(db)

    editor = reaktoro.ChemicalEditor(db)

    gaseous_species = ["C1(g)", "C4(g)", "C10(g)"]
    oil_species = ["C1(liq)", "C4(liq)", "C10(liq)"]

    eos_params = reaktoro.CubicEOSParams(
        model=reaktoro.CubicEOSModel.PengRobinson,
        phase_identification_method=reaktoro.PhaseIdentificationMethod.GibbsEnergyAndEquationOfStateMethod,
    )

    editor.addGaseousPhase(gaseous_species).setChemicalModelCubicEOS(eos_params)
    editor.addLiquidPhase(oil_species).setChemicalModelCubicEOS(eos_params)

    system = reaktoro.ChemicalSystem(editor)

    problem = reaktoro.EquilibriumProblem(system)

    problem.setTemperature(temperature, 'degF')
    problem.setPressure(pressure, 'psi')
    problem.setElementAmount('C1', composition[0])
    problem.setElementAmount('C4', composition[1])
    problem.setElementAmount('C10', composition[2])

    options = reaktoro.EquilibriumOptions()
    options.hessian = reaktoro.GibbsHessian.Exact
    options.optimum.max_iterations = 10000
    options.optimum.tolerance = 1e-8
    # options.optimum.output.active = True

    state = reaktoro.equilibrate(problem, options)

    gas_phase_molar_fraction = state.phaseAmount('Gaseous') / molar_base
    liquid_molar_fraction = state.phaseAmount('Liquid') / molar_base
    phase_fractions = np.array([gas_phase_molar_fraction, liquid_molar_fraction])
    assert phase_fractions == pytest.approx(F_expected, rel=5e-3)

    c1_gas_fraction = state.speciesAmount('C1(g)') / phase_fractions[0]
    c4_gas_fraction = state.speciesAmount('C4(g)') / phase_fractions[0]
    c10_gas_fraction = state.speciesAmount('C10(g)') / phase_fractions[0]
    equilibrium_gas_composition = np.array([
        c1_gas_fraction,
        c4_gas_fraction,
        c10_gas_fraction
    ])
    assert equilibrium_gas_composition == pytest.approx(
        y_expected,
        rel=2e-2
    )

    c1_liq_fraction = state.speciesAmount('C1(liq)') / phase_fractions[1]
    c4_liq_fraction = state.speciesAmount('C4(liq)') / phase_fractions[1]
    c10_liq_fraction = state.speciesAmount('C10(liq)') / phase_fractions[1]
    equilibrium_liq_composition = np.array([
        c1_liq_fraction,
        c4_liq_fraction,
        c10_liq_fraction
    ])
    assert equilibrium_liq_composition == pytest.approx(
        x_expected,
        rel=2e-2
    )
