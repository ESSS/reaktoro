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
import pytest
from reaktoro import *


def test_ReactionParams():
    reaction = ReactionParams()
    coefficients = reaction.analytic
    assert not coefficients
    coefficients.append(1)
    assert coefficients
    assert reaction.analytic
    assert reaction.analytic[0] == 1
    assert coefficients is reaction.analytic


def test_SpeciesThermoData():
    species = SpeciesThermoData()
    assert species.properties is None
    assert species.reaction is None
    assert species.phreeqc is None

    species.properties = SpeciesThermoInterpolatedProperties()
    species.reaction = ReactionThermoInterpolatedProperties()
    species.phreeqc = SpeciesThermoParamsPhreeqc()
    species.phreeqc.reaction = ReactionParams()
    assert species.properties is not None
    assert species.reaction is not None
    assert species.phreeqc is not None
    assert species.phreeqc.reaction is not None


def test_FluidSpeciesThermoData():
    fluid_species = FluidSpeciesThermoData()
    assert fluid_species.hkf is None
    fluid_species.hkf = FluidSpeciesThermoParamsHKF()
    assert fluid_species.hkf is not None


def test_AqueousSpeciesThermoData():
    aqueous_species = AqueousSpeciesThermoData()
    assert aqueous_species.hkf is None
    aqueous_species.hkf = AqueousSpeciesThermoParamsHKF()
    assert aqueous_species.hkf is not None


def test_GaseousSpeciesThermoData():
    gaseous_species = GaseousSpeciesThermoData()
    assert gaseous_species.hkf is None
    gaseous_species.hkf = GaseousSpeciesThermoParamsHKF()
    assert gaseous_species.hkf is not None


@pytest.mark.parametrize(
    "thermodata",
    [
        GaseousSpeciesThermoData, AqueousSpeciesThermoData, MineralSpeciesThermoData
    ]
)
def test_SpeciesThermoParamsNIST(thermodata):
    species_thermodata = thermodata()
    assert species_thermodata.nist is None
    species_thermodata.nist = SpeciesThermoParamsNIST()
    assert species_thermodata.nist is not None


def test_MineralSpeciesThermoParamsHKF():
    hkf = MineralSpeciesThermoParamsHKF()
    assert len(hkf.a) == 0
    hkf.a.append(1)
    assert len(hkf.a) == 1

    a = hkf.a
    assert a[0] == 1

    del a
    assert hkf.a[0] == 1

    a = hkf.a
    a[0] = 2
    assert hkf.a[0] == 2
    hkf.a[0] = 3
    del hkf
    assert a[0] == 3


def test_SpeciesThermoParamsNIST_getter_and_setters():
    nist = SpeciesThermoParamsNIST()

    # Getters with default values

    G0 = nist.G0
    assert G0 == 0  # default

    H0 = nist.H0
    assert H0 == 0  # default

    Cp = nist.Cp
    assert Cp == 0  # default

    Cp_a = nist.Cp_a
    assert Cp_a == 0  # default

    Cp_b = nist.Cp_b
    assert Cp_b == 0  # default

    Cp_c = nist.Cp_c
    assert Cp_c == 0  # default

    # Getters and setters with custom values

    nist.G0 = 1
    assert nist.G0 == 1

    nist.H0 = 2
    assert nist.H0 == 2

    nist.Cp = 3
    assert nist.Cp == 3

    nist.Cp_a = 4
    assert nist.Cp_a == 4

    nist.Cp_b = 5
    assert nist.Cp_b == 5

    nist.Cp_c = 6
    assert nist.Cp_c == 6


@pytest.mark.parametrize(
    "thermodata",
    [
        GaseousSpeciesThermoData, AqueousSpeciesThermoData, MineralSpeciesThermoData
    ]
)
def test_SpeciesThermoParamsNIST_in_aqueous_species_thermodata(thermodata):
    species_thermo_data = thermodata()
    assert species_thermo_data.nist is None
    species_thermo_data.nist = SpeciesThermoParamsNIST()
    assert species_thermo_data.nist is not None

    nist = species_thermo_data.nist
    species_thermo_data.nist.G0 = 1.0
    assert nist.G0 == 1.0

    del species_thermo_data
    assert nist.G0 == 1.0


def test_MineralSpeciesThermoData():
    thermo_data = MineralSpeciesThermoData()

    assert thermo_data.hkf is None
    thermo_data.hkf = MineralSpeciesThermoParamsHKF()
    assert thermo_data.hkf is not None

    hkf = thermo_data.hkf
    thermo_data.hkf.a.append(1)
    assert hkf.a[0] == 1
    del hkf
    assert thermo_data.hkf.a[0] == 1

    hkf = thermo_data.hkf
    del thermo_data
    assert hkf.a[0] == 1


def test_MineralSpecies():
    mineral_species = MineralSpecies()
    thermo_data = mineral_species.thermoData()
    thermo_data.hkf = MineralSpeciesThermoParamsHKF()

    mineral_species.thermoData().hkf.a.append(1)
    del mineral_species
    assert thermo_data.hkf.a[0] == 1


def test_LiquidSpeciesThermoData():
    liquid_species = LiquidSpeciesThermoData()
    assert liquid_species.hkf is None
    liquid_species.hkf = LiquidSpeciesThermoParamsHKF()
    assert liquid_species.hkf is not None

