// Reaktoro is a unified framework for modeling chemically reactive systems.
//
// Copyright (C) 2014-2018 Allan Leal
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this library. If not, see <http://www.gnu.org/licenses/>.

#include "SpeciesThermoStateNIST.hpp"

// C++ includes
#include <cmath>

// Reaktoro includes
#include <Reaktoro/Common/Constants.hpp>
#include <Reaktoro/Common/Exception.hpp>
#include <Reaktoro/Common/NamingUtils.hpp>
#include <Reaktoro/Common/ThermoScalar.hpp>
#include <Reaktoro/Thermodynamics/Models/SpeciesThermoState.hpp>
#include <Reaktoro/Thermodynamics/Species/AqueousSpecies.hpp>
#include <Reaktoro/Thermodynamics/Species/FluidSpecies.hpp>
#include <Reaktoro/Thermodynamics/Species/MineralSpecies.hpp>
#include <Reaktoro/Thermodynamics/Water/WaterConstants.hpp>
#include <Reaktoro/Thermodynamics/Water/WaterThermoState.hpp>
#include <Reaktoro/Thermodynamics/Water/WaterThermoStateUtils.hpp>

namespace Reaktoro {
namespace {

/// The reference temperature assumed in the HKF equations of state (in units of K)
const double referenceTemperature = 298.15;

/// The reference temperature assumed in the HKF equations of state (in units of Pascal)
const double referencePressure = 1.0e5;

const double kJToJ = 1e3;

template<typename SpeciesType>
auto checkSpeciesDataNIST(const SpeciesType& species) -> void
{
    const auto& nist = *species.thermoData().nist;

    const std::string error = "Unable to calculate the thermodynamic properties of species " +
        species.name() + " using the NIST database parameters.";

    if(!std::isfinite(nist.G0))
        RuntimeError(error, "Missing `G0` data for this species in the database");

    if(!std::isfinite(nist.H0))
        RuntimeError(error, "Missing `H0` data for this species in the database");

    if(!std::isfinite(nist.Cp))
        RuntimeError(error, "Missing `Cp` data for this species in the database");
}

} // namespace


// This approach uses IAPWS (Wagner and Pruss) to calculate water (solvent) properties.
// It is currently not used, could be useful in the future.
auto speciesThermoStateSolventNIST(Temperature T, Pressure P, const AqueousSpecies& species) -> SpeciesThermoState
{
    // Check if all minimum NIST parameters are available
    checkSpeciesDataNIST(species);

    // Get the NIST thermodynamic data of the species
    const auto& nist = *species.thermoData().nist;

    // Calculate H2O properties with Wagner & Pruss
    WaterThermoState wt = waterThermoStateWagnerPruss(T, P, StateOfMatter::Liquid);

    // Calculate H2O properties with Wagner & Pruss at reference
    const auto Ttr = 273.15;  // unit: K
    const auto Ptr = referencePressure;  // unit: Pascal
    WaterThermoState wtr = waterThermoStateWagnerPruss(Ttr, Ptr, StateOfMatter::Liquid);

    // Reference values
    const auto& state_ref = genericSpeciesThermoStateNIST(Ttr, Ptr, species);
    const auto Vtr = wtr.volume * waterMolarMass;  // unit: m3/mol
    const auto Str = state_ref.entropy; // unit: J/(mol*K)
    const auto Gtr = state_ref.gibbs_energy; // unit: J/mol
    const auto Htr = state_ref.enthalpy; // unit: J/mol
    const auto Utr = Htr - Ptr*Vtr;
    const auto Atr = Utr - Ttr*Str;

    // Heat capacities
    const auto Cp = wt.cp * waterMolarMass;
    const auto Cv = wt.cv * waterMolarMass;

    // Computed with Wagner & Pruss
    const auto Sw = waterMolarMass * wt.entropy;         // unit: J/(mol*K)
    const auto Hw = waterMolarMass * wt.enthalpy;        // unit: J/mol
    const auto Uw = waterMolarMass * wt.internal_energy; // unit: J/mol

    // Calculate the standard molal thermodynamic properties of the aqueous species
    const auto S  = Sw + Str;
    const auto H  = Hw + Htr;
    const auto U  = Uw + Utr;
    const auto G  = Hw - T * (Sw + Str) + Ttr * Str + Gtr;
    const auto A  = Uw - T * (Sw + Str) + Ttr * Str + Atr;
    const auto V  = wt.volume * waterMolarMass;

    SpeciesThermoState state;
    state.entropy          = S;
    state.enthalpy         = H;
    state.internal_energy  = U;
    state.gibbs_energy     = G;
    state.helmholtz_energy = A;
    state.volume           = V;
    state.heat_capacity_cp = Cp;
    state.heat_capacity_cv = Cv;

    return state;
}

template<typename SpeciesType>
auto genericSpeciesThermoStateNIST(Temperature T, Pressure P, const SpeciesType& species) -> SpeciesThermoState
{
    // Check if all minimum NIST parameters are available
    checkSpeciesDataNIST(species);

    // Get the NIST thermodynamic data of the species
    const auto& nist = *species.thermoData().nist;

    // Auxiliary variables
    const auto R = universalGasConstant;
    const auto Tr = referenceTemperature;
    const auto T_theta = 200.0;  // in K
    const auto G0 = nist.G0 * kJToJ;
    const auto H0 = nist.H0 * kJToJ;
    // Since customizations for E-UNIQUAC are being used, we calculate S0 this way to have consistency.
    // This is valid according to Gibbs-Helmholtz equation with P = cte. This is a fine assumption
    // since it just for the reference state, which is constant.
    const auto S0 = (H0 - G0) / Tr;
    const auto Cp0 = nist.Cp;
    const auto a = std::isfinite(nist.Cp_a) ? nist.Cp_a : Cp0;
    const auto b = std::isfinite(nist.Cp_b) ? nist.Cp_b : 0.0;
    const auto c = std::isfinite(nist.Cp_c) ? nist.Cp_c : 0.0;

    // Calculate the integrals of the heal capacity function of the gas from Tr to T at constant pressure Pr
    const auto CpdT   = a*(T - Tr) + 0.5*b*(T*T - Tr*Tr) + c*log((T - T_theta) / (Tr - T_theta));
    const auto CpdlnT = a*log(T/Tr) + b*(T - Tr) + c*(log((T - T_theta) / (Tr - T_theta)) + log(Tr / T)) / T_theta;

    // Calculate the standard molal thermodynamic properties of the gas
    auto V  = R*T/P; // the ideal gas molar volume (in units of m3/mol), this is a harsh simplification since V is
                     // not provided by NIST. This could be improved in the future, but this value is not used
                     // in equilibrium calculations.
    auto G  = G0 - S0 * (T - Tr) + CpdT - T * CpdlnT;
    auto H  = H0 + CpdT;
    auto S  = S0 + CpdlnT;
    auto U  = H - P*V;
    auto A  = U - T*S;
    auto Cp = a + b*T + c/(T - T_theta);

    SpeciesThermoState state;
    state.volume           = V;
    state.gibbs_energy     = G;
    state.enthalpy         = H;
    state.entropy          = S;
    state.internal_energy  = U;
    state.helmholtz_energy = A;
    state.heat_capacity_cp = Cp;
    state.heat_capacity_cv = state.heat_capacity_cp; // approximate Cp = Cv for an aqueous solution

    return state;
}

auto aqueousSpeciesThermoStateNIST(Temperature T, Pressure P, const AqueousSpecies& species) -> SpeciesThermoState
{
    return genericSpeciesThermoStateNIST(T, P, species);
}

auto gaseousSpeciesThermoStateNIST(Temperature T, Pressure P, const FluidSpecies& species) -> SpeciesThermoState
{
    SpeciesThermoState state = genericSpeciesThermoStateNIST(T, P, species);
    const auto R = universalGasConstant;
    state.heat_capacity_cv = state.heat_capacity_cp - R;

    return state;
}

auto liquidSpeciesThermoStateNIST(Temperature T, Pressure P, const FluidSpecies& species) -> SpeciesThermoState
{
    return genericSpeciesThermoStateNIST(T, P, species);
}

auto mineralSpeciesThermoStateNIST(Temperature T, Pressure P, const MineralSpecies& species) -> SpeciesThermoState
{
    return genericSpeciesThermoStateNIST(T, P, species);
}

} // namespace Reaktoro
