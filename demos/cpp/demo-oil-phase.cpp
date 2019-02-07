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

#include <Reaktoro/Reaktoro.hpp>
using namespace Reaktoro;

// NOTE: Copies from ChemicalEditor [PROTOTYPING]

auto lnActivityConstants(const AqueousPhase& phase) -> ThermoVectorFunction
{
    // The ln activity constants of the aqueous species
    ThermoVector ln_c(phase.numSpecies());

    // The index of solvent water species
    const Index iH2O = phase.indexSpeciesAnyWithError(alternativeWaterNames());

    // Set the ln activity constants of aqueous species to ln(55.508472)
    ln_c = std::log(1.0 / waterMolarMass);

    // Set the ln activity constant of water to zero
    ln_c[iH2O] = 0.0;

    ThermoVectorFunction f = [=](Temperature T, Pressure P) mutable {
        return ln_c;
    };

    return f;
}

auto lnActivityConstants(const GaseousPhase& phase) -> ThermoVectorFunction
{
    // The ln activity constants of the gaseous species
    ThermoVector ln_c(phase.numSpecies());

    ThermoVectorFunction f = [=](Temperature T, Pressure P) mutable {
        ln_c = log(P * 1e-5); // ln(Pbar)
        return ln_c;
    };

    return f;
}



// NOTE: Copy & Pasted from ChemicalEditor [PROTOTYPING]
template<typename PhaseType>
auto convertPhase(const PhaseType& phase, const Database& database, std::vector<double> const& pressures, std::vector<double> const& temperatures) -> Phase
{
    // The number of species in the phase
    const unsigned nspecies = phase.numSpecies();

    // Define the lambda functions for the calculation of the essential thermodynamic properties
    Thermo thermo(database);

    std::vector<ThermoScalarFunction> standard_gibbs_energy_fns(nspecies);
    std::vector<ThermoScalarFunction> standard_enthalpy_fns(nspecies);
    std::vector<ThermoScalarFunction> standard_volume_fns(nspecies);
    std::vector<ThermoScalarFunction> standard_heat_capacity_cp_fns(nspecies);
    std::vector<ThermoScalarFunction> standard_heat_capacity_cv_fns(nspecies);

    // Create the ThermoScalarFunction instances for each thermodynamic properties of each species
    for(unsigned i = 0; i < nspecies; ++i)
    {
        const std::string name = phase.species(i).name();

        standard_gibbs_energy_fns[i]     = [=](double T, double P) { return thermo.standardPartialMolarGibbsEnergy(T, P, name); };
        standard_enthalpy_fns[i]         = [=](double T, double P) { return thermo.standardPartialMolarEnthalpy(T, P, name); };
        standard_volume_fns[i]           = [=](double T, double P) { return thermo.standardPartialMolarVolume(T, P, name); };
        standard_heat_capacity_cp_fns[i] = [=](double T, double P) { return thermo.standardPartialMolarHeatCapacityConstP(T, P, name); };
        standard_heat_capacity_cv_fns[i] = [=](double T, double P) { return thermo.standardPartialMolarHeatCapacityConstV(T, P, name); };
    }

    // Create the interpolation functions for thermodynamic properties of the species
    ThermoVectorFunction standard_gibbs_energies_interp     = interpolate(temperatures, pressures, standard_gibbs_energy_fns);
    ThermoVectorFunction standard_enthalpies_interp         = interpolate(temperatures, pressures, standard_enthalpy_fns);
    ThermoVectorFunction standard_volumes_interp            = interpolate(temperatures, pressures, standard_volume_fns);
    ThermoVectorFunction standard_heat_capacities_cp_interp = interpolate(temperatures, pressures, standard_heat_capacity_cp_fns);
    ThermoVectorFunction standard_heat_capacities_cv_interp = interpolate(temperatures, pressures, standard_heat_capacity_cv_fns);
    ThermoVectorFunction ln_activity_constants_func         = lnActivityConstants(phase);

    // Define the thermodynamic model function of the species
    PhaseThermoModel thermo_model = [=](PhaseThermoModelResult& res, Temperature T, Pressure P)
    {
        // Calculate the standard thermodynamic properties of each species
        res.standard_partial_molar_gibbs_energies     = standard_gibbs_energies_interp(T, P);
        res.standard_partial_molar_enthalpies         = standard_enthalpies_interp(T, P);
        res.standard_partial_molar_volumes            = standard_volumes_interp(T, P);
        res.standard_partial_molar_heat_capacities_cp = standard_heat_capacities_cp_interp(T, P);
        res.standard_partial_molar_heat_capacities_cv = standard_heat_capacities_cv_interp(T, P);
        res.ln_activity_constants                     = ln_activity_constants_func(T, P);

        return res;
    };

    // Create the Phase instance
    Phase converted = phase;
    converted.setThermoModel(thermo_model);

    return converted;
}


int main()
{
    // Database db("supcrt07-organics.xml");
    Database db("supcrt98");

    // NOTE: Copy & Pasted from ChemicalEditor {
    // The default temperatures for the interpolation of the thermodynamic properties (in units of celsius)
    std::vector<double> temperatures{ 0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300 };

    // The default pressures for the interpolation of the thermodynamic properties (in units of bar)
    std::vector<double> pressures{ 1, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000 };

    // Convert the temperatures and pressures to units of kelvin and pascal respectively
    for (auto& x : temperatures) { x = x + 273.15; }
    for (auto& x : pressures) { x = x * 1.0e+5; }
    // }



    ChemicalEditor editor(db);
    editor.addAqueousPhase("H O C");

    //// Approach 1:
    //editor.addGaseousPhase({"H2O(g)", "CH4(g)" /* , "C2H6(g) " */ });
    //ChemicalSystem system(editor);

     // Approach 2:
    auto gas_species = std::vector<GaseousSpecies>{
        db.gaseousSpecies("H2O(g)"),
        db.gaseousSpecies("CH4(g)"),
    };
    auto mixture = GaseousMixture(gas_species);
    auto gas = GaseousPhase(mixture);

    std::vector<Phase> phases;
    phases.push_back(convertPhase(editor.aqueousPhase(), db, pressures, temperatures));
    phases.push_back(convertPhase(gas, db, pressures, temperatures));

    ChemicalSystem system(phases);


    EquilibriumProblem problem(system);
    //problem.setTemperature(4.0, "degC");
    //problem.setPressure(35.0, "bar");
    problem.add("H2O", 1, "kg");
    problem.add("CH4", 100, "g");

    ChemicalState state = equilibrate(problem);

    std::cout << state << std::endl;
}
