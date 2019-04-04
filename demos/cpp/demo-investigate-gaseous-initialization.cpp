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

// This demo was made to investigate the initialization of Gaseous Phase.
// Which is direct related with the convergence of the problem, study why

#include <Reaktoro/Reaktoro.hpp>
using namespace Reaktoro;

auto solve(ChemicalEditor& editor) -> void {

	ChemicalSystem system(editor);

	EquilibriumProblem problem(system);
	problem.add("O2", 1, "umol");
	problem.add("CO2(g)", 1, "mol");
	problem.add("H2O(g)", 1, "mol");
	problem.add("H2S(g)", 1, "mol");
	problem.setTemperature(79, "degC");
	problem.setPressure(1, "bar");

	ChemicalState state(system);// = equilibrate(problem);

	EquilibriumSolver solver(system);

	auto res = solver.solve(state, problem);

	std::cout << "result: state " << (res.optimum.succeeded ? "converged" : "did not converge") << std::endl;
	std::cout << state << std::endl;
}

int main()
{
	Database db("supcrt98.xml");

	ChemicalEditor editor1(db);
	editor1.addAqueousPhaseWithElements("H O C S");
	editor1.addGaseousPhase({"H2O(g)", "CO2(g)", "H2S(g)"});

	std::cout << "ChemicalEditor editor1(db); \n" <<
		"editor1.addAqueousPhaseWithElements(\"H O C S\"); \n" <<
		"editor1.addGaseousPhase({ \"H2O(g)\", \"CO2(g)\", \"H2S(g)\" });" << std::endl;
	solve(editor1);
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;


	ChemicalEditor editor2(db);
	editor2.addAqueousPhaseWithElements("H O C S");
	editor2.addGaseousPhase({  "CO2(g)", "H2O(g)", "H2S(g)" });

	std::cout << "ChemicalEditor editor1(db); \n" <<
		"editor1.addAqueousPhaseWithElements(\"H O C S\"); \n" <<
		"editor1.addGaseousPhase({ \"CO2(g)\", \"H2O(g)\", \"H2S(g)\" });" << std::endl;
	solve(editor2);
}
