import pytest

from reaktoro import *

def test_Cubic_EOS_multiple_roots():
    
    database = Database("supcrt98.xml")

    editor = ChemicalEditor(database)
    editor.addAqueousPhaseWithElementsOf("H2O Fe(OH)2 Fe(OH)3 NH3")
    editor.addGaseousPhaseWithElementsOf("NH3")
    editor.addMineralPhase("Magnetite")

    system = ChemicalSystem(editor)
    
    state = ChemicalState(system)

    solver = EquilibriumSolver(system)

    Temperature = 298.15
    Pressure = 100000.0
    b = [3.0,
         122.01687012,
         1.0,
         63.50843506,
         0.0]
   
    solver.approximate(state, Temperature, Pressure, b)
    
    state.properties()
    