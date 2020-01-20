from reaktoro import *

def test_Cubic_EOS_multiple_roots():
    """This problem leads to the following CubicEOS roots
    PR - Z1 = 1.00027728
         Z2 = 0.0001655
         Z3 = -0.0011024
    since bmix = 1.635e-05 -> Z3 is an invalid root 
    and since Z3 < Z2 < Z1 -> Z2 is an invalid root.
    Reaktoro should remove Z3, Z2 and proceed instead of removing only
    Z3 and raising the exception "Logic error: it was expected Z roots of size 3, but got: 2".    
    """
    
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
    
