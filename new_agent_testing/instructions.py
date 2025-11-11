'''
This file contains the tasks and instructions for new agent testing.
'''


### Task Instructions
base_instruction = """Answer this question using simulations: """ #(ALWAYS USE 1/100 cycles and up to 5 molecules for speed.IGNORE the low accuracy!): 

mol_s = "methane" #molecules_s = ["CO2", "N2", "methane", "ethane"]
mol_l = "n-pentane"#molecules_l = ["n-pentane", "n-hexane", "n-heptane"]
rosenbluth = "0.0197439" #rosenbluth = ["0.0197439", "0.0029442", "0.0004450"] # from Aastha
framework = "IRMOF-13"
f_hvf = 0.877

parameters = {
    "framework" : framework,
    "hvf" : f_hvf,
    "molecule" : mol_s,
    "molecule_l" : mol_l,
    "rosenbluth" : rosenbluth,
}

### Multistep Tasks
ads_dil = "Determine the adsorption enthalpy of {molecule} on {framework} using a simulation at infinite dilution"
ads_dil_l = "Determine the adsorption enthalpy of {molecule_l} on {framework} using a simulation at infinite dilution"  
ads_1 = "Determine the adsorption enthalpy of {molecule} on {framework}"
ads_l = "Determine the adsorption enthalpy of {molecule_l} on {framework}"
ads_2 = "Compare the adsorption enthalpies of {molecule} and {molecule_l} on {framework}"
h = "Determine the henry coefficient of {molecule} on {framework}"
h_l = "Determine the henry coefficient of {molecule_l} on {framework}"
h_2 = "Determine the henry coefficient of {molecule} and {molecule_l} on {framework}"
tasks_multistep = [ads_dil, ads_1, ads_2, h, h_2]


### Single Step Tasks
add_hvf = " given the helium void fraction of {hvf}"
add_rb_1 = " and the ideal gas rosenbluth weight of {rosenbluth} for {molecule}"
hvf = "Calculate the helium void fraction of {framework}"
surface = "Determine the surface area of {framework}"
rosenbluth_1 = "Calculate the ideal Rosenbluth weights for {molecule_l}"

tasks_framework = [hvf, surface]                                                    # framework
tasks_n1_s = [i + add_hvf for i in [ads_dil, ads_1, h]]                             # molecule, framework, hvf
tasks_n1_l = [rosenbluth_1] + [i + add_hvf + add_rb_1 for i in [ads_dil_l, ads_l, h_l]] # molecule, framework, hvf
tasks_n2_sl = [i + add_hvf + add_rb_1 for i in [ads_2, h_2]]                        # molecule, molecule2, framework, hvf

def task_prompt(task, instruction=base_instruction, parameters=parameters):
    return instruction + task.format(**parameters)

tasks_1 = {i: task_prompt(task) for i, task in enumerate(tasks_framework + tasks_n1_s + tasks_n1_l)}
tasks_n = {i: task_prompt(task) for i, task in enumerate(tasks_n2_sl)}

### Instruction Dictionaries

steps = "NumberOfCycles                500\nNumberOfInitializationCycles  100"
widom_steps = "NumberOfCycles                500\nNumberOfInitializationCycles  0"

hint_hvf = f"""To compute the helium void fraction, set up a MC simulation with helium as the adsorbate. Use the following parameters:
{widom_steps}
Component 0 MoleculeName       helium
            MoleculeDefinition local
            WidomProbability   1.0
            CreateNumberOfMolecules 0

"""
hint_surface = f"""To compute the surface area of the framework, use the following parameters:
{steps}
Framework 0
FrameworkName framework
UnitCells [int] [int] [int]
SurfaceAreaProbeDistance Sigma

Component 0 MoleculeName             Argon
            MoleculeDefinition       local
            SurfaceAreaProbability   1.0
            CreateNumberOfMolecules  0

IMPORTANT: never use something like ComputeSurfaceArea yes or similar keywords. The instruction must be followed exactly.
"""

hint_rosenbluth = f"""To compute the ideal gas rosenbluth weight for a molecule, set up a MC simulation with the following parameters:
{widom_steps}
Component 0 MoleculeName              [molecule name]
            MoleculeDefinition        local
            WidomProbability          1.0
            CreateNumberOfMolecules   0
"""

hint_diluted = f"""To compute the adsorption enthalpy at infinite dilution, set up a MC simulation with the following parameters:
{steps}
Component 0 MoleculeName             [molecule name]
            MoleculeDefinition       local
            TranslationProbability   1.0
            ReinsertionProbability   1.0
            RotationProbability      1.0
            CreateNumberOfMolecules  1
"""

hint_ads = f"""To compute the adsorption enthalpy, set up a GCMC simulation with the following parameters:
{steps}
ComputeNumberOfMoleculesHistogram yes
ComputeEnergyHistogram yes

ExternalTemperature 298.0
ExternalPressure 1e3

Component 0 MoleculeName             [molecule name]
            MoleculeDefinition       local
		    IdealGasRosenbluthWeight [rosenbluth]
            TranslationProbability   0.5
		    RotationProbability      0.5
            ReinsertionProbability   0.5
		    PartialReinsertionProbability 1
            SwapProbability          1.0
            CreateNumberOfMolecules  0
"""

hint_ads_n2 = f"""To compute the adsorption enthalpy of two molecules, set up a GCMC simulation with the following parameters:
{steps}
ComputeNumberOfMoleculesHistogram yes
ComputeEnergyHistogram yes

Component 0 MoleculeName               [molecule name 0]
            MoleculeDefinition         local
	          IdealGasRosenbluthWeight   [rosenbluth]
            MolFraction                [fraction component 0]
            TranslationProbability     0.5
            RotationProbability.       0.5
            RegrowProbability          0.5
            IdentityChangeProbability  1.0
              NumberOfIdentityChanges  2
              IdentityChangesList      0 1
            SwapProbability            1.0
            CreateNumberOfMolecules    0

Component 1 MoleculeName               [molecule name 1]
            MoleculeDefinition         local
	          IdealGasRosenbluthWeight   [rosenbluth]
            MolFraction                [fraction component 1]
            TranslationProbability     0.5
	        RotationProbability	       0.5
            RegrowProbability          0.5
            IdentityChangeProbability  1.0
              NumberOfIdentityChanges  2
              IdentityChangesList      0 1
            SwapProbability            1.0
            CreateNumberOfMolecules    0
"""

hint_henry = f"""To compute the henry coefficient, set up a MC simulation with the following parameters:
{widom_steps}
Component 0 MoleculeName              [molecule name]
            MoleculeDefinition        local
            IdealGasRosenbluthWeight  [rosenbluth]
            WidomProbability          1.0
            CreateNumberOfMolecules   0
"""

hint_cbmc = """If a molecule has torsions, use CBMC for better sampling. This requires a ideal gas rosenbluth weight calculation first. Molecules without torsions have a rosenbluth weight of 1.0 (methane is treated as single-atom entity!)"""
hint_hvf_add = "Use the helium void fraction to correct the adsorption properties. Ignore for henry coefficient calculations." 

instructions_1 = {
    0 : hint_hvf,
    1 : hint_surface,
    2 : hint_diluted,
    3 : hint_ads + hint_hvf_add + hint_hvf,
    4 : hint_henry + hint_hvf_add + hint_cbmc,
    5 : hint_cbmc + hint_rosenbluth,
    6 : hint_diluted+hint_cbmc + hint_rosenbluth+hint_hvf_add + hint_hvf,
    7 : hint_ads+hint_cbmc + hint_rosenbluth+hint_hvf_add + hint_hvf,
    8 : hint_henry +hint_cbmc + hint_rosenbluth+hint_hvf_add + hint_hvf,
}

instructions_n = {
    0: hint_ads_n2 + hint_cbmc + hint_rosenbluth + hint_hvf_add + hint_hvf,
    1: hint_henry + hint_cbmc + hint_rosenbluth + hint_hvf_add + hint_hvf,
}