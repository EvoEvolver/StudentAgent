# Hint Dictionaries
import json

steps = "NumberOfCycles                5000\nNumberOfInitializationCycles  1000\n"
few_steps = "NumberOfCycles                500\nNumberOfInitializationCycles  100\n"
widom_steps = "NumberOfCycles                50000\nNumberOfInitializationCycles  0\n"

box = """Box 0
BoxLengths 30 30 30
"""

framework = """Framework 0
FrameworkName framework
UnitCells [int] [int] [int] # from framework loader tool
"""

T = "ExternalTemperature 300"
p = "ExternalPressure 1e5"

widom = """
            WidomProbability   1.0
            CreateNumberOfMolecules 0
"""

moves = """
            TranslationProbability   1.0
            RotationProbability      1.0
            ReinsertionProbability   1.0
            PartialReinsertionProbability 1.0
"""
swap = """
            SwapProbability          1.0
"""

mol = """
Component 0 MoleculeName             [molecule name]
            MoleculeDefinition local
"""
n = "            CreateNumberOfMolecules"

change = """            MolFraction                [fraction component i]
            IdentityChangeProbability  1.0
              NumberOfIdentityChanges  2
              IdentityChangesList      0 1
"""

hint_hvf = f"""To compute the helium void fraction, set up a MC simulation with helium as the adsorbate. Use the following parameters:
{widom_steps}
{framework}
{T}
{p}
{mol}
{widom}
"""

hint_rosenbluth = f"""To compute the ideal gas rosenbluth weight for a molecule, set up a MC simulation with the following parameters:
{widom_steps}
{box}
{T}
{mol}
{widom}
"""

hint_diluted = f"""To compute the adsorption enthalpy at infinite dilution, set up a MC simulation with the following parameters:
{steps}
{framework}
{T}
{mol}
{moves}
{swap}
{n} 1

IMPORTANT:
The enthalpy of adsorption can be estimated as: ∆H= ⟨Uhg⟩−⟨Uh⟩−⟨Ug⟩−RT
where ⟨Uhg⟩, ⟨Uh⟩, and ⟨Ug⟩are the average energy of the guest molecule inside the host-framework, the
average energy of the host-framework, and the average energy of a single guest-molecule in the gas phase,
respectively. The term RT is the enthalpy per particle of the ideal bulk phase. It accounts for the work to
push the gas adsorbates into the fluid phase when it desorbs.

For a rigid framework, ⟨Uh⟩ = 0.
For a rigid adsorbate, ⟨Ug⟩ = 0 (for flexible adsorbates, Ug has to be determined separately).
The RASPA output file provides ⟨Uhg⟩ as 'Total energy' which includes tail corrections (which are not applicable for only one molecule!).
Therefore, subtract the tail correction energy from the total energy (or directly add up 'Adsorbate-Adsorbate energy' and 'Host-Adsorbate energy'.
"""


hint_ads = f"""To compute the adsorption enthalpy, set up a GCMC simulation with the following parameters:
{steps}
{framework}
HeliumVoidFraction [real]
{p}
{T}
{mol}
{moves}
{swap}
{n} 0
"""

hint_ads_n2 = f"""To compute the adsorption enthalpy of two molecules, set up a GCMC simulation with the following parameters:
{steps}
{framework}
{p}
{T}
{mol}
{moves}
{swap}
{change}
{n} 0

{mol.replace("0","1")}
{moves}
{swap}
{change}
{n} 0

"""

hint_henry = f"""To compute the henry coefficient, set up a MC simulation with the following parameters:
{widom_steps}
{framework}
{mol}
{widom}
"""

hint_total = f"""To compute the total energy of a molecule in the gas phase, set up a NVT MC simulation in a box with the following parameters:
{steps}
{box}
{mol}
{moves}
{n} 1
"""

hint_large = """IMPORTANT: If a molecule has torsions, CBMC is used for better sampling.
This requires a ideal gas rosenbluth weight calculation first.
Molecules that are treated without torsions have a rosenbluth weight of 1.
Add the rosenbluth weight to the molecule definition in the format:

Component 0 MoleculeName             [molecule name]
            IdealGasRosenbluthWeight [real]
"""

hint_multi_iso = "IMPORTANT: The helium void fraction has to be obtained from a prior simulation to the simulation input!"
hint_multi = "IMPORANT: This is a multi-step simulation. You first have to determine properties such as the helium void fraction, ideal gas Rosenbluth weight or internal energy in prior simulations, then use it for the final simulation to compute the desired property."


# Assemble hints:

hints = {
    "total": hint_total,
    "rosenbluth": hint_rosenbluth,
    "henry": hint_henry,
    "hvf": hint_hvf,
    "ads_dil": hint_diluted,
    "ads_iso": hint_ads,
    "sl": hint_ads_n2,
    "l": hint_large,
    "multi": hint_multi,
    "multi_iso": hint_multi_iso,
}


def main():
    json.dump(hints, open("hints.json", "w"), indent=4)


if __name__ == "__main__":
    main()
