# Hint Dictionaries
import json

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
hint_multi = "This is a multi-step simulation. You first have to determine properties such as the helium void fraction or ideal gas Rosenbluth weight in prior simulations, then use it for the final simulation to compute the desired property."

hints = {
    "henry": hint_henry + hint_hvf_add,
    "hvf": hint_hvf,
    "surface": hint_surface,
    "ads_dil": hint_diluted,  # + hint_hvf_add + hint_hvf, # TODO: check if necessary!
    "ads_iso": hint_ads + hint_hvf_add + hint_hvf,
    "sl": hint_ads_n2,
    "l": hint_cbmc + hint_rosenbluth,
    "multi": hint_multi,
}

"""
instructions_1 = {
    0: hint_hvf,
    1: hint_surface,
    2: hint_diluted,
    3: hint_ads + hint_hvf_add + hint_hvf,
    4: hint_henry + hint_hvf_add + hint_cbmc,
    5: hint_cbmc + hint_rosenbluth,
    6: hint_diluted + hint_cbmc + hint_rosenbluth + hint_hvf_add + hint_hvf,
    7: hint_ads + hint_cbmc + hint_rosenbluth + hint_hvf_add + hint_hvf,
    8: hint_henry + hint_cbmc + hint_rosenbluth + hint_hvf_add + hint_hvf,
}

instructions_n = {
    0: hint_ads_n2 + hint_cbmc + hint_rosenbluth + hint_hvf_add + hint_hvf,
    1: hint_henry + hint_cbmc + hint_rosenbluth + hint_hvf_add + hint_hvf,
}
"""

if __name__ == "__main__":
    for key, hint in hints.items():
        print(f"--- {key} ---")
        print(hint)

    json.dump(hints, open("hints.json", "w"), indent=4)
