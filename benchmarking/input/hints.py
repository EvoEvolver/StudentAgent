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

The enthalpy of adsorption can be estimated as:
∆H= ⟨Uhg⟩−⟨Uh⟩−⟨Ug⟩−RT
where ⟨Uhg⟩, ⟨Uh⟩, and ⟨Ug⟩are the average energy of the guest molecule inside the host-framework, the
average energy of the host-framework, and the average energy of a single guest-molecule in the gas phase,
respectively. The term RT is the enthalpy per particle of the ideal bulk phase. It accounts for the work to
push the gas adsorbates into the fluid phase when it desorbs.

For a rigid framework, ⟨Uh⟩ = 0.
For a rigid adsorbate, ⟨Ug⟩ = 0.
The RASPA output file provides ⟨Uhg⟩ as 'Total energy' which includes tail corrections (which are not applicable for only one molecule!).
Therefore, subtract the tail correction energy from the total energy (or directly add up 'Adsorbate-Adsorbate energy' and 'Host-Adsorbate energy'.
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
            ReinsertionProbability      0.5
            PartialReinsertionProbability 0.5
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
            ReinsertionProbability      0.5
            PartialReinsertionProbability 0.5
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

hint_total = f"""To compute the total energy of a molecule in the gas phase, set up a NVT MC simulation in a box with the following parameters:
{widom_steps}
Component 0 MoleculeName              [molecule name]
            MoleculeDefinition        local
            
"""

hint_cbmc = """If a molecule has torsions, use CBMC for better sampling. This requires a ideal gas rosenbluth weight calculation first. Molecules without torsions have a rosenbluth weight of 1.0 (methane is treated as single-atom entity!)"""
hint_hvf_add = "Use the helium void fraction to correct the adsorption properties. Ignore for henry coefficient calculations."
hint_multi = "This is a multi-step simulation. You first have to determine properties such as the helium void fraction or ideal gas Rosenbluth weight in prior simulations, then use it for the final simulation to compute the desired property."


hints = {
    "total": hint_total + hint_cbmc + hint_rosenbluth,
    "henry": hint_henry + hint_hvf_add,
    "hvf": hint_hvf,
    "ads_dil": hint_diluted,  # + hint_hvf_add + hint_hvf, # TODO: check if necessary!
    "ads_iso": hint_ads + hint_hvf_add + hint_hvf,
    "sl": hint_ads_n2,
    "l": hint_cbmc + hint_rosenbluth,
    "multi": hint_multi,
}


if __name__ == "__main__":
    for key, hint in hints.items():
        print(f"--- {key} ---")
        print(hint)

    json.dump(hints, open("hints.json", "w"), indent=4)
