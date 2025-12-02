"""
This script prepares the prior knowledge that can be provided to the agent.
"""

# First Version

intro = """
I will teach you about RASPA and how to perform various Monte Carlo (MC) simulations using it. I will cover the following topics:

- What is RASPA and what are its applications
- How to setup a simulation using the provided tools and environment
- Monte Carlo (MC) Simulations in general:
	- What is a MC and how to setup RASPA for these simulations
	- What are the available MC moves
    - Prerequisite simulations
- Specific MC simulation examples:
	- MC in one box: total energy, RDF, angle distributions
	- multiple components
	- MC on a framework: helium void fraction, Rosenbluth value, adsorption at infinite solution, adsorption isotherm, Henry coefficient, adsorption binary mixture, adsorption selectivities

Try not only to memorize the content, but also to understand the underlying principles and how to apply them in practice.
It is essential to build and connect different aspects during the teaching.
Try now to setup a robust knowledge structure with a lot of connections which you can extend during the teaching.

I will provide examples and explanations for each topic.
Dont ask for clarifications unless I ask you explitly (which I will do!).
"""


raspa_general = """
RASPA is a classical molecular simulation software specialized on simulations of porous systems and their interactions with liquids or gases.
RASPA allows for different kinds of simulations for various different purposes.
"""
tools_setup = """
To calculate properties with RASPA, FIRST ask you memory.
Important information are always: simulation input details, simulation prerequisites, output analysis.
If prerequisites are needed (e.g. helium void fraction or ideal rosenbluth weight), ALWAYS first concentrate on simulating these. Then, use the results in a next simulation!

For each individual simulation, you need to generate several files with your tools before running the simulation:
1. Molecules:
    - Identify the molecules (gas/liquid) to simulate.
    - <molecule loader tool>: Automatically generate molecular definition files and corresponding force field and pseudoatoms files for one or multiple molecules.
2. Box / framework:
    - Identify the system for the simulation: empty box or porous material (MOF, zeolite, ...).
    - If box: specify in the simulation.input file later
    - If material: <framwork loader tool>: Load the structure .cif files. If you cannot load a structure file, ask for it.
3. simulation input file:
    - Identify the goal of the simulation and ask your memory to find the required settings.
    - <input file tool>: based on the template in the tool description, generate a input file using knowledge from your memory!
4. Run the simulation:
    <execute raspa tool>: run the simulation and automatically generate a new, empty folder for the next simulation.
5. Output:
    - <output tool>: parse relevant information from the *.output file and search for the required properties
    - Some simulations generate additional folders with properties. You can inspect them if you want but else you can ignore the content mostly.
"""


raspa_mc_general = """This is a quick overview of Monte Carlo simulations for molecular simulations:
**Monte Carlo simuations**
Monte Carlo (MC) simulations sample the distribution of states of a systems to determine average, thermodynamic properties.
The transitions between the states are random and have to be chosen to be able to reach every possible state of the system.
In practice, these transitions are called MC moves which typically include the translation, rotation, insertion/deletion of particles for NVT, NpT ensembles or GCMC.
The selection of MC moves implicitly defines the ensemble for the system.

**Configurational bias MC (CBMC)**
For ALL molecules with torsions/bends, reinsertion moves are unlikely to succeed.
CBMC introduces a much more effictive alternative (in RASPA: PartialReinsertionProbability/CBMBProbability) for these insertions.
To use this, the ideal gas Rosenbluth weight is required as an additional input (see prerequisite simulations later) to correctly use these biased moves.
This value is 1 for small molecules and exponentially decreases for molecules with a lot of torsions.
"""


moves_additional = """\n\n
Here are additional expert annotations for the MC moves to consider:
- TranslationProbability: Translation Probability can be safely used for all simulations in a framework
- RotationProbability: Rotation Probability can be safely used for all simulations in a framework
- CBMCProbability / PartialReinsertionProbability: CBMC Probability can be safely used for all simulations involving molecules with any torsion parameters in the molecule definition file
- ReinsertionProbability: Reinsertion Probability can be safely used for all simulations in a framework
- SwapProbability: Swap Probability can be safely used for all simulations to calculate adsorption isotherms using GCMC simulations
- WidomProbability: WidomProbability can be safely used for all simulations to calculate Henry’s constant, IdealGasRosenbluthWeight, and helium void fraction
- IdentityChangeProbability: IdentitySwapProbability can be safely used for all simulations to calculate adsorption isotherms of mixtures(more than one component) using GCMC simulations
"""


prerequisites_details = """**Prerequite simulations**
Two types of prerequisites are essential to consider before EVERY simulation.

HeliumVoidFraction:
ALWAYS if a framework is specified.
This values need to be added to every framework definition.
It is simulated by Widom insertions of Helium on the framework.
The HeliumVoidFraction (between 0 and 1) corresponds to the fraction of empty space.
Its value corresponds to the average widom rosenbluth weight in the RASPA output!

IdealGasRosenbluthWeight:
ALWAYS if CBMC is used (see above).
This values need to be added to every compound definition with CBMC moves.
It is simulated by Widom insertions of the individual compound in a box.
This simulation can have multiple compounds in parallel since Widom insertions only sample the chemical potential without really inserting a particle.
The IdealGasRosenbluthWeight (between 0 and 1) corrects the MC move acceptance probabilities if biasing is used.
Molecules without rotations (e.g. methane) have a value of 1 which is exponentially decreasing to 0 for larger molecules.
Its value corresponds to the average widom rosenbluth weight in the RASPA output!

IMPORTANT: examples for both simulations will be provided later!
"""


"""Details regarding the duration setting of a simulation input file:
<keyword>
<name>NumberOfCycles </name>
<type>[int]</type>
<description>
\
The number of cycles for the production run.
For Monte Carlo a cycle consists of $N$ steps, where $N$ is the amount of
molecules with a minimum of 20 steps. This means that on average during each cycle on each molecule a
Monte Carlo move has been attempted (either successful or unsuccessful). For MD the number of cycles
is simply the amount of integration steps.
</description>
    </keyword>

<keyword>
<name>NumberOfInitializationCycles </name>
<type>[int]</type>
<description>
\
The number of cycles used to initialize the system using Monte Carlo. This can be used for both Monte Carlo
as well as Molecular Dynamics to quickly equilibrate the positions of the atoms in the system.
</description>
    </keyword>

<keyword>
<name>NumberOfEquilibrationCycles </name>
<type>[int]</type>
<description>
\
For Molecular Dynamics  it is the number of MD steps to equilibrate the velocities in the systems. After this
equilibration the production run is started. For Monte Carlo, in particular CFMC, the equilibration-phase is used
to measure the biasing factors.
</description>
    </keyword>
"""


"""This is a list of molecule properties and monte carlo movescan be assigned to each molecule/component in the simulation.input file to specify the simulation:
<keyword>
<name>Component </name>
<type>[int] MoleculeName [string]</type>
<description>
\
Reads in the definition of component [int] using the file `\emph{molecule-name-string}.def' from the
directory `\$\{RASPA\_DIR\}/share/raspa/molecules/\emph{molecule-definitions-string}'.
</description>
    </keyword>

<keyword>
<name>MoleculeDefinitions </name>
<type>[string]</type>
<description>
\
The type of the molecule. For example, there could an OPLS version of the molecule, or a TraPPE version, etc. This \emph{molecule-definitions-string} is actually the directory name
under which the molecule file is found in `\$\{RASPA\_DIR\}/share/raspa/molecules/'.
</description>
    </keyword>

<keyword>
<name>MolFraction </name>
<type>[real]</type>
<description>
\
The mol fraction of this component in the mixture. The values can be specified relative to other components, as the fractions are normalized afterwards.
The partial pressures for each component are computed from the total pressure and the mol fraction per component.
</description>
    </keyword>

<keyword>
<name>FugacityCoefficient </name>
<type>[real]</type>
<description>
\
The fugacity coefficient for the current component. For values 0 (or by not specifying this line), the fugacity coefficients are automatically computed using the Peng-Robinson
equation of state. Note the critical pressure, critical temperature, and acentric factor need to be specified in the molecule file.
</description>
    </keyword>

<keyword>
<name>IdealGasRosenbluthWeight </name>
<type>[real]</type>
<description>
\
The ideal Rosenbluth weight is the growth factor of the CBMC algorithm for a single chain in an empty box. The value only depends on temperature and therefore needs to be computed
only once. For adsorption, specifying the value in advance is convenient because the applied pressure does not need to be corrected afterwards (the Rosenbluth weight corresponds to a shift
in the chemical potential reference value, and the chemical potential is directly obtained from the fugacity). For equimolar mixtures this is essential.
</description>
    </keyword>

<keyword>
<name>GibbsSwapProbability </name>
<type>[real]</type>
<description>
\
The relative probability to attempt a Gibbs swap MC move for the current component. The `GibbsSwapMove' transfers a randomly selected particle from one box to the other
(50\% probability to transfer a particle from box I to II, an 50\% visa versa).
</description>
    </keyword>

<keyword>
<name>TranslationProbability </name>
<type>[real]</type>
<description>
\
The relative probability to attempt a translation move for the current component. A random displacement is chosen in the allowed directions (see `TranslationDirection').
Note that the internal configuration of the molecule is unchanged by this move. The maximum displacement is scaled during the simulation to achieve an acceptance
ratio of 50\%.
</description>
    </keyword>

<keyword>
<name>RandomTranslationProbability </name>
<type>[real]</type>
<description>
\
The relative probability to attempt a random translation move for the current component. The displacement is chosen such that any position in the box can reached. It is therefore
similar as reinsertion, but `reinsertion' changes the internal conformation of a molecule and uses biasing.
</description>
    </keyword>

<keyword>
<name>RotationProbability </name>
<type>[real]</type>
<description>

The relative probability to attempt a random rotation move for the current component. The rotation is around the starting bead. A random vector on a sphere
is generated, and the rotation is random around this vector.
</description>
    </keyword>

<keyword>
<name>CBMCProbability </name>
<type>[real]</type>
<description>
\
The relative probability to attempt a partial reinsertion move for the current component. Part of the molecule is regrown, while part of the molecule can remain fixed.
The list of partial reinsertion moves is specified in the `molecule.def' file.
</description>
    </keyword>

<keyword>
<name>ReinsertionProbability </name>
<type>[real]</type>
<description>
\
The relative probability to attempt a full reinsertion move for the current component. Multiple first beads are chosen, and one of these is selected according to its Boltzmann weight.
The remaining part of the molecule is grown using biasing. This move is very useful, and often necessary, to change the internal configuration of flexible molecules.
</description>
    </keyword>

<keyword>
<name>SwapProbability </name>
<type>[real]</type>
<description>
\
The relative probability to attempt a insertion or deletion move. Whether to insert or delete is decided randomly with a probability of 50\% for each.
The swap move imposes a chemical equilibrium between the system and an imaginary particle reservoir for the current component. The move starts with multiple first bead, and
grows the remainder of the molecule using biasing.
</description>
    </keyword>

<keyword>
<name>WidomProbability </name>
<type>[real]</type>
<description>
\
The relative probability to attempt a Widom particle insertion move for the current component. The Widom particle insertion moves measure the chemical potential
and can be directly related to Henry coefficients and heats of adsorption.
</description>
    </keyword>

<keyword>
<name>SurfaceAreaProbability </name>
<type>[real]</type>
<description>
\
The relative probability to attempt a surface-area move for the current component.
</description>
    </keyword>

<keyword>
<name>ReinsertionInPlaceProbability </name>
<type>[real]</type>
<description>
\
The relative probability to attempt a reinsertion-in-place move for the current component. The reinsertion position is the current position of the starting bead of the randomly selected
molecule. Alternatively, one can use the partial reinsertion move leaving one bead fixed. The move is very useful to sample configuration on a plane for dcTST to change
the internal configuration, e.g. bonds, bends, torsions, etc.
</description>
    </keyword>

<keyword>
<name>IdentityChangeProbability </name>
<type>[real]</type>
<description>
\
The relative probability to attempt an identity-change move for the current component. A molecule of type $A$ is reinsertion, in the same place as the starting bead of $A$, as type $B$ using
the starting bead of component $B$. The $A-B$ list is defined using `IdentityChangesList' defining $B$ for each component $A$, i.e. the current component can be reinserted into
any component defined in the `IdentityChangesList' list, and from that list the component is chosen randomly.
</description>
    </keyword>

<keyword>
<name>NumberOfIdentityChanges </name>
<type>[int]</type>
<description>
\
The number of `IdentityChangesList' elements for the current component.
</description>
    </keyword>

<keyword>
<name>IdentityChangesList </name>
<type>[list-of-int]</type>
<description>
\
The list of components that the current component can be changed into. The identity-change move will randomly choose the new component from this list.
</description>
    </keyword>

<keyword>
<name>GibbsIdentityChangeProbability </name>
<type>[real]</type>
<description>
\
The relative probability to attempt an identity change for the current component in the Gibbs ensemble.
It is a very useful move to for mixture of $n$ components. Out of the $n$ components, two components $i\not=j$ are selected at random.
At random, it is selected to switch the identity of component $i$
in box $I$ or in box $II$, and the identity of the component $j$ in the other box.
In each box, a particle is selected at random which matches the desired identity.
</description>
    </keyword>

<keyword>
<name>NumberOfGibbsIdentityChanges </name>
<type>[int]</type>
<description>
\
The number of `GibbsIdentityChangesList' elements for the current component.
</description>
    </keyword>

<keyword>
<name>GibbsIdentityChangesList </name>
<type>[list-of-int]</type>
<description>
\
The list of components that the current component can be changed into. The Gibbs-identity-change move will randomly choose the new component from this list.
</description>
    </keyword>

<keyword>
<name>CreateNumberOfMolecules </name>
<type>[int]</type>
<description>
\
The number of molecule to create for the current component. Note these molecules are \emph{in addition} to anything read in by using a restart-file. Usually, when the restart-file
is used the amount here should be put back to zero. A warning, putting this value unreasonably high results in an infinite loop. The routine accepts molecules that are grown causing
no overlap (energy smaller than `EnergyOverlapCriteria'). Also the initial starting configurations are far from optimal and substantial equilibration is needed to reduce the energy.
However, the CBMC growth is able to reach very high densities.
</description>
    </keyword>


Here are additional expert annotations for the MC moves to consider:
- TranslationProbability: Translation Probability can be safely used for all simulations in a framework
- RotationProbability: Rotation Probability can be safely used for all simulations in a framework
- CBMCProbability / PartialReinsertionProbability: CBMC Probability can be safely used for all simulations involving molecules with any torsion parameters in the molecule definition file
- ReinsertionProbability: Reinsertion Probability can be safely used for all simulations in a framework
- SwapProbability: Swap Probability can be safely used for all simulations to calculate adsorption isotherms using GCMC simulations
- WidomProbability: WidomProbability can be safely used for all simulations to calculate Henry’s constant, IdealGasRosenbluthWeight, and helium void fraction
- IdentityChangeProbability: IdentitySwapProbability can be safely used for all simulations to calculate adsorption isotherms of mixtures(more than one component) using GCMC simulations
"""


"""
This is a list of properties and their settings that RASPA can calculate in a simualtion. They will produce extra folders with files specifying the format of the property output:
<keyword>
<name>ComputeNumberOfMoleculesHistogram </name>
<type>[yes|no]</type>
<description>
<name>WriteNumberOfMoleculesHistogramEvery </name>
<type>[int]</type>
<description>
\
Output the histogram every [int] cycles.
</description>


<name>NumberOfMoleculesRange </name>
<type>[real]</type>
<description>
\
The range of the histograms.
</description>


<name>NumberOfMoleculesHistogramSize </name>
<type>[int]</type>
<description>
\
The number of elements of the histograms.
</description>

</description>
    </keyword>

<keyword>
<name>ComputeDistanceHistograms </name>
<type>[yes|no]</type>
<description>
<name>WriteDistanceHistogramEvery </name>
<type>[int]</type>
<description>
\
Output the distance histograms every [int] cycles.
</description>


<name>MaxRangeDistanceHistogram </name>
<type>[real]</type>
<description>
\
The range of the histograms.
</description>


<name>NumberOfElementsDistanceHistogram </name>
<type>[int]</type>
<description>
\
The number of elements of the histograms.
</description>


<name>DistanceHistogramDefinition </name>
<type>[F|A|C] [int] [int] [F|A|C] [int] [int]</type>
<description>
\
Define a distance histogram between two atoms.
</description>

</description>
    </keyword>

<keyword>
<name>ComputeBendAngleHistograms </name>
<type>[yes|no]</type>
<description>
<name>WriteBendAngleHistogramEvery </name>
<type>[int]</type>
<description>
\
Output the distance histograms every [int] cycles.
</description>


<name>MaxRangeBendAngleHistogram </name>
<type>[real]</type>
<description>
\
</description>


<name>NumberOfElementsBendAngleHistogram </name>
<type>[int]</type>
<description>
\
</description>


<name>BendAngleHistogramDefinition </name>
<type>[F|A|C] [int] [int] [F|A|C] [int] [int] [F|A|C] [int] [int]</type>
<description>
\
</description>

</description>
    </keyword>

<keyword>
<name>ComputeDihedralAngleHistograms </name>
<type>[yes|no]</type>
<description>
<name>WriteDihedralAngleHistogramEvery </name>
<type>[int]</type>
<description>
\
Output the distance histograms every [int] cycles.
</description>


<name>MaxRangeDihedralAngleHistogram </name>
<type>[real]</type>
<description>
\
</description>


<name>NumberOfElementsDihedralAngleHistogram </name>
<type>[int]</type>
<description>
\
</description>


<name>DihedralAngleHistogramDefinition \small{</name>
<type>[F|A|C] [int] [int] [F|A|C] [int] [int] [F|A|C] [int] [int] [F|A|C] [int] [int]}</type>
<description>
\
</description>

</description>
    </keyword>

<keyword>
<name>ComputeAngleBetweenPlanesHistograms </name>
<type>[yes|no]</type>
<description>
<name>WriteAngleBetweenPlanesHistogramEvery </name>
<type>[int]</type>
<description>
\
Output the distance histograms every [int] cycles.
</description>


<name>MaxRangeAngleBetweenPlanesHistogram </name>
<type>[real]</type>
<description>
\
</description>


<name>NumberOfElementsAngleBetweenPlanesHistogram </name>
<type>[int]</type>
<description>
\
</description>


<name>AngleBetweenPlanesHistogramDefinition </name>
<type>[F|A|C] [int] [int] [F|A|C] [int] [int] [F|A|C] [int] [int]
        [F|A|C] [int] [int] [F|A|C] [int] [int] [F|A|C] [int] [int]</type>
<description>
\
</description>

</description>
    </keyword>

<keyword>
<name>ComputePSD </name>
<type>[yes|no]</type>
<description>
<name>WritePSDEvery </name>
<type>[int]</type>
<description>
\
Output the PSD every [int] cycles.
</description>


<name>PSDProbeDistance </name>
<type>[Minimum|Sigma]</type>
<description>
\
Sets whether to use the minimum of the potential $\sigma^{1/6}$ as the probe distance or whether to use $\sigma$.
</description>


<name>HistogramSizePoreSizeDistribution </name>
<type>[int]</type>
<description>
\
default: 100.
</description>


<name>MaxRangePoreSizeDistribution </name>
<type>[real]</type>
<description>
\
default: 10.
</description>

</description>
    </keyword>

<keyword>
<name>ComputeRDF </name>
<type>[yes|no]</type>
<description>
<name>WriteRDFEvery </name>
<type>[int]</type>
<description>
\
Output the RDF every [int] cycles.
</description>

</description>
    </keyword>

<keyword>
<name>ComputeMSD </name>
<type>[yes|no]</type>
<description>
<name>WriteMSDEvery </name>
<type>[int]</type>
<description>
\
     Output the MSD every [int] cycles.
</description>


<name>SampleMSDEvery </name>
<type>[int]</type>
<description>
\
    Samples every [int] integration steps.
</description>


<name>ComputeIndividualMSD </name>
<type>[yes|no]</type>
<description>
\
    Computes the msd, not only per component, but also per molecule.
</description>


<name>NumberOfBlocksMSD </name>
<type>[int]</type>
<description>
\
    The  number of blocks for the order-$n$ correlation measurement. Each block represent a different time-scale of sampling.
</description>


<name>NumberOfBlockElementsMSD </name>
<type>[int]</type>
<description>
\
    The number of elements in each block. For example, if the number is 10, then the first block samples: $1,2,3,\dots,10$, the second block
    $10,20,30,\dots,100$, the third block $100,200,300,\dots,1000$, etc.
</description>

</description>
    </keyword>

<keyword>
<name>ComputeDensityHistograms </name>
<type>[yes|no]</type>
<description>
\
Sets whether or not to compute a density histogram for the current system. For example, during adsorption it keeps track of the amount of molecules.
</description>
    </keyword>

<keyword>
<name>ComputeEnergyHistogram </name>
<type>[yes|no]</type>
<description>
<name>WriteEnergyHistogramEvery </name>
<type>[int]</type>
<description>
\
Sets to print the energy histogram of the system every [int] cycles.
</description>


<name>EnergyHistogramSize </name>
<type>[int]</type>
<description>
\
Sets the number of elements of the histogram.
</description>


<name>EnergyHistogramLowerLimit </name>
<type>[real]</type>
<description>
\
Sets the lower limit of the histogram.
</description>

<name>EnergyHistogramUpperLimit </name>
<type>[real]</type>
<description>
\
Sets the upper limit of the histogram.
</description>

</description>
    </keyword>

<keyword>
<name>ComputeEndToEndDistanceHistograms </name>
<type>[yes|no]</type>
<description>
\
Sets whether or not to compute a histogram for end-to-end distances of molecules for the current system.
</description>
    </keyword>

<keyword>
<name>ComputeMoleculeProperties </name>
<type>[yes|no]</type>
<description>
\
Sets whether or not to compute properties of molecules like average bond-lengths, average bend-angles etc. for the current system.
</description>
    </keyword>

<keyword>
<name>PrintMoleculePropertiesEvery </name>
<type>[int]</type>
<description>
\
Sets to print the properties of molecules every [int] cycles.
</description>
    </keyword>

<keyword>
<name>ComputeSurfaceArea </name>
<type>[yes|no]</type>
<description>
<name>SurfaceAreaProbeAtom </name>
<type>[string]</type>
<description>
\
</description>


<name>SurfaceAreaSamplingPointsPerSphere </name>
<type>[int]</type>
<description>
\
Sets the number of points to sampling a sphere per iteration.
</description>


<name>SurfaceAreaProbeDistance </name>
<type>[Minimum|Sigma]</type>
<description>
\
Sets whether to use the minimum of the potential $\sigma^{1/6}$ as the probe distance or whether to use $\sigma$.
</description>

</description>
    </keyword>
"""


"""
These are relevant simulation input parameters when using an empty box.
<keyword>
<name>$\begin{array}{l}\text{Box </name>
<type>[int]}\\ \text{[real] [real] [real]}\end{array}</type>
<description>
\
Set the system [int] to type `Box' (other option is `Framework' when a framework is present).
The cell dimensions of rectangular box of system [int] in Angstroms.
</description>
    </keyword>

<keyword>
<name>$\begin{array}{l}\text{BoxAngles </name>
<type>[int]}\\ \text{[real] [real] [real]}\end{array}</type>
<description>
\
Set the system [int] to type `Box' (other option is `Framework' when a framework is present).
The cell angles of rectangular box of system [int] in Angstroms.
</description>
    </keyword>
"""

"""These are relevant simulation input parameters when using a framework/material via a .cif file.
<keyword>
<name>Framework </name>
<type>[int]</type>
<description>
\
Set the system [int] to type `Framework' (other option is `Box' when no framework is present).
All other options listed in the section framework parameters refer to this system, so make sure this is before any other framework options.
</description>
    </keyword>

<keyword>
<name>FrameworkName </name>
<type>[string]</type>
<description>
\
Loads the framework with name [string]. Several frameworks can be read per system,
which is useful for to study interpenetration of frameworks. Here the frameworks
are allowed to move independently from each other.
</description>
    </keyword>

<keyword>
<name>HeliumVoidFraction </name>
<type>[real]</type>
<description>
\
The void fraction as measure by probing the structure with helium a room temperature. This quantity has to be obtained from a separate
simulation and is essential to compute the \emph{excess}-adsorption during the simulation.
</description>
    </keyword>

<keyword>
<name>UnitCells </name>
<type>[int] [int] [int]</type>
<description>
\
The number of unit cells in x,y, and z direction for the system. The full cell will contain the unit cells, and periodic boundary conditions
will be applied on the box level (\emph{not} on a unit cell level).
</description>
    </keyword>

<keyword>
<name>FrameworkDefinitions </name>
<type>[string]</type>
<description>
\
The force field name [string] of the flexible framework. The file is read even when `FlexibleFramework no' is specified (the reason is that
framework bond-dipoles are defined using the `framework.def' file).
</description>
    </keyword>

<keyword>
<name>AddAtomNumberCodeToLabel </name>
<type>[yes|no]</type>
<description>
\
Writing structure-files: the number is added to the framework atom-types, e.g.\ `O' are mapped to `O1', `O2', `O3', etc.
</description>
    </keyword>
"""


# Second Version

raspa_knowledge = {}

raspa_knowledge[
    "intro"
] = """You will learn how to use RASPA to perform various Monte Carlo (MC) simulations.
You have access to a general overview of these aspects:
- How to setup the simulation environment using the provided tools
- Monte Carlo (MC) simulations in RASPA (MC moves)
- Specific MC simulation workflows
"""


raspa_knowledge[
    "setup"
] = """Each single simulation requires the following steps:
1. Load molecule definition file(s)
2. Load framework cif, if the simulation is not in an empty box
3. Setup simulation input file using details from memory
4. Execute RASPA
5. Output analysis using details from memory
"""

raspa_knowledge[
    "mc"
] = """**Monte Carlo (MC)** simulations sample the distribution of states of a systems to determine average, thermodynamic properties.
The transitions between the states are random and are implemented as MC moves (e.g. translation, rotation, insertion/deletion of particles).
The selection of MC moves implicitly defines the thermodynamic ensemble for the system.

**Configurational bias MC (CBMC)** is applied for ALL molecules with torsions/bends.
Since reinsertion moves are unlikely to succeed, partial reinsertion moves are used instead.
For accurate sampling, the ideal gas Rosenbluth weight is required as an additional input.
This value is 1 for small molecules and exponentially decreases for molecules with a lot of torsions.
"""
raspa_knowledge[
    "mc_moves"
] = """**Detailed MC moves**

<keyword>
    <name>MolFraction </name>
    <type>[real]</type>
    <description>
    The mol fraction of this component in the mixture. The values can be specified relative to other components, as the fractions are normalized afterwards.
    The partial pressures for each component are computed from the total pressure and the mol fraction per component.
    </description>
</keyword>

<keyword>
    <name>FugacityCoefficient </name>
    <type>[real]</type>
    <description>
    The fugacity coefficient for the current component. For values 0 (or by not specifying this line), the fugacity coefficients are automatically computed using the Peng-Robinson
    equation of state. Note the critical pressure, critical temperature, and acentric factor need to be specified in the molecule file.
    </description>
</keyword>

<keyword>
    <name>IdealGasRosenbluthWeight </name>
    <type>[real]</type>
    <description>
    The ideal Rosenbluth weight is the growth factor of the CBMC algorithm for a single chain in an empty box. The value only depends on temperature and therefore needs to be computed
    only once. For adsorption, specifying the value in advance is convenient because the applied pressure does not need to be corrected afterwards (the Rosenbluth weight corresponds to a shift
    in the chemical potential reference value, and the chemical potential is directly obtained from the fugacity). For equimolar mixtures this is essential.
    </description>
</keyword>

<keyword>
    <name>TranslationProbability </name>
    <type>[real]</type>
    <description>
    The relative probability to attempt a translation move for the current component. A random displacement is chosen in the allowed directions (see `TranslationDirection').
    Note that the internal configuration of the molecule is unchanged by this move. The maximum displacement is scaled during the simulation to achieve an acceptance
    ratio of 50\%.
    </description>
</keyword>

<keyword>
    <name>RandomTranslationProbability </name>
    <type>[real]</type>
    <description>
    The relative probability to attempt a random translation move for the current component. The displacement is chosen such that any position in the box can reached. It is therefore
    similar as reinsertion, but `reinsertion' changes the internal conformation of a molecule and uses biasing.
    </description>
</keyword>

<keyword>
    <name>RotationProbability </name>
    <type>[real]</type>
    <description>
    The relative probability to attempt a random rotation move for the current component. The rotation is around the starting bead. A random vector on a sphere
    is generated, and the rotation is random around this vector.
    </description>
</keyword>

<keyword>
    <name>PartialReinsertionProbability</name>
    <type>[real]</type>
    <description>
    The relative probability to attempt a partial reinsertion move for the current component. Part of the molecule is regrown, while part of the molecule can remain fixed.
    The list of partial reinsertion moves is specified in the `molecule.def' file.
    </description>
</keyword>

<keyword>
    <name>ReinsertionProbability </name>
    <type>[real]</type>
    <description>
    The relative probability to attempt a full reinsertion move for the current component. Multiple first beads are chosen, and one of these is selected according to its Boltzmann weight.
    The remaining part of the molecule is grown using biasing. This move is very useful, and often necessary, to change the internal configuration of flexible molecules.
    </description>
</keyword>

<keyword>
    <name>SwapProbability </name>
    <type>[real]</type>
    <description>
    The relative probability to attempt a insertion or deletion move. Whether to insert or delete is decided randomly with a probability of 50\% for each.
    The swap move imposes a chemical equilibrium between the system and an imaginary particle reservoir for the current component. The move starts with multiple first bead, and
    grows the remainder of the molecule using biasing.
    </description>
</keyword>

<keyword>
    <name>WidomProbability </name>
    <type>[real]</type>
    <description>
    The relative probability to attempt a Widom particle insertion move for the current component. The Widom particle insertion moves measure the chemical potential
    and can be directly related to Henry coefficients and heats of adsorption.
    </description>
</keyword>

<keyword>

<keyword>
    <name>IdentityChangeProbability </name>
    <type>[real]</type>
    <description>
    The relative probability to attempt an identity-change move for the current component. A molecule of type $A$ is reinsertion, in the same place as the starting bead of $A$, as type $B$ using
    the starting bead of component $B$. The $A-B$ list is defined using `IdentityChangesList' defining $B$ for each component $A$, i.e. the current component can be reinserted into
    any component defined in the `IdentityChangesList' list, and from that list the component is chosen randomly.
    </description>
</keyword>

<keyword>
    <name>NumberOfIdentityChanges </name>
    <type>[int]</type>
    <description>
    The number of `IdentityChangesList' elements for the current component.
    </description>
</keyword>

<keyword>
    <name>IdentityChangesList </name>
    <type>[list-of-int]</type>
    <description>
    The list of components that the current component can be changed into. The identity-change move will randomly choose the new component from this list.
    </description>
</keyword>

<keyword>
    <name>CreateNumberOfMolecules </name>
    <type>[int]</type>
    <description>
    The number of molecule to create for the current component. Note these molecules are \emph{in addition} to anything read in by using a restart-file. Usually, when the restart-file
    is used the amount here should be put back to zero. A warning, putting this value unreasonably high results in an infinite loop. The routine accepts molecules that are grown causing
    no overlap (energy smaller than `EnergyOverlapCriteria'). Also the initial starting configurations are far from optimal and substantial equilibration is needed to reduce the energy.
    However, the CBMC growth is able to reach very high densities.
    </description>
</keyword>
"""
raspa_knowledge[
    "mc_moves_expert"
] = """Here are additional expert annotations for the MC moves to consider:
- TranslationProbability can be safely used for all simulations in a framework
- RotationProbability can be safely used for all simulations in a framework
- PartialReinsertionProbability can be safely used for all simulations involving molecules with any torsion parameters in the molecule definition file
- ReinsertionProbability can be safely used for all simulations in a framework
- SwapProbability can be safely used for all simulations to calculate adsorption isotherms using GCMC simulations
- WidomProbability can be safely used for all simulations to calculate Henry’s constant, IdealGasRosenbluthWeight, and helium void fraction
- IdentityChangeProbability can be safely used for all simulations to calculate adsorption isotherms of mixtures(more than one component) using GCMC simulations
"""


if __name__ == "__main__":
    import json

    path = "knowledge.json"
    json.dump(raspa_knowledge, open(path, "w"), indent=4)
