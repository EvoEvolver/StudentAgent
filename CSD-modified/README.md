### 1. CR_data_CSD_modified_20250227.csv: the information of CoRE MOF 2024 ASR, FSR and Ion datasets.
meaning of columns name:
*  number: index of all structures
*  coreid: CORE MOF ID, year + metal + topology + dimention + extension + No.
*  refcode: REFCODE name of CSD and Supporting Information
*  name: common name
*  mofid-v1: MOFid(https://pubs.acs.org/doi/full/10.1021/acs.cgd.9b01050)
*  mofid-v2: MOFid v2
*  LCD (Å): pore-limiting diameter by Zeo++(https://www.zeoplusplus.org/)
*  PLD (Å): largest cavity diameter by Zeo++(https://www.zeoplusplus.org/)
*  LFPD (Å): largest free pore diameter by Zeo++(https://www.zeoplusplus.org/)
*  Density (g/cm3): crystal density by Zeo++(https://www.zeoplusplus.org/)
*  ASA (A2), ASA (m2/cm3) and ASA (m2/g): accessible surface area by Zeo++(https://www.zeoplusplus.org/)
*  NASA (A2), NASA (m2/cm3) and NASA (m2/g): non-accessible surface area by Zeo++(https://www.zeoplusplus.org/)
*  PV (A3) and PV (cm3/g): pore volume by Zeo++(https://www.zeoplusplus.org/)
*  VF: void fraction by Zeo++(https://www.zeoplusplus.org/)
*  NAV (A3) and NPV (cm3/g): non-accessible pore volume by Zeo++(https://www.zeoplusplus.org/)
*  NAV_VF: non-accessible void fraction by Zeo++(https://www.zeoplusplus.org/)
*  structure_dimension: dimensionality of each framework by Zeo++(https://www.zeoplusplus.org/)
*  topology(SingleNodes): topology defined by single node(https://scipost.org/10.21468/SciPostChem.1.2.005)
*  topology(AllNodes):topology defined by all nodes(https://scipost.org/10.21468/SciPostChem.1.2.005)
*  catenation: number of nets
*  dimension_by_topo: dimension determined by CrystalNets.jl
*  hall: space group
*  number_spacegroup: space group by number
*  Metal Types: metal in the framework
*  Has OMS: contain open metal site or not
*  OMS Types: which metal is OMS
*  Charge: Charge source(https://pubs.acs.org/doi/10.1021/acs.jctc.4c00434)
*  average_atomic_mass: average atomic mass of structure
*  Heat_capacity@300K (J/g/K): machine learning predicted heat capacity at 300 K (https://www.nature.com/articles/s41563-022-01374-3)
*  std @ 300 K (J/g/K): the uncertainty of machine learning predicted heat capacity at 300 K (https://www.nature.com/articles/s41563-022-01374-3)
*  Heat_capacity@350K (J/g/K): machine learning predicted heat capacity at 350 K (https://www.nature.com/articles/s41563-022-01374-3)
*  std @ 350 K (J/g/K): the uncertainty of machine learning predicted heat capacity at 350 K (https://www.nature.com/articles/s41563-022-01374-3)
*  Heat_capacity@400K (J/g/K): machine learning predicted heat capacity at 400 K (https://www.nature.com/articles/s41563-022-01374-3)
*  std @ 400 K (J/g/K): the uncertainty of machine learning predicted heat capacity at 400 K (https://www.nature.com/articles/s41563-022-01374-3)
*  k_cp (J/g/K/K): slope parameter, heat capacity = k_cp * temperature + cp0
*  cp0 (J/g/K): intercept parameter, heat capacity = k_cp * temperature + cp0
*  Pearson product-moment correlation coefficients: PPMCC of heat capacity by ML predicted with fitted
*  natoms: number of atoms in unit cell
*  Source: obtained source (CSD or SI)
*  DOI: Digital Object Identifier of the paper reported structure
*  Year: publication year
*  Time: publication time
*  Publication: publisher
*  Extension: classification of structure after curation
*  unmodified: whether the structure is unmodified according to comparing with original structure
*  Thermal_stability(℃): machine learning predicted decomposed temperature
*  Solvent_stability: machine learning predicted the probility of solvent stability (stable when value >0.5)
*  Water_stability: machine learning predicted the probility of water stability (stable when value >0.5)
*  KH_water: classification of hydrophilicity and hydrophobicity MOFs by Gibbs Ensemble Monte Carlo (GEMC) Simulation

### 2. CoREMOF2024DB_CSD.zip: database in this work
*  CR: computaion-ready
  *  ASR: all solvent removed
  *  FSR: free solvent removed
  *  Ion: structure with ions for charge balance
*  NCR: not computaion-ready

### 3. NCR_data_CSD_modified_20250227.xlsx: details of all structures by Chen_Manz and mofchecker for each NCR cases.