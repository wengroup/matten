# Datasets 
This directory contains an example dataset of 100 elasticity tensors of crystals:

## example_crystal_elasticity_tensor_n100.json

This is a subset of the elasticity tensor dataset used in the paper: `An equivariant 
graph neural network for the elasticity tensors of all seven crystal systems`, 
by Wen et al., https://doi.org/10.1039/d3dd00233k 

Explanation and the full dataset is available at: https://doi.org/10.5281/zenodo.8190849


## si_nmr_data.json 

This is the dataset used in the paper: `Machine Learning Full NMR Chemical Shift Tensors 
of Silicon Oxides with Equivariant Graph Neural Networks`, by Venetos et al. https://doi.org/10.1021/acs.jpca.2c07530

### Explanation 
- structure: pymatgen structure, consisting of N atoms.
- species: atomic number of the atoms in the structure.
- nmr_tensor: list of M 3x3 tensor matrix, each 3x3 tensor correspond to one atom. 1<= M <= N.
- atom_selector: indices of atoms the tensor correspond to.
  For example, [True, False, False, True, False, False] means the tensors correspond to 
  atoms 0, and 3. This also means the size of `nmr_tensor` is M=2.
- Qn: polymerization of the SiO4 in the structure.
- sigma_iso: isotropic tensor value.
