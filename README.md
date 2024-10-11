# matten

This repo contains the Materials Tensor (MatTen) model for predicting tensorial
properties of crystals such as the elasticity tensor.

MatTen is an equivariant graph neural network built using [e3nn](https://github.com/e3nn/e3nn).

## Install

Follow the official documentation to install [pytorch>=2.0.0](https://pytorch.org/get-started/locally/).
Then

```
git clone https://github.com/wengroup/matten.git
pip install -e ./matten
```

If you get package version conflicts, try the below command to install the dependencies
with strict version requirements.

```
pip install -e "./matten[strict]"
```

## Use the pretrained model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/wengroup/matten/blob/main/notebooks/predict_colab.ipynb)

```python
from pymatgen.core import Structure
from matten.predict import predict


def get_structure():
    a = 5.46
    lattice = [[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]]
    basis = [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]
    Si = Structure(lattice, ["Si", "Si"], basis)

    return Si


structure = get_structure()

elasticity_tensor = predict(structure)
```

The `predict` function returns an elasticity tensor. To make predictions for multiple
crystals, pass a list of structures to `predict`.

## Data

- An example of 100 crystals is available in the [datasets](./datasets) directory.
- The full dataset is available at: https://doi.org/10.5281/zenodo.8190849

## Train the model (using your own data)

See instructions [here](./scripts/README.md).

## Reference

> Wen, M., Horton, M. K., Munro, J. M., Huck, P., & Persson, K. A. (2024). An equivariant graph neural network for the elasticity tensors of all seven crystal systems. Digital Discovery, 3(5), 869–882. doi: 10.1039/D3DD00233K

```
@article{matten,
  author = {Wen, Mingjian and Horton, Matthew K. and Munro, Jason M. and Huck, Patrick and Persson, Kristin A.},
  title = {An equivariant graph neural network for the elasticity tensors of all seven crystal systems},
  journal = {Digital Discovery},
  volume = {3},
  number = {5},
  pages = {869--882},
  year = {2024},
  publisher = {RSC},
  doi = {10.1039/D3DD00233K}
}
```
