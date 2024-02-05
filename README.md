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

## Reference

```
@article{matten,
  title = {An equivariant graph neural network for the elasticity tensors of all seven crystal systems},
  author = {Wen, Mingjian and Horton, Matthew K. and Munro, Jason M. and Huck, Patrick and Persson, Kristin A.},
  doi = {10.1039/D3DD00233K},
  publisher = {Digital Discovery},
  year = {2024},
}
```
