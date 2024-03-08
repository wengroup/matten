# scripts

This directory contains scripts to train/evaluate/test the models.

## First try

To train an example model, run the below command in this directory:

```bash
python train_materials_tensor.py
```

This will train a toy model on a very small example dataset for only 10 epochs and then stop.

After training, you can see the metrics and the location of the best model.

### Under the hood

All training configurations (data, optimizer, model etc.) are defined in the [./configs/materials_tensor.yaml](./configs/materials_tensor.yaml) file. Particularly you can see from the `data` section (copied below) that the data file we are using is `example_crystal_elasticity_tensor_n100.json` and it is stored at the [../datasets/](../datasets). Here we use the same file for training, validation and testing. This is just for demonstration purposes. In a real-world scenario, you would have separate files for each of these.

```yaml
data:
  root: ../datasets/
  r_cut: 5.0
  trainset_filename: example_crystal_elasticity_tensor_n100.json
  valset_filename: example_crystal_elasticity_tensor_n100.json
  testset_filename: example_crystal_elasticity_tensor_n100.json
  reuse: false
  loader_kwargs:
    batch_size: 32
    shuffle: true
```

Feel free to change other settings and see how the model behaves. You might find [this config](../pretrained/20230627/config_final.yaml) file useful for reference, which is the final configuration used to train the model in the paper.

You might also want to uncomment the `logger` section to use [Weights and Biases](https://wandb.ai/site) for logging, which makes tracking the training process much easier.

## Training with your own data

All you need to do is to prepare your data as json file(s), similar to `example_crystal_elasticity_tensor_n100.json`.
It needs a list of pymatgen `structure`s and the corresponding `elastic_tensor_full` for the structures.
See this [notebook](../notebooks/prepare_data.ipynb) for more details on how to prepare the data.
