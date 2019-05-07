# Adversarial Autoencoders for Compact Representations of 3D Point Clouds

Authors: Maciej Zamorski, Maciej Zięba, Piotr Klukowski, Rafał Nowak, Karol Kurach, Wojciech Stokowiec, and Tomasz Trzciński

![mainimg](https://github.com/MaciejZamorski/3d-AAE/blob/master/BinaryRepresentations.png)

## Introduction
This is a PyTorch implementation for a family of 3dAAE models, a novel framework for learning continuous and binary representations of 3d point clouds based on Adversarial Autoencoder model, as presented in:

M. Zamorski, M. Zięba, et al., Adversarial Autoencoders for Compact Representations of 3D Point Clouds, [arXiv preprint](https://arxiv.org/abs/1811.07605) (2018)
## Citation
```
@article{zamorski2018adversarial,
  title={Adversarial Autoencoders for Compact Representations of 3D Point Clouds},
  author={Zamorski, Maciej and Zi{\k{e}}ba, Maciej and Klukowski, Piotr and Nowak, Rafa{\l} and Kurach, Karol and Stokowiec, Wojciech and Trzci{\'n}ski, Tomasz},
  journal={arXiv preprint arXiv:1811.07605},
  year={2018}
}
```

## Requirements
Stored in `requirements.txt`, Python dependencies are:
```
h5py
matplotlib
numpy
pandas
git+https://github.com/szagoruyko/pyinn.git@master
torch==0.4.1
```

## Usage
### Training
Run an experiment with:

`python3.6 experiments/train.py --config settings.json`

where

`train.py` - one of the training scripts from the `experiments` directory

`settings.json` - JSON file with training settings and hyperparameter values created as shown in example `settings/hyperparams.json`

### Evaluation
`python3.6 evaluation/find_best_epoch_on_validation.py --config settings.json`

Calculates JSD distance between sampled point clouds and the validation set and presents the best epoch.

`python3.6 evaluation/generate_data_for_metrics.py --config settings.json`

Produce reconstructed and generated point clouds in a form of NumPy array to be used with validation methods from ["Learning Representations and Generative Models For 3D Point Clouds" repository](https://github.com/optas/latent_3d_points/blob/master/notebooks/compute_evaluation_metrics.ipynb)