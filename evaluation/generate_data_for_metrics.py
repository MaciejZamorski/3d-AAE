import argparse
import json
import logging
import random
from importlib import import_module
from os.path import join

import numpy as np
import torch
from torch.distributions import Beta
from torch.utils.data import DataLoader

from datasets.shapenet.shapenet import ShapeNetDataset
from loggers.basic_logger import setup_logging
from utils.util import find_latest_epoch, cuda_setup


def main(eval_config):
    # Load hyperparameters as they were during training
    train_results_path = join(eval_config['results_root'], eval_config['arch'],
                              eval_config['experiment_name'])
    with open(join(train_results_path, 'config.json')) as f:
        train_config = json.load(f)

    random.seed(train_config['seed'])
    torch.manual_seed(train_config['seed'])
    torch.cuda.manual_seed_all(train_config['seed'])

    setup_logging(join(train_results_path, 'results'))
    log = logging.getLogger(__name__)

    weights_path = join(train_results_path, 'weights')
    if eval_config['epoch'] == 0:
        epoch = find_latest_epoch(weights_path)
    else:
        epoch = eval_config['epoch']
    log.debug(f'Starting from epoch: {epoch}')

    device = cuda_setup(eval_config['cuda'], eval_config['gpu'])
    log.debug(f'Device variable: {device}')
    if device.type == 'cuda':
        log.debug(f'Current CUDA device: {torch.cuda.current_device()}')

    #
    # Dataset
    #
    dataset_name = train_config['dataset'].lower()
    if dataset_name == 'shapenet':
        dataset = ShapeNetDataset(root_dir=train_config['data_dir'],
                                  classes=train_config['classes'], split='test')
    elif dataset_name == 'faust':
        from datasets.dfaust import DFaustDataset
        dataset = DFaustDataset(root_dir=train_config['data_dir'],
                                classes=train_config['classes'], split='test')
    elif dataset_name == 'mcgill':
        from datasets.mcgill import McGillDataset
        dataset = McGillDataset(root_dir=train_config['data_dir'],
                                classes=train_config['classes'], split='test')
    else:
        raise ValueError(f'Invalid dataset name. Expected `shapenet` or '
                         f'`faust`. Got: `{dataset_name}`')
    classes_selected = ('all' if not train_config['classes']
                        else ','.join(train_config['classes']))
    log.debug(f'Selected {classes_selected} classes. Loaded {len(dataset)} '
              f'samples.')

    if 'distribution' in train_config:
        distribution = train_config['distribution']
    elif 'distribution' in eval_config:
        distribution = eval_config['distribution']
    else:
        log.warning('No distribution type specified. Assumed normal = N(0, 0.2)')
        distribution = 'normal'

    #
    # Models
    #
    arch = import_module(f"model.architectures.{eval_config['arch']}")
    E = arch.Encoder(train_config).to(device)
    G = arch.Generator(train_config).to(device)

    #
    # Load saved state
    #
    E.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_E.pth')))
    G.load_state_dict(torch.load(join(weights_path, f'{epoch:05}_G.pth')))

    E.eval()
    G.eval()

    num_samples = len(dataset.point_clouds_names_test)
    data_loader = DataLoader(dataset, batch_size=num_samples,
                             shuffle=False, num_workers=4,
                             drop_last=False, pin_memory=True)

    # We take 3 times as many samples as there are in test data in order to
    # perform JSD calculation in the same manner as in the reference publication
    noise = torch.FloatTensor(3 * num_samples, train_config['z_size'], 1)
    noise = noise.to(device)

    X, _ = next(iter(data_loader))
    X = X.to(device)

    np.save(join(train_results_path, 'results', f'{epoch:05}_X'), X)

    for i in range(3):
        if distribution == 'normal':
            noise.normal_(0, 0.2)
        else:
            noise_np = np.random.beta(train_config['z_beta_a'],
                                      train_config['z_beta_b'],
                                      noise.shape)
            noise = torch.tensor(noise_np).float().round().to(device)
        with torch.no_grad():
            X_g = G(noise)
        if X_g.shape[-2:] == (3, 2048):
            X_g.transpose_(1, 2)

        np.save(join(train_results_path, 'results', f'{epoch:05}_Xg_{i}'), X_g)

    with torch.no_grad():
        z_e = E(X.transpose(1, 2))
        if isinstance(z_e, tuple):
            z_e = z_e[0]
        X_rec = G(z_e)
    if X_rec.shape[-2:] == (3, 2048):
        X_rec.transpose_(1, 2)

    np.save(join(train_results_path, 'results', f'{epoch:05}_Xrec'), X_rec)


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='File path for evaluation config')
    args = parser.parse_args()

    evaluation_config = None
    if args.config is not None and args.config.endswith('.json'):
        with open(args.config) as f:
            evaluation_config = json.load(f)
    assert evaluation_config is not None

    main(evaluation_config)
