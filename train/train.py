#!/usr/bin/env python3

import os
import multiprocessing
import sys
import json

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import config
import r2av
import re
import glob
from data_vessels import VesselsDataset
from transformations import (
    random_affine, random_hsv, random_vertical_flip, random_horizontal_flip,
    to_torch_tensors, pad_images_unet, random_cutout
)
from data_utils import SubsetRandomSampler, SubsetSequentialSampler, get_sets



torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True



def create_dataloaders(
        dataset, data, train_idx, test_idx, transforms_train,
        transforms_test
):
    """ Creates dataloaders.

    Args:
        dataset: class for dataset creation.
        data: data info.
        train_idx: training images.
        test_idx: test images.
        transforms_train: transformations to apply to train data.
        transforms_test: transformations to apply to test data.

    Returns:
        Train and test data loaders.
    """

    dataset_train = dataset(data, transforms.Compose(transforms_train))
    dataset_test = dataset(data, transforms.Compose(transforms_test))

    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetSequentialSampler(test_idx)

    train_loader = DataLoader(
        dataset_train, batch_size=1, sampler=train_sampler, num_workers=2,
        drop_last=True, pin_memory=True
    )
    test_loader = DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=2,
        drop_last=False, pin_memory=True
    )

    return train_loader, test_loader


def train(training_path, train_idx, test_idx):
    """ Trains the network.

    Args:
        training_path: path to save training results.
        train_idx: list of images for training.
        test_idx: list of images for validation.
    """
    transforms_train = []
    transforms_train.extend([
        pad_images_unet,
        random_hsv,
        random_affine,
        random_vertical_flip,
        random_horizontal_flip,
        random_cutout,
        to_torch_tensors,
    ])

    transforms_test = []
    transforms_test.extend([
        pad_images_unet,
        to_torch_tensors,
    ])

    train_loader, test_loader = create_dataloaders(
        dataset=VesselsDataset,
        data=config.data,
        train_idx=list(train_idx),
        test_idx=list(test_idx),
        transforms_train=transforms_train,
        transforms_test=transforms_test
    )
    
    if config.use_checkpoint:
        checkpoint_path = config.checkpoint_path
        scheduler_path = config.scheduler_path
    else:
        checkpoint_path = training_path
        scheduler_path = training_path

    cnnet = r2av.R2Vessels(
        base_channels=config.base_channels,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_iterations=config.num_iterations,
        gpu_id=config.gpu_id,
        model=config.model,
        criterion=config.criterion,
        base_criterion=config.base_criterion,
        add_criterion=config.add_criterion,
        learning_rate=config.learning_rate,
        checkpoint_path=checkpoint_path,
        scheduler_path = scheduler_path,
        use_checkpoint=config.use_checkpoint,
    )

    # Train the model
    cnnet.training(
        train_loader=train_loader,
        test_loader=test_loader,
        scheduler_patience=sys.maxsize,
        stopping_patience=200,
        path_to_save=training_path,
        save_period=30 # 100
    )


def split(seq, num):
    """Splits a list into parts with the same length (approximately).
    """
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


def train_sets(sets):
    """ Runs trainings with different sets. """
    current = multiprocessing.current_process()
    process_id = str(current.pid)

    instances = []

    for i, set_ in enumerate(sets):
        if i not in config.active_folds:
            continue
        train_imgs = set_['training']
        val_imgs = set_['validation']

        generator_pth = config.model
        if config.model == 'RRWNet':
            generator_pth += '_{}it'.format(config.num_iterations)
        pattern = '{}/{}_folds/{}_lr{:.0e}_{}_bc{}/{}'
        criterion_str = config.criterion
        if config.base_criterion is not None:
            criterion_str += '-' + config.base_criterion
        if config.add_criterion is not None:
            criterion_str += '-' + config.add_criterion
        train_path = pattern.format(
            config.training_folder,
            config.num_folds,
            generator_pth,
            config.learning_rate,
            criterion_str,
            config.base_channels,
            i
        )

        if not os.path.exists(train_path):
            os.makedirs(train_path)

        # shutil.copy2('./config.py', train_path)
        # Convert config module to dict to save it as a json file
        args_dict = {k: v for k, v in vars(config.args).items() if not k.startswith('__')}
        with open(os.path.join(train_path, 'config.json'), 'w') as f:
            json.dump(args_dict, f, indent=4)

        current_instance = {
            'i': i,
            'train_path': train_path,
            'train_imgs': train_imgs,
            'val_imgs': val_imgs
        }

        instances.append(current_instance)

    print(instances)

    for instance in instances:
        print('\n', instance['train_path'])
        print(
            '[PID: {}] TRAINING SET ({}): {}/{}'.format(
                process_id, instance['i'],
                instance['train_imgs'],
                instance['val_imgs']
            )
        )

        train(
            instance['train_path'],
            instance['train_imgs'],
            instance['val_imgs']
        )

        
    


def multi_train():
    """Creates and runs processes, each one with its own data
    (train - val).
    """
    processes = []

    folds = get_sets(
        config.images,
        num_imgs_val=config.num_imgs_val
    )

    sets = split(folds, config.n_proc)

    # Runs processes with given sets
    for i in range(len(sets)):
        # Run process:
        p = multiprocessing.Process(target=train_sets, args=(sets[i],))
        p.start()  # start process
        processes.append(p)  # append process to process list

    # Wait for all processes
    for p in processes:
        p.join()
    
    
    

def clean_checkpoints(directory):
    # 定义匹配 `scheduler_xx.pth` 和 `checkpoint_xx.pth` 的正则表达式
    scheduler_pattern = re.compile(r'scheduler_(\d+)\.pth')
    checkpoint_pattern = re.compile(r'checkpoint_(\d+)\.pth')

    # 获取所有匹配的文件
    scheduler_files = glob.glob(os.path.join(directory, 'scheduler_*.pth'))
    checkpoint_files = glob.glob(os.path.join(directory, 'checkpoint_*.pth'))

    def get_latest_file(files, pattern):
        """ 解析文件名中的 iteration 数字，找到最新的文件 """
        if not files:
            return None, []
        
        parsed_files = [(f, int(pattern.search(os.path.basename(f)).group(1))) for f in files if pattern.search(f)]
        if not parsed_files:
            return None, []
        
        # 按 iteration 排序
        parsed_files.sort(key=lambda x: x[1])
        
        # 获取最新的文件和需要删除的文件
        latest_file = parsed_files[-1][0]
        files_to_delete = [f[0] for f in parsed_files[:-1]]

        return latest_file, files_to_delete

    # 找到需要删除的文件
    _, schedulers_to_delete = get_latest_file(scheduler_files, scheduler_pattern)
    _, checkpoints_to_delete = get_latest_file(checkpoint_files, checkpoint_pattern)

    # 删除不需要的文件
    for f in schedulers_to_delete + checkpoints_to_delete:
        os.remove(f)
        print(f"Deleted: {f}")


if __name__ == "__main__":
    multi_train()
