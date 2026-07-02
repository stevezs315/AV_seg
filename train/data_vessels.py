from pathlib import Path
import os
from os.path import join
import re
from collections import defaultdict

from torch.utils.data import Dataset
import numpy as np
from skimage import io

import config



def read_image(file):
    img = io.imread(file)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    return img


def read_file(file):
    if '.npy' in file:
        data = np.load(file)
    elif '.npz' in file:
        data = np.load(file)['data']
    else:
        data = read_image(file)
        data = data.astype(np.float32)
        if data.max() > 1.0:
            data /= 255.0
    return data


class Img2ImgDataset:

    def __init__(self, data):
        self.target_paths = self._resolve_paths(data['data_folder'], data['target']['path'])
        self.original_paths = self._resolve_paths(data['data_folder'], data['original']['path'])
        self.mask_paths = self._resolve_paths(data['data_folder'], data['mask']['path'])

        self.target_path = self.target_paths[0]
        self.original_path = self.original_paths[0]
        self.mask_path = self.mask_paths[0]
        # self.centerness_path = []
        # for i in range(3):
        #     self.centerness_path.append(os.path.join(data['data_folder'], data['centerness']['path']
        #                                             , data['centerness']['dil'][i]))
        # self.dilation_path = []
        # for i in range(3):
        #     self.dilation_path.append(os.path.join(data['data_folder'], data['dilation']['path']
        #                                             , data['dilation']['dil'][i]))
        self.target_pattern = data['target']['pattern']
        self.original_pattern = data['original']['pattern']
        self.mask_pattern = data['mask']['pattern']

        # self.centerness_a_pattern = data['centerness']['pattern_a']
        # self.centerness_v_pattern = data['centerness']['pattern_v']
        # self.centerness_ves_pattern = data['centerness']['pattern_ves']

        # self.dilation_a_pattern = data['dilation']['pattern_a']
        # self.dilation_v_pattern = data['dilation']['pattern_v']
        # self.dilation_ves_pattern = data['dilation']['pattern_ves']

        self._make_dataset()

    def _resolve_paths(self, data_folder, paths):
        if isinstance(paths, (list, tuple)):
            return [os.path.join(data_folder, path) for path in paths]
        return [os.path.join(data_folder, paths)]

    def _make_dataset(self):
        target = re.compile(self.target_pattern)
        orig = re.compile(self.original_pattern)
        mask = re.compile(self.mask_pattern)
        # centerness_a = re.compile(self.centerness_a_pattern)
        # centerness_v = re.compile(self.centerness_v_pattern)
        # centerness_ves = re.compile(self.centerness_ves_pattern)
        
        # dilation_a = re.compile(self.dilation_a_pattern)
        # dilation_v = re.compile(self.dilation_v_pattern)
        # dilation_ves = re.compile(self.dilation_ves_pattern)

        number = re.compile('[0-9]+')

        self.targets = defaultdict(dict)
        self.origs = defaultdict(dict)
        self.masks = defaultdict(dict)
        # self.centerness_a_map = defaultdict(dict)
        # self.centerness_v_map = defaultdict(dict)
        # self.centerness_ves_map = defaultdict(dict)
        # self.dilation_a_map = defaultdict(dict)
        # self.dilation_v_map = defaultdict(dict)
        # self.dilation_ves_map = defaultdict(dict)

        # print(self.centerness_a_map)
        # paths = [self.target_path, self.original_path, self.mask_path, 
        #          self.centerness_path[0], self.centerness_path[0], self.centerness_path[0],
        #          self.dilation_path[0], self.dilation_path[0], self.dilation_path[0]]
        paths = [self.target_paths, self.original_paths, self.mask_paths]
        # patterns = [target, orig, mask, centerness_a, centerness_v, centerness_ves,
        #             dilation_a, dilation_v, dilation_ves]
        patterns = [target, orig, mask]
        # data_dicts = [self.targets, self.origs, self.masks, 
        #               self.centerness_a_map, self.centerness_v_map, self.centerness_ves_map,
        #             self.dilation_a_map, self.dilation_v_map, self.dilation_ves_map]
        data_dicts = [self.targets, self.origs, self.masks]

        for path_list, pattern, data_dict in zip(paths, patterns, data_dicts):
            for path in path_list:
                for file_name in os.listdir(path):
                    if config.dataset == 'RITE-train' or config.dataset.startswith('LES-AV'):
                        n = number.findall(file_name)
                        if n:
                            n = int(n[0])
                            if pattern.match(file_name):
                                data_dict[n] = os.path.join(path, file_name)
                    else:
                        if pattern.match(file_name):
                            data_dict[Path(file_name).stem] = os.path.join(path, file_name)
        # print(self.origs)
        # print(self.targets)
        # print(self.masks)
        # print(self.centerness_a_map)
        # print(self.centerness_v_map)
        # print(self.centerness_ves_map)

class VesselsDataset(Dataset, Img2ImgDataset):

    def __init__(self, data, transform=None):
        Img2ImgDataset.__init__(self, data)
        self.transform = transform
        self.vessels = self.targets
        self.retinos = self.origs
        # This allows no continuous keys in the dict:
        self.indices = [n for n in self.retinos.keys()]

    def __len__(self):
        return len(self.retinos)

    def __getitem__(self, index):
        _index = index
        retino = self.retinos[_index]
        vessel = self.vessels[_index]
        mask = self.masks[_index]
        # c_artery = self.centerness_a_map[_index]
        # c_vein = self.centerness_v_map[_index]
        # c_vessel = self.centerness_ves_map[_index]
        # d_artery = self.dilation_a_map[_index]
        # d_vein = self.dilation_v_map[_index]
        # d_vessel = self.dilation_ves_map[_index]
        
        # print(retino)
        assert isinstance(retino, str)
        assert isinstance(vessel, str)
        assert isinstance(mask, str)
        # assert isinstance(c_artery, str)
        # assert isinstance(d_artery, str)

        r = read_file(retino)
        m = read_file(mask)
        v = read_file(vessel)
        # c = []
        # d = []
        # for i in range(len(self.centerness_path)):
        #     c_a = read_file(join(self.centerness_path[i], c_artery))
        #     c_v = read_file(join(self.centerness_path[i], c_vein))
        #     c_ves = read_file(join(self.centerness_path[i], c_vessel))
        #     c_merge = np.concatenate((c_a, c_v, c_ves), axis=2)
        #     c.append(c_merge)

        #     d_a = read_file(join(self.dilation_path[i], d_artery))
        #     d_v = read_file(join(self.dilation_path[i], d_vein))
        #     d_ves = read_file(join(self.dilation_path[i], d_vessel))
        #     d_merge = np.concatenate((d_a, d_v, d_ves), axis=2)
        #     d.append(d_merge)
        #     # print(self.centerness_path[i],i)
        # raise ValueError
        # item = [r, v, m, c[0], c[1], c[2], d[0], d[1], d[2]]
        item = [r, v, m]
        if self.transform is not None:
            item = self.transform(item)
        return [_index, item]

