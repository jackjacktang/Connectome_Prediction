import os
import numpy as np
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
# from data.image_folder import make_dataset
# from PIL import Image
import torch.nn.functional as F


class ConnectomeEpi1000Dataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt, fold_case):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.fold_case = fold_case
        self.sub_list = self.fold_case['0'] + self.fold_case['1']

        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        # print(self.sub_list)
        conn = torch.load(self.root + self.sub_list[index] + '_norm.dat')
        conn = torch.squeeze(conn, 0)
        if self.sub_list[index] in self.fold_case['0']:
            gt = torch.tensor(np.array(0))
        elif self.sub_list[index] in self.fold_case['1']:
            gt = torch.tensor(np.array(1))
        else:
            print('error')
            exit()
        # print(gt.size())
        # return {'pre': pre_conn, 'tx': None, 'gt': gt, 'path': self.sub_list[index]}
        return {'conn': conn, 'gt': gt, 'path': self.sub_list[index]}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.sub_list)
    #
    # def set_fold_case(self, fold_case):
    #     self.sub_list = fold_case

