import os
import numpy as np
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
# from data.image_folder import make_dataset
# from PIL import Image
import torch.nn.functional as F

# group_1 and group_2 are subjects belong to different groups
# five_fold_conn contains fold subject information

class ConnectomeEpiDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        train_fold = opt.name[-1]

        fold_no = opt.name[-1]
        self.sub_list = five_fold_conn['fold_'+fold_no+'_train']


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
        conn = torch.load(self.root + self.sub_list[index][0] + '_norm.dat')
        conn = torch.squeeze(conn, 0)

        if self.sub_list[index][0] in group_2:
            gt = torch.tensor(np.array(1))
        elif self.sub_list[index][0] in group_1:
            gt = torch.tensor(np.array(0))
        else:
            print('error')
            exit()
        # print(gt.size())
        # return {'pre': pre_conn, 'tx': None, 'gt': gt, 'path': self.sub_list[index]}
        return {'conn': conn, 'gt': gt, 'path': self.sub_list[index]}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.sub_list)
