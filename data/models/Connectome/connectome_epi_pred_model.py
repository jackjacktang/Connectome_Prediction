"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from .base_model import BaseModel
from . import networks
from .loss import *


class ConnectomeEpiPredModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(dataset_mode='connectome')  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset.
        parser.add_argument('--analysis_dir', type=float, default=10.0, help='path to store analysis results')

        return parser

    def __init__(self, opt):
        """Initialize this Lesion Inpaint class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['pred']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['output', 'gt_img']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        self.model_names = ['G']

        self.criterion = nn.CrossEntropyLoss()

        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'connect', 32, 'batch',
                                      not opt.no_dropout, opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids)


        if self.isTrain:  # only defined during training time
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.

            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'  # use <direction> to swap data_A and data_B
        if self.isTrain:
            self.conn = input['conn'].to(self.device)  # get masked brain, i.e. lesion areas have value of 0
            self.gt = input['gt'].to(self.device)
            self.image_paths = input['path']  # get image paths
        else:
            # print('reach here for testing')
            self.conn = input['conn'].to(self.device)  # get masked brain, i.e. lesion areas have value of 0
            # self.tx = input['tx'].to(self.device)
            # self.pre_tx = torch.cat((self.pre, self.tx), 1)

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        # if self.isTrain:
        #     self.input = torch.cat((self.brain, self.lesion), dim=1)
        self.label_out = self.netG(self.conn)

        return self.label_out


    def backward_G(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_pred = self.criterion(self.label_out, self.gt)
        self.loss_pred.backward()       # calculate gradients of network G w.r.t. loss_G


    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()

        self.optimizer_G.zero_grad()  # set GE's gradients to zero
        self.backward_G()
        self.optimizer_G.step()

    def clear_grad(self):
        self.optimizer_G.zero_grad()