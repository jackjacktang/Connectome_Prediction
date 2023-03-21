import numpy as np
import os
import sys
import ntpath
import time
# from . import util, html
from tensorboardX import SummaryWriter


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


# def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
#     """Save images to the disk.
#
#     Parameters:
#         webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
#         visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
#         image_path (str)         -- the string is used to create image paths
#         aspect_ratio (float)     -- the aspect ratio of saved images
#         width (int)              -- the images will be resized to width x width
#
#     This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
#     """
#     image_dir = webpage.get_image_dir()
#     short_path = ntpath.basename(image_path[0])
#     name = os.path.splitext(short_path)[0]
#
#     webpage.add_header(name)
#     ims, txts, links = [], [], []
#
#     for label, im_data in visuals.items():
#         im = util.tensor2im(im_data)
#         image_name = '%s_%s.png' % (name, label)
#         save_path = os.path.join(image_dir, image_name)
#         util.save_image(im, save_path, aspect_ratio=aspect_ratio)
#         ims.append(image_name)
#         txts.append(label)
#         links.append(image_name)
#     webpage.add_images(ims, txts, links, width=width)


class Recorder():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.writer = SummaryWriter(log_dir=os.path.join(opt.checkpoints_dir, opt.name), filename_suffix=opt.name)
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def plot_current_losses(self, current_iters, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        for k, v in losses.items():
            self.writer.add_scalar('loss/' + k, v, current_iters)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.4f, data: %.4f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.4f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
