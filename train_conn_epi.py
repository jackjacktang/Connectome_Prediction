"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
import torch
import numpy as np
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
# from util.visualizer import Visualizer
from util.recorder import Recorder
# group_1 and group_2 are subjects belong to different groups
# five_fold_conn contains fold subject information

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

if __name__ == '__main__':

    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    recorder = Recorder(opt)
    total_iters = 0                # the total number of training iterations

    best_test = 0
    best_vim = 0
    best_mul = 0
    best_epoch = 0
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            # if total_iters % opt.print_freq == 0:
            #     t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            losses = model.get_current_losses()
            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                # model.compute_visuals()
                # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                recorder.plot_current_losses(total_iters, losses)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk

                t_comp = (time.time() - iter_start_time) / opt.batch_size
                t_data = iter_start_time - iter_data_time
                # visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                # if opt.display_id > 0:
                #     visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                recorder.print_current_losses(epoch, total_iters, losses, t_comp, t_data)


            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()


        # find best weights for folds
        correct_1 = 0
        correct_2 = 0
        wrong_1 = 0
        wrong_2 = 0
        fold_no = opt.name[-1]
        test_fold = opt.name[-1]
        test_subs = five_fold_conn['fold_'+test_fold+'_test']

        test_batch = torch.zeros((len(test_subs), 3486))
        test_gt_batch = np.zeros((len(test_subs), 1))
        case_no = 0
        while case_no < len(test_subs):

            conn = torch.load(opt.dataroot + test_subs[case_no][0] + '_norm.dat')
            test_batch[case_no] = conn

            if test_subs[case_no][0] in group_2:
                test_gt_batch[case_no] = 1
            elif test_subs[case_no][0] in group_1:
                test_gt_batch[case_no] = 0
            else:
                print('invalid label')

            case_no += 1

        test_gt_batch_tensor = torch.unsqueeze(torch.from_numpy(test_gt_batch), 0)



        model.set_input({'conn': test_batch, 'gt': test_gt_batch_tensor, 'path': 'N/A'})
        set_requires_grad(model.netG, False)
        logits = model.forward()
        output_prob = logits.softmax(dim=1)
        pred_lbl = np.argmax(output_prob.data.cpu().numpy(), axis=1)
        test_gt_batch = test_gt_batch[:, 0]

        for pred, gt in zip(pred_lbl, test_gt_batch):
            if pred == 1 and gt == 1:
                correct_2 += 1
            elif pred == 0 and gt == 1:
                wrong_2 += 1
            elif pred == 0 and gt == 0:
                correct_1 += 1
            elif pred == 1 and gt == 0:
                wrong_1 += 1

        print(correct_1, correct_2, wrong_1, wrong_2)
        g1_acc = correct_1 / (correct_1 + wrong_1)
        g2_acc = correct_2 / (correct_2 + wrong_2)
        test_acc = (correct_1 + correct_2) / (correct_1 + correct_2 + wrong_1 + wrong_2)
        print('g1 acc: {}, g2 acc: {}, test acc: {}'.format(g1_acc, g2_acc, test_acc))
        if epoch >= 50 and test_acc > best_test:
            print('saving the best model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'best'
            model.save_networks(save_suffix)
            best_test = test_acc
            best_epoch = epoch
            best_1 = g1_acc
            best_2 = g2_acc
        elif epoch >= 50 and test_acc == best_test:
            print('saving the best model (epoch %d, total_iters %d)' % (epoch, total_iters))
            save_suffix = 'best'
            model.save_networks(save_suffix)
        else:
            if epoch - best_epoch >= 50:
                set_requires_grad(model.netG, True)
                break

        set_requires_grad(model.netG, True)


        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
