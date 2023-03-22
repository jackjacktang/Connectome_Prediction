"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
import nibabel as nib
import numpy as np
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
# from util.visualizer import save_images
# from util import html
from util.stats import *
from skimage import measure
import subprocess
import time

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

# row and col refer to index not conenctome spreadsheet
def find_ind(ind, region_no):
    ones = np.ones((region_no, region_no))
    ones = np.triu(ones)
    np.fill_diagonal(ones, 0)
    lookup = np.where(ones.astype(bool))

    row, col = lookup[0][ind], lookup[1][ind]
    return row, col


def update_dict(lookup, fold, entry_key, update_val):
    if fold not in lookup.keys():
        lookup[fold] = dict()
    if entry_key in lookup[fold].keys():
        lookup[fold][entry_key] += update_val
    else:
        lookup[fold][entry_key] = update_val


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.n_fold = 5
    opt.epoch = 'best'

    view = opt.view

    fold_no = 5

    pred_acc = dict()
    edge_no = opt.nregions

    grad_edge_gp1 = np.zeros((int(fold_no), edge_no*edge_no))
    grad_edge_gp2 = np.zeros((int(fold_no), edge_no*edge_no))
    correct_sub_edge_gp1 = np.array([])
    correct_sub_edge_gp2 = np.array([])

    for fold in range(1, int(fold_no) + 1):
    # for fold in range(1, 2):
        print('testing phase on fold {}'.format(fold))

        device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

        opt.name = 'epi_predict_fold' + str(fold)

        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers

        # print(model.get_weights())

        time_spent = []

        case_no = 1
        test_subs = five_fold_conn['fold_'+str(fold)+'_test']

        entry = ''
        correct_gp1 = 0
        correct_gp2 = 0
        wrong_gp1 = 0
        wrong_gp2 = 0

        test_batch = torch.zeros((len(test_subs), edge_no*edge_no))
        test_gt_batch = np.zeros((len(test_subs), 1))
        case_no = 0

        # *********  Calculate Confusion Matrix  *********
        while case_no < len(test_subs):

            conn_path = opt.dataroot
            conn_ = torch.load(conn_path + test_subs[case_no][0] + '_norm.dat')
            test_batch[case_no] = conn_

            # print(test_subs)
            if test_subs[case_no][1] == 0:
                test_gt_batch[case_no] = 0
            elif test_subs[case_no][1] == 1:
                test_gt_batch[case_no] = 1

            case_no += 1

        test_gt_batch_tensor = torch.unsqueeze(torch.from_numpy(test_gt_batch), 0)
        # print(test_gt_batch.size())

        model.set_input({'conn': test_batch, 'gt': test_gt_batch_tensor})
        set_requires_grad(model.netG, True)
        model.conn.requires_grad = True
        model.conn.retain_grad()
        model.netG.zero_grad()
        logits = model.forward()
        output_prob = logits.softmax(dim=1)
        # print(output_prob.data.cpu().numpy())
        pred_lbl = np.argmax(output_prob.data.cpu().numpy(), axis=1)
        test_gt_batch = test_gt_batch[:, 0]

        print(pred_lbl, test_gt_batch)

        for pred, gt in zip(pred_lbl, test_gt_batch):
            if pred == 0 and gt == 0:
                correct_gp1 += 1
            elif pred == 1 and gt == 0:
                wrong_gp1 += 1
            elif pred == 1 and gt == 1:
                correct_gp2 += 1
            elif pred == 0 and gt == 1:
                wrong_gp2 += 1

        # print(correct_gp1, correct_gp2, wrong_gp1, wrong_gp2)
        update_dict(pred_acc, str(fold), 'correct_gp1', correct_gp1)
        update_dict(pred_acc, str(fold), 'correct_gp2', correct_gp2)
        update_dict(pred_acc, str(fold), 'wrong_gp1', wrong_gp1)
        update_dict(pred_acc, str(fold), 'wrong_gp2', wrong_gp2)

        # *********  Finish Calculate Confusion Matrix  *********


        # *********  Grad Analysis  *********
        # *********  Network weights on edge/node  *********
        sum_grad_edge_gp1 = np.zeros(edge_no*edge_no)
        sum_grad_edge_gp2 = np.zeros(edge_no*edge_no)

        # grad for gp1
        for sub in range(len(test_subs)):
            model.set_input({'conn': test_batch, 'gt': test_gt_batch_tensor})
            model.conn.requires_grad = True
            model.conn.retain_grad()
            logits = model.forward()
            logits[sub, 0].backward()
            sum_grad_edge_gp1 += model.conn.grad.cpu().numpy()[sub]

            if pred_lbl[sub] == test_gt_batch[sub] and test_gt_batch[sub] == 0:
                edge_weight_gp1 = model.conn.grad.cpu().numpy()[sub] * test_batch[sub].numpy()
                edge_weight_gp1 = np.expand_dims(edge_weight_gp1, 0)
                if len(correct_sub_edge_gp1) == 0:
                    correct_sub_edge_gp1 = edge_weight_gp1
                else:
                    correct_sub_edge_gp1 = np.concatenate((correct_sub_edge_gp1, edge_weight_gp1), axis=0)

        # grad for gp2
        for sub in range(len(test_subs)):
            model.set_input({'conn': test_batch, 'gt': test_gt_batch_tensor})
            model.conn.requires_grad = True
            model.conn.retain_grad()
            logits = model.forward()
            logits[sub, 1].backward()
            sum_grad_edge_gp2 += model.conn.grad.cpu().numpy()[sub]

            if pred_lbl[sub] == test_gt_batch[sub] and test_gt_batch[sub] == 1:
                edge_weight_gp2 = model.conn.grad.cpu().numpy()[sub] * test_batch[sub].numpy()
                edge_weight_gp2 = np.expand_dims(edge_weight_gp2, 0)
                if len(correct_sub_edge_gp2) == 0:
                    correct_sub_edge_gp2 = edge_weight_gp2
                else:
                    correct_sub_edge_gp2 = np.concatenate((correct_sub_edge_gp2, edge_weight_gp2), axis=0)


        grad_edge_gp1[fold-1] = sum_grad_edge_gp1 / len(test_subs)
        grad_edge_gp2[fold-1] = sum_grad_edge_gp2 / len(test_subs)


    # Summarize Confusion Matrix
    correct_gp1 = 0
    correct_gp2 = 0
    wrong_gp1 = 0
    wrong_gp2 = 0
    for k in pred_acc.keys():
        top = pred_acc[k]['correct_gp1'] + pred_acc[k]['correct_gp2']
        bot = top + pred_acc[k]['wrong_gp1'] + pred_acc[k]['wrong_gp2']
        print("Fold{} Acc: {}".format(k, top/bot))
        correct_gp1 += pred_acc[k]['correct_gp1']
        correct_gp2 += pred_acc[k]['correct_gp2']
        wrong_gp1 += pred_acc[k]['wrong_gp1']
        wrong_gp2 += pred_acc[k]['wrong_gp2']

    gp1_acc = correct_gp1 / (correct_gp1 + wrong_gp1)
    gp2_acc = correct_gp2 / (correct_gp2 + wrong_gp2)
    test_acc = (correct_gp1 + correct_gp2) / (correct_gp1 + correct_gp2 + wrong_gp1 + wrong_gp2)
    print(correct_gp1, wrong_gp1, correct_gp2, wrong_gp2)
    print('gp1 acc: {}, gp2 acc: {}'.format(gp1_acc, gp2_acc))
    print('Overall Acc', test_acc)

    # Summarize Grad Edge/Node
    average_grad_edge_gp1 = np.average(grad_edge_gp1, axis=0)
    average_grad_edge_gp2 = np.average(grad_edge_gp2, axis=0)
    print(average_grad_edge_gp1.shape)

    grad_node_gp1 = np.zeros(edge_no)
    grad_node_gp2 = np.zeros(edge_no)

    for ind in range(len(average_grad_edge_gp1)):
        row, col = find_ind(ind)
        grad_node_gp1[row] += average_grad_edge_gp1[ind]
        grad_node_gp1[col] += average_grad_edge_gp1[ind]

    for ind in range(len(average_grad_edge_gp2)):
        row, col = find_ind(ind)
        grad_node_gp2[row] += average_grad_edge_gp2[ind]
        grad_node_gp2[col] += average_grad_edge_gp2[ind]

    grad_node_gp1 /= edge_no
    grad_node_gp2 /= edge_no

    average_correct_sub_edge_gp1 = np.average(correct_sub_edge_gp1, axis=0)
    average_correct_sub_edge_gp2 = np.average(correct_sub_edge_gp2, axis=0)

    print(average_correct_sub_edge_gp1.shape)

    correct_sub_node_gp1 = np.zeros(edge_no)
    correct_sub_node_gp2 = np.zeros(edge_no)

    for ind in range(len(average_correct_sub_edge_gp1)):
        row, col = find_ind(ind)
        correct_sub_node_gp1[row] += average_correct_sub_edge_gp1[ind]
        correct_sub_node_gp1[col] += average_correct_sub_edge_gp1[ind]

    for ind in range(len(average_correct_sub_edge_gp2)):
        row, col = find_ind(ind)
        correct_sub_node_gp2[row] += average_correct_sub_edge_gp2[ind]
        correct_sub_node_gp2[col] += average_correct_sub_edge_gp2[ind]

    correct_sub_node_gp1 /= edge_no
    correct_sub_node_gp2 /= edge_no

    # average_grad_edge_0/average_grad_edge_1  84 * 84
    # grad_node_0/grad_node_1 1 * 84
    # average_correct_sub_edge_0/average_correct_sub_edge_1 84 * 84
    # correct_sub_node_0/correct_sub_node_1   1 * 84

    ana_path = '/home/jack/MRgFUS/Epilipsy_Data_1000/ana/'

    # average_grad_edge_0 = torch.from_numpy(average_grad_edge_0)
    # average_grad_edge_1 = torch.from_numpy(average_grad_edge_1)
    # torch.save(ana_path + 'grad_node_0.dat', grad_node_0)
    # grad_node_0 = torch.from_numpy(grad_node_0)
    # grad_node_1 = torch.from_numpy(grad_node_1)
    # average_correct_sub_edge_0 = torch.from_numpy(average_correct_sub_edge_0)
    # average_correct_sub_edge_1 = torch.from_numpy(average_correct_sub_edge_1)
    # correct_sub_node_0 = torch.from_numpy(correct_sub_node_0)
    # correct_sub_node_1 = torch.from_numpy(correct_sub_node_1)

    print(average_grad_edge_gp1.shape, average_grad_edge_gp2.shape)
    print(grad_node_gp1.shape, grad_node_gp2.shape)
    print(average_correct_sub_edge_gp1.shape, average_correct_sub_edge_gp2.shape)
    print(correct_sub_node_gp1.shape, correct_sub_node_gp2.shape)

    np.save(ana_path + 'average_grad_edge_gp1', average_grad_edge_gp1)
    np.save(ana_path + 'average_grad_edge_gp2', average_grad_edge_gp2)
    np.save(ana_path + 'grad_node_gp1', grad_node_gp1)
    np.save(ana_path + 'grad_node_gp2', grad_node_gp2)
    np.save(ana_path + 'average_correct_sub_edge_gp1', average_correct_sub_edge_gp1)
    np.save(ana_path + 'average_correct_sub_edge_gp2', average_correct_sub_edge_gp2)
    np.save(ana_path + 'correct_sub_node_gp1', correct_sub_node_gp1)
    np.save(ana_path + 'correct_sub_node_gp2', correct_sub_node_gp2)





