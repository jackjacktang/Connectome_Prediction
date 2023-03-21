import os
import subprocess

fold = 5
for i in range(5, 6):
    cmd = ['python3', '../train_conn.py',
    '--dataroot', '/home/jack/MRgFUS/Connectome/processed_vector/NoAblationPreTxNorm/',
    '--checkpoints_dir', '../checkpoints/' + str(fold) + '_fold',
    '--gpu_ids', '0',
    '--name', 'connect_lesion' + str(i),
    '--model', 'connectome_lesion',
    '--input_nc', '2',
    '--output_nc', '1',
    '--init_type', 'kaiming',
    '--no_dropout',
    '--netD', 'pixel',
    '--pool_size', '10',
    '--dataset_mode', 'connectome',
    '--num_threads', '4',
    '--batch_size', '12',
    '--no_html',
    '--beta1', '0.99',
    '--lr', '0.0001',
    '--n_epochs', '500',
    '--display_freq', '50',
    '--print_freq','50',
    '--save_latest_freq', '50',
    '--save_epoch_freq', '200']

    subprocess.run(cmd)
