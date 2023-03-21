# set -ex

python3 ../train.py \
--dataroot /home/jack/MRgFUS/Connectome/processed_vector/NoAblationPreTxNorm/ \
--checkpoints_dir ../checkpoints/10_fold \
--gpu_ids 0 \
--name connect_lesion10 \
--model connectome_lesion \
--input_nc 1 \
--output_nc 2 \
--init_type kaiming \
--no_dropout \
--netD pixel \
--pool_size 10 \
--dataset_mode connectome \
--num_threads 4 \
--batch_size 8 \
--no_html \
--beta1 0.99 \
--lr 0.0001 \
--n_epochs 1000 \
--display_freq 50 \
--print_freq 50 \
--save_latest_freq 50 \
--save_epoch_freq 200 \

