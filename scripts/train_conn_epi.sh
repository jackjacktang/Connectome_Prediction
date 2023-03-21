# set -ex
python3 ../train_conn_epi.py \
--dataroot /home/jack/MRgFUS/Epilipsy_Data/tensor/ \
--checkpoints_dir ../checkpoints/epi_predict_five_fold \
--gpu_ids 0 \
--name epi_predict_fold1 \
--model connectome_epi_pred \
--input_nc 1 \
--output_nc 2 \
--init_type kaiming \
--no_dropout \
--netD pixel \
--pool_size 10 \
--dataset_mode connectome_epi \
--num_threads 4 \
--batch_size 4 \
--no_html \
--beta1 0.99 \
--lr 0.0001 \
--n_epochs 3000 \
--display_freq 500 \
--print_freq 500 \
--save_latest_freq 100 \
--save_epoch_freq 500 \

