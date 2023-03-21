set -ex
python3 ../test_conn.py \
--dataroot /home/jack/MRgFUS/Connectome/original/NoAblationPreTx/ \
--checkpoints_dir ../checkpoints \
--gpu_ids 0 \
--name unet_noablation_side \
--model connectome_pred \
--input_nc 2 \
--output_nc 1 \
--norm batch \
--view ax
