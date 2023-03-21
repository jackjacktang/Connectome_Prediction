set -ex
python3 ../test_conn_epi_patient_updated.py \
--dataroot /home/jack/MRgFUS/Epilipsy_Data/tensor/ \
--checkpoints_dir ../checkpoints/epi_predict_five_fold \
--gpu_ids 0 \
--name epi_predict_patient \
--model connectome_epi_pred \
--input_nc 2 \
--output_nc 1 \
--norm batch \
--epoch best \
--view ax
