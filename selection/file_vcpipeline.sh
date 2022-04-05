#!/bin/zsh
exp_name=$1
params_name=$2
python fromfile_applyvc.py ../../diagnostic_measures/voxceleb_newwavs/ $params_name $exp_name
python create_melfs.py $exp_name/augmented_dataset $exp_name/augmented_melfs
python vc_K_creation.py $exp_name/augmented_melfs $exp_name/Ks
python vc_L_creation.py $exp_name/augmented_melfs $exp_name/Ls
python exp_testing.py $exp_name/Ks $exp_name/Ls

