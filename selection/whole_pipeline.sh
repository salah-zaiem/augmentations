#!/bin/zsh
exp_name=$1
python apply_augment_iemocap.py ../IEMOCAP_wav/ $exp_name
python create_melfs.py $exp_name/augmented_dataset $exp_name/augmented_melfs
python iemocap_K_creation.py $exp_name/augmented_melfs $exp_name/Ks
python iemocap_L_creation.py $exp_name/augmented_melfs $exp_name/Ls
python exp_testing.py $exp_name/Ks $exp_name/Ls

