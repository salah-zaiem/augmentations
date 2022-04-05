import os 
import sys 

import numpy as np
import os
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp
from collections import Counter
import cca_core
melfs_dir = sys.argv[1]
outdir=sys.argv[2]
# Now building the K matrix : 
melfs_files = os.listdir(melfs_dir)
melfs_paths = [os.path.join(melfs_dir, x) for x in melfs_files]
speakers = list(np.unique([x.split("_")[1] for x in melfs_files]))
print(speakers)
from gaussian_relative_nonUniform_downsample_uniform import downsample

gaussian_downsampling = partial(downsample, n_samples= 30, std_ratio=0.07,
                                std_slope=0.1)


def load_vector(rank, vector_dicts, norm_dicts, melfs_dir, list_considered) :
    if rank in vector_dicts :
        return vector_dicts[rank], norm_dicts[rank], vector_dicts, norm_dicts
    else : 
        filename= list_considered[rank]
        fname=  filename.split("/")[-1].split(".")[0] + ".npy"
        loaded= np.load(os.path.join(melfs_dir, fname))
        acts = gaussian_downsampling(loaded)
        norm = np.linalg.norm(acts)
        norm_dicts[rank] = norm
        vector_dicts[rank] = acts
        return acts, norm, vector_dicts, norm_dicts


def svd_transform(melfs1, size=20) :
    cacts1 = melfs1 - np.mean(melfs1, axis=1, keepdims=True)

    U1, s1, V1 = np.linalg.svd(cacts1, full_matrices=False)

    svacts1 = np.dot(s1[:size]*np.eye(size), V1[:size])
   
    return svacts1
def per_speaker_matrix(speaker, paths) : 
    melfs_paths =[paths[x] for x in range(len(paths)) if melfs_files[x].split("_")[1]==speaker]
    N=len(melfs_paths)
    K_matrix = np.zeros((N,N))
    #print(f"size of the matrix : {N}")
    all_values = []
    vector_dicts ={}
    norm_dicts = {}
    #N=number_words
    K_matrix = np.zeros((N,N))
    all_values = []
    #list_considered = words_dict[word][0:number_words]
    list_considered = melfs_paths
    for i in tqdm(range(N)):
        svacts1, norm1, vector_dicts, norm_dicts = load_vector(i,
                                                               vector_dicts,norm_dicts,
                                                               melfs_dir,
                                                               list_considered)
        # can also compute as svacts1 = np.dot(U1.T[:20], cacts1)
        for j in range(i+1): 
            svacts2, norm2, vector_dicts, norm_dicts = load_vector(j,
                                                               vector_dicts,norm_dicts,
                                                               melfs_dir,
                                                                   list_considered)
            scalar = np.trace(svacts1.T @ svacts2)
            value = scalar / (norm1*norm2)


            K_matrix[i,j]=value
            if i!=j : 
                all_values.append(value)

    for i in range(N):
        for j in range(i,N): 
            K_matrix[i,j] = K_matrix[j,i]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    np.save(os.path.join(outdir, "K_matrix_"+speaker+".npy"), K_matrix)
part_func = partial(per_speaker_matrix,paths = melfs_paths) 
#for speaker in tqdm(speakers) : 
#    per_speaker_matrix(speaker, [melfs_paths[x] for x in range(len(melfs_paths)) if melfs_files[x].split("_")[0]==speaker])
parallel=False    
if parallel :
    v = mp.cpu_count()
    p = Pool(min(v, len(speakers)))
    print(f"working with {min(v,len(speakers))} cpus")
    r = list(tqdm(p.imap(part_func, speakers), total=len(speakers)))
else : 
    for speaker in speakers :
        print(speaker)
        part_func(speaker)
#Start by stocking the melfs


