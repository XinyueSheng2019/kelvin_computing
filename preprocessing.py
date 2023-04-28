# from random import shuffle
import pandas as pd 
import numpy as np 
from astropy import visualization
import h5py
import json

# set the seed
from numpy.random import seed
import tensorflow as tf
# seed(0)
# tf.random.set_seed(0)

def open_with_h5py(filepath):
    imageset = np.array(h5py.File(filepath, mode = 'r')['imageset'])
    labels = np.array(h5py.File(filepath, mode = 'r')['label'])
    metaset = np.array(h5py.File(filepath, mode = 'r')['metaset'])
    idx_set = np.array(h5py.File(filepath, mode = 'r')['idx_set'])
    return imageset, labels, metaset, idx_set

def single_transient_preprocessing(image, meta):
    pre_image = image.reshape(1, image.shape[0], image.shape[1], image.shape[-1])
    pre_meta = meta.reshape(1,meta.shape[0])
    return pre_image, pre_meta

def preprocessing(filepath, label_dict, hash_table, output_path):

    imageset, labels, metaset, idx_set = open_with_h5py(filepath)
    test_num_dict = label_dict["test_num"]

    for k in label_dict['classify'].keys():
        if label_dict['classify'][k] not in label_dict['label'].values():
            ab_idx = np.where(labels == label_dict["classify"][k])
            imageset, metaset, labels = np.delete(imageset, ab_idx, 0), np.delete(metaset, ab_idx, 0), np.delete(labels, ab_idx, 0)
            idx_set = np.delete(idx_set, ab_idx, 0)

    # seperate training and test sets from hash table
    test_obj_dict = {}
    test_idx = []
    for k in test_num_dict.keys():
        obj_idx = np.where(labels == label_dict["label"][k])[0]
        np.random.shuffle(obj_idx)
        test_k_idx = obj_idx[:test_num_dict[k]]
        test_idx += test_k_idx.tolist()

        k_idx_set = idx_set[test_k_idx]
        test_obj_dict[k] = {}
        for j in k_idx_set:
            test_obj_dict[k][hash_table[str(int(j))]["ztf_id"]] = str(int(j))
        
    # write test_obj_dict.json
    with open(output_path + "testset_obj.json", "w") as outfile:
        json.dump(test_obj_dict, outfile, indent = 4)

    train_imageset, train_metaset, train_labels = np.delete(imageset, test_idx, 0), np.delete(metaset, test_idx, 0), np.delete(labels, test_idx, 0)
    test_imageset, test_metaset, test_labels = np.take(imageset, test_idx, 0), np.take(metaset, test_idx, 0), np.take(labels, test_idx, 0)

    train_imageset = np.nan_to_num(train_imageset)
    train_metaset = np.nan_to_num(train_metaset)
    test_imageset = np.nan_to_num(test_imageset)
    test_metaset = np.nan_to_num(test_metaset)

    return train_imageset, train_metaset, train_labels, test_imageset, test_metaset, test_labels



if __name__ == '__main__':

    label_path = 'label_dict.json'
    label_dict = open(label_path,'r')
    label_dict = json.loads(label_dict.read())

    hash_path = 'hash_table.json'
    hash_table = open(hash_path, 'r')
    hash_table = json.loads(hash_table.read())

    filepath = 'r_peak_set.hdf5'

    preprocessing(filepath, label_dict, hash_table)
    

        # print(np.all(np.isnan(imageset)))

    # image normalization: in the build_dataset process
    # print(imageset[1,1,:].shape)
    # imageset = np.apply_along_axis(zscale, 2, imageset)
    # imageset = np.apply_along_axis(image_normalization, 2, imageset)

    # meta normailization: we have the NormalizationLayer in the model, so this step is removed.

    # re-label based on requirements
    # for k in relabel_dict["classify"].keys():
    #     for j in relabel_dict["classify"][k]:
    #         labels[labels == j+100] = relabel_dict["relabel"][k]
    