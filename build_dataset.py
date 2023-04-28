#! ~/anaconda3/envs/astro_py8/bin/python python3
# from ctypes.wintypes import tagMSG
from logging import raiseExceptions
from multiprocessing import managers
from tkinter.tix import COLUMN
import pandas as pd
import numpy as np
import json
import os
import re

import tensorflow as tf
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from tensorflow.keras import models
from astropy.io import fits
from astropy import visualization
from astropy.nddata import Cutout2D
from astropy.wcs.wcs import WCS
import h5py

from multiprocessing import Process, Manager, Value


obj_re = re.compile('ZTF')
sci_re = re.compile('sci')
diff_re = re.compile('diff')
ref_re = re.compile('ref')

def check_shape(img):
    if img.shape != (60,60) or np.all(np.isnan(img)):
        return False
    else:
        return True

def check_bogus(model, img, threhold = 0.5):
    img = img.reshape(1,60,60,1)
    result = model.predict(img)[0][1]
    return True if result >= threhold else False
    

def cutout_img(data, header, ra, dec, size = 60):
    # cutout the image to proposed size
    pixels = WCS(header=header).all_world2pix(ra,dec,1)
    pixels = [int(x) for x in pixels]
    cutout = Cutout2D(data, position=pixels, size=size)
    return cutout.data

def get_shaped_image_simple(img, size = 60, tolerance = 2):
    '''
    roughly cut and complement the image
    '''
    for i in [0,1]:
        if img.shape[i] > size:
            if i == 0:   
                img = img[:size,:]
            else:
                img = img[:,:size]
        if size - img.shape[i] <= tolerance and size - img.shape[i] > 0:
            while img.shape[i] < 60:
                if i == 0:
                    img = np.append(img, img[-1,:].reshape(1,60), axis = i)
                else:
                    img = np.append(img, img[:,-1].reshape(60, 1), axis = i)
    return img


def get_shaped_image(filename, ra, dec):
    '''
    cutout img with 60x60
    leave those imgs with very varying sizes orginally.
    '''
    if filename[-2:]!='fz':
        f = fits.open(filename,ignore_missing_end=True)  # open a FITS file
        hdr = f[0].header 
        data = f[0].data
        if hdr['NAXIS1'] >= 60 and hdr['NAXIS2'] >= 60:
            data = cutout_img(data, hdr, ra, dec)
        return data

    else:
        f = fits.open(filename,ignore_missing_end=True)
        f.verify('fix')
        hdr = f[1].header
        data = f[1].data
        if hdr['NAXIS1'] >=60 and hdr['NAXIS2'] >= 60:
            data = cutout_img(data, hdr, ra, dec)
        return data
    

def save_to_h5py(dataset, metaset, labels, idx_set, filepath):
    print('image shape: ', dataset.shape)
    print('meta shape: ', metaset.shape)
    print('label shape: ', labels.shape)

    f1 = h5py.File(filepath, "w")
    f1.create_dataset("imageset", dataset.shape, dtype='f', data=dataset)
    f1.create_dataset("label", labels.shape, dtype='i', data=labels)
    f1.create_dataset("metaset", metaset.shape, dtype='f', data=metaset)
    f1.create_dataset("idx_set", idx_set.shape, dtype='f', data=idx_set)
    f1.close()


def add_obj_meta(obj, obj_path, filefracday):
    meta = pd.read_csv(obj_path + '/' + obj + '/obj_meta4ML.csv')
    d_row = meta.loc[meta.filefracday == int(filefracday)]
    # print(d_row)
    # d_row = d_row.fillna(0) # replace with zero, more discussion !!!
    new_row = [d_row['candi_mag'].values[0],d_row['disc_mag'].values[0], d_row['delta_t_discovery'].values[0], d_row['delta_t_recent'].values[0], d_row['delta_mag_discovery'].values[0], d_row['delta_mag_recent'].values[0], d_row['delta_host_mag'].values[0]] 

    return new_row

def add_host_meta(obj, host_path, only_complete = True):

    def add_mag(line, band):
        if line[band+'Ap'] != np.nan:
            return line[band+'Ap']
        elif line[band+'PSF'] != np.nan:
            return line[band+'PSF']
        else:
            return 0 # replace with zero, more discussion !!!

    # print(host_path+'/'+obj+'.csv')
    if os.path.exists(host_path+'/'+obj+'.csv'):
        meta = pd.read_csv(host_path+'/'+obj+'.csv') # .fillna(0)
        line = meta.iloc[0]
        # if Ap is not avalidable, replace with PSF
        h_row = []
        for b in ['g','r','i','z','y', 'g-r_', 'r-i_']:
            h_row.append(add_mag(line, b))
        host_meta = h_row
        host_meta = np.array(host_meta)
        return np.nan_to_num(host_meta).tolist()
    elif only_complete == True:
        return None
    else:
        host_meta = [0]*7
        return np.array(host_meta)
    
        
def add_sherlock_info(sherlock_table, ztf_id, properties = list, only_complete = True):
    line = sherlock_table[sherlock_table.object_id == ztf_id]
    sherlock_meta = []
    if len(line) == 0:
        if only_complete == True:
            return None
        else:
            return np.array([0]*len(properties))
    else:
        for p in properties:
            sherlock_meta.append(line[p].values[0]) 
        return sherlock_meta

def zscale(img):
    # where to set up pencentages.
    vmin = visualization.ZScaleInterval().get_limits(img)[0]
    # vmax = visualization.ZScaleInterval().get_limits(img)[1]
    # img[img > vmax] = vmax
    img[img < vmin] = vmin 
    img = np.nan_to_num(img, nan = vmin)
    return img


def image_normal(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))


def img_reshape(img):
    return img.reshape(img.shape[0], img.shape[1], 1)

def meta_normal(meta):
    pass

def get_obs_image(obj_path, filefracday, no_diff, f, BClassifier):
    '''
    get images from one observation.
    '''
    obs_listdir = os.listdir(obj_path + '/'+f+'/' + filefracday)
    sci_img = list(filter(sci_re.match, obs_listdir))[0]
    diff_img = list(filter(diff_re.match, obs_listdir))[0]
    band_path = obj_path + '/'+ f
    band_listdir = os.listdir(band_path)
    ref_img = list(filter(ref_re.match,band_listdir))[0]
    sci_data = get_shaped_image_simple(fits.getdata(obj_path + '/'+f+'/' + filefracday+'/' + sci_img))
    diff_data = get_shaped_image_simple(fits.getdata(obj_path + '/'+f+'/' + filefracday+'/' + diff_img))
    ref_data = get_shaped_image_simple(fits.getdata(obj_path + '/'+f+'/' + ref_img))
    
    if no_diff is True:
        if check_shape(sci_data) and check_shape(ref_data):
            sci_data = img_reshape(image_normal(zscale(sci_data))) 
            ref_data = img_reshape(image_normal(zscale(ref_data)))
            if check_bogus(BClassifier, sci_data) and check_bogus(BClassifier, ref_data):  
                comb_data = np.concatenate((sci_data, ref_data), axis = -1)
                return comb_data
            else:
                return None
        else:
            return None
    else:
        if check_shape(sci_data) and check_shape(diff_data) and check_shape(ref_data):
            sci_data = img_reshape(image_normal(zscale(sci_data))) 
            ref_data = img_reshape(image_normal(zscale(ref_data)))
            diff_data = img_reshape(image_normal(zscale(diff_data)))
            if check_bogus(BClassifier, sci_data) and check_bogus(BClassifier, ref_data) and check_bogus(BClassifier, diff_data):
                comb_data = np.concatenate((sci_data, ref_data), axis = -1)
                comb_data = np.concatenate((comb_data, diff_data), axis = -1)
                return comb_data
            else:
                return None
        else:
            return None

def get_single_transient_peak(ztf_id, image_path, host_path, band = 'r', no_diff = True, BClassifier = None):
  
    if band == 'r':
        f = '2'
    elif band == 'g':
        f = '1'
    else:
        raise Exception('WARNING: unvalid band!')

    sci_re = re.compile('sci')
    diff_re = re.compile('diff')
    ref_re = re.compile('ref')

    obj_path = os.path.join(image_path, ztf_id)
    j = obj_path + '/mag_with_img.json'
    j = open(j, 'r')
    mag_wg = json.loads(j.read())

    mj = obj_path + '/image_meta.json'
    mj = open(mj, 'r')
    meta = json.loads(mj.read())
    
    candids = mag_wg["candidates_with_image"]['f' + f]

    if len(candids) >= 1 and meta['f'+f]["obj_with_no_ref"] is False:
        # if the object doesn't have reference image, skip

        mags = np.array([[m['magpsf'],m["filefracday"]] for m in candids])
        idx = np.argmin(mags[:,0])
        filefracday = mags[idx][1]
        # print(obj)
        if filefracday not in meta['f'+f]["obs_with_no_diff"]:
            # if the peak image doesn't have difference images, skip
            obs_listdir = os.listdir(obj_path + '/'+f+'/' + filefracday)
            sci_img = list(filter(sci_re.match, obs_listdir))[0]
            diff_img = list(filter(diff_re.match, obs_listdir))[0]
            band_path = obj_path + '/'+ f
            band_listdir = os.listdir(band_path)
            ref_img = list(filter(ref_re.match,band_listdir))[0]
            sci_data = get_shaped_image_simple(fits.getdata(obj_path + '/'+f+'/' + filefracday+'/' + sci_img))
            diff_data = get_shaped_image_simple(fits.getdata(obj_path + '/'+f+'/' + filefracday+'/' + diff_img))
            ref_data = get_shaped_image_simple(fits.getdata(obj_path + '/'+f+'/' + ref_img))
            
            if no_diff is True:
                if check_shape(sci_data) and check_shape(ref_data):
                    sci_data = img_reshape(image_normal(zscale(sci_data))) 
                    ref_data = img_reshape(image_normal(zscale(ref_data)))
                    if check_bogus(BClassifier, sci_data) and check_bogus(BClassifier, ref_data):  
                        comb_data = np.concatenate((sci_data, ref_data), axis = -1)
            else:
                if check_shape(sci_data) and check_shape(diff_data) and check_shape(ref_data):
                    sci_data = img_reshape(image_normal(zscale(sci_data))) 
                    ref_data = img_reshape(image_normal(zscale(ref_data)))
                    diff_data = img_reshape(image_normal(zscale(diff_data)))
                    if check_bogus(BClassifier, sci_data) and check_bogus(BClassifier, ref_data) and check_bogus(BClassifier, diff_data):
                        comb_data = np.concatenate((sci_data, ref_data), axis = -1)
                        comb_data = np.concatenate((comb_data, diff_data), axis = -1)
               
            # add meta data
            meta_data = add_obj_meta(ztf_id, image_path, filefracday) + add_host_meta(ztf_id, host_path)
       
            return np.array(comb_data), np.array(meta_data)

def multi_worker_task(obj, f, image_path, host_path, sherlock_table, BClassifier, label_dict, mp_meta_set, mp_image_set, mp_label_set, mp_hash_table, mp_idx_set, mp_idx):
    obj_path = os.path.join(image_path, obj)
    j = obj_path + '/mag_with_img.json'
    j = open(j, 'r')
    mag_wg = json.loads(j.read())

    mj = obj_path + '/image_meta.json'
    mj = open(mj, 'r')
    meta = json.loads(mj.read())
    candids = mag_wg["candidates_with_image"]['f' + f]

    if len(candids) >= 1 and meta['f'+f]["obj_with_no_ref"] is False:
        print(obj)
        ffds = np.array([m["filefracday"] for m in candids])
        for ffd in ffds:
            if ffd not in meta['f'+f]["obs_with_no_diff"]:
                comb_data = get_obs_image(obj_path, ffd, True, f, BClassifier)
                if comb_data is not None:    
                    obj_meta = add_obj_meta(obj, image_path, ffd)
                    host_meta = add_host_meta(obj, host_path, True)
                    sher_meta = add_sherlock_info(sherlock_table, obj, ['separationArcsec'], True)
                   
                    if host_meta is not None and sher_meta is not None:   
                        meta_data = obj_meta + host_meta + sher_meta
                        mp_meta_set.append(meta_data)
                        mp_image_set.append(comb_data)
                        mp_label_set.append(label_dict[meta['label']])
                        mp_hash_table[mp_idx.value] = {'ztf_id': obj, 'ffd':ffd, 'type': meta['label'], 'label':label_dict[meta['label']]}
                        mp_idx_set.append(mp_idx.value)
                        mp_idx.value += 1
            

def single_band_all_db(image_path, host_path, sherlock_path, output_path, label_dict, band = 'r', no_diff = True, only_complete = True,  BClassifier = None):
    '''
    get all observations for each transient, and treat each of them as a sample.
    '''
    

    file_names = os.listdir(image_path) 
    file_names = list(filter(obj_re.match, file_names))

    sherlock_table = pd.read_csv(sherlock_path)
    
    image_set = []
    meta_set = []
    label_set = []
    idx_set = []
    hash_table = {}

    if band == 'r':
        f = '2'
    elif band == 'g':
        f = '1'
    else:
        raise Exception('WARNING: unvalid band!')


    manager = Manager()
    mp_image_set = manager.list()
    mp_meta_set = manager.list()
    mp_label_set = manager.list()
    mp_idx_set = manager.list()
    mp_hash_table = manager.dict()
    mp_idx = Value('i', 0)

    n = 0
    p = []
    for obj in file_names:
        p.append(Process(target=multi_worker_task, args = (obj, f, image_path, host_path, sherlock_table, BClassifier, label_dict, mp_meta_set, mp_image_set, mp_label_set, mp_hash_table, mp_idx_set, mp_idx)))
        p[n].start()
        p[n].join()
        n += 1
      
    image_set += mp_image_set
    meta_set += mp_meta_set
    label_set += mp_label_set
    idx_set += mp_idx_set
    hash_table = {**hash_table, **mp_hash_table}
        
    
    save_to_h5py(np.array(image_set), np.array(meta_set), np.array(label_set), np.array(idx_set), output_path + 'data.hdf5')
    with open(output_path + "hash_table.json", "w") as outfile:
        json.dump(hash_table, outfile, indent = 4)
    
def single_band_peak_db(image_path, host_path, sherlock_path, output_path, label_dict, band = 'r', no_diff = True, only_complete = True,  BClassifier = None):
    '''
    This function will choose:
    1. the observations at the peak day - science, reference, and difference images
    2. Meta data for each object
    3. label
    Then, they are stored in a named h5py file with the keywords: image_set, meta_set, labels
    '''
    obj_re = re.compile('ZTF')
    sci_re = re.compile('sci')
    diff_re = re.compile('diff')
    ref_re = re.compile('ref')

    file_names = os.listdir(image_path) 
    file_names = list(filter(obj_re.match, file_names))

    sherlock_table = pd.read_csv(sherlock_path)
    
    image_set = []
    meta_set = []
    label_set = []
    idx_set = []
    hash_table = {}

    mp_idx = 0
    
    if band == 'r':
        f = '2'
    elif band == 'g':
        f = '1'
    else:
        raise Exception('WARNING: unvalid band!')

    # new_TDE = pd.read_csv('../../data/TDE_20221017.csv')['object_id'].tolist() # debug for TDE

    for obj in file_names[:10]:
        obj_path = os.path.join(image_path, obj)
        j = obj_path + '/mag_with_img.json'
        j = open(j, 'r')
        mag_wg = json.loads(j.read())

        mj = obj_path + '/image_meta.json'
        mj = open(mj, 'r')
        meta = json.loads(mj.read())
        
        candids = mag_wg["candidates_with_image"]['f' + f]

        if len(candids) >= 1 and meta['f'+f]["obj_with_no_ref"] is False:
            # if the object doesn't have reference image, skip

            mags = np.array([[m['magpsf'],m["filefracday"]] for m in candids])
            idx = np.argmin(mags[:,0])
            filefracday = mags[idx][1]
            print(obj)
            if filefracday not in meta['f'+f]["obs_with_no_diff"]:
                comb_data = get_obs_image(obj_path, filefracday, no_diff, f, BClassifier)
                if comb_data is not None:
                    obj_meta = add_obj_meta(obj, image_path, filefracday)
                    host_meta = add_host_meta(obj, host_path, only_complete)
                    sher_meta = add_sherlock_info(sherlock_table, obj, ['separationArcsec'], only_complete)
                    print(host_meta, sher_meta)
                    if host_meta is not None and sher_meta is not None:   
                        meta_data = obj_meta + host_meta + sher_meta
                        meta_set.append(meta_data)
                        image_set.append(comb_data)
                        label_set.append(label_dict[meta['label']])
                        hash_table[mp_idx] = {'ztf_id': obj, 'type': meta['label'], 'label':label_dict[meta['label']]}
                        idx_set.append(mp_idx)
                        mp_idx += 1  
                    else:
                        continue
                else:
                    continue
    
    save_to_h5py(np.array(image_set), np.array(label_set), np.array(meta_set), np.array(idx_set), output_path + 'data.hdf5')
    with open(output_path + "hash_table.json", "w") as outfile:
        json.dump(hash_table, outfile, indent = 4)
    
def gr_band_peak_db(image_path, host_path, output_path, label_dict, no_diff = True, BClassifier = None):
    '''
    This function will choose:
    1. the observations at the peak day - science, reference, and difference images
    2. Meta data for each object
    3. label
    Then, they are stored in a named h5py file with the keywords: image_set, meta_set, labels
    '''
    obj_re = re.compile('ZTF')
    sci_re = re.compile('sci')
    diff_re = re.compile('diff')
    ref_re = re.compile('ref')

    file_names = os.listdir(image_path) 
    file_names = list(filter(obj_re.match, file_names))
    
    image_set = []
    meta_set = []
    label_set = []
    idx_set = []
    hash_table = {}

    mp_idx = 0
    
    def stack_image(stack_list):
        comb_data = stack_list[0]
        n = 1
        while n < len(stack_list):
            comb_data = np.concatenate((comb_data, stack_list[n]), axis = -1)
            n += 1
        return comb_data


    for obj in file_names:
        obj_path = os.path.join(image_path, obj)
        j = obj_path + '/mag_with_img.json'
        j = open(j, 'r')
        mag_wg = json.loads(j.read())

        mj = obj_path + '/image_meta.json'
        mj = open(mj, 'r')
        meta = json.loads(mj.read())

        print(obj)

        stack_list = []
        stack_meta = []
        
        flag = True  # only store data when two bands are available
        for f in ['1','2']:
            candids = mag_wg["candidates_with_image"]['f' + f]

            if len(candids)>=1 and meta['f'+f]["obj_with_no_ref"] is False:
                # if the object doesn't have reference image, skip

                mags = np.array([[m['magpsf'],m["filefracday"]] for m in candids])
                idx = np.argmin(mags[:,0])
                
                filefracday = mags[idx][1]
                if filefracday not in meta['f'+f]["obs_with_no_diff"]:
                    # if the peak image doesn't have difference images, skip
                    obs_listdir = os.listdir(obj_path + '/'+f+'/' + filefracday)
                    sci_img = list(filter(sci_re.match, obs_listdir))[0]
                    diff_img = list(filter(diff_re.match, obs_listdir))[0]
                    band_path = obj_path + '/'+ f
                    band_listdir = os.listdir(band_path)
                    ref_img = list(filter(ref_re.match,band_listdir))[0]
                    sci_data = get_shaped_image_simple(fits.getdata(obj_path + '/'+f+'/' + filefracday+'/' + sci_img))
                    diff_data = get_shaped_image_simple(fits.getdata(obj_path + '/'+f+'/' + filefracday+'/' + diff_img))
                    ref_data = get_shaped_image_simple(fits.getdata(obj_path + '/'+f+'/' + ref_img))
                    if no_diff is True:
                        if check_shape(sci_data) and check_shape(ref_data):
                            sci_data = img_reshape(image_normal(zscale(sci_data))) 
                            ref_data = img_reshape(image_normal(zscale(ref_data)))
                            if check_bogus(BClassifier, sci_data) and check_bogus(BClassifier, ref_data):  
                                comb_data = np.concatenate((sci_data, ref_data), axis = -1)
                                stack_list.append(comb_data)
                                # add meta data
                                meta_data = add_host_meta(obj, host_path) + add_obj_meta(obj, image_path, filefracday) 
                                stack_meta += meta_data
                            else:
                                flag = False
                        else:
                            flag = False
                    else:
                        if check_shape(sci_data) and check_shape(ref_data) and check_shape(diff_data):
                            sci_data = img_reshape(image_normal(zscale(sci_data)))
                            ref_data = img_reshape(image_normal(zscale(ref_data)))
                            diff_data = img_reshape(image_normal(zscale(diff_data)))
                            if check_bogus(BClassifier, sci_data) and check_bogus(BClassifier, ref_data) and check_bogus(BClassifier, diff_data):
                                comb_data = np.concatenate((sci_data, ref_data), axis = -1) 
                                comb_data = np.concatenate((comb_data, diff_data), axis = -1)
                                stack_list.append(comb_data)
                                # add meta data
                                meta_data = add_host_meta(obj, host_path) + add_obj_meta(obj, image_path, filefracday)
                                stack_meta += meta_data 
                            else:
                                flag = False
                        else:
                            flag = False        
                else:
                    flag = False
            else:
                flag = False

        if flag:
            comb_data = stack_image(stack_list)
            image_set.append(comb_data)
            label_set.append(label_dict[meta['label']])
            meta_set.append(stack_meta)

            hash_table[mp_idx] = {'ztf_id': obj, 'type': meta['label'], 'label': label_dict[meta['label']]}
            idx_set.append(mp_idx)
            mp_idx += 1
    
    save_to_h5py(np.array(image_set), np.array(label_set), np.array(meta_set), np.array(idx_set), output_path + 'data.hdf5')
    with open(output_path + "hash_table.json", "w") as outfile:
        json.dump(hash_table, outfile, indent = 4)



if __name__ == '__main__':
    band = 'r'
    image_path = '../../../data/image_sets_v3'
    host_path = '../../../data/host_info_r5'
    output_path = '../model_with_data/r_band/all_set_check_bogus_20230321'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    label_path = '../model_labels/label_dict.json'
    label_dict = open(label_path,'r')
    label_dict = json.loads(label_dict.read())

    sherlock_path = '../../../data/ztf_sherlock_matches/ztf_sherlock_host.csv'

    BClassifier = models.load_model('../../bogus_classifier/models/bogus_model_without_zscale')

    single_band_all_db(image_path, host_path, sherlock_path, output_path, label_dict["classify"], band = band, no_diff= True, BClassifier = BClassifier)


    # single_band_peak_db(image_path, host_path, sherlock_path, output_path, label_dict["classify"], 'r', True, True, BClassifier)
    # gr_band_peak_db(image_path, host_path, output_path, label_dict["classify"], no_diff = True)

