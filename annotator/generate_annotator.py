import json, sys, settings
import lasair, os
from astropy.io import fits
# from astropy.io.fits import getdata
import numpy as np
sys.path.append("..") 
import build_dataset
import ztf_image_pipeline 
from sherlock_host_pipeline import get_potential_host
from main import predict_new_transient
from obj_meta_pipeline import collect_meta
from host_meta_pipeline import PS1catalog_host
from obj_meta_pipeline import collect_meta, get_host_mag
from preprocessing import preprocessing, single_transient_preprocessing,open_with_h5py
from tensorflow.keras import models

NEEDLE_PATH = '../../model_with_data/r_band/peak_set_check_bogus_with_distance_only_complete__equal_test_20230223'
LABEL_PATH = '../../model_labels/label_dict_equal_test.json'
BCLASSIFIER_PATH = '/Users/xinyuesheng/Documents/astro_projects/scripts/bogus_classifier/models/bogus_model_without_zscale'
NEEDLE_OBJ_PATH = 'needle_objects'
LABEL_LIST = ['SN', 'SLSN-I', 'TDE']

BClassifier = models.load_model(BCLASSIFIER_PATH)


def get_obj_meta(candidates, candi_idx, disc_mjd, disc_mag, host_mag):
    
    # new_row = [d_row['candi_mag'].values[0],d_row['disc_mag'].values[0], d_row['delta_t_discovery'].values[0], d_row['delta_t_recent'].values[0], d_row['delta_mag_discovery'].values[0], d_row['delta_mag_recent'].values[0], d_row['delta_host_mag'].values[0]] 
    
    candi_mag = candidates[candi_idx]['magpsf']
    delta_t_discovery = round(candidates[candi_idx]['jd'] - disc_mjd - 2400000.5, 5)
    delta_mag_discovery = round(candi_mag - disc_mag, 5)
    if candi_idx < len(candidates) - 1: 
        delta_t_recent = round(candidates[candi_idx]['jd'] - candidates[candi_idx + 1]['jd'], 5)
        delta_mag_recent = round(candi_mag - candidates[candi_idx + 1]['magpsf'], 5)
    else:
        delta_t_recent = 0.0
        delta_mag_recent = 0.0
    delta_host_mag = round(candi_mag - host_mag, 5)

    return [candi_mag, disc_mag, delta_t_discovery, delta_t_recent, delta_mag_discovery, delta_mag_recent, delta_host_mag] 
   




def collect_data_from_lasair(objectId, objectInfo, band = 'r'):
    if band == 'g':
        fid = 1
    elif band == 'r':
        fid = 2

    candidates = objectInfo['candidates']
   
    candidates = [x for x in candidates if 'image_urls' in x.keys() and x['fid'] == fid]
    
    
    disdate = objectInfo['objectData']['discMjd']
    discFilter = objectInfo['objectData']['discFilter']

    # print(objectInfo)

    
    

    if len(candidates) > 1:
        mags = np.array([m['magpsf']for m in candidates])
        idx = np.argmin(mags)

        if discFilter == band:
            discMag = objectInfo['objectData']['discMag']
            discMag = float(discMag.strip(r'\u')[0])
        else:
            discMag = candidates[-1]['magpsf']

        flag = False
        
        while flag == False and idx < len(candidates)-1 and idx >= 0:
            peak_urls = candidates[idx]['image_urls']
            science_url = peak_urls['Science']
            template_url = peak_urls['Template']
            # difference_url = peak_urls['Difference']

            obsjd_path = ztf_image_pipeline.create_path(NEEDLE_OBJ_PATH, objectId)
            sci_fname = "sci_ztf_peak.fits"
            sci_filename = obsjd_path + '/' + sci_fname
            ref_fname = 'ref_ztf_peak.fits'
            ref_filename = obsjd_path + '/' + ref_fname

            if not os.path.exists(sci_filename):
                os.system("curl -o %s %s" % (sci_filename, science_url))
                os.system("curl -o %s %s" % (ref_filename, template_url))
            
            if os.path.getsize(sci_filename) < 800 or os.path.getsize(ref_filename) < 800:
                flag = False
                idx += 1
            else:
                flag = True
                
        if flag == True:
        
            sci_data = build_dataset.get_shaped_image_simple(fits.getdata(sci_filename))
            ref_data = build_dataset.get_shaped_image_simple(fits.getdata(ref_filename))
            
            if build_dataset.check_shape(sci_data) and build_dataset.check_shape(ref_data):
                sci_data = build_dataset.img_reshape(build_dataset.image_normal(build_dataset.zscale(sci_data))) 
                ref_data = build_dataset.img_reshape(build_dataset.image_normal(build_dataset.zscale(ref_data)))
                if build_dataset.check_bogus(BClassifier, sci_data) and build_dataset.check_bogus(BClassifier, ref_data):
                    img_data = np.concatenate((sci_data, ref_data), axis = -1)
                    if img_data is not None:    
                        
                        host_ra, host_dec = objectInfo['sherlock']['raDeg'], objectInfo['sherlock']['decDeg']
                        PS1catalog_host(_id = objectId, _ra = host_ra, _dec = host_dec, save_path=NEEDLE_OBJ_PATH + '/hosts')
                        host_meta = build_dataset.add_host_meta(objectId, host_path = NEEDLE_OBJ_PATH + '/hosts', only_complete = True)
                        if host_meta is None:
                            print('Host meta not found.')
                            return None, None
                        else:
                            obj_meta = get_obj_meta(candidates, idx, disdate, discMag, host_meta[1])
                            sher_meta = [objectInfo['sherlock']['separationArcsec']]
                            host_ra, host_dec = objectInfo['sherlock']['raDeg'], objectInfo['sherlock']['decDeg']
                            meta_data = np.array(obj_meta + host_meta + sher_meta)  
                            return img_data, meta_data
        else:
            print('object %s images no found' % objectId)
            return None, None

    else:
        print('candidates for %s not found.\n' % objectId)
        return None, None
    
    # img_data, meta_data = single_transient_preprocessing(img_data, meta_data)
    


def needle_prediction(img_data, meta_data):
    img_data, meta_data = single_transient_preprocessing(img_data, meta_data)
    TSClassifier = models.load_model(NEEDLE_PATH + '/model_256_3_256_3')
    results = TSClassifier.predict({'image_input': img_data, 'meta_input': meta_data})
    return results


# This function deals with an object once it is received from Lasair
def handle_object(objectId, L, topic_out, threhold = 0.75):
    # from the objectId, we can get all the info that Lasair has
    objectInfo = L.objects([objectId])[0]
    if not objectInfo:
        return 0

    img_data, meta_data = collect_data_from_lasair(objectId, objectInfo, band = 'r')
    if img_data is None and meta_data is None:
        print('object %s failed to be annocated.' % objectId)
    else:
        results = needle_prediction(img_data, meta_data)
    
        print(results)
        
        classdict      = {'SN': str(results[0][0]), 'SLSN-I': str(results[0][1]), 'TDE': str(results[0][2])} 
        if np.max(results[0]) >= threhold:
            classification = LABEL_LIST[np.argmax(results[0])]
        else:
            classification = 'unclear'
        explanation    = 'NEEDLE prediction.'
        # now we annotate the Lasair data with the classification
        L.annotate(
            topic_out, 
            objectId, 
            classification,
            version='0.1', 
            explanation=explanation, 
            classdict=classdict, 
            url='')
        print(objectId, '-- annotated!')

    # get all images

    # print(objectInfo.keys())
    # print(objectInfo['objectId'])
    
    # objectInfo.keys():
    #  -- objectData: about the object and its features
    #  -- candidates: the lightcurve of detections and nondetections
    #  -- sherlock: the sherlock information
    #  -- TNS: any crossmatch with the TNS database

    # analyse object here. The following is a toy annotation
    # use NEEDLE to predict the class

    # STEP 1: add raw image and metadata
    # STEP 2: quality test - preprocessing
    # STEP 3: feed into NEEDLE and get results
    

        
    return 1

#####################################
# first we set up pulling the stream from Lasair
# a fresh group_id gets all, an old group_id starts where it left off
group_id = settings.GROUP_ID

# a filter from Lasair, example 'lasair_2SN-likecandidates'
topic_in = settings.TOPIC_IN

# kafka consumer that we can suck from
consumer = lasair.lasair_consumer('kafka.lsst.ac.uk:9092', group_id, topic_in)

# the lasair client will be used for pulling all the info about the object
# and for annotating it
L = lasair.lasair_client(settings.API_TOKEN)

# TOPIC_OUT is an annotator owned by a user. API_TOKEN must be that users token.
topic_out = settings.TOPIC_OUT

# just get a few to start
max_alert = 100

n_alert = n_annotate = 0
while n_alert < max_alert:
    msg = consumer.poll(timeout=20)
    if msg is None:
        break
    if msg.error():
        print(str(msg.error()))
        break
    jsonmsg = json.loads(msg.value())
    objectId       = jsonmsg['objectId']

    n_alert += 1
    n_annotate += handle_object(objectId, L, topic_out)

print('Annotated %d of %d objects' % (n_annotate, n_alert))