#! /usr/bin/env python

from logging import raiseExceptions
import pandas as pd 
import numpy as np
import os
import json
from multiprocessing import Pool,cpu_count
import lasair


token = 'f7b4b64c53168512a4bcba06827c6c0015e9c9f6'

def get_json(ztf_id, path):

	L = lasair.lasair_client(token)
	c = L.objects([ztf_id])[0]
	try: # remove non-detections
		temp_list = []
		for cd in c['candidates']:
			if 'candid' in cd.keys():
				temp_list.append(cd)
		c['candidates'] = temp_list
	except raiseExceptions:
		print(c)
	save_path = path + '/' + str(ztf_id) + '.json'
	json_object = json.dumps(c, indent=4)

	outfile = open(save_path, "w") # Writing to sample.json
	outfile.write(json_object)	
	outfile.close()

	return c

if __name__ == '__main__':

	# ztf_info = pd.read_csv('../data/TDE_20221017.csv')
	# id_list = ztf_info['object_id'].tolist()
	# path = '../TDE_20221017'
	# if not os.path.exists(path):
	# 	os.makedirs(path)
	# print(f'starting computations on {cpu_count()} cores')
	# with Pool() as pool:
	# 	pool.starmap(get_json, zip(id_list, np.repeat(path, len(id_list))))
	ztf_id = 'ZTF23aadsfsh'
	path = '../test'
	if not os.path.exists(path):
		os.makedirs(path)
	get_json(ztf_id, path)
	




# https://lasair-ztf.lsst.ac.uk/api/objects/?objectIds=ZTF22aadghqe&token=f7b4b64c53168512a4bcba06827c6c0015e9c9f6&format=json

