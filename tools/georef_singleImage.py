import numpy as np
import cv2
import socket
import matplotlib as mpl
#mpl.use('Agg')
from matplotlib import pyplot as plt
import glob
import os
import multiprocessing
import pdb
import sys

#homebrewed
sys.path.append('../nadir_vis_image/')
import geoRef_vis_ff2 as visible2
sys.path.append('../')
import fireScene2D as fS2D

####################################
def read_cf_location(cf_file):    
    cf_gps_utm = np.zeros([3,4])

    f = open(cf_file)
    lines = f.readlines()
    f.close()
    for i_line, line in enumerate(lines[1:]):
        cf_gps_utm[:,i_line] = line.rstrip().split(' ')[0:3] 

    return cf_gps_utm


####################################
if __name__ == "__main__":
####################################
    
    fireName           = 'fipaper'
    calibration_matrix = None
    resolution         = .005
    domain_size        = 12.

    root_path       = '/home/paugam/Desktop/image_josh/'
    cf_file         = root_path + 'cf_location.txt'
    outDir          = root_path + 'geo_image/'
    path_instrument = root_path

    suffix_image_name_vis = 'layout_'

    utm_pts = read_cf_location(cf_file)

    #define the grid
    center = utm_pts.mean(axis=1)[0:2]
    resolution, grid_e, grid_n = fS2D.defineGrid(center[0],center[1], box_size = domain_size, res = resolution)

    img_ref_filenames = glob.glob(root_path + 'layout_*.png')

    for img_ref_filename in img_ref_filenames:

        img_ref_raw = visible2.load_image(img_ref_filename, calibration_matrix=calibration_matrix, flag_numpy=False)
        nx_img_raw, ny_img_raw = img_ref_raw.shape[:2]

        M, img = visible2.geoRef_fromgcp(fireName, img_ref_filename, img_ref_raw, \
                                grid_e, grid_n, path_instrument,         \
                                utm_pts, suffix_image_name_vis, outDir)
        
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['font.family'] = 'Comic Sans MS'
        mpl.rcParams['axes.linewidth'] = 1
        mpl.rcParams['axes.labelsize'] = 14.
        mpl.rcParams['xtick.labelsize'] = 18.
        mpl.rcParams['ytick.labelsize'] = 18.
        mpl.rcParams['figure.subplot.left'] = .0
        mpl.rcParams['figure.subplot.right'] = 1.
        mpl.rcParams['figure.subplot.top'] = 1.
        mpl.rcParams['figure.subplot.bottom'] = .0
        mpl.rcParams['figure.subplot.hspace'] = 0.05
        mpl.rcParams['figure.subplot.wspace'] = 0.05   
        
        fig = plt.figure(figsize=(10,10))
        ax = plt.subplot(111)
        ax.imshow(np.transpose(img,[1,0,2]),origin='lower')
        plt.axis('off')
        outname = img_ref_filename.split('.')[0]+'_geo.png'
        fig.savefig(outDir+os.path.basename(outname),dpi=600)
