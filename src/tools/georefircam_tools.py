
import numpy as np 
import sys
import os 
import glob 
import pdb 
import cv2
import transformation 

#################################################
def get_cam_loc_angle(rvec,tvec):
    
    rotM_cam = cv2.Rodrigues(rvec)[0]
    cameraPosition = np.array(-(np.matrix(rotM_cam).T * np.matrix(tvec)))
   
    tmp = transformation.euler_from_matrix(rotM_cam,'rzyz')
    cameraAngle = [180 + 180/3.14*tmp[1], -tmp[2]*180/3.14, 180+(tmp[0]*180/3.14)]
    
    return cameraPosition, cameraAngle


############################################
def load_good_lwir_frame_selection(flag_restart, dir_out_npy, path_data, georefMode='SH'):

    if (not(os.path.isfile(dir_out_npy+'../lwir_time_info.npy'))) | (flag_restart == False):
        print('')
        print('loop over lwir ... ', end=' ')
        sys.stdout.flush()
        
        lwir_goeref00    = sorted(glob.glob(dir_out_npy+'*'+georefMode+'*.npy'))
        
        lwir_time = []
        lwir_id = []  

        badId_lwir = get_bad_idLwir(dir_out_npy) 
        lwir_goeref = []
        for file_ in lwir_goeref00:
            id_lwir =  int( os.path.basename(file_).split('.')[0].split('_')[-2] )
            if id_lwir in badId_lwir: continue
            lwir_goeref.append(file_)
            lwir_time.append( np.load(file_, allow_pickle=True, encoding='latin1')[0][1] )
            lwir_id.append( id_lwir)
       
        lwir_time = np.array(lwir_time)
        lwir_id = np.array(lwir_id,dtype=np.int)    

        np.save(dir_out_npy+'../lwir_time_info.npy',[lwir_time,lwir_id])
        np.save(dir_out_npy+'../lwir_selection_filename.npy',lwir_goeref)  
    
    else:
        lwir_time, lwir_id = np.load(dir_out_npy+'../lwir_time_info.npy')
        lwir_goeref_        = np.load(dir_out_npy+'../lwir_selection_filename.npy')
        
        lwir_goeref = []
        for ii,lwir_goeref__ in enumerate(lwir_goeref_):
            try:
                lwir_goeref.append( lwir_goeref__.replace('/media/paugam/goulven/data/', path_data) )
            except:
                lwir_goeref.append( lwir_goeref__.decode().replace('/media/paugam/goulven/data/', path_data) )
        lwir_goeref = np.array(lwir_goeref)

    print('done ')

    return np.array(lwir_id,dtype=np.int), np.array(lwir_time), np.array(lwir_goeref)


############################################3
def load_good_mir_frame_selection(flag_restart, dir_out_npy, plotname, path_data, georefMode='SH'):

    if (not(os.path.isfile(dir_out_npy+'../mir_time_info.npy'))) | (flag_restart == False):
        print('')
        print('loop over mir ... ', end=' ')
        sys.stdout.flush()

        mir_goeref00    = sorted(glob.glob(dir_out_npy+'*'+georefMode+'*.npy'))

        mir_time = []
        mir_id = []
        camera_position = []
        camera_angle = []

        try:
            [badId_mir,badMirTime],[goodMirId,goodMirTime] = np.load(dir_out_npy+'{:s}_badgoodMirIdTime.npy'.format(plotname),allow_pickle=True)
        except:
            [badId_mir,badMirTime],[goodMirId,goodMirTime] = np.load(dir_out_npy+'{:s}_badgoodMirIdTime.npy'.format(plotname),allow_pickle=True, encoding='latin1')

        mir_goeref = []
        for file_ in mir_goeref00:
            id_mir =  int( os.path.basename(file_).split('.')[0].split('_')[-2] )
            if id_mir in badId_mir: continue
            mir_goeref.append(file_)
            try:
                info_ = np.load(file_, allow_pickle=True)[0]
            except:
                info_ = np.load(file_, allow_pickle=True, encoding='latin1')[0]

            mir_time.append(info_[1])
            mir_id.append( id_mir)

            rvec, tvec = info_[-2], info_[-1]
            loc, angle = get_cam_loc_angle(rvec,tvec)
            camera_position.append(loc[:,0])
            camera_angle.append(angle)\
        

        mir_time = np.array(mir_time)
        mir_id = np.array(mir_id,dtype=np.int)

        np.save(dir_out_npy+'../mir_time_info.npy',[mir_time,mir_id])
        np.save(dir_out_npy+'../mir_selection_filename.npy',mir_goeref)
        
        item= (0.,0.,0.,0.,0.,0.,0.,'mm')
        cameraInfo = np.array([item]*len(camera_position),dtype=np.dtype([('time',float),('x_image',float),('y_image',float),('z_image',float),\
                                                       ('azimuthAngle_image',float),('viewAngle_image',float),('tilt_image',float),('srf','U400')]))
        cameraInfo = cameraInfo.view(np.recarray)
        
        camera_position = np.array(camera_position)
        camera_angle    = np.array(camera_angle)

        cameraInfo.time = mir_time
        cameraInfo.x_image = camera_position[:,0]
        cameraInfo.y_image = camera_position[:,1]
        cameraInfo.z_image = camera_position[:,2]
        cameraInfo.viewAngle_image    = camera_angle[:,0]
        cameraInfo.azimuthAngle_image      = camera_angle[:,1]
        cameraInfo.tilt_image = camera_angle[:,2]
        cameraInfo.srf = '../data_static/Camera/agema550/SpectralResponseFunction/agema550.txt'

        np.save(dir_out_npy+'../mir_pose.npy', cameraInfo)

    else:
        mir_time, mir_id = np.load(dir_out_npy+'../mir_time_info.npy')
        mir_goeref_       = np.load(dir_out_npy+'../mir_selection_filename.npy')

        mir_goeref = []
        for ii,mir_goeref__ in enumerate(mir_goeref_):
            try:
                mir_goeref.append( mir_goeref__.replace('/media/paugam/goulven/data/', path_data) )
            except:
                mir_goeref.append( mir_goeref__.decode().replace('/media/paugam/goulven/data/', path_data) )
        mir_goeref = np.array(mir_goeref)

    print('done')

    return np.array(mir_id,dtype=np.int), np.array(mir_time), np.array(mir_goeref)


#############################################################################
def get_bad_idLwir(dirIn):

    filenames = glob.glob(dirIn+'badFrameID*')
    badId_lwir = []
    for filename in filenames:
        badId_lwir_ = np.load(filename)
        [badId_lwir.append(id) for id in badId_lwir_ ]

    return sorted(badId_lwir)

