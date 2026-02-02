'''
collection of fct to print DART configuration files
'''


from __future__ import print_function
from __future__ import division
from builtins import range
from past.utils import old_div
import os
import sys
import shutil
import numpy as np
import math
import pdb 
import struct 
import netCDF4
import subprocess
import glob 
from struct import unpack
from xml.dom import minidom
import itertools
import multiprocessing
import importlib
import dart_tools_dao
import pickle 

importlib.reload(dart_tools_dao)

# Dart parameters
DART_HOME = ""  # set here the default path of your DART folder
DART_HOME = os.path.expanduser(DART_HOME)
if len(DART_HOME) == 0:
    try:
        DART_HOME = os.environ['DART_HOME']
    except KeyError:
        DART_HOME = None
if DART_HOME is None:
    raise EnvironmentError("You need to set your DART_HOME path, either by setting the variable above or through system environment.")

DART_LOCAL = ""  # set here the default path of your DART folder
DART_LOCAL = os.path.expanduser(DART_LOCAL)
if len(DART_LOCAL) == 0:
    try:
        DART_LOCAL = os.environ['DART_LOCAL']
    except KeyError:
        DART_LOCAL = None
if DART_LOCAL is None:
    raise EnvironmentError("You need to set your DART_LOCAL path, either by setting the variable above or through system environment.")


DART_VERSION = '5.7.9'
DART_BUILD = 'v1176'

#homebrewed
sys.path.append(DART_LOCAL+'/MySrc/LoadOpticalProperties/')
import loadOpticProp2CoeffDiff
import rteModel
from readRadiativeBudgetFigures import computeExitanceRadiativeBudgetFigure

sys.path.append('./tools/')
import myPickle

#NAME_PHASE_FUN_CO2 = 'CO2'
#NAME_PHASE_FUN_CO = 'CO'
#NAME_PHASE_FUN_H2O = 'H2O'
#NAME_PHASE_FUN_VEG = 'Wildcard'
NAME_PHASE_GND = 'Loam_gravelly_brown'
NAME_PHASE_GND_MOD = 'Loam_gravelly_brown_mod'
NAME_PHASE_BOX = 'Box'
NAME_TEMP_FUN = 'thermal_function_290_310'
NAME_TEMP_BOX = '0K'

#PHASE_FUN_ID_CO2 = 0
#PHASE_FUN_ID_CO = 1
#PHASE_FUN_ID_H2O = 3
#PHASE_FUN_ID_VEG = 4
PHASE_GND_ID = 0
PHASE_GND_MOD_ID = 1
PHASE_BOX_ID_in = 2
PHASE_BOX_ID_out = 3
TEMP_FUN_ID = 0
TEMP_BOX_ID = 1

# Other
NL = '\n'

# Physics constants
#T_AMBIENT = 293.15
MCO2 = 12 + 2*16    # g/mol
MCO = 12 + 16       # g/mol
MH2O = 2*1 + 16     # g/mol
m_1_soot = 1.e-21    # kg
NAVOGADRO = 6.022e23    # mol-1


#NAME_PHASE_FUN_SOOT = 'soot_poitou'
#PHASE_FUN_ID_SOOT = 5




##############################################################
def getTriangleIT_from_op(dirIn,opfront,opBack):
    trianglesFile = dirIn + 'triangles.txt'
    size = os.path.getsize(trianglesFile) // 102
    trianglesId = []
    with open(trianglesFile, "rb") as tf:
        for id in range(size):
            # is quad
            isQuad = tf.read(1)
            # coords x 9
            coords = []
            for i in range(9):
                coords.append(unpack("d", tf.read(8))[0])
            # front optical property type
            frontType = unpack("i", tf.read(4))[0]
            # front optical property index
            frontOpticalPropertyID = unpack("i", tf.read(4))[0]
            # front temperature index
            frontTemperatureID = tf.read(4)
            # double face
            doubleFace = tf.read(1)
            # back optical property type
            backType = unpack("i", tf.read(4))[0]
            # back optical property index
            backOpticalPropertyID = unpack("i", tf.read(4))[0]
            # back temperature index
            backTemperatureID = tf.read(4)
            # surface type
            surfaceType = tf.read(4)
            
            if (frontOpticalPropertyID == opfront) & (backOpticalPropertyID == opBack):
                trianglesId.append(id)
                #print (frontOpticalPropertyID, backOpticalPropertyID)
    return trianglesId


#definition of spectral bad
#for image, only 3.9 micron
###################################################################
def spectral_bands_definition(spectralbandConfigName,getFRP=False,flag_backgrdSim=False):
    
    from spectralBAndsDefinition import spectral_band_39, spectral_band_FrancoisFRP, spectral_band_largeIR, spectral_band_dual_IR, spectral_band_verylargeIR
    from spectralBAndsDefinition import getBandsSpecs

    if getFRP:
        spectral_band_selected = spectral_band_FrancoisFRP
    elif flag_backgrdSim: 
        spectral_band_selected = spectral_band_39
    else: 
        spectral_band_selected = spectral_band_39 # default
        
        if spectralbandConfigName == 'FrancoisFRP':
            spectral_band_selected = spectral_band_FrancoisFRP

        elif spectralbandConfigName == 'Agema550MIR':
            spectral_band_selected = spectral_band_39
        
        elif spectralbandConfigName == 'dualIR':
            spectral_band_selected = spectral_band_dual_IR

        elif spectralbandConfigName == 'largeIR':
            spectral_band_selected = spectral_band_largeIR

        elif spectralbandConfigName == 'verylargeIR':
            spectral_band_selected = spectral_band_verylargeIR

    bandspecs = getBandsSpecs(spectral_band_selected[1])

    return spectral_band_selected, bandspecs


#####################################################
def ensure_dir(f):
    import os
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


#############################################################
def dump_dart_xml(inputConfig, root_postproc, name_simu, 
                  time,time00,i_time, nx,ny,nz,dx,dy,dz, xs,ys,zs, Firescene, outputDir, flag_parallel, getFRP=False,cameraInfo=None,T_ambient=None,
                  flag_test_geo=False,):
   
    #copy template simulation to DART_maket directory
    template_dir = 'template_FT'
    if inputConfig.params_DART['useLux']:
        if name_simu == 'fds_poolFire_kero': 
            template_dir = template_dir.replace('FT', 'lux_radio')
        else:
            template_dir = template_dir.replace('FT', 'lux')
    if flag_test_geo:
        template_dir += '_noAtm_grdBB'

    print('----------')
    print('template = '+template_dir)
    print('----------')

    if time == None:
        print('***pb, no time given to create dart config')
        pdb.set_trace()

    root_dart_simulation = DART_LOCAL+'simulations/'   #inputConfig.params_DART['root_dart_simulation']
    root_dart_tools      = DART_HOME + 'tools/linux/' #inputConfig.params_DART['root_dart_tools']

    dir_out  = root_postproc + 'DART_maket/simulations/'
    
    if inputConfig.params_DART['flag_run_sensitivity']:
        suffix = '_sens'
    else:
        suffix = ''
    if getFRP:
        suffix += '_box'
        #set all temperature on the edge of the domain to zeros for the box
        for key in ['Temp_veg', 'Temp_veg2', 'Temp']:
            Firescene[key][ :, :,-1] = 0
            Firescene[key][ :,-1, :] = 0
            Firescene[key][ :, 0, :] = 0
            Firescene[key][-1, :, :] = 0
            Firescene[key][ 0, :, :] = 0

    DARTimageflag = '{:s}_'.format(inputConfig.params_DART['dart_config_bands']) \
                    if ('dart_config_bands' in inputConfig.params_DART.keys()) else ''

    run_name = '{:s}t_{:03d}_{:02d}_s'.format(DARTimageflag,*np.array(math.modf(round(time,2))*np.array([100,1]),dtype=int)[::-1])+suffix
    run_name_original = dir_out + run_name
    
    #copy template simulation configuration
    if os.path.isdir(run_name_original+'/input/'):
        shutil.rmtree(run_name_original+'/input/')
    if os.path.isdir(run_name_original+'/output/'):
        shutil.rmtree(run_name_original+'/output/')
    ensure_dir(run_name_original)
    ensure_dir(run_name_original+'/input/')
    
    if i_time <=0: 
        template_dir_ = '../data_static/DART/'+template_dir
        flag_run_phase = True
    else:
        run_name00 = '{:s}_{:03d}_{:02d}_s'.format(name_simu,*np.array(math.modf(round(time00,2))*np.array([100,1]),dtype=int)[::-1])+suffix
        run_name_original00 = root_dart_simulation + run_name00
        template_dir_ = run_name_original00
        flag_run_phase = False
        #pdb.set_trace()
        #template_dir_ = os.path.dirname(os.path.abspath(__file__)) + '/../data_static/DART/'+template_dir
        #flag_run_phase = True

    for file_ in glob.glob(template_dir_+'/input/*.txt'): #temperature file needed by phases in lux mode
        shutil.copy(file_, run_name_original+'/input/')
    for file_ in glob.glob(template_dir_+'/input/*.xml'):
        shutil.copy(file_, run_name_original+'/input/')
    ensure_dir(run_name_original+'/output/')
    for file_ in glob.glob(template_dir_+'/output/*'):
        if 'BAND' not in file_:
            if os.path.isfile(file_):
                shutil.copy(file_, run_name_original+'/output/')
            else:
                shutil.copytree(file_+'/', run_name_original+'/output/'+file_.split('/')[-1], )

    outputDir_simu = run_name_original + '/input/'
   
    try: 
        spectralbandConfigName = inputConfig.params_DART['spectralbandConfigName']
        if inputConfig.params_DART['flag_run_sensitivity']: spectralbandConfigName =  'Agema550MIR'
    except: 
        spectralbandConfigName = None
  
    try: 
        fv_thresholds = [inputConfig.params_DART['fv_threshold'], inputConfig.params_DART['aerosol_tracer_threshold']]
    except: 
        fv_thresholds = [inputConfig.params_DART['fv_threshold'], None]

    dart_mode = 'Lux' if inputConfig.params_DART['useLux'] else 'FT'

    if flag_run_phase:
        #write phase (to set up camera)
        #----
        phase_xml_filename = phase_xml(template_dir_, outputDir_simu,cameraInfo,getFRP, 
                                       inputConfig.params_DART['flag_run_sensitivity']&inputConfig.params_sensAna['run_backgrd_simulation'],
                                       spectralbandConfigName, 
                                       nproc=inputConfig.params_DART['nproc'])
      
        #set target pixel size in phase.xml
        with open(phase_xml_filename,'r') as f:
            lines = f.readlines()

        with open(phase_xml_filename,'w') as f:
            dxy = .5*(dx+dy)
            for line in lines:
                line_ = line
                
                if (inputConfig.params_DART['useLux']) & ('pixelSize="1.0"' in line_):
                    line_ = line_.replace(r'pixelSize="1.0"',r'pixelSize="{:.3f}"'.format(dxy))

                if (inputConfig.params_DART['useLux']) & ('maximumRenderingTime="240"' in line_):
                    line_ = line_.replace(r'maximumRenderingTime="240"',r'maximumRenderingTime="{:.0f}"'.format(inputConfig.params_DART['lux_max_time']) )

                if (inputConfig.params_DART['useLux']) & ('targetRayDensityPerPixel="5000"' in line_):
                    line_ = line_.replace(r'targetRayDensityPerPixel="5000"',r'targetRayDensityPerPixel="{:.0f}"'.format(inputConfig.params_DART['lux_rayDensity']) )
               
                if (inputConfig.params_DART['useLux']) & ('meshGrid="0.1"' in line_) & ('lux_radiativeBudget_meshgrid' in inputConfig.params_DART.keys()):
                    line_ = line_.replace(r'meshGrid="0.1"',r'meshGrid="{:.2f}"'.format(inputConfig.params_DART['lux_radiativeBudget_meshgrid']) )

                
                f.write(line_)


    if not(inputConfig.params_DART['useDAO']) :
        print ('DART mode is : '+ dart_mode)
        print ('----------')
        print ('run maket')
        print ('----------')
        
        #write maket
        #----
        maket_xml(outputDir_simu,nx,ny,nz,dx,dy,dz,Firescene,flag_test_geo=flag_test_geo)
        
        #write plot 
        #----
        Firescene.Temp_veg[0,0,-1] = 1000 # trick to force the full fds domain to be loaded in DART
        if (Firescene.Temp_veg.max() != 0.) | (Firescene.Temp.max() > 1200.):
            plot_max_height = plot_xml(template_dir_,inputConfig, name_simu, outputDir_simu,nx,ny,nz,dx,dy,dz,Firescene,T_ambient, fv_thresholds, flag_parallel)
        else:
            plot_max_height = plot_xml_empty(inputConfig, outputDir_simu,nx,ny,nz,dx,dy,dz,Firescene)
        
        print ( '   max plot height: {:.2f}'.format(plot_max_height))
        #create box for calculation of FRP
        if getFRP:
            object_3d_xml(outputDir_simu,nx,ny,nz,dx,dy,dz,Firescene,plot_max_height)   
 
    
    else:
        
        dxy = .5*(dx+dy)
        #daoxy = nx*dx
        #ndaox=1
        #daoz = nz*dz
        #ndaoz = 1
        daoxy = dx
        ndaox= nx
        daoz = dz
        ndaoz = nz
        

        print ('DART mode is : '+ dart_mode)
        print ('----------')     
        print ('run DAO ')
        print ('----------')     
        dir_3DFS = root_postproc + inputConfig.params_3DFS['dir_output']
        flag_extraFDS = True if 'Acetone_dirData' in inputConfig.params_model.keys() else False
        curtainLoc = inputConfig.params_model['curtainLoc'] if 'curtainLoc' in inputConfig.params_model.keys() else None
     
        if True:
            #with open(dir_3DFS+ '3DfS_t_{:04d}_{:02d}.pickle'.format( *np.array(math.modf(round(time,2))*np.array([100,1]),dtype=int)[::-1] ), 'rb') as infile:
            #    xs, ys, zs, _,_ = pickle.load(infile)
            #xs,ys,zs,_, _ = myPickle.decompress(dir_3DFS+ '3DfS_t_{:04d}_{:02d}.pickle.pbz2'.format( *np.array(math.modf(round(time,2))*np.array([100,1]),dtype=int)[::-1] ))
            #_ = None
            loadedData = [xs,ys,zs,None,Firescene]
            output = dart_tools_dao.run_dao_noLoading(inputConfig, name_simu, daoxy, daoz, time, root_postproc, dir_3DFS, dir_out, 
                                                      fv_thresholds, T_ambient, curtainLoc, flag_extraFDS, getFRP, loadedData, flag_run_phase, flag_debug=False)
            
            if 'setup dart simu done' not in output:
                print('something fishy in dart_tools_dao')
                print('stop in dart_tools.py')
                print ('###############')
                pdb.set_trace()
         
        else:
            print ('run dart_tools_dao with DART python: '+ DART_HOME)
            process  = subprocess.Popen([DART_HOME+"/bin/python/python3.8", 
                                         "dart_tools_dao.py", "-name", name_simu, 
                                         "-dxy", '{:}'.format(daoxy),
                                         "-dz",  '{:}'.format(daoz), 
                                         "-t",  '{:}'.format(time),
                                         "-dirPostproc",  root_postproc,
                                         "-dir3dFS",      dir_3DFS, 
                                         "-dirOut",       dir_out, 
                                         "-tempA", '{:}'.format(T_ambient), 
                                         "-extraFDS", '{:}'.format(flag_extraFDS),
                                         "-fv", '{:},{:}'.format(*fv_thresholds),
                                         "-box", '{:}'.format(getFRP),
                                         "-curtainLoc", '{:}'.format(curtainLoc),
                                         "-onecell", '{:}'.format(False), #inputConfig.params_DART['useLux']), 
                                                                          #This cannot work for now, as temperature grid is the same as the voxel, hard set to False
                                         ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate()

            if b'setup dart simu done' not in output:
                print('something fishy in dart_tools_dao')
                print('stop in dart_tools.py')
                print ('###############')
                for line in output.split(b'\n'): 
                    print (line.decode())
                print ('###############')
                for line in error.split(b'\n'): 
                    print (line.decode())
                pdb.set_trace()
            else: 
                for line in output.split(b'\n'):
                    for line_ in line.split(b'\r'):
                        #if b'There are' in line_: print(line_.decode("utf-8")) 
                        print(line_.decode("utf-8")) 
            
    return True


#############################################################
def phase_xml(template_dir, outputDir_simu, cameraInfo, getFRP, flag_backgrdSim, spectralbandConfigName=None, nproc=2):

    '''
    load phase.xml from the template and add camera info
    It requires that the template is having airborne camera ticked and one camera set up
    '''
    #template_dir = 'template'
    #if flag_test_geo:
    #    template_dir = template_dir + '_noAtm_grdBB'

    phasexml_template_file = template_dir+'/input/phase.xml'
    f = open(phasexml_template_file,'r')
    phasexml_template = f.readlines()
    f.close()


    #set number of thread for DART
    ################################
    phasexml_template_old = phasexml_template
    for i_line, line in enumerate(phasexml_template):
        phasexml_template_old[i_line] = line.replace('nbThreads="22"','nbThreads="{:d}"'.format(nproc))
    phasexml_template = phasexml_template_old


    #Insert definition of the pinhole camera or remove all call to pinhole if no camera wanted
    ################################
    idx_to_keep = []
    if cameraInfo is None:
        check = ['Pinhole', 'InsideSensor', 'InsideSensor', 'InsideSensor']
        flag_check = [False, False, False, False]
        camera_def = [[],[],[],[]]
        camera_def_start_line = []
    else:
        check = ['Pinhole', 'InsideSensor', 'InsideSensor', 'InsideSensor']
        flag_check = [False, False, False, False]
        camera_def = [[],[],[],[]]
        camera_def_start_line = []

    phasexml_template_old = phasexml_template
    idx_to_remove = []
    for icheck, check_ in enumerate(check):
        for i_line, line in enumerate(phasexml_template):
   
            if cameraInfo is None:
                if 'SensorImageSimulation' in line:
                        phasexml_template_old[i_line] = line.replace('SensorImageSimulation="1"','SensorImageSimulation="0"')

            if '<'+check_+' ' in line:
                if i_line in camera_def_start_line: continue
                flag_check[icheck] = True
                camera_def_start_line.append(i_line)

            if flag_check[icheck]:  
                camera_def[icheck].append(line)
                idx_to_remove.append(i_line)
            
            if (not(flag_check[0])) & (not(flag_check[1])) & (not(flag_check[2])):
                if (i_line not in idx_to_keep) & (i_line not in idx_to_remove): idx_to_keep.append(i_line)

            if ('</'+check_+'>' in line) & (flag_check[icheck]):
                flag_check[icheck] = False
                if icheck < len(check)-1: break

    if len(camera_def) == 0:
        print() 
        print('missing airborne camera in template phase.xml in ', template_dir +'/input/')
    phasexml_template = [phasexml_template_old[idx] for idx in idx_to_keep]


    #define the spectral band depending of the set-up: ie simulation for total exitance, or only 3.9 micron
    ########################
    phasexml_template_old = phasexml_template
    check = 'SpectralIntervalsProperties'
    spectral_def = []; idx_to_keep = []
    flag_check=False
    for i_line, line in enumerate(phasexml_template):

        if ('order1Products="0"' in line) & (getFRP):
            phasexml_template_old[i_line] = line.replace('order1Products="0"','order1Products="1"')
        if ('radiativeBudgetProducts="0"' in line) & (getFRP):
            phasexml_template_old[i_line] = line.replace('radiativeBudgetProducts="0"','radiativeBudgetProducts="1"')
            
        if '<'+check in line:
            flag_check = True
        
        if flag_check:  
            spectral_def.append(line)
        else:
            idx_to_keep.append(i_line)

        if '</'+check in line:
            flag_check = False
    phasexml_template = [phasexml_template_old[idx] for idx in idx_to_keep]
    if len(spectral_def) != 4:
        print('should have only one band in the template')
        pdb.set_trace()

    #dump phase.xml
    ###############
    phase_xml_file = outputDir_simu + '/phase.xml'
    f = open(phase_xml_file,'w')
    flag_check = False 
    for line in phasexml_template:
        if '<SensorImageSimulation' in line: 
            f.write(line)
            if cameraInfo is not None:
                #and now write camera info
                for i_cam in range(cameraInfo.shape[0]):
                    if cameraInfo.dart_icam[i_cam]!=-999: continue
                    cc_input = 0
                    if cameraInfo.type[i_cam] == 'pinhole':
                        flag_check=np.zeros(10)
                        for line_camera_def in camera_def[0]:
                            line_ = line_camera_def
                            
                            if 'sensorPosX="0.0"' in line_:
                                line_ = line_.replace('sensorPosX="0.0"','sensorPosX="{:.2f}"'.format(cameraInfo.x[i_cam])); cc_input +=1
                                flag_check[0] = 1 
                            if 'sensorPosY="0.0"' in line_:
                                line_ = line_.replace('sensorPosY="0.0"','sensorPosY="{:.2f}"'.format(cameraInfo.y[i_cam])); cc_input +=1
                                flag_check[1] = 1 
                            if 'sensorPosZ="1000.0"' in line_:
                                line_ = line_.replace('sensorPosZ="1000.0"','sensorPosZ="{:.2f}"'.format(cameraInfo.z[i_cam])); cc_input +=1
                                flag_check[2] = 1 
                            if 'cameraPhi="45.0"' in line_:
                                line_ = line_.replace('cameraPhi="45.0"','cameraPhi="{:.2f}"'.format(cameraInfo.phi[i_cam])); cc_input +=1
                                flag_check[3] = 1 
                            if 'cameraTheta="30.0"' in line_:
                                line_ = line_.replace('cameraTheta="30.0"','cameraTheta="{:.2f}"'.format(cameraInfo.theta[i_cam])); cc_input +=1
                                flag_check[4] = 1 
                            if 'nbPixelsX="60"' in line_:
                                line_ = line_.replace('nbPixelsX="60"','nbPixelsX="{:d}"'.format(int(cameraInfo.nx[i_cam]))); cc_input +=1
                                flag_check[5] = 1 
                            if 'nbPixelsY="50"' in line_:
                                line_ = line_.replace('nbPixelsY="50"','nbPixelsY="{:d}"'.format(int(cameraInfo.ny[i_cam]))); cc_input +=1
                                flag_check[6] = 1 
                            if 'aovX="30.0"' in line_:
                                line_ = line_.replace('aovX="30.0"','aovX="{:.2f}"'.format(cameraInfo.fov[i_cam])); cc_input +=1
                                flag_check[7] = 1 
                            if 'aovY="25.0' in line_:
                                line_ = line_.replace('aovY="25.0"','aovY="{:.2f}"'.format(1.*cameraInfo.ny[i_cam]/cameraInfo.nx[i_cam]*cameraInfo.fov[i_cam])); cc_input +=1
                                flag_check[8] = 1 
                            
                            #if 'sizeImageX="40"' in line_:
                            #    line_ = line_.replace('sizeImageX="40"','sizeImageX="{:3.1f}"'.format(cameraInfo.sizeX[i_cam])); cc_input +=1
                            #    flag_check[5] = 1 
                            #if 'sizeImageY="60"' in line_:
                            #    line_ = line_.replace('sizeImageY="60"','sizeImageY="{:3.1f}"'.format(cameraInfo.sizeY[i_cam])); cc_input +=1
                            #    flag_check[6] = 1 
                            if 'cameraRotation="0.0"' in line_:
                                line_ = line_.replace('cameraRotation="0.0"','cameraRotation="{:3.1f}"'.format(cameraInfo.rot_intr[i_cam])); cc_input +=1
                                flag_check[9] = 1 
                            f.write(line_)
                        if cc_input != 10: 
                                print('pinhole need to be set up with default value in the template')
                                print(flag_check)
                                pdb.set_trace()
                    
                    if cameraInfo.type[i_cam] == 'insideScenePinHole':
                        flag_check=np.zeros(11)
                        for line_camera_def in camera_def[1]:
                            line_ = line_camera_def
                            
                            if 'sensorPosX="36.0"' in line_:
                                line_ = line_.replace('sensorPosX="36.0"','sensorPosX="{:.4f}"'.format(cameraInfo.x[i_cam])); cc_input +=1
                                flag_check[0] = 1 
                            if 'sensorPosY="36.0"' in line_:
                                line_ = line_.replace('sensorPosY="36.0"','sensorPosY="{:.4f}"'.format(cameraInfo.y[i_cam])); cc_input +=1
                                flag_check[1] = 1 
                            if 'sensorPosZ="1.6"' in line_:
                                line_ = line_.replace('sensorPosZ="1.6"','sensorPosZ="{:.4f}"'.format(cameraInfo.z[i_cam])); cc_input +=1
                                flag_check[2] = 1 

                            if 'sensorDirectionPhi="45.0"' in line_:
                                line_ = line_.replace('sensorDirectionPhi="45.0"','sensorDirectionPhi="{:.4f}"'.format(cameraInfo.phi[i_cam])); cc_input +=1
                                flag_check[3] = 1 
                            if 'sensorDirectionTetha="98.0"' in line_:
                                line_ = line_.replace('sensorDirectionTetha="98.0"','sensorDirectionTetha="{:.4f}"'.format(cameraInfo.theta[i_cam])); cc_input +=1
                                flag_check[4] = 1 
                            
                            if 'nbPixelsHeight="1000"' in line_:
                                line_ = line_.replace('nbPixelsHeight="1000"','nbPixelsHeight="{:d}"'.format(int(cameraInfo.sizeY[i_cam]))); cc_input +=1
                                flag_check[5] = 1 
                            if 'nbPixelsWidth="1000"' in line_:
                                line_ = line_.replace('nbPixelsWidth="1000"','nbPixelsWidth="{:d}"'.format(int(cameraInfo.sizeX[i_cam]))); cc_input +=1
                                flag_check[6] = 1 
                            
                            if 'height="0.4"' in line_:
                                line_ = line_.replace('height="0.4"','height="{:.4f}"'.format(cameraInfo.imgHeight[i_cam])); cc_input +=1
                                flag_check[7] = 1 
                            if 'width="0.54"' in line_:
                                line_ = line_.replace('width="0.54"','width="{:.4f}"'.format(cameraInfo.imgWidth[i_cam])); cc_input +=1
                                flag_check[8] = 1 
                            if 'tethaOrientation="0.0"' in line_:
                                line_ = line_.replace('tethaOrientation="0.0"','tethaOrientation="{:.4f}"'.format(cameraInfo.rot_intr[i_cam])); cc_input +=1
                                flag_check[9] = 1 
                            if 'focaleDistance="0.1"' in line_:
                                line_ = line_.replace('focaleDistance="0.1"','focaleDistance="{:.4f}"'.format(cameraInfo.focal[i_cam])); cc_input +=1
                                flag_check[10] = 1 

                            f.write(line_)

                        if cc_input != 11: 
                                print('insideScene to be set up with default value in the template, cc_input=', cc_input )
                                print(flag_check)
                                pdb.set_trace()


                    if cameraInfo.type[i_cam] == 'insideSceneFishEye':
                        flag_check=np.zeros(10)
                        for line_camera_def in camera_def[2]:
                            line_ = line_camera_def
                            
                            if 'sensorPosX="36.0"' in line_:
                                line_ = line_.replace('sensorPosX="36.0"','sensorPosX="{:.4f}"'.format(cameraInfo.x[i_cam])); cc_input +=1
                                flag_check[0] = 1 
                            if 'sensorPosY="36.0"' in line_:
                                line_ = line_.replace('sensorPosY="36.0"','sensorPosY="{:.4f}"'.format(cameraInfo.y[i_cam])); cc_input +=1
                                flag_check[1] = 1 
                            if 'sensorPosZ="1.6"' in line_:
                                line_ = line_.replace('sensorPosZ="1.6"','sensorPosZ="{:.4f}"'.format(cameraInfo.z[i_cam])); cc_input +=1
                                flag_check[2] = 1 

                            if 'sensorDirectionPhi="45.0"' in line_:
                                line_ = line_.replace('sensorDirectionPhi="45.0"','sensorDirectionPhi="{:.4f}"'.format(cameraInfo.phi[i_cam])); cc_input +=1
                                flag_check[3] = 1 
                            if 'sensorDirectionTetha="98.0"' in line_:
                                line_ = line_.replace('sensorDirectionTetha="98.0"','sensorDirectionTetha="{:.4f}"'.format(cameraInfo.theta[i_cam])); cc_input +=1
                                flag_check[4] = 1 
                            
                            if 'nbPixelsPerAxis="1001"' in line_:
                                line_ = line_.replace('nbPixelsPerAxis="1001"','nbPixelsPerAxis="{:d}"'.format(int(cameraInfo.fishEyeNbrPix[i_cam]))); cc_input +=1
                                flag_check[5] = 1 
                            if 'zenithMaximum="90.0"' in line_:
                                line_ = line_.replace('zenithMaximum="90.0"','zenithMaximum="{:.4f}"'.format(cameraInfo.fishEyeZenithMax[i_cam])); cc_input +=1
                                flag_check[6] = 1 
                            if 'zenithMinimum="0.0"' in line_:
                                line_ = line_.replace('zenithMinimum="0.0"','zenithMinimum="{:.4f}"'.format(cameraInfo.fishEyeZenithMin[i_cam])); cc_input +=1
                                flag_check[7] = 1
                            if 'radius="0.01"' in line_:
                                line_ = line_.replace('radius="0.01"', 'radius="{:.4f}"'.format(cameraInfo.fishEyeradius[i_cam])); cc_input +=1
                                flag_check[8] = 1
                            if 'tethaOrientation="0.0"' in line_:
                                line_ = line_.replace('tethaOrientation="0.0"','tethaOrientation="{:.4f}"'.format(cameraInfo.fishEyeTheta[i_cam])); cc_input +=1
                                flag_check[9] = 1

                            f.write(line_)

                        if cc_input != 10: 
                                print('insideSceneFishEye to be set up with default value in the template, cc_input=', cc_input )
                                print(flag_check)
                                pdb.set_trace()

                    if cameraInfo.type[i_cam] == 'insideSceneOrtho':
                        flag_check=np.zeros(10)
                        for line_camera_def in camera_def[3]:
                            line_ = line_camera_def
                            
                            if 'sensorPosX="36.0"' in line_:
                                line_ = line_.replace('sensorPosX="36.0"','sensorPosX="{:.4f}"'.format(cameraInfo.x[i_cam])); cc_input +=1
                                flag_check[0] = 1 
                            if 'sensorPosY="36.0"' in line_:
                                line_ = line_.replace('sensorPosY="36.0"','sensorPosY="{:.4f}"'.format(cameraInfo.y[i_cam])); cc_input +=1
                                flag_check[1] = 1 
                            if 'sensorPosZ="1.6"' in line_:
                                line_ = line_.replace('sensorPosZ="1.6"','sensorPosZ="{:.4f}"'.format(cameraInfo.z[i_cam])); cc_input +=1
                                flag_check[2] = 1 

                            if 'sensorDirectionPhi="45.0"' in line_:
                                line_ = line_.replace('sensorDirectionPhi="45.0"','sensorDirectionPhi="{:.4f}"'.format(cameraInfo.phi[i_cam])); cc_input +=1
                                flag_check[3] = 1 
                            if 'sensorDirectionTetha="98.0"' in line_:
                                line_ = line_.replace('sensorDirectionTetha="98.0"','sensorDirectionTetha="{:.4f}"'.format(cameraInfo.theta[i_cam])); cc_input +=1
                                flag_check[4] = 1 
                            
                            if 'height="0.72"' in line_:
                                line_ = line_.replace('height="0.72"','height="{:.4f}"'.format(cameraInfo.imgHeight[i_cam])); cc_input +=1
                                flag_check[5] = 1 
                            if 'width="1.28"' in line_:
                                line_ = line_.replace('width="1.28"','width="{:.4f}"'.format(cameraInfo.imgWidth[i_cam])); cc_input +=1
                                flag_check[6] = 1 
                            
                            if 'nbPixelsHeight="720"' in line_:
                                line_ = line_.replace('nbPixelsHeight="720"','nbPixelsHeight="{:d}"'.format(int(cameraInfo.sizeX[i_cam]))); cc_input +=1
                                flag_check[7] = 1 
                            if 'nbPixelsWidth="1280"' in line_:
                                line_ = line_.replace('nbPixelsWidth="1280"','nbPixelsWidth="{:d}"'.format(int(cameraInfo.sizeY[i_cam]))); cc_input +=1
                                flag_check[8] = 1 
                            if 'tethaOrientation="0.0"' in line_:
                                line_ = line_.replace('tethaOrientation="0.0"','tethaOrientation="{:.4f}"'.format(cameraInfo.rot_intr[i_cam])); cc_input +=1
                                flag_check[9] = 1 

                            f.write(line_)

                        if cc_input != 10: 
                                print('insideScene ortho to be set up with default value in the template, cc_input=', cc_input )
                                print(flag_check)
                                pdb.set_trace()



        elif '<SpectralIntervals' in line:
            f.write(line)
            
            spectral_band_selected, _ = spectral_bands_definition( spectralbandConfigName, getFRP, flag_backgrdSim)

            spectral_band_XX = spectral_band_selected[1]
            for line_ in spectral_band_XX:
                f.write(line_)
        
        #elif '<CommonParameters' in line:
        elif 'commonSkylCheckBox="1" irraDef="0"/>' in line:
            f.write(line)
            line_skyl = []
            nbre_band = spectral_band_selected[0]
            for i_band in range(nbre_band): # 15 band in spectral_band_FrancoisFRP
                line_skyl.append('<SpectralIrradianceValue Skyl="0" bandNumber="{:d}" irradiance="0"/>'.format(i_band) )
            for line_ in line_skyl:
                f.write(line_+'\n')

        elif '<SpectralIrradianceValue' in line:
            continue

        else:
            f.write(line)
   
    f.close() 

    return phase_xml_file


#############################################################
def maket_xml(outputDir_simu,nx,ny,nz,dx,dy,dz,Firescene, flag_test_geo=False):
    
    phase_grd = NAME_PHASE_GND
    phase_grd_id = PHASE_GND_ID
    if flag_test_geo:
        phase_grd    = 'box_inside'
        phase_grd_id = 2

    Lx = nx*dx
    Ly = ny*dy
    Lz = nz*dz
    
    fm = open(os.path.join(outputDir_simu,'maket.xml'),'w')
    
    maket_desc =    '<?xml version="1.0" encoding="UTF-8"?>' + NL + \
                    '<DartFile build="{1}" version="{0}">'.format(DART_VERSION, DART_BUILD) + NL + \
                    '    <Maket dartZone="0" exactlyPeriodicScene="0" useRandomGenerationSeed="0">' + NL + \
                    '        <Scene>' + NL + \
                    '            <CellDimensions roundToExact="0" x="{:.3f}" z="{:.3f}"/>'.format(dx,dz) + NL + \
                    '            <SceneDimensions x="{0:.4f}" y="{1:.4f}"/>'.format(Lx,Ly) + NL + \
                    '        </Scene>' + NL + \
                    '        <RandomGenerationParameters generationSeed="733426921"/>' + NL + \
                    '        <Soil>' + NL + \
                    '            <OpticalPropertyLink ident="{0}" indexFctPhase="{1}" type="0"/>'.format(phase_grd,phase_grd_id) + NL + \
                    '            <ThermalPropertyLink' + NL + \
                    '                idTemperature="{0}" indexTemperature="{1}"/>'.format(NAME_TEMP_FUN,TEMP_FUN_ID) + NL + \
                    '            <Topography presenceOfTopography="1">' + NL + \
                    '            <TopographyProperties fileName="DEM.mp#"/>' + NL + \
                    '            </Topography>' + NL + \
                    '            <DEM_properties createTopography="1">' + NL + \
                    '            <DEMGenerator caseDEM="5" outputFileName="DEM.mp#">' + NL + \
                    '            <DEM_5 dataEncoding="0" dataFormat="8" fileName="dem_raster.img"/>' + NL + \
                    '            </DEMGenerator>' + NL + \
                    '            </DEM_properties>' + NL + \
                    '        </Soil>' + NL + \
                    '        <LatLon altitude="0.0" latitude="0.0" longitude="0.0"/>' + NL + \
                    '    </Maket>' + NL + \
                    '</DartFile>'
    
    fm.write(maket_desc)
    
    fm.close()

    # and create a raster file for a fake DEM. This is just a trick to assign the ground T to the first level of the temperature.txt
    create_raster_dem(outputDir_simu,nx,ny,nz,dx,dy,dz)


#############################################################
def create_raster_dem(outputDir_simu,nx,ny,nz,dx,dy,dz):

    zero_level = 1.0663275133993011e-05
    dem = np.zeros([nx,ny]) + 0.9*dz
    dem[0,0] = zero_level

    print('   dimension of the DEM: {:d}x{:d}'.format(nx,ny))

    file_dem = outputDir_simu + '/dem_raster.img'
    f = open(file_dem,'wb')
    for i in range(nx):
        for j in range(ny):
            f.write(struct.pack('d',dem[i,j]))
    f.close()


#############################################################
def object_3d_xml(outputDir_simu,nx,ny,nz,dx,dy,dz,Firescene,plot_max_height):
    
    xpos = nx*dx/2.
    ypos = ny*dy/2.
    zpos = 0.#-dz/2.
    
    xscale = (nx-1)*dx/2.   # because the original box size is 2.
    yscale = (ny-1)*dy/2.   # because the original box size is 2.
    zscale = (dz + max([plot_max_height,2*dz]))/2.       # because box is partly below groud level
    #zscale = nz*dz/2.       # because box is partly below groud level
    
    fobj = open(os.path.join(outputDir_simu,'object_3d.xml'),'w')
    
    box_desc =[ '<?xml version="1.0" encoding="UTF-8"?>\n',
                '<DartFile version="5.5.3">\n',
                '    <object_3d>\n',
                '        <Types>\n',
                '            <DefaultTypes>\n',
                '                <DefaultType indexOT="101" name="Default_Object" typeColor="125 0 125"/>\n',
                '                <DefaultType indexOT="102" name="Leaf" typeColor="0 175 0"/>\n',
                '            </DefaultTypes>\n',
                '            <CustomTypes>\n',
                '                <Type indexOT="103" name="box" typeColor="125 0 125"/>\n',
                '            </CustomTypes>\n',
                '        </Types>\n',
                '        <ObjectList>\n',
                '            <Object file_src="3D_Objects/box.obj" hasGroups="1"\n',
                '                hidden="0" isDisplayed="1" name="Object" num="0" objectColor="125 0 125">\n',
                '                <GeometricProperties>\n',
                '                    <PositionProperties xpos="{0}" ypos="{1}" zpos="{2}"/>\n'.format(xpos,ypos,zpos),
                '                    <Dimension3D xdim="2.0000009536743164" ydim="2.0" zdim="2.0"/>\n',
                '                    <ScaleProperties \n',
                '                        xscale="{0}" yscale="{1}" zscale="{2}"\n'.format(xscale,yscale,zscale),
                '                        xScaleDeviation="0" yScaleDeviation="0.0" zScaleDeviation="0.0" />\n',
                '                    <RotationProperties xRotDeviation="0.0" xrot="0.00"\n',
                '                        yRotDeviation="0.0" yrot="0.00"\n',
                '                        zRotDeviation="0.0" zrot="0.00"/>\n',
                '                </GeometricProperties>\n',
                '                <ObjectOpticalProperties doubleFace="0" isLAICalc="0"\n',
                '                    isSingleGlobalLai="0" sameOPObject="0"/>\n',
                '                <ObjectTypeProperties sameOTObject="0"/>\n',
                '                <Groups>\n',
                '                    <Group hasElements="0" hidden="0" isLAICalc="0"\n',
                '                        name="Cube.004_Cube.005_Face2" num="1">\n',
                '                        <GroupOpticalProperties doubleFace="1" sameOPGroup="1">\n',
                '                            <OpticalPropertyLink ident="box_outside"\n',
                '                                indexFctPhase="3" type="0"/>\n',
                '                            <ThermalPropertyLink idTemperature="0K" indexTemperature="1"/>\n',
                '                            <BackFaceOpticalProperty>\n',
                '                                <OpticalPropertyLink ident="box_inside"\n',
                '                                    indexFctPhase="2" type="0"/>\n',
                '                            </BackFaceOpticalProperty>\n',
                '                            <BackFaceThermalProperty>\n',
                '                                <ThermalPropertyLink idTemperature="0K" indexTemperature="1"/>\n',
                '                            </BackFaceThermalProperty>\n',
                '                        </GroupOpticalProperties>\n',
                '                        <GroupTypeProperties sameOTGroup="1">\n',
                '                            <ObjectTypeLink identOType="box" indexOT="103"/>\n',
                '                        </GroupTypeProperties>\n',
                '                    </Group>\n',
                '                    <Group hasElements="0" hidden="0" isLAICalc="0"\n',
                '                        name="Cube.002_Cube.003_Face1" num="2">\n',
                '                        <GroupOpticalProperties doubleFace="1" sameOPGroup="1">\n',
                '                            <OpticalPropertyLink ident="box_outside"\n',
                '                                indexFctPhase="3" type="0"/>\n',
                '                            <ThermalPropertyLink idTemperature="0K" indexTemperature="1"/>\n',
                '                            <BackFaceOpticalProperty>\n',
                '                                <OpticalPropertyLink ident="box_inside"\n',
                '                                    indexFctPhase="2" type="0"/>\n',
                '                            </BackFaceOpticalProperty>\n',
                '                            <BackFaceThermalProperty>\n',
                '                                <ThermalPropertyLink idTemperature="0K" indexTemperature="1"/>\n',
                '                            </BackFaceThermalProperty>\n',
                '                        </GroupOpticalProperties>\n',
                '                        <GroupTypeProperties sameOTGroup="1">\n',
                '                            <ObjectTypeLink identOType="box" indexOT="103"/>\n',
                '                        </GroupTypeProperties>\n',
                '                    </Group>\n',
                '                    <Group hasElements="0" hidden="0" isLAICalc="0"\n',
                '                        name="Cube.003_Cube.004_Face5" num="3">\n',
                '                        <GroupOpticalProperties doubleFace="1" sameOPGroup="1">\n',
                '                            <OpticalPropertyLink ident="box_outside"\n',
                '                                indexFctPhase="3" type="0"/>\n',
                '                            <ThermalPropertyLink idTemperature="0K" indexTemperature="1"/>\n',
                '                            <BackFaceOpticalProperty>\n',
                '                                <OpticalPropertyLink ident="box_inside"\n',
                '                                    indexFctPhase="2" type="0"/>\n',
                '                            </BackFaceOpticalProperty>\n',
                '                            <BackFaceThermalProperty>\n',
                '                                <ThermalPropertyLink idTemperature="0K" indexTemperature="1"/>\n',
                '                            </BackFaceThermalProperty>\n',
                '                        </GroupOpticalProperties>\n',
                '                        <GroupTypeProperties sameOTGroup="1">\n',
                '                            <ObjectTypeLink identOType="box" indexOT="103"/>\n',
                '                        </GroupTypeProperties>\n',
                '                    </Group>\n',
                '                    <Group hasElements="0" hidden="0" isLAICalc="0"\n',
                '                        name="Cube.005_Cube.006_Face4" num="4">\n',
                '                        <GroupOpticalProperties doubleFace="1" sameOPGroup="1">\n',
                '                            <OpticalPropertyLink ident="box_outside"\n',
                '                                indexFctPhase="3" type="0"/>\n',
                '                            <ThermalPropertyLink idTemperature="0K" indexTemperature="1"/>\n',
                '                            <BackFaceOpticalProperty>\n',
                '                                <OpticalPropertyLink ident="box_inside"\n',
                '                                    indexFctPhase="2" type="0"/>\n',
                '                            </BackFaceOpticalProperty>\n',
                '                            <BackFaceThermalProperty>\n',
                '                                <ThermalPropertyLink idTemperature="0K" indexTemperature="1"/>\n',
                '                            </BackFaceThermalProperty>\n',
                '                        </GroupOpticalProperties>\n',
                '                        <GroupTypeProperties sameOTGroup="1">\n',
                '                            <ObjectTypeLink identOType="box" indexOT="103"/>\n',
                '                        </GroupTypeProperties>\n',
                '                    </Group>\n',
                '                    <Group hasElements="0" hidden="0" isLAICalc="0"\n',
                '                        name="Cube.001_Cube.002_Face3" num="5">\n',
                '                        <GroupOpticalProperties doubleFace="1" sameOPGroup="1">\n',
                '                            <OpticalPropertyLink ident="box_outside"\n',
                '                                indexFctPhase="3" type="0"/>\n',
                '                            <ThermalPropertyLink idTemperature="0K" indexTemperature="1"/>\n',
                '                            <BackFaceOpticalProperty>\n',
                '                                <OpticalPropertyLink ident="box_inside"\n',
                '                                    indexFctPhase="2" type="0"/>\n',
                '                            </BackFaceOpticalProperty>\n',
                '                            <BackFaceThermalProperty>\n',
                '                                <ThermalPropertyLink idTemperature="0K" indexTemperature="1"/>\n',
                '                            </BackFaceThermalProperty>\n',
                '                        </GroupOpticalProperties>\n',
                '                        <GroupTypeProperties sameOTGroup="1">\n',
                '                            <ObjectTypeLink identOType="box" indexOT="103"/>\n',
                '                        </GroupTypeProperties>\n',
                '                    </Group>\n',
                '                </Groups>\n',
                '            </Object>\n',
                '        </ObjectList>\n',
                '        <ObjectFields/>\n',
                '   </object_3d>\n',
                '</DartFile>\n']
    
    for line in box_desc:
        fobj.write(line)
    
    fobj.close()


#####################################################
def plot_xml_empty(inputConfig, outputDir_simu,nx,ny,nz,dx,dy,dz,Firescene):
    
    fx = open(os.path.join(outputDir_simu,'plots.xml'),'w')
    topxml = '<?xml version="1.0" encoding="UTF-8"?>' + NL\
        + '<DartFile version="{0}">'.format(DART_VERSION) + NL\
        + '    <Plots addExtraPlotsTextFile="0" isVegetation="0">' + NL\
        + '        <ImportationFichierRaster/>' + NL
    bottomxml = '    </Plots>' + NL\
        + '</DartFile>' + NL
    fx.write(topxml)
    fx.write(bottomxml)
    fx.close()
   
    ft = open(os.path.join(outputDir_simu,'temperatures.txt'),'w') 
    #dump ground temperature in the file temperature.txt (first layer)
    line_t = []
    for i in range(nx):
        for j in range(ny):
            dT = Firescene.Temp_grd[i,j,0]
            if dT<0.1:    # DART does not accept null temperatures for non empty cells
                dT = 0.1
            #ft.write("{0:8.3f} ".format(dT))
            line_t.append("{0:8.3f} ".format(dT))
            #ft.write('\n')
            #line_t.append('\n')
        #ft.write('\n')
        line_t.append('\n')
    line_t.append('\n')

    for k in range(nz):
        for i in range(nx):
            for j in range(ny):
                
                # Write the temperature of the voxel (here voxel == cell) in temperature.txt
                try:
                    if Firescene.MassDry[i,j,k] > 0:
                        dT = Firescene.Temp_veg[i,j,k]
                    else:
                        dT = Firescene.Temp[i,j,k]
                except AttributeError:
                    dT = Firescene.Temp[i,j,k]
                if dT<0.1:    # DART does not accept null temperatures for non empty cells
                    dT = 0.1
                
                #ft.write("{0:8.3f} ".format(dT))
                line_t.append("{0:8.3f} ".format(dT))
                # end of writing in temperature.txt
            #ft.write('\n')
            line_t.append('\n')
        #ft.write('\n')
        line_t.append('\n')

    for line_ in line_t:
        ft.write(line_)
    
    ft.close()

    return 0


#############################################################
def get_lut_gas(lut, specie, ):
    idx_specie = []
    lut_temp = []
    for ii, name in enumerate(lut.name):
        if specie == name.split('_')[0].lower():
            idx_specie.append(ii)
            lut_temp.append( float(lut.name[ii].split('_')[1].split('k')[0]) )
   
    lut = lut[(idx_specie,)]
    lut_temp = np.array(lut_temp)
   
    lut_out = np.array([('m',0,'m','m',0.)]*lut.shape[0],dtype=np.dtype([ (x,y[0]) for x,y in lut.dtype.fields.items()] + [('temp',np.dtype(float))]))
    lut_out = lut_out.view(np.recarray)
    lut_out.id = lut.id
    lut_out.ident = lut.ident
    lut_out.name = lut.name
    lut_out.temp = lut_temp

    return lut_out

#############################################################
def getNameAndIdFromGasAndTemp(lut, specie, temperature = None):

    if specie == 'soot': 
        idx = np.where(lut.name == 'soot_poitou')
        return lut.ident[idx][0], lut.id[idx][0]
    
    elif specie == 'vegetation': 
        idx = np.where(lut.name == 'rayleigh_1_0')
        return lut.ident[idx][0], lut.id[idx][0]

    else: #deal with gas
        idx_ = np.abs( lut.temp - temperature).argmin()

        return lut.ident[idx_], lut.id[idx_]


#############################################################
def star_get_plot_description (arg):
    return get_plot_description(*arg)

#############################################################
def get_plot_description (k,i,j, dx, dy, dz, Firescene, lut, soot_fv_max, soot_fv_threshold, aerosol_fv_max, aerosol_fv_threshold, T_ambient, nx, ny, nz):
   
    max_height_plot = None
    nplot_veg       = 0
    nplot_gas       = 0
    nplot_plu       = 0
    line_xx = []
    line_tt = []
    
    lut_co2, lut_co, lut_h2o, lut_opticProp_coeff_diff, lut_opticProp_coeff_diff = lut

    '''
    Each cell is a plot (a truncated cylinder with a base and a height)
    Each plot needs :
    - the coordonates of each corner (4) of the base
    - the height of the base
    - the height of the plot
    - the number of different particles and for each one :
        - the number density
        - the index and name of the optical property file (the same as in
        "Optical & temperature properties")
    - the index and name of the temperature function (the same as in
    "Optical & temperature properties") which is ignored since the
    temperature.txt file will be provided to DART
    '''
    
    # Base ABCD of the plot (A,B,C,D in the anticlockwise)
    xA = i*dx      + dx*0.01
    yA = j*dy      + dy*0.01
    xB = (i+1)*dx  - dx*0.01
    yB = j*dy      + dy*0.01
    xC = (i+1)*dx  - dx*0.01
    yC = (j+1)*dy  - dy*0.01
    xD = i*dx      + dx*0.01
    yD = (j+1)*dy  - dy*0.01
    
    # height_base = (k+1)*dz  # flame "floating" above the ground
    height_base = (k+1)*dz + dz*0.01 # Ronan: add k+1, this is for the DEM
    height_plot = dz - 2*dz*0.01
    
    nb_part = 4
    m_CO2  = Firescene.CO2[i,j,k]   * 1.e3 #g
    m_CO   = Firescene.CO[i,j,k]    * 1.e3
    try:
        m_H2O = Firescene.H2O[i,j,k]    * 1.e3 
    except: 
        m_H2O = 0. 

    #m_soot = Firescene.soot[i,j,k] #* 1.e3
    
    dCO2 = round(old_div(m_CO2, MCO2) * NAVOGADRO )#* 1.e-15)     # because densities are in 1e15 m-3 in DART inputs
    dCO  = round(old_div(m_CO, MCO)  * NAVOGADRO )#* 1.e-15)
    dH2O = round(old_div(m_H2O, MH2O) * NAVOGADRO )# * 1.e-15)
    #dsoot = round(m_soot / m_1_soot )#* 1.e-15)          ############################## check optical properties of soot
  
    NAME_PHASE_FUN_CO2 ,PHASE_FUN_ID_CO2 = getNameAndIdFromGasAndTemp(lut_co2, 'co2', temperature=Firescene.Temp[i,j,k])
    NAME_PHASE_FUN_CO,  PHASE_FUN_ID_CO  = getNameAndIdFromGasAndTemp(lut_co,  'co' , temperature=Firescene.Temp[i,j,k])
    NAME_PHASE_FUN_H2O, PHASE_FUN_ID_H2O = getNameAndIdFromGasAndTemp(lut_h2o, 'h2o', temperature=Firescene.Temp[i,j,k])

    try:
        NAME_PHASE_FUN_SOOT, PHASE_FUN_ID_SOOT = getNameAndIdFromGasAndTemp(lut_opticProp_coeff_diff, 'soot' )  
    except: 
        pdb.set_trace()

    NAME_PHASE_FUN_VEG, PHASE_FUN_ID_VEG   = getNameAndIdFromGasAndTemp(lut_opticProp_coeff_diff, 'vegetation' ) 
  
    #try:
    #    ksoot_fds = Firescene.kappa_fds[i,j,k]
    #except AttributeError:
    #    ksoot_fds = 0
    #try:
    #    ksoot_poitou = Firescene.kappa_poitou[i,j,k]
    #except AttributeError:
    #    ksoot_poitou = 0
    #try:
    #    ksoot_wfds = Firescene.kappa_wfds[i,j,k]
    #except AttributeError:
    #    ksoot_wfds = 0
    
   
    #now we just have one optical properties model for soot
    
    # RONAN: 
    sootXMLvalue = Firescene.fv[i,j,k]
    aerosolXMLvalue = Firescene.fv_aerosol[i,j,k]
    #if whichSootProperties == 0:
    #    sootXMLvalue = dsoot
    #    soottest = m_soot
    #elif whichSootProperties == 1:
    #    sootXMLvalue = ksoot_fds
    #elif whichSootProperties == 2:
    #    sootXMLvalue = ksoot_poitou
    #    soottest = ksoot_poitou
    #elif whichSootProperties == 3:
    #    sootXMLvalue = ksoot_wfds
    #else:
    #    sootXMLvalue = dsoot
    #    soottest = m_soot               

    
    # If there is no particle, do not create any plot (otherwise computing_time++)
    # if (dCO2+dCO+dsoot+dH2O)>0.0:
    #                print '---{0}'.format(soottest)
    #                print '---{0}'.format(maxksoot)
    #if 0 == 0:    ############################################# just to reduce the number of plots
   

    #
    # add flame materials
    #
    #if (Firescene.fv[i,j,k] > soot_fv_max * soot_fv_threshold) & (Firescene.Temp_veg[i,j,k] <= T_ambient): #remove filter on Tambient when we have 
    if Firescene.fv[i,j,k] > soot_fv_max * soot_fv_threshold:    ############################################# just to reduce the number of plots
        max_height_plot = height_base+height_plot
        # Formatting of the plot description in xml
        plot_desc = '        <Plot form="0" hidden="0" isDisplayed="1" repeatedOnBorder="1" type="3">' + NL\
                +   '            <Polygon2D>' + NL
        # type 3 : air plot
        plot_desc += '                <Point2D x="{0:.4f}" y="{1:.4f}"/>'.format(xA,yA) + NL\
                +    '                <Point2D x="{0:.4f}" y="{1:.4f}"/>'.format(xB,yB) + NL\
                +    '                <Point2D x="{0:.4f}" y="{1:.4f}"/>'.format(xC,yC) + NL\
                +    '                <Point2D x="{0:.4f}" y="{1:.4f}"/>'.format(xD,yD) + NL\
                +    '            </Polygon2D>' + NL\
                +    '            <PlotAirProperties nbParticule="{0}" verticalFillMode="0">'.format(nb_part) + NL\
                +    '            <AirGeometry baseheight="{0}" height="{1}" stDev="0.0"/>'.format(height_base,height_plot) + NL
                
        # soot
        plot_desc += '                <AirOpticalProperties extinctionCoefficient="{0}">'.format(sootXMLvalue) + NL\
                +    '                    '\
                +    '<AirOpticalPropertyLink ident="{0}" indexFctPhase="{1}"/>'.format(NAME_PHASE_FUN_SOOT,PHASE_FUN_ID_SOOT) + NL\
                +    '                </AirOpticalProperties>' + NL
        
        # Temperature
        plot_desc += '                '\
                +    '<ThermalPropertyLink idTemperature="{0}" indexTemperature="{1}"/>'.format(NAME_TEMP_FUN,TEMP_FUN_ID) + NL\
                +    '                '\
                +    '<GroundThermalPropertyLink idTemperature="{0}" indexTemperature="{1}"/>'.format(NAME_TEMP_FUN,TEMP_FUN_ID) + NL

        if dCO> 0: 
            # CO
            plot_desc += '                <AirOpticalProperties extinctionCoefficient="{0}">'.format(dCO) + NL\
                    +    '                    '\
                    +    '<AirOpticalPropertyLink ident="{0}" indexFctPhase="{1}"/>'.format(NAME_PHASE_FUN_CO,PHASE_FUN_ID_CO) + NL\
                    +    '                </AirOpticalProperties>' + NL
        if dCO2 > 0: 
            # CO2
            plot_desc += '                <AirOpticalProperties extinctionCoefficient="{0}">'.format(dCO2) + NL\
                    +    '                    '\
                    +    '<AirOpticalPropertyLink ident="{0}" indexFctPhase="{1}"/>'.format(NAME_PHASE_FUN_CO2,PHASE_FUN_ID_CO2) + NL\
                    +    '                </AirOpticalProperties>' + NL
                
        if dH2O>0: 
            # Water
            plot_desc += '                <AirOpticalProperties extinctionCoefficient="{0}">'.format(dH2O) + NL\
                    +    '                    '\
                    +    '<AirOpticalPropertyLink ident="{0}" indexFctPhase="{1}"/>'.format(NAME_PHASE_FUN_H2O,PHASE_FUN_ID_H2O) + NL\
                    +    '                </AirOpticalProperties>' + NL
        
        plot_desc += '            </PlotAirProperties>' + NL\
                +    '        </Plot>' + NL
        # end of formatting
        
        # Write the description in plots.xml
        line_xx.append(plot_desc)
        #fx.write(plot_desc)
        nplot_gas = 1
    
    #
    # plume aerosols
    #
    try: 
        #if Firescene.fv_aerosol[i,j,k] > aerosol_fv_max * aerosol_fv_threshold:    ############################################# just to reduce the number of plots
        if Firescene.flag_aerosol[i,j,k] >= 0.5:  
            max_height_plot = height_base+height_plot
            
            # Formatting of the plot description in xml
            plot_desc = '        <Plot form="0" hidden="0" isDisplayed="1" repeatedOnBorder="1" type="3">' + NL\
                    +   '            <Polygon2D>' + NL
            # type 3 : air plot
            plot_desc += '                <Point2D x="{0:.4f}" y="{1:.4f}"/>'.format(xA,yA) + NL\
                    +    '                <Point2D x="{0:.4f}" y="{1:.4f}"/>'.format(xB,yB) + NL\
                    +    '                <Point2D x="{0:.4f}" y="{1:.4f}"/>'.format(xC,yC) + NL\
                    +    '                <Point2D x="{0:.4f}" y="{1:.4f}"/>'.format(xD,yD) + NL\
                    +    '            </Polygon2D>' + NL\
                    +    '            <PlotAirProperties nbParticule="{0}" verticalFillMode="0">'.format(nb_part) + NL\
                    +    '            <AirGeometry baseheight="{0}" height="{1}" stDev="0.0"/>'.format(height_base,height_plot) + NL
            
            if dCO2 > 0:
                # CO2
                plot_desc += '                <AirOpticalProperties extinctionCoefficient="{0}">'.format(dCO2) + NL\
                        +    '                    '\
                        +    '<AirOpticalPropertyLink ident="{0}" indexFctPhase="{1}"/>'.format(NAME_PHASE_FUN_CO2,PHASE_FUN_ID_CO2) + NL\
                        +    '                </AirOpticalProperties>' + NL
            
            # Temperature
            plot_desc += '                '\
                    +    '<ThermalPropertyLink idTemperature="{0}" indexTemperature="{1}"/>'.format(NAME_TEMP_FUN,TEMP_FUN_ID) + NL\
                    +    '                '\
                    +    '<GroundThermalPropertyLink idTemperature="{0}" indexTemperature="{1}"/>'.format(NAME_TEMP_FUN,TEMP_FUN_ID) + NL

            if dCO > 0:
                # CO
                plot_desc += '                <AirOpticalProperties extinctionCoefficient="{0}">'.format(dCO) + NL\
                        +    '                    '\
                        +    '<AirOpticalPropertyLink ident="{0}" indexFctPhase="{1}"/>'.format(NAME_PHASE_FUN_CO,PHASE_FUN_ID_CO) + NL\
                        +    '                </AirOpticalProperties>' + NL
           
            if aerosolXMLvalue > 0: 
                # aerosol
                plot_desc += '                <AirOpticalProperties extinctionCoefficient="{0}">'.format(aerosolXMLvalue) + NL\
                        +    '                    '\
                        +    '<AirOpticalPropertyLink ident="{0}" indexFctPhase="{1}"/>'.format(NAME_PHASE_FUN_SOOT,PHASE_FUN_ID_SOOT) + NL\
                        +    '                </AirOpticalProperties>' + NL
                    
            # Water
            plot_desc += '                <AirOpticalProperties extinctionCoefficient="{0}">'.format(dH2O) + NL\
                    +    '                    '\
                    +    '<AirOpticalPropertyLink ident="{0}" indexFctPhase="{1}"/>'.format(NAME_PHASE_FUN_H2O,PHASE_FUN_ID_H2O) + NL\
                    +    '                </AirOpticalProperties>' + NL
            
            
            plot_desc += '            </PlotAirProperties>' + NL\
                    +    '        </Plot>' + NL
            
            # end of formatting
            
            # Write the description in plots.xml
            line_xx.append(plot_desc)
            #fx.write(plot_desc)
            nplot_plu = 1
    except AttributeError as e:
        #print "AttributeError raised: {1}".format(e.strerror)
        pass


    if True:
        #
        # Vegetation plot
        #
        try:
            #if (Firescene.Temp_veg[i,j,k] > T_ambient) & (k==1):
            #if (Firescene.MassDry[i,j,k] > 0) :
            if Firescene.Temp_veg[i,j,k] > 0:
                max_height_plot = height_base+height_plot     
                kveg = Firescene.kappa_veg[i,j,k]
                
                # Formatting of the plot description in xml
                plot_desc = '        <Plot form="0" hidden="0" isDisplayed="1" repeatedOnBorder="1" type="3">' + NL\
                        +   '            <Polygon2D>' + NL
                    # type 3 : air plot
                plot_desc += '                <Point2D x="{0:.4f}" y="{1:.4f}"/>'.format(xA,yA) + NL\
                        +    '                <Point2D x="{0:.4f}" y="{1:.4f}"/>'.format(xB,yB) + NL\
                        +    '                <Point2D x="{0:.4f}" y="{1:.4f}"/>'.format(xC,yC) + NL\
                        +    '                <Point2D x="{0:.4f}" y="{1:.4f}"/>'.format(xD,yD) + NL\
                        +    '            </Polygon2D>' + NL\
                        +    '            <PlotAirProperties nbParticule="{0}" verticalFillMode="0">'.format(1) + NL\
                        +    '            <AirGeometry baseheight="{0}" height="{1}" stDev="0.0"/>'.format(height_base,height_plot) + NL
                                                
                # Vegetation air plot
                plot_desc += '                <AirOpticalProperties extinctionCoefficient="{0}">'.format(kveg) + NL\
                        +    '                    '\
                        +    '<AirOpticalPropertyLink ident="{0}" indexFctPhase="{1}"/>'.format(NAME_PHASE_FUN_VEG,PHASE_FUN_ID_VEG) + NL\
                        +    '                </AirOpticalProperties>' + NL
                
                # Temperature
                plot_desc += '                '\
                        +    '<ThermalPropertyLink idTemperature="{0}" indexTemperature="{1}"/>'.format(NAME_TEMP_FUN,TEMP_FUN_ID) + NL\
                        +    '                '\
                        +    '<GroundThermalPropertyLink idTemperature="{0}" indexTemperature="{1}"/>'.format(NAME_TEMP_FUN,TEMP_FUN_ID) + NL
                
                plot_desc += '            </PlotAirProperties>' + NL\
                        +    '        </Plot>' + NL
                # end of formatting
                
                # Write the description in plots.xml
                #fx.write(plot_desc)
                line_xx.append(plot_desc)
                nplot_veg = 1
       

        except AttributeError as e:
            #print "AttributeError raised: {1}".format(e.strerror)
            pass
    

    # Write the temperature of the voxel (here voxel == cell) in temperature.txt
    dT = Firescene.Temp[i,j,k]
    if Firescene.Temp_veg[i,j,k] > 0:
        dT = Firescene.Temp_veg[i,j,k]
        #print 'Particle temp. {0:3.2f}K\tcell temp. {1:3.2f}K\tdiff. {2:3.2f}'.format(Firescene.Temp[i,j,k],dT,Firescene.Temp[i,j,k]-dT)
    

    #if dT<0.1:    # DART does not accept null temperatures for non empty cells
    #    dT = 0.1
    
    #ft.write("{0:8.3f} ".format(dT))
    line_tt.append("{0:8.3f} ".format(dT))
    # end of writing in temperature.txt
    
    if (j == ny -1) & (i == nx -1):
        #ft.write('\n')
        line_tt.append('\n')
        
    if j == ny -1:
        #ft.write('\n')
        line_tt.append('\n')
    
    return max_height_plot, nplot_veg, nplot_gas, nplot_plu, line_xx, line_tt


#############################################################
def plot_xml(template_dir, inputConfig, name_simu,outputDir_simu,nx,ny,nz,dx,dy,dz,Firescene,T_ambient, fv_thresholds, flag_parallel):
    
    Lx = nx*dx
    Ly = ny*dy
    Lz = nz*dz
    max_height_plot = -999
    
    shutil.copy2(template_dir+'/input/coeff_diff.xml', outputDir_simu+'coeff_diff.xml')
   
    #load Gas optical Prop
    #database = 'fluid_Gas.db'
    #database = 'fluid_Gas_4D_small_05cm_L001cm.db'
    database =  inputConfig.params_DART['OPdataBase']
    if os.path.isfile(DART_LOCAL + '/database/' + database):
        databaseName = DART_LOCAL + '/database/' + database
    elif os.path.isfile(DART_HOME + '/database/' + database):
        databaseName = DART_HOME + '/database/' + database
    else:
        print ('missing database: ', database)	

    onlySelectedTemp = None
    if 'removeGas' in inputConfig.params_model.keys():
        if inputConfig.params_model['removeGas']: 
            onlySelectedTemp = np.array([1000])
    lut_opticProp_coeff_diff = loadOpticProp2CoeffDiff.loadOpticProp2CoeffDiff(outputDir_simu, databaseName, onlySelectedTemp= onlySelectedTemp)
    lut_co2 = get_lut_gas(lut_opticProp_coeff_diff, 'co2')
    lut_h2o = get_lut_gas(lut_opticProp_coeff_diff, 'h2o')
    lut_co  = get_lut_gas(lut_opticProp_coeff_diff, 'co' )
    
    fx = open(os.path.join(outputDir_simu,'plots.xml'),'w')
    ft = open(os.path.join(outputDir_simu,'temperatures.txt'),'w')
    
    topxml = '<?xml version="1.0" encoding="UTF-8"?>' + NL\
        + '<DartFile build="{1}" version="{0}">'.format(DART_VERSION,DART_BUILD) + NL\
        + '    <Plots addExtraPlotsTextFile="0" isVegetation="0">' + NL\
        + '        <ImportationFichierRaster/>' + NL
    
    bottomxml = '    </Plots>' + NL\
        + '</DartFile>' + NL
    
    fx.write(topxml)
    
    nplot_gas = 0
    nplot_veg = 0
    nplot_plu = 0
    
   # try:
   #     maxksoot = Firescene.kappa_poitou.max()
   # except AttributeError:
   #     maxksoot = Firescene.soot.max()
    #ksoot_poitou = Firescene.kappa_poitou
    
    soot_fv_max = Firescene.fv[:,:,:].max()
    aerosol_fv_max = Firescene.fv_aerosol[:,:,:].max()
    #fv_threshold = 0.01
    soot_fv_threshold = fv_thresholds[0]
    aerosol_fv_threshold = fv_thresholds[1]

    index = np.where( Firescene.fv > soot_fv_max * soot_fv_threshold)
    print('   max       soot fv = {:2.5e}'.format(soot_fv_max)) 
    print('   threshold soot fv = {:2.5e}'.format(soot_fv_max * soot_fv_threshold)) 
    print('   expected nbre plot in the flame =', len(index[0]))
    
    if aerosol_fv_threshold is not None: 
        #index = np.where( Firescene.fv_aerosol > aerosol_fv_max * aerosol_fv_threshold)
        #print('   max       aerosol fv = {:2.5e}'.format(aerosol_fv_max)) 
        #print('   threshold aerosol fv = {:2.5e}'.format(aerosol_fv_max * aerosol_fv_threshold)) 
        #print('   expected nbre plot in the plume =', len(index[0]))
        print('   ## use now flag_aerosol to determine pt in the plume')
        print('   expected nbre plot in the plume =', len(np.where(Firescene.flag_aerosol>0.5)[0]))
    else: 
        if Firescene.fv_aerosol.max()!=0:  
            print ('stop here, there is aerosol concentration but no aerosol_fv_threshold defined in config file')
            sys.exit()
    #dump ground temperature in the file temperature.txt (first layer)
    line_t = []
    for i in range(nx):
        for j in range(ny):
            dT = Firescene.Temp_grd[i,j,0]
            if dT<0.1:    # DART does not accept null temperatures for non empty cells
                dT = 0.1
            #ft.write("{0:8.3f} ".format(dT))
            line_t.append("{0:8.3f} ".format(dT))
            #ft.write('\n')
            #line_t.append('\n')
        #ft.write('\n')
        line_t.append('\n')
    line_t.append('\n')

    print("           \r", end='')
    line_x = []
   

    lut = [lut_co2, lut_co, lut_h2o, lut_opticProp_coeff_diff, lut_opticProp_coeff_diff]
    args = []
    for [k,i,j] in itertools.product(range(nz), range(nx), range(ny)):
        args.append( [k,i,j,dx, dy, dz, Firescene, lut, soot_fv_max,    soot_fv_threshold,   \
                                                        aerosol_fv_max, aerosol_fv_threshold,\
                      T_ambient, nx, ny, nz])

    #flag_parallel = False
    if flag_parallel: 
       
            # set up a pool to run the parallel processing
            cpus = multiprocessing.cpu_count()
            pool = multiprocessing.Pool(processes=cpus)

            # then the map method of pool actually does the parallelisation  
            results = pool.map(star_get_plot_description, args)
            pool.close()
            pool.join()
           
            #for [idx, flame_depth_here,tmp_here] in results:
            for arg,res in zip(args,results):

                max_height_plot_, nplot_veg_, nplot_gas_, nplot_plu_, line_xx, line_tt = res
           
                for line_ in line_xx: 
                    line_x.append(line_)
                for line_ in line_tt: 
                    line_t.append(line_)

                if max_height_plot_ != None: max_height_plot = max([max_height_plot_, max_height_plot])
                nplot_gas += nplot_gas_
                nplot_plu += nplot_plu_
                nplot_veg += nplot_veg_

    else:   
        ii = 0
        ii_max = nx*ny*nz
        for arg in args:
            print(" {0:3d}%\r".format(int(100*float(ii)/ii_max)), end='')
            ii+=1
            
            max_height_plot_, nplot_veg_, nplot_gas_, nplot_plu_, line_xx, line_tt = star_get_plot_description(arg)
      
            for line_ in line_tt: 
                line_t.append(line_)

            if max_height_plot_ != None: max_height_plot = max([max_height_plot_, max_height_plot])
            
            #if nplot_veg ==1 : 
            #    continue
            #else: 
            #    print (arg[:3])
            
            for line_ in line_xx: 
                line_x.append(line_)

            nplot_gas += nplot_gas_
            nplot_plu += nplot_plu_
            nplot_veg += nplot_veg_

            
    '''
    # # Plot for the ground under the flame
    #ids = np.transpose(np.nonzero(Firescene.CO2[:,:,0] + \
    #                               Firescene.CO[:,:,0]  + \
    #                               Firescene.soot[:,:,0])) # indexes where there is something atground level (k=0)
    ids = np.transpose(np.nonzero(Firescene.Temp_grd[:,:,0])) # indexes where there is something atground level (k=0)
    i_min = ids[0,0]
    i_max = ids[len(ids)-1,0]
    j_min = ids[0,1]
    j_max = ids[len(ids)-1,1]
    x_min = i_min*dx
    x_max = (i_max+1)*dx
    y_min = j_min*dy
    y_max = (j_max+1)*dy
    
    plot_desc = '        <Plot form="0" type="0">' + NL\
        + '            <Polygon2D>' + NL\
        + '                <Point2D x="{0}" y="{1}"/>'.format(x_min,y_min) + NL\
        + '                <Point2D x="{0}" y="{1}"/>'.format(x_max,y_min) + NL\
        + '                <Point2D x="{0}" y="{1}"/>'.format(x_max,y_max) + NL\
        + '                <Point2D x="{0}" y="{1}"/>'.format(x_min,y_max) + NL\
        + '            </Polygon2D>' + NL\
        + '            <GroundOpticalPropertyLink ident="{0}" indexFctPhase="{1}" type="0"/>'.format(NAME_PHASE_GND_MOD,PHASE_GND_ID) + NL\
        + '            <GroundThermalPropertyLink idTemperature="{0}" indexTemperature="{1}"/>'.format(NAME_TEMP_FUN,TEMP_FUN_ID) + NL\
        + '        </Plot>' + NL
    
    fx.write(plot_desc)
    '''

    print('   There are {0} plots with vegetation.'.format(nplot_veg))
    print('   There are {0} plots with gas or soot.'.format(nplot_gas))
    print('   There are {0} plots in the plume.'.format(nplot_plu))

    for line_ in line_t:
       ft.write(line_)

    for line_ in line_x:
        fx.write(line_)

    fx.write(bottomxml)
    
    fx.close()
    ft.close()

    return max_height_plot


#############################################################
def dump_netcdf(outputDir,name_simu,x,y,z,Firescene,time,flag_write='append'):

    filename_out = os.path.join(outputDir,"firescene_"+name_simu+".nc")

    if flag_write == 'init' :
        if os.path.isfile(filename_out):
            os.remove(filename_out)
        ncfile = netCDF4.Dataset(filename_out,'w')
        
        ncfile.description = 'example of 3D Fire scene for a fuelf type of 100% cured matted grass with a moisture content of 11%'
       
        # Global attributes
        setattr(ncfile, 'created', 'R. Paugam') 
        setattr(ncfile, 'company', 'KCL Geog')
        setattr(ncfile, 'title', '3Dfirescene')

        # dimensions
        ncfile.createDimension('x',x.shape[0])
        ncfile.createDimension('y',y.shape[0])
        ncfile.createDimension('z',z.shape[0])
        ncfile.createDimension('time',None)

        # variables
        ncx = ncfile.createVariable('x', 'f8', ('x',))
        setattr(ncx, 'long_name', 'x')
        setattr(ncx, 'standard_name', 'x')
        setattr(ncx, 'units','m')

        ncy = ncfile.createVariable('y', 'f8', ('y',))
        setattr(ncy, 'long_name', 'y')
        setattr(ncy, 'standard_name', 'y')
        setattr(ncy, 'units','m')
        
        ncz = ncfile.createVariable('z', 'f8', ('z',))
        setattr(ncz, 'long_name', 'z')
        setattr(ncz, 'standard_name', 'z')
        setattr(ncz, 'units','m')
            
        ncTime = ncfile.createVariable('time', 'f8', ('time',))
        setattr(ncTime, 'long_name', 'time')
        setattr(ncTime, 'standard_name', 'time')
        setattr(ncTime, 'units','seconds since fire ignition')

        ncT    = ncfile.createVariable('T',    'f8', ('time','z', 'y', 'x',), fill_value=-999.) 
        ncT.long_name     = 'Temperature' 
        ncT.standard_name = 'T'
        ncT.units         = 'K' 
        ncT.range         = np.array([270,2000])

        ncCO2  = ncfile.createVariable('CO2',  'f8', ('time','z', 'y', 'x',), fill_value=-999. )
        ncCO2.long_name     = 'CO2 concentration'
        ncCO2.standard_name = 'CO2' 
        ncCO2.units         = '??' 
        ncCO2.range         = np.array([Firescene.CO2.min(),Firescene.CO2.max()])

        ncsoot = ncfile.createVariable('sootfv', 'f8', ('time','z', 'y', 'x',), fill_value=-999. ) 
        ncsoot.long_name      = 'soot volume fraction' 
        ncsoot.standard_name  = 'fv' 
        ncsoot.units          =  '-'
        ncsoot.range          = np.array([Firescene.fv.min(), Firescene.fv.max()]) 

        ncCO   = ncfile.createVariable('CO',   'f8', ('time','z', 'y', 'x',), fill_value=-999. ) 
        ncCO.long_name     = 'CO concentration'
        ncCO.standard_name =  'CO'
        ncCO.units         = '?? '
        ncCO.range          = np.array([Firescene.CO.min(), Firescene.CO.max()]) 
        
        ncH2O   = ncfile.createVariable('H2O',   'f8', ('time','z', 'y', 'x',), fill_value=-999. )
        ncH2O.long_name      =  'H2O concentration'
        ncH2O.standard_name  = 'H2O'
        ncH2O.units          = '??' 
        ncH2O.range          = np.array([Firescene.H2O.min(), Firescene.H2O.max()]) 
        
        ncflag   = ncfile.createVariable('flag',   'f8', ('time','z', 'y', 'x',), fill_value=-999. ) 
        ncflag.long_name     =  'location flame'
        ncflag.standard_name = 'flameLoc'
        ncflag.units         = '-'
        ncflag.range          = np.array([0,1])
        
        ncflagVeg   = ncfile.createVariable('flag_veg',   'f8', ('time','z', 'y', 'x',), fill_value=-999. ) 
        ncflagVeg.long_name     = 'location vegetation'
        ncflagVeg.standard_name = 'vegLoc'
        ncflagVeg.units         =  '-'
        ncflag.range          = np.array([0,1])
                
        ncTveg    = ncfile.createVariable('T_veg',    'f8', ('time','z', 'y', 'x',), fill_value=-999. ) 
        ncTveg.long_name     = 'Temperature vegetation'
        ncTveg.standard_name = 'Tveg'
        ncTveg.units         = 'K'
        ncTveg.range          = np.array([Firescene.Temp_veg.min(), Firescene.Temp_veg.max()]) 
        
        ncflagAerosol   = ncfile.createVariable('flag_aerosol',   'f8', ('time','z', 'y', 'x',), fill_value=-999. ) 
        ncflagAerosol.long_name     =  'location plume aerosol'
        ncflagAerosol.standard_name = 'AerosolLoc'
        ncflagAerosol.units         ='-'
        ncflagAerosol.range         = np.array([0,1])

        ncx[:] = x
        ncy[:] = y
        ncz[:] = z       
        
        i_time = 0

    elif flag_write == 'append':
        ncfile = netCDF4.Dataset(filename_out,'a')
        
        ncx    = ncfile.variables['x']
        ncy    = ncfile.variables['y']
        ncz    = ncfile.variables['z']
        ncTime = ncfile.variables['time']

        ncT       = ncfile.variables['T']
        ncCO2     = ncfile.variables['CO2']
        ncsoot    = ncfile.variables['sootfv']
        ncCO      = ncfile.variables['CO']
        ncH2O      = ncfile.variables['H2O']
        ncflag    = ncfile.variables['flag']
        ncflagVeg = ncfile.variables['flag_veg']
        ncflagAerosol = ncfile.variables['flag_aerosol']
        ncTveg    = ncfile.variables['T_veg']
        
        i_time = ncT.shape[0]

    else:
        print('issue with flag_write_netcdf')
        pdb.set_trcae()

    print('#################')
    print(i_time, time)
    ncTime[i_time] = time

    def for_display_in_netcdf(arr):
        nx,ny,nz = arr.shape
        
        #arr_out = np.zeros([nx,ny,nz])
        #for kk in range(nz):
        #    tmp = arr[:,:,kk]
        #    arr_out[:,:,kk] = tmp.T

        return np.swapaxes(arr,0,2)

    ncT[i_time,:,:,:]       = for_display_in_netcdf(Firescene.Temp[:,:,:])
    ncCO2[i_time,:,:,:]     = for_display_in_netcdf(Firescene.CO2[:,:,:])
    ncCO[i_time,:,:,:]      = for_display_in_netcdf(Firescene.CO[:,:,:])
    ncH2O[i_time,:,:,:]      = for_display_in_netcdf(Firescene.H2O[:,:,:])
    ncsoot[i_time,:,:,:]    = for_display_in_netcdf(Firescene.fv[:,:,:])
    ncflag[i_time,:,:,:]    = for_display_in_netcdf(Firescene.flag[:,:,:])
    ncflagVeg[i_time,:,:,:] = for_display_in_netcdf(Firescene.flag_veg[:,:,:])
    ncTveg[i_time,:,:,:]    = for_display_in_netcdf(Firescene.Temp_veg[:,:,:])
    ncflagAerosol[i_time,:,:,:] = for_display_in_netcdf(Firescene.flag_aerosol[:,:,:])

    ncfile.close()

    #and copy a version without hdf for paraview
    #filename_out_2 = os.path.join(outputDir,"firescene_"+name_simu+"_nohdf.nc")
    #subprocess.call(["nccopy", "-k", "1", filename_out, filename_out_2 ])

    return 0


#############################################################
def dump_netcdf_atm(outputDir,name_simu,x,y,z,Atmscene,time,flag_write='append'):

    filename_out = os.path.join(outputDir,"atmscene_"+name_simu+".nc")

    if flag_write == 'init' :
        if os.path.isfile(filename_out):
            os.remove(filename_out)
        ncfile = netCDF4.Dataset(filename_out,'w')
        
        ncfile.description = 'example of 3D Fire scene for a fuelf type of 100% cured matted grass with a moisture content of 11%'
       
        # Global attributes
        setattr(ncfile, 'created', 'R. Paugam') 
        setattr(ncfile, 'company', 'KCL Geog')
        setattr(ncfile, 'title', '3Dfirescene')

        # dimensions
        ncfile.createDimension('x',x.shape[0])
        ncfile.createDimension('y',y.shape[0])
        ncfile.createDimension('z',z.shape[0])
        ncfile.createDimension('time',None)

        # variables
        ncx = ncfile.createVariable('x', 'f8', ('x',))
        setattr(ncx, 'long_name', 'x')
        setattr(ncx, 'standard_name', 'x')
        setattr(ncx, 'units','m')

        ncy = ncfile.createVariable('y', 'f8', ('y',))
        setattr(ncy, 'long_name', 'y')
        setattr(ncy, 'standard_name', 'y')
        setattr(ncy, 'units','m')
        
        ncz = ncfile.createVariable('z', 'f8', ('z',))
        setattr(ncz, 'long_name', 'z')
        setattr(ncz, 'standard_name', 'z')
        setattr(ncz, 'units','m')
            
        ncTime = ncfile.createVariable('time', 'f8', ('time',))
        setattr(ncTime, 'long_name', 'time')
        setattr(ncTime, 'standard_name', 'time')
        setattr(ncTime, 'units','seconds since fire ignition')

        ncT    = ncfile.createVariable('T',    'f8', ('time','z', 'y', 'x',), fill_value=-999.) 
        ncT.long_name     = 'Temperature' 
        ncT.standard_name = 'T'
        ncT.units         = 'K' 
        ncT.range         = np.array([270,2000])

        ncsv = ncfile.createVariable('tracer', 'f8', ('time','z', 'y', 'x',), fill_value=-999. ) 
        ncsv.long_name      = 'passive tracer' 
        ncsv.standard_name  = 'sv' 
        ncsv.units          =  '-'
        ncsv.range          = np.array([0,1]) 

        ncH2O   = ncfile.createVariable('H2O',   'f8', ('time','z', 'y', 'x',), fill_value=-999. )
        ncH2O.long_name      =  'H2O concentration'
        ncH2O.standard_name  = 'H2O'
        ncH2O.units          = '??' 
        ncH2O.range          = np.array([Atmscene.H2O.min(), Atmscene.H2O.max()]) 
        
        ncflag_grd   = ncfile.createVariable('flag_grd',   'f8', ('time','z', 'y', 'x',), fill_value=-999. ) 
        ncflag_grd.long_name     =  'location heat release'
        ncflag_grd.standard_name = 'heatLoc'
        ncflag_grd.units         = '-'
        ncflag_grd.range          = np.array([0,1])
        
        ncU   = ncfile.createVariable('u',   'f8', ('time','z', 'y', 'x',), fill_value=-999. ) 
        ncU.long_name     = 'u velocity'
        ncU.standard_name = 'u'
        ncU.units         =  '-'
        ncU.range          = np.array([Atmscene.u.min(),Atmscene.u.max()])
                
        ncV   = ncfile.createVariable('v',   'f8', ('time','z', 'y', 'x',), fill_value=-999. ) 
        ncV.long_name     = 'v velocity'
        ncV.standard_name = 'v'
        ncV.units         =  '-'
        ncV.range          = np.array([Atmscene.v.min(),Atmscene.v.max()])
        
        ncW   = ncfile.createVariable('w',   'f8', ('time','z', 'y', 'x',), fill_value=-999. ) 
        ncW.long_name     = 'w velocity'
        ncW.standard_name = 'w'
        ncW.units         =  '-'
        ncW.range          = np.array([Atmscene.w.min(),Atmscene.w.max()])

        ncx[:] = x
        ncy[:] = y
        ncz[:] = z       
        
        i_time = 0

    elif flag_write == 'append':
        ncfile = netCDF4.Dataset(filename_out,'a')
        
        ncx    = ncfile.variables['x']
        ncy    = ncfile.variables['y']
        ncz    = ncfile.variables['z']
        ncTime = ncfile.variables['time']

        ncT       = ncfile.variables['T']
        ncsv      = ncfile.variables['tracer']
        ncH2O      = ncfile.variables['H2O']
        ncflag_grd    = ncfile.variables['flag_grd']
        ncU = ncfile.variables['u']
        ncV = ncfile.variables['v']
        ncW = ncfile.variables['w']
        
        i_time = ncT.shape[0]

    else:
        print('issue with flag_write_netcdf')
        pdb.set_trcae()

    print('#################')
    print(i_time, time)
    ncTime[i_time] = time

    def for_display_in_netcdf(arr):
        nx,ny,nz = arr.shape
        
        #arr_out = np.zeros([nx,ny,nz])
        #for kk in range(nz):
        #    tmp = arr[:,:,kk]
        #    arr_out[:,:,kk] = tmp.T

        return np.swapaxes(arr,0,2)

    ncT[i_time,:,:,:]       = for_display_in_netcdf(Atmscene.temp[:,:,:])
    ncsv[i_time,:,:,:]     = for_display_in_netcdf(Atmscene.tracer[:,:,:])
    ncH2O[i_time,:,:,:]      = for_display_in_netcdf(Atmscene.H2O[:,:,:])
    ncflag_grd[i_time,:,:,:]    = for_display_in_netcdf(Atmscene.flag_grd[:,:,:])
    ncU[i_time,:,:,:]    = for_display_in_netcdf(Atmscene.u[:,:,:])
    ncV[i_time,:,:,:]    = for_display_in_netcdf(Atmscene.v[:,:,:])
    ncW[i_time,:,:,:]    = for_display_in_netcdf(Atmscene.w[:,:,:])

    ncfile.close()

    #and copy a version without hdf for paraview
    #filename_out_2 = os.path.join(outputDir,"firescene_"+name_simu+"_nohdf.nc")
    #subprocess.call(["nccopy", "-k", "1", filename_out, filename_out_2 ])

    return 0

###############################
def saveasnpy(basenameMpsharp,flag_image=False):
    '''
    ##### saveasnpy #####
    save an ilwis image created by dart as a .npy (numpy format) file with additional information.
     
    input
    basenamempsharp   path to the ilwis file to read. do not put the file extension at the end of the path.
    
    output
    dictionnary in the numpy format file containing:
      - the list of values
      - the size of the image (in pixels)
      - the size of the pixels (in m)
      - the spectral band name in dart
      - the iteration name in dart
      - the type of data
    
    to read this dictionnary:
    dartimage = numpy.load('nameofthenpyfile').item()
    dartimage['dx'] to access the x size of a pixel.
    ''' 
    # read the ILWIS files
    tableout,nbRows,nbColumns = mpsharpToAscii(basenameMpsharp,False)
    
    # find additionnal information in the path and in the file 'dart.txt' of the simulation
    head0,tail0 = os.path.split(basenameMpsharp)
    if not(os.path.exists(head0)):
        print('The input path does not seem to exist.')
        sys.exit(0)
    
    if not flag_image:
        head1,tail1 = os.path.split(head0)
        head2,tail2 = os.path.split(head1)
        iterName = tail2
        head3,tail3 = os.path.split(head2)
        typeProductName = tail3
        head4,tail4 = os.path.split(head3)
        bandName = tail4
        with open(os.path.join(head4,'dart.txt'),'r') as fdt:
            fdt.readline()
            fdt.readline()
            fdt.readline()
            lsplitted = fdt.readline().split('*')
            
            try:
                buf = lsplitted[2].split()
                sizePixel = []
                for elt in buf:
                    sizePixel.append(float(elt))
            except: 
                pdb.set_trace()

        # creation of the dictionary
        dartImage = {'imageValues':tableout,'nx':nbRows,'ny':nbColumns,'dx':sizePixel[0],'dy':sizePixel[1], \
            'iterName':iterName,'typeProductName':typeProductName,'bandName':bandName}
        # TODO: check that there is no inversion between x and y
   
    else:
        iterName = 'na'
        bandName = head0.split('/')[-3]
        typeProductName = head0.split('/')[-2]
        dartImage = {'imageValues':tableout,'nx':nbRows,'ny':nbColumns,'dx':-999,'dy':-999, \
            'iterName':iterName,'typeProductName':typeProductName,'bandName':bandName}



    # save
    np.save(basenameMpsharp + '.npy',dartImage)


###########################################################
def mpsharpToAscii(finput, dumpascii = True):
    '''
    #### mpsharpToAscii #####
     read an ILWIS image and return the list of values and its x and y size. If requested, save the list as a text file.
     
     INPUT
     finput        path to the ILWIS file to read. DO NOT put the file extension at the end of the path.
     dumpascii     boolean. True to request the matrix to be saved as a text file, False otherwise.
     
     OUTPUT
     tableout,nbRows,nbColumns + text file (if requested)
     tableout      list of the values of the ILWIS image, rows after rows.
     nbRows        number of rows of the image.
     nbColumns     number of columns of the image.
    #########
    '''
    NBBYTESDOUBLE = 8
   
    head,tail = os.path.split(finput)
    if not(os.path.exists(head)):
        wd = os.getcwd()
        fileBaseName = tail
    else:
        wd = os.path.abspath(head)
        fileBaseName = tail
    
    
    # To get the number of row and columns
    fileMapSize = os.path.join(wd,fileBaseName+'.mpr')
    fms = open(fileMapSize,'r')
    
    l = fms.readline()
    while not('[Map]' in l):    # skip all the lines before finding '[Map]'
        l = fms.readline()
    while not('Size' in l):     # skip all the lines before finding 'Size'
        l = fms.readline()
    
    if l == '':
        print('Pb when reading \'{0}\': the number of rows and columns could not be found.'.format(fileBaseName+'.mpr'))
        sys.exit(0)
    else:       # extracts the x and y sizes of the image
        lsplitted = l.split('=')
        lsplitted = lsplitted[1].split()
        fms.close()
        nbColumns = float(lsplitted[1])
        nbRows = float(lsplitted[0])
    
    # Datafile
    filePath = os.path.join(wd,fileBaseName+'.mp#')
    
    byteTable = []
    
    with open(filePath,'rb') as f:
        
        bytes = f.read(NBBYTESDOUBLE)
        
        # read 8 bytes per 8 bytes and store them into byteTable
        byteTable.append(bytes)
        
        while bytes:
            bytes = f.read(NBBYTESDOUBLE)
            byteTable.append(bytes)
    
    idcol = 0.
    idrow = 0.
    bufstr = ''
    tableout = []
    
    for bbs in byteTable:
        floatnumber = float(unpack('<d',bbs)[0])    # to convert each 8bytes element into a float
        # the binary file was created using C++. 'd' means that a C++ double was written. '<' means
        # the little-endian format was used by the C++ (i don't know why this one...but it works !)
        
        bufstr += '{0:16.8f}'.format(floatnumber)
        tableout.append(floatnumber)
        idcol += 1.
        
        if (idcol >= nbColumns):
            bufstr += '\n'
            idcol = 0.
            idrow += 1.
        
        if (idrow >= nbRows):
            break
    
#    print len(byteTable[int(nbColumns*nbRows):])
#    print '\'{0}\''.format(byteTable[len(byteTable)-1])
    
    # Write the matrix of float as a text file.
    if dumpascii:
        fout = open(fileBaseName + '.txt','w')
        fout.write(bufstr)
        fout.close()
    
    return tableout,nbRows,nbColumns


##################################################
def load_DART_output(inputConfig,root_postproc,FireName,time_requested,sensor='',product='Tapp',band='BAND0',i_cam=1,vz=0,va=0,flag_image_type='direction',flag_reso=None):

    root_dart_simulation = DART_LOCAL+'simulations/'   #inputConfig.params_DART['root_dart_simulation']
    root_dart_tools      = DART_HOME + 'tools/linux/' #inputConfig.params_DART['root_dart_tools']
    
    #dir_in  = root_postproc + 'DART_maket/'
    
    if inputConfig is not None:
        if inputConfig.params_DART['flag_run_sensitivity']:
            suffix = '_sens'
        else:
            suffix = ''
    else:
        suffix = ''

    DARTimageflag = '{:s}_'.format(inputConfig.params_DART['dart_config_bands']) \
                    if ('dart_config_bands' in inputConfig.params_DART.keys()) else ''

    if time_requested is None:
        run_name = FireName
    else:
        run_name = '{:s}t_{:03d}_{:02d}_s'.format(DARTimageflag,*np.array(math.modf(round(time_requested,2))*np.array([100,1]),dtype=int)[::-1])+suffix
        run_name = run_name.replace('{:s}t_'.format(DARTimageflag),FireName+'_')

    if (product != 'Radiance') & ('ITER' not in sensor):
        product=''
    
    suffix = ''
    #if flag_image_type == 'direction':
    #    suffix = 'ImageReechantillonnee({0:.9f},{0:.9f})_OS1/'.format(flag_reso)

    dir_im =  root_dart_simulation + run_name + '/output/'+band+'/'+product+'/'+sensor+'/IMAGES_DART/'+suffix
    if flag_image_type == 'pinhole':
        vz_ = round(vz,1)
        va_ = round(va,1)
        #image = 'ima_camera{0:02d}_VZ={1:03d}_{2:01d}_VA={3:03d}_{4:01d}'.format( i_cam,\
        #       int(vz_), int(10*(round(vz_-int(vz_),1) ) ),  \
        #       int(va_), int(10*(round(va_-int(va_),1) ) )   )
        image =os.path.basename( glob.glob(dir_im + 'ima_camera{0:03d}'.format(i_cam)+'*.mpr')[0]).split('.mp')[0]
    
    elif 'insideScene' in flag_image_type: 
        '''
        vz_ = vz
        va_ = va
        vz_ = (math.pi - vz_/180 * math.pi) * 180 /math.pi
        if vz_ == 0.: 
            va_ = 0
        else:
            if va_ >= 180: 
                va_ = (va_/180 * math.pi - math.pi )
            else: 
                va_ = (va_/180 * math.pi + math.pi )
            
            if va_ > 2*math.pi: va_ -= 2*math.pi
            elif va_ < 0: va_ = math.pi + va_

            va_ = va_ * 180 / math.pi 
        
        vz_ = round(vz_,3)
        va_ = round(va_,3)

        image = 'camera_IS_{0:02d}_VZ={1:03d}_{2:01d}_VA={3:03d}_{4:01d}'.format( i_cam,\
               int(vz_), int(10*(round(vz_-int(vz_),1) ) ),  \
               int(va_), int(10*(round(va_-int(va_),1) ) )   )
        ''' 
        image =os.path.basename( glob.glob(dir_im + 'camera_IS_{0:03d}'.format(i_cam)+'*.mpr')[0]).split('.mp')[0]
        #print(image)
        #if i_cam == 1:
        #    print("################")
        #    print("################ to solve later, constante name of file for image 2 ###############")
        #    print("################")
        #    image = 'camera_52_VZ=090_0_VA=270_0'

    elif flag_image_type == 'direction':
           image = 'ima{0:02d}_VZ={1:03d}_{2:01d}_VA={3:03d}_{4:01d}'.format( i_cam,\
               int(vz_), int(10*(round(vz_-int(vz_),1) ) ),  \
               int(va_), int(10*(round(va_-int(va_),1) ) )   )
    else:
        print('')
        print('issue with flag in load_DART_output', flag_image_type)
        print('')
        pdb.set_trace()
        
    #load product
    ##############
    npy_image =  dir_im + image + '.npy'
    fileisThere = os.path.isfile(npy_image)
    if (not fileisThere):
        #print 'convert  file'
        filename = dir_im + image
        flag_image = (sensor == 'SENSOR')  | (flag_image_type != 'direction') #| (flag_image_type == 'direction')
        saveasnpy(filename,flag_image=flag_image)

    tmp = np.load(npy_image, allow_pickle=True).item()
  
    if (flag_image_type == 'direction'):
        
        nx = int(tmp['ny'])
        ny = int(tmp['nx'])
        dx = tmp['dy']
        dy = tmp['dx']
        Lx = nx*dx
        Ly = ny*dy
        
        value = np.array(tmp['imageValues'])
        #print 'load '+ product + ' ' + tmp['typeProductName']+ ' from '+ image
        out_ = value.reshape(ny,nx)

        out = np.zeros([nx,ny],dtype=[('x',float),\
                                       ('y',float),\
                                       (tmp['typeProductName'],float) ])
        out = out.view(np.recarray)
        out.x = np.array([np.arange(.5*dx,Lx,dx)] * ny).T
        out.y = np.array([np.arange(.5*dy,Ly,dy)] * nx)
        out[tmp['typeProductName']] =  out_[::-1].T

        return out
    
    else:
        nx = int(tmp['nx'])
        ny = int(tmp['ny'])
        value = np.array(tmp['imageValues'])
        #print('load '+ product + ' ' + tmp['typeProductName']+ ' from '+ image)
        return value.reshape(nx,ny)[::-1].T


###############################################################
def get_surfaceVoxel(dir_output):

    rb_file = dir_output + 'simulation.properties.txt'
    with open(rb_file) as f:
        lines = f.readlines()
    for line in lines:
        if 'maket.voxel.size.x:' in line: sizeX = float(line.split('x:')[1])
        if 'maket.voxel.size.y:' in line: sizeY = float(line.split('y:')[1])

    return sizeX * sizeY


###############################################################
def compute_true_FRP(inputConfig, FireName, root_postproc, time_requested ):
   
    '''
    sum up radiative budget of the box set up arounf the domain simiulation to extamate the true FRP
    '''

    root_dart_simulation = DART_LOCAL+'simulations/'   #inputConfig.params_DART['root_dart_simulation']
    root_dart_tools      = DART_HOME + 'tools/linux/' #inputConfig.params_DART['root_dart_tools']

    #dir_in  = root_postproc + 'DART_maket/'
    
    DARTimageflag = '{:s}_'.format(inputConfig.params_DART['dart_config_bands']) \
                    if ('dart_config_bands' in inputConfig.params_DART.keys()) else ''

    run_name = '{:s}t_{:03d}_{:02d}_s'.format(DARTimageflag,*np.array(math.modf(round(time_requested,2))*np.array([100,1]),dtype=int)[::-1])+'_sens_box'
    run_name = run_name.replace('{:s}t_'.format(DARTimageflag),FireName+'_')

    dir_input  =  root_dart_simulation + run_name + '/input/'
    dir_output =  root_dart_simulation + run_name + '/output/'

    xmldoc = minidom.parse(dir_input+'phase.xml')
    bands = xmldoc.getElementsByTagName('SpectralIntervalsProperties')
    nbre_band = len(bands)
    bands_center=[]; bands_width = []
    for i_band, band in enumerate(bands):
        bands_center.append(float(band.attributes['meanLambda'].value))
        bands_width.append(float(band.attributes['deltaLambda'].value))

    #load triangle ID, the box is set with ID 103
    '''
    triangle_file_info = dir_output + 'triangles.txt'
    reader = asciitable.NoHeader()
    reader.data.splitter.delimiter = ' '
    reader.data.splitter.process_line = None
    data = reader.read(triangle_file_info)
    triangle_ID=np.array(data['col18'])
    idx_triangle_box = np.where(triangle_ID == 103)[0]
    '''
    
    nbre_band = len(glob.glob(dir_output+'BAND*'))

    #for every band of the box simulation, load the radiative budget for the box
    exitance = 0
    
    if inputConfig.params_DART['useDAO']: #Omar idea, need temperature grid to not use voxel
   
        '''
        surface_area = get_surfaceVoxel(dir_output)

        for i_band in range(nbre_band):
            band = 'BAND{:d}/'.format(i_band)
            rb_file = dir_output + band + 'RADIATIVE_BUDGET/ITERX/RadiativeBudget_3D'
            with open(rb_file) as f:
                lines = np.array(f.readlines())
           
            ii_ = np.where( lines == '* TotalExit *\n')[0][0]

            pdb.set_trace()
            exitance += float(lines[ii_+1]) * bands_width[i_band] * surface_area  # flux is compute with 2D surface of volume dixit Omar
        '''
        band_name = 'BAND{:d}/'.format(i_band)
        rb_file = dir_output + band_name + 'RADIATIVE_BUDGET/ITERX/RadiativeBudgetFigures.txt'
        scene_file = dir_output + 'scene.scn'
        exitance = computeExitanceRadiativeBudgetFigure(rb_file, scene_file, PHASE_BOX_ID_out, PHASE_BOX_ID_in)
    
    else:
        idx_triangle_box = getTriangleIT_from_op(dir_output, PHASE_BOX_ID_out, PHASE_BOX_ID_in)

        for i_band in range(nbre_band):
        
            band = 'BAND{:d}/'.format(i_band)
            rb_file = dir_output + band + 'RADIATIVE_BUDGET/ITERX/RadiativeBudgetFigures.txt'
            # Skip the first line if it's a header, adjust skiprows accordingly
            data = np.loadtxt(rb_file, delimiter='\t', skiprows=1)

            # intecept_energy = data[:, 0]  # if you need col1
            # surface_area    = data[:, 3]  # if you need col4
            intecept_energy = data[:, 4]    # col5 (0-based index)
            surface_area    = data[:, 8]    # col9 (0-based index)
            
            #reader = asciitable.NoHeader()
            #reader.data.splitter.delimiter = '\t'
            #reader.data.splitter.process_line = None
            #reader.data.start_line = 1
            #data = reader.read(rb_file)
            #intecept_energy = np.array(data['col1'])
            #surface_area    = np.array(data['col4'])
            #intecept_energy = np.array(data['col5'])
            #surface_area    = np.array(data['col9'])

            for idx in idx_triangle_box:
                exitance += intecept_energy[idx] * bands_width[i_band] * surface_area[idx]
   
    return exitance * 1.e-3 # kW


########################################################################
def load_bdf_image_location(inputConfig,run_name,i_cam, radiance):


    root_dart_simulation = DART_LOCAL+'simulations/'   #inputConfig.params_DART['root_dart_simulation']
    root_dart_tools      = DART_HOME + 'tools/linux/' #inputConfig.params_DART['root_dart_tools']

    dir_dartSimu = root_dart_simulation + run_name + '/output/'
    simuProperties = dir_dartSimu + 'simulation.properties.txt'
    f = open(simuProperties,'r')
    lines = f.readlines()
    f.close()
    
    for i_line, line in enumerate(lines):
        if ('ima_camera{:03d}'.format(i_cam+1) in line):
            break
    nx_cam, ny_cam = radiance.shape

    if inputConfig.params_DART['useLux']:
        gcp_world = np.zeros([4,3])
        i_start = i_line#+5
        flag_found = False
        for i in range(i_start,i_start+19):
            #i_pt = int(old_div((i-i_start),2))
            template_gcp = 'groundBoundaryPoints'
            #template_gcp = 'imageCorners'
            try:
                if (template_gcp in lines[i]) & ('.x' in lines[i]):
                    i_pt = int(lines[i].split('point')[1].split('.')[0])
                    gcp_world[i_pt,0] = float(lines[i].split('.x:')[1].strip())  
                    flag_found = True
                if (template_gcp in lines[i]) & ('.y' in lines[i]):
                    i_pt = int(lines[i].split('point')[1].split('.')[0])
                    gcp_world[i_pt,1] = float(lines[i].split('.y:')[1].strip())  
                    flag_found = True
            except: 
                pdb.set_trace()

        gcp_cam = np.array([[0,ny_cam],[nx_cam,ny_cam],[0,0],[nx_cam,0]])

    else: 
        gcp_world = np.zeros([4,3])
        i_start = i_line+5
        flag_found = False
        for i in range(i_start,i_start+9):
            i_pt = int(old_div((i-i_start),2))
            if '.x' in lines[i]:
                gcp_world[i_pt,0] = float(lines[i].split('.x:')[1])  
                flag_found = True
            if '.y' in lines[i]:
                gcp_world[i_pt,1] = float(lines[i].split('.y:')[1])  
                flag_found = True  

        gcp_cam = np.array([[0,ny_cam],[nx_cam,ny_cam],[0,0],[nx_cam,0]])

    if not flag_found:
        print('')
        print('did not find gcp location in ', simuProperties)
        print('stop')
        print('')
        pdb.set_trace()

    #gcp_cam = np.array([[0,0],[0,ny_cam-1],[nx_cam-1,0],[nx_cam-1,ny_cam-1]])
    #gcp_cam = np.array([[0,ny_cam-1],[nx_cam-1,ny_cam-1],[0,0],[nx_cam-1,0]])
    #gcp_cam = np.array([[0,ny_cam],[nx_cam,ny_cam],[0,0],[nx_cam,0]])

    return gcp_cam, gcp_world
