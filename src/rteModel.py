from __future__ import print_function
import numpy as np 
import matplotlib.pyplot as plt
import glob 
import sys
import os
import shutil
import pdb 
import subprocess
import math
import datetime 

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


#####################################################
def ensure_dir(f):
    import os
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

'''
#####################################################
def test_if_dartSim_done(inputConfig,root_postproc,FireName,time_requested,getFRP=False):
    
    root_dart_simulation = DART_LOCAL+'simulations/'   #inputConfig.params_DART['root_dart_simulation']
    root_dart_tools      = DART_HOME + 'tools/linux/' #inputConfig.params_DART['root_dart_tools']


    
    if inputConfig.params_DART['flag_run_sensitivity']:
        suffix = '_sens'
    else:
        suffix = ''
    if getFRP: suffix += '_box'

    DARTimageflag = '{:s}_'.format(inputConfig.params_DART['dart_config_bands']) \
                    if ('dart_config_bands' in inputConfig.params_DART.keys()) else ''
    
    run_name = '{:s}t_{:03d}_{:02d}_s'.format(DARTimageflag,*np.array(math.modf(round(time_requested,2))*np.array([100,1]),dtype=int)[::-1])+suffix
    run_name_original = root_dart_simulation + run_name.replace('{:s}t_'.format(DARTimageflag),FireName+'_')
    
    dir_Savedmaket  = root_postproc + 'DART_maket/simulations/' + run_name

    if os.path.isfile(run_name_original+'/output/dart.txt'):
        f = open(run_name_original+'/output/dart.txt','r')
        lines = f.readlines()
        f.close()
        flag_crash = False
        for line_ in lines:
            if 'error' in line_: flag_crash = True
        if flag_crash: 
            return False, dir_Savedmaket, run_name_original
        else:
            return True, dir_Savedmaket, run_name_original

    else:
        return False, dir_Savedmaket, run_name_original
'''

#####################################################
def run_dart(inputConfig,dir_in,FireName,time_requested=None,flag_set_up_box4_RadFlux=False,flag_test_geo=False, 
             flag2run = 'all') :#DirectionPhase):
    '''
    run DART on the output of 3DfireScene
    '''
    indent = '   '

    template_dir = 'template'
    if inputConfig.params_DART['useLux']:
        template_dir = template_dir + '_lux'
    if flag_test_geo:
        template_dir += '_noAtm_grdBB'

    if time_requested == None:
        print('***pb, no time given to run_dart')
        pdb.set_trace()

    root_dart_simulation = DART_LOCAL+'simulations/'   #inputConfig.params_DART['root_dart_simulation']
    root_dart_tools      = DART_HOME + 'tools/linux/' #inputConfig.params_DART['root_dart_tools']

    #dir_in  = root_postproc + 'DART_maket/simulations/'
    
    if inputConfig.params_DART['flag_run_sensitivity']:
        suffix = '_sens'
    else:
        suffix = ''

    DARTimageflag = '{:s}_'.format(inputConfig.params_DART['dart_config_bands']) \
                    if ('dart_config_bands' in inputConfig.params_DART.keys()) else ''
    run_name = '{:s}t_{:03d}_{:02d}_s'.format(DARTimageflag,*np.array(math.modf(round(time_requested,2))*np.array([100,1]),dtype=int)[::-1])+suffix
    run_name_original = dir_in + run_name
    
    run_name = run_name.replace('{:s}t_'.format(DARTimageflag),FireName+'_')
    if flag_set_up_box4_RadFlux:
        run_name_original = run_name_original + '_box'
        run_name = run_name + '_box'
 
    if (flag2run == 'all') | (flag2run == 'directionPhase') | (flag2run == 'dirSetup'):
        #copy template simulation configuration
        if os.path.isdir(root_dart_simulation+run_name):
            shutil.rmtree(root_dart_simulation+run_name)
        shutil.copytree(run_name_original+'/', root_dart_simulation+run_name)
   
    '''
    ensure_dir(root_dart_simulation+run_name)
    ensure_dir(root_dart_simulation+run_name+'/input/')
    ensure_dir(root_dart_simulation+run_name+'/output/')
    for file_ in glob.glob('../data_static/DART/'+template_dir+'/input/*.xml'):
        shutil.copy(file_, root_dart_simulation+run_name+'/input/')


    #and copy the temperature, dem, maket and plots xml files
    shutil.copy(run_name_original+'/maket.xml', root_dart_simulation+run_name+'/input/')
    shutil.copy(run_name_original+'/temperature.txt', root_dart_simulation+run_name+'/input/')
    shutil.copy(run_name_original+'/plots.xml', root_dart_simulation+run_name+'/input/')
    shutil.copy(run_name_original+'/phase.xml', root_dart_simulation+run_name+'/input/')
    shutil.copy(run_name_original+'/coeff_diff.xml', root_dart_simulation+run_name+'/input/')
    if flag_set_up_box4_RadFlux:
        shutil.copy(run_name_original+'/object_3d.xml', root_dart_simulation+run_name+'/input/')
    #for file_ in glob.glob(root_dart_simulation+'rose_XXX/input/DEM.*'):
    #    shutil.copy(file_, root_dart_simulation+run_name+'/input/')
    shutil.copy(run_name_original+'/dem_raster.img', root_dart_simulation+run_name+'/input/')

    ##run the DEM creator to transform the rater file to DART input file
    #dart_dem = rout_dart_tools + 'dart-dem.sh'
    #process = subprocess.Popen([dart_dem, run_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #process.wait()
    #print process.stderr.read()
    #print process.stdout.read()
    '''

    if flag2run == 'dirSetup': 
        return 

    datetime_beforeDart = datetime.datetime.now()

    #run DART
    path_here = os.getcwd()+'/'
    os.chdir(root_dart_tools)
  
    print('   run DART {:s}...'.format(flag2run), end=' ')
    sys.stdout.flush()
    if (flag2run == 'all') :   
        process = subprocess.Popen(['./dart-full.sh', run_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
    elif (flag2run == 'directionPhase'):
        process = subprocess.Popen(['./XMLUpgrader.sh', run_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        #process = subprocess.Popen(['../../bin/DARTDemGenerator.exe', run_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #output, error = process.communicate()
        #pdb.set_trace()
        process = subprocess.Popen(['./dart-directions.sh', run_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        process = subprocess.Popen(['./dart-phase.sh', run_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
    elif (flag2run == 'atmosphere'): 
            process = subprocess.Popen(["../../bin/atmosphereMaket.exe", "-path", run_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate()
    elif (flag2run == 'onlyDart'):
            #print('PASS ONLYDART')
            process = subprocess.Popen(['./dart-only.sh', run_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            output, error = process.communicate()
    else: 
        print()
        print('********  stop in rteModel.run_dart') 
        print('          bad flag2run = ', flag2run)
    #process = subprocess.Popen(['./dart-full.sh', run_name], stderr=subprocess.PIPE).wait()
   
    #MERDE
    return 

    #process = subprocess.Popen(['./dart-full.sh', run_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE).wait()
    #process.wait()

    #a bit shit, below is to remove know warning that can occur during DART run
    known_warning = ['Warning: 3D temperature file : temperature.txt unconsistent with the maket dimensions: There are more temperatures than cells.',        \
                     'Warning: For at least one direction other than incident sun direction, the integral of the aerosol phase function over all scattering '+\
                               'directions except forward direction is not coherent',                                                                         \
                     'Hence, scattering function is normalized. For removing the warning (more accurate results), increase the integral step',                \
                     ]\
                     + \
                     ['java.lang.NullPointerException', \
                             '\tat cesbio.dart.controleversionxml.traitements.regles.Version_5_6_2.addWaterAmountOptions(Unknown Source)',\
                             '\tat sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)', \
                             '\tat sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)', \
                             '\tat sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)', \
                             '\tat java.lang.reflect.Method.invoke(Method.java:497)',\
                             '\tat cesbio.dart.controleversionxml.traitements.MonteeDeVersion.appliquerReglesVersionnage(Unknown Source)', \
                             '\tat cesbio.dart.controleversionxml.traitements.MonteeDeVersion.monterDeVersion(Unknown Source)', \
                             '\tat cesbio.dart.controleversionxml.lanceur.DARTMonteeDeVersion.ControlerVersionSimulation(Unknown Source)', \
                             '\tat cesbio.dart.controleversionxml.lanceur.Main.main(Unknown Source)'\
                    ]\
                    +\
                    ['Lower corner coordinates not found. Reset to default: (0., 0., 0.). This is only a problem if you are using the sub-zone option']\
                    +\
                    ['Lower corner coordinates not found. Reset to default: (0., 0., 0.). This is only a problem if you are using the sub-zone option.']\
                    +\
                    ['Warning: 3D temperature file : temperatures.txt unconsistent with the maket dimensions: There are more temperatures than cells.']

    outputs_print = output.decode("utf-8")
    errors_print = error.decode("utf-8")
   
    errors = error.decode("utf-8").split('\n')
   
    
    if '' in errors:
        errors.remove('')

    if len(errors) == 0:
        flag_stop = False
    else:
        if inputConfig.params_DART['useLux']:
            flag_stop = True
            if  errors[-1][-21:] == '100%, rendering done.' : # check end of Lux simulation  
                flag_stop = False
                
        else:
            flag_stop = True
            error_count = len(errors)
            i_error_count = 0
            for error in errors:
                for known_w in known_warning:
                    if error in known_w:
                        i_error_count += 1     
            if i_error_count == error_count:
                flag_stop = False

    if flag_stop:
        for line_ in output.decode().split('\n'):
            print(line_)
        print('')
        print('')
        for line_ in errors:
            print(line_)
        print('')
        print('DART crashes ...')
        print('')
        os.chdir(path_here)
        pdb.set_trace()
    
    print('done in {:s}'.format(str(datetime.timedelta(seconds=(datetime.datetime.now()-datetime_beforeDart).total_seconds()))))
    #print process.stderr.read()
    #print process.stdout.read()

    os.chdir(path_here)

    with open(root_dart_simulation+run_name+'/output.txt','w') as f: 
        f.writelines(outputs_print)

    with open(root_dart_simulation+run_name+'/error.txt','w') as f: 
        f.writelines(errors_print)


############################
def run_srte39():

    print('nothing yet')
    '''
               if False:
                #if not flag_box:
                    #print Firescene.shape
                    #print Lz
                        
                    #######
                    # run SRTE39
                    #######
                    
                    # clean directory
                    outputDir_SRTE = '../data_out/SRTE/'+name_simu+'/'
                    ensure_dir(outputDir_SRTE)
                    ensure_dir(outputDir_SRTE+'FRP_pict/')
                    ensure_dir(outputDir_SRTE+'Netcdf/')
                    ensure_dir(outputDir_SRTE+'Npy/')


                    #nbre_cam = len(nxp)
                    Lxp = np.zeros(nbreCam); Lyp = np.zeros(nbreCam); r_pc = np.zeros(nbreCam)
                    for i_cam in range(nbreCam):
                        r_pc[i_cam] =  distance_cam[i_cam] #/ math.cos(theta_pc[i_cam] * 3.14/180.)
                        
                        dd = 2 * (r_pc[i_cam] * math.tan(.5*fov * 3.14/180.)) # discance along longest lenght of the frame

                        if nxp[i_cam] > nyp[i_cam]:
                            Lxp[i_cam] =  dd 
                            Lyp[i_cam] =  dd  * nyp[i_cam]/nxp[i_cam] #/ math.cos(theta_pc[i_cam] * 3.14/180.)
                        else:
                            Lyp[i_cam] =  dd 
                            Lxp[i_cam] =  dd  * nxp[i_cam]/nyp[i_cam] #/ math.cos(theta_pc[i_cam] * 3.14/180.)

                    #print r_pc
                    #print Lxp
                    #print Lyp

                    cam_spec={'nxp':nxp[:nbreCam],'nyp':nyp[:nbreCam],'Lxp':Lxp[:nbreCam],'Lyp':Lyp[:nbreCam],'phi_pc':phi_pc[:nbreCam],'theta_pc':theta_pc[:nbreCam],\
                            'r_pc':r_pc[:nbreCam],'x_pc':vec_shift[:nbreCam,0],'y_pc':vec_shift[:nbreCam,1],'z_pc':vec_shift[:nbreCam,2]}
                    # no need here time = float(time_str.split('_')[0]) + float(time_str.split('_')[1])/100
                    
                    #call SRTE39
                    file_name_prefix = 'mirCam172'
                    FRPmir_arr, FRPtrue_arr, Temp_arr, x_arr, y_arr, z_arr =  SRTE39.render_images(outputDir_SRTE,file_name_prefix,time,Firescene,Lx,Ly,Lz,xs,ys,zs,dxs,dys,dzs,cam_spec)

                    time_frp_model[0,i_time] = time
                    time_frp_model[1,i_time] = FRPmir_arr[0] # first cam is nadir 
                    time_frp_model[2,i_time] = FRPtrue_arr[0] 
                    time_frp_model[3,i_time] = FRPmir_arr[1] 
                    time_frp_model[4,i_time] = FRPtrue_arr[1] 

                    nadir_image = Temp_arr[0]
                    offnadir_image = Temp_arr[1]
                    
                    #regrid the modelled image on the native grid of fireScene
                    #do it only for the nadir image now
                    x_mm = x_arr[0]
                    y_mm = y_arr[0]
                    z_mm = z_arr[0]
                    nadir_image_native_res = np.zeros(Firescene.shape[0:2])
                    for j in range(Firescene.shape[1]):
                        for i in range(Firescene.shape[0]):
                            index_mm = np.where( (x_mm >= xs[i]) & (x_mm < xs[i]+dx) &\
                                                 (y_mm >= ys[j]) & (y_mm < ys[j]+dy)  )

                            wavelength = 3.9 + np.zeros(len(index_mm[0]))
                            mean_radiance = np.array([planck_radiance(wavelength,nadir_image[index_mm]).mean()])
                            nadir_image_native_res[i,j] = planck_temperature(np.array([3.9]), mean_radiance)[0]

                    #save regridded image + raw SRTE image in npy format
                    file_npy = outputDir_SRTE+'Npy/' + file_name_prefix + '_{0:04d}'.format(int(round(time,0)))
                    tosave =  [x_in,  y_in,         firefront2D_in,       \
                               x_arr, y_arr, z_arr, Temp_arr,             \
                               xs,    ys,           nadir_image_native_res]
                    np.save(file_npy,tosave)

    '''

###########################
if __name__ == "__main__":
###########################
    
    print('see driver.py')

