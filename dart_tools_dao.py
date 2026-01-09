from __future__ import division
from past.utils import old_div
import sys
import os
import random as rnd
import numpy as np
import struct
import shutil 
import pdb 
import math 
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage import io
import importlib
import argparse
import multiprocessing

#homebrewed
sys.path.append(os.environ['HOME']+'/Src/dart-fire/src/loadExtraScene/')
import read_fds_bf
import read_fds_prof
import read_fds_obst
import rteModel
sys.path.append('./tools/')
import myPickle

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

# add the DAO python folder to current system path
sys.path.append(os.path.join(DART_HOME, "bin", "python_script", "DAO"))
# import DAO toolkit

try:
    import dao
    from dao.tools.Mesh import Mesh
except: 
    sys.path.append('../DAO/')
    import dao
    from dao.tools.Mesh import Mesh

#homebrewed
sys.path.append(DART_LOCAL+'/MySrc/LoadOpticalProperties/')
import loadOpticProp2CoeffDiff
sys.path.append(DART_LOCAL+'/MySrc/DAO/')
import addTempWall2coeffdiff


global xco2Temp 
xco2Temp = []

# Physics constants
#T_AMBIENT = 293.15
MCO2 = 12 + 2*16    # g/mol
MCO = 12 + 16       # g/mol
MH2O = 2*1 + 16     # g/mol
m_1_soot = 1.e-21    # kg
NAVOGADRO = 6.022e23    # mol-1
MAcetone = 58.08    # g/mol
marge = 0.01

minMassDry  = 1.e-6 #kg
minMassChar = 1.e-8 #kg
minMassAsh  = 1.e-8 #kg


global temp_threshold_default
temp_threshold_default = 300.

def str2Bool(input_):
    if input_ in ['true', 'TRUE' , 'True' , '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh']:
        return True
    else:
        return False

def str2float(input_):
    if input_ == 'None': 
        return None
    else:
        return float(input_)

#####################################################
def Y2X(Yco2, MMolco2,
        Yco,  MMolco,
        Yh2o, MMolh2o,
        MMolair):
    
    totMol = (Yh2o/MMolh2o+Yco2/MMolco2+Yco/MMolco+ (1-Yco2-Yco)/MMolair )  #h2o is not in the MMolair of dry air
    return np.where(totMol>0, Yco2/MMolco2 / totMol, 0 ), \
           np.where(totMol>0, Yco/MMolco   / totMol, 0 ), \
           np.where(totMol>0, Yh2o/MMolh2o / totMol, 0 )  #xco2,xco,xh2o 

#############################################################
def create_raster_dem(outputDir_simu,nx,ny,nz,dx,dy,dz):

    z0_dart = (1.-marge)*dz  # MERDE
    
    #zero_level = 1.0663275133993011e-05
    dem = np.zeros([nx,ny]) + z0_dart
    dem[0,0] = 0

    file_dem = outputDir_simu + '/dem_raster.tif'
    io.imsave( file_dem, dem, plugin='tifffile',check_contrast=False)
     
    #file_dem = outputDir_simu + '/dem_raster.img'
    #f = open(file_dem,'wb')
    #for i in range(nx):
    #    for j in range(ny):
    #        f.write(struct.pack('d',dem[i,j]))
    #f.close()
    
    return z0_dart, dem

##############################################################
def addBndfAsTriangle2Mockup(mockup, iplan_arr, bndf_arr, bndf_xyz, times, time, id_opti=0, id_tempInit=2, onecellDim=None):
    
    # some aliases on dao types
    TriangleProperty = dao.TriangleProperty
    Triangle = dao.Triangle
    Turbid = dao.Turbid
    Fluid = dao.Fluid
    Matrix4 = dao.Matrix4

    itime = np.abs(times-time).argmin()
    if onecellDim is None:
        nx,ny,nz = mockup.getMockupDimension()
        dxy,dz = mockup.getCellSize()
    else: 
        nx,ny,nz = onecellDim[:3]
        dxy,dz   = onecellDim[3:]

    dxfds =  bndf_xyz[0][0][1,0,0]-bndf_xyz[0][0][0,0,0]

    xc = nx*dxy/2+0.0001; yc = ny*dxy/2+0.0001; zc = 0.9*dz+0.0001
    idT_ = id_tempInit
    for iplan in iplan_arr: 

        val  = bndf_arr[iplan][:,:,:,itime]
        x,y,z = bndf_xyz[iplan]
        
        if np.unique(x).shape[0] == 1: rotAxe = 'y'
        if np.unique(y).shape[0] == 1: rotAxe = 'x'
        rotAngle = 90

        for x_,y_,z_,T_ in zip(np.squeeze(x)[:-1,:-1].flatten(), np.squeeze(y)[:-1,:-1].flatten(), np.squeeze(z)[:-1,:-1].flatten(), np.squeeze(val)[:-1,:-1].flatten()):
            scale = 1#00
            #print (x_, y_, z_)
            # add a square
            planeProperty = TriangleProperty(TriangleProperty.WALL, True, TriangleProperty.LAMBERTIAN, id_opti, idT_)
            #transform = Matrix4().scale(scale*dxfds,scale*dxfds,scale*dxfds).rotate(rotAngle,rotAxe,radians=False)\
            #                     .translate((scale*dxfds)/2,0,(scale*dxfds)/2).translate(xc+x_,yc+y_,zc+z_+0.01)
            
            
            if rotAxe == 'x':
                transform = Matrix4().rotate(rotAngle,rotAxe,radians=False).scale(scale*dxfds,scale*dxfds,scale*dxfds)\
                                    .translate((scale*dxfds)/2,0,(scale*dxfds)/2).translate(xc+x_,yc+y_,zc+z_)
            elif rotAxe == 'y':
                transform = Matrix4().rotate(rotAngle,rotAxe,radians=False).scale(scale*dxfds,scale*dxfds,scale*dxfds)\
                                     .translate(0,(scale*dxfds)/2,(scale*dxfds)/2).translate(xc+x_,yc+y_,zc+z_)
             
            
            obj = dao.OBJloader(os.path.join(DART_HOME, "database", "3D_Objects","Square.obj"))
            obj.load()
            obj.addToMockup(mockup, planeProperty, transform)
        
            idT_ += 1

    return mockup


#############################################################
def addCurtain(mockup, x,y,z ,id_opti=0, id_temp=0):
    
    # some aliases on dao types
    TriangleProperty = dao.TriangleProperty
    Triangle = dao.Triangle
    Turbid = dao.Turbid
    Fluid = dao.Fluid
    Matrix4 = dao.Matrix4

    nx,ny,nz = mockup.getMockupDimension()
    dxy,dz = mockup.getCellSize()

    if x == 0: 
        rotAxe = 'y'
        scalex = 1 
        scaley = ny*dxy
        scalez = (nz+1)*dz
    elif x == dxy*nx: 
        rotAxe = 'y'
        scalex = 1 
        scaley = ny*dxy
        scalez = (nz+1)*dz
    elif y == 0: 
        rotAxe = 'x'
        scalex = nx*dxy 
        scaley = 1
        scalez = (nz+1)*dz
    elif y == ny*dxy: 
        rotAxe = 'x'
        scalex = nx*dxy 
        scaley = 1
        scalez = (nz+1)*dz
    
    rotAngle = 90

    scale = 1
    # add a square
    planeProperty = TriangleProperty(TriangleProperty.WALL, True, TriangleProperty.LAMBERTIAN, id_opti, id_temp)
    
    
    transform = Matrix4().rotate(rotAngle,rotAxe,radians=False).scale(scalex,scaley,scalez)\
                        .translate(x,y,z)
     
    obj = dao.OBJloader(os.path.join(DART_HOME, "database", "3D_Objects","Square.obj"))
    obj.load()
    obj.addToMockup(mockup, planeProperty, transform)

    return mockup


#############################################################
def get_lut_gas(lut):
    idx_specie = []
    lut_temp = []
    lut_xh2o = []
    lut_xco2 = []
    lut_xco = []
    for ii, name in enumerate(lut.name):
        if name[:2] == 'gt': # only consider gas
            idx_specie.append(ii)
            lut_temp.append( float(lut.name[ii].split('_')[0][2:-1])       )
            lut_xh2o.append(    float(lut.name[ii].split('_')[1].split('x')[1])*1.e-10  )
            lut_xco2.append(    float(lut.name[ii].split('_')[1].split('x')[2])*1.e-10  )
            lut_xco.append(    float(lut.name[ii].split('_')[1].split('x')[3])*1.e-10  )
   
    lut = lut[(idx_specie,)]
    lut_temp = np.array(lut_temp)
    lut_xh2o = np.array(lut_xh2o)
    lut_xco2 = np.array(lut_xco2)
    lut_xco = np.array(lut_xco)
   
    lut_out = np.array([('m',0,'m','m',0.,0.,0.,0.)]*lut.shape[0],dtype=np.dtype(  [ (x,y[0]) for x,y in lut.dtype.fields.items()] 
                                                                           + [('temp',np.dtype(float)),('xh2o',np.dtype(float)),('xco2',np.dtype(float)),('xco',np.dtype(float))]))
    lut_out = lut_out.view(np.recarray)
    lut_out.type = lut.type
    lut_out.id = lut.id
    lut_out.ident = lut.ident
    lut_out.name = lut.name
    lut_out.temp = lut_temp
    lut_out.xh2o = lut_xh2o
    lut_out.xco2 = lut_xco2
    lut_out.xco = lut_xco

    return lut_out

#############################################################
def get_lut_veg(lut):
    idx_veg = []
    for ii, type_ in enumerate(lut.type):
        if type_ ==  'turbid':
            idx_veg.append(ii)
   
    lut = lut[(idx_veg)]
    
    return lut.view(np.recarray)

#############################################################
def getNameAndIdFromGasAndTemp(lut, specie, temperature = None, xmolar=None):

    if specie == 'soot': 
        #idx = np.where(lut.name == 'soot_poitou')
        idx = np.where(lut.name == 'soot_bordbarhostika')
        return lut.ident[idx][0], lut.id[idx][0]
    
    elif specie == 'vegetation':  # for fluid veg
        idx = np.where(lut.name == 'rayleigh_1_0')
        #idx = np.where(lut.name == 'hg_1_1_1_1_1')
        #pdb.set_trace()
        return lut.ident[idx][0], lut.id[idx][0]
    
    elif specie == 'dryVeg':  
        idx = np.where(lut.name == 'grass_rye')
        return lut.ident[idx][0], lut.id[idx][0]
    
    elif specie == 'charVeg':  
        idx = np.where(lut.name == 'reflect_equal_0_trans_equal_0_0')
        return lut.ident[idx][0], lut.id[idx][0]
    
    elif specie == 'ashVeg':  
        idx = np.where(lut.name == 'grass_rye')
        return lut.ident[idx][0], lut.id[idx][0]

    elif specie == 'gas': 
        #idx_tempmin = np.where( np.abs( lut.temp - temperature) == np.abs( lut.temp - temperature).min() )
        dist = ( (lut.temp - temperature)**2+\
                 (lut.xh2o - xmolar[0]  )**2+\
                 (lut.xco2 - xmolar[1]  )**2+\
                 (lut.xco  - xmolar[2]  )**2 )
        idx_ = dist.argmin()
        return lut.ident[idx_], lut.id[idx_]

    else: 
        print('bas specie in getNameAndIdFromGasAndTemp')
        sys.exit()


#############################################################
def star_addPlotDescription2Mockup(param):
    return addPlotDescription2Mockup(*param)

#------------------------------------------------------------
def addPlotDescription2Mockup(flag_useLux, flag_lux_vegModel ,flag_onlyGas, flag_removeVeg, flag_removeFlame, flag,\
                              i, j, k, i0, j0, k0, dxy, dz, dxyD, dzD,  
                              scene, lut,
                              temp_threshold,
                              soot_fv_max, soot_fv_threshold, 
                              aerosol_tracer_max, aerosol_tracer_threshold, 
                              T_ambient ) :
  
    # some aliases on dao types
    TriangleProperty = dao.TriangleProperty
    Triangle = dao.Triangle
    Turbid = dao.Turbid
    Fluid = dao.Fluid
    Matrix4 = dao.Matrix4

    TEMP_FUN_ID = 0
    lut_gas, lut_veg, lut_opticProp_coeff_diff = lut
    nplot_gas_soot, nplot_gas_co2, nplot_gas_h2o, nplot_gas_co, nplot_plu = 0, 0, 0, 0, 0,
    nplot_veg,nplot_char,nplot_ash = 0, 0, 0

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
    #xA = i*dxy      + dxy*0.001
    #yA = j*dxy      + dxy*0.001
    #xB = (i+1)*dxy  - dxy*0.001
    #yB = j*dxy      + dxy*0.001
    #xC = (i+1)*dxy  - dxy*0.001
    #yC = (j+1)*dxy  - dxy*0.001
    #xD = i*dxy      + dxy*0.001
    #yD = (j+1)*dxy  - dxy*0.001
    
    # height_base = (k+1)*dz  # flame "floating" above the ground
    height_base = (k-k0)*dz + dzD*1.01 # Ronan: add k+1, this is for the DEM
    height_plot = dz - dzD*0.02
    max_height_plot = -999 
    
    #nb_part = 4
    if flag == 'fire': 
        #m_CO2  = scene.CO2[i,j,k]   * 1.e3 #g
        #m_CO   = scene.CO[i,j,k]    * 1.e3
        #m_H2O = scene.H2O[i,j,k]    * 1.e3 
        try:
            xCO2 = scene.xCO2[i,j,k]
            xCO  = scene.xCO[i,j,k]
            xH2O = scene.xH2O[i,j,k]
        
        except: 
            MMolh2o = 2*1 + 16     # g/mol
            MMolco2 = 12 + 2*16    # g/mol
            MMolco  = 12 + 16      # g/mol
            MMolair   = 28.96      # g/mol
            YCO2 = 0  #to improve...
            YCO = 0
            YH2O = 0

            xCO2, xCO, xH2O = Y2X(YCO2, MMolco2,
                                  YCO,  MMolco,
                                  YH2O, MMolh2o,
                                  MMolair)
    
        sootXMLvalue = scene.fv[i,j,k]
        aerosolXMLvalue = 0 
    
    elif flag == 'atm':
        MMolh2o = 2*1 + 16     # g/mol
        MMolco2 = 12 + 2*16    # g/mol
        MMolco  = 12 + 16      # g/mol
        MMolair   = 28.96      # g/mol
        YCO2  = 0.
        YCO   = 0.
        YH2O = scene.rvap[i,j,k] #/  1.e3 
        sootXMLvalue = None 
        aerosolXMLvalue = 0 
        
        xCO2, xCO, xH2O = Y2X(YCO2, MMolco2,
                              YCO,  MMolco,
                              YH2O, MMolh2o,
                              MMolair)

    else:
        pdb.set_trace()


    #m_soot = scene.soot[i,j,k] #* 1.e3
    
    #dCO2 = round(old_div(m_CO2, MCO2) * NAVOGADRO )#* 1.e-15)     # because densities are in 1e15 m-3 in DART inputs
    #dCO  = round(old_div(m_CO, MCO)  * NAVOGADRO )#* 1.e-15)
    #dH2O = round(old_div(m_H2O, MH2O) * NAVOGADRO )# * 1.e-15)
    #dsoot = round(m_soot / m_1_soot )#* 1.e-15)          ############################## check optical properties of soot
    
    #
    # Vegetation plot
    #
    key_temp_veg = 'Temp_veg'
    #if 'lux_vegModel' in inputConfig.params_DART.keys():
    #    if inputConfig.params_DART['lux_vegModel'] == 'turbid': key_temp_veg = 'Temp_veg2'
    #if flag_lux_vegModel:
    lux_vegModel = flag_lux_vegModel #flag_lux_vegModel #inputConfig.params_DART['lux_vegModel']
    #else: 
    #    lux_vegModel = 'fluid' # if not defined, for example in FT mode, then we use default fluid
    

    fluid = []
    turbid= []
    loct   = []
    locf   = []

    #
    # add flame materials
    #
    if flag == 'fire': 
        #try: 
        #if (scene.fv[i,j,k] >= soot_fv_max * soot_fv_threshold): 
        #if (scene.hrrpuv[i,j,k] >= 1.e-2):  
        fluid_ = []
        turbid_ = []
        
        #if (scene.Temp[i,j,k] >= temp_threshold) & (not(flag_removeFlame)):  
        if (not(flag_removeFlame)):  
        ############################################# just to reduce the number of plots
            max_height_plot = height_base+height_plot
        
            flag_skip_soot = False
            #if ('onlyGas' in inputConfig.params_model.keys()):
            #    if inputConfig.params_model['onlyGas']:  
            if (flag_onlyGas) or (sootXMLvalue==0):
                flag_skip_soot = True
           
            if not(flag_skip_soot):
                NAME_PHASE_FUN_SOOT, PHASE_FUN_ID_SOOT = getNameAndIdFromGasAndTemp(lut_opticProp_coeff_diff, 'soot' )  
                fluid_.append(Fluid(density=sootXMLvalue, fluidOpticalPropertyID=PHASE_FUN_ID_SOOT, temperatureID=TEMP_FUN_ID))
                nplot_gas_soot = 1
           
            if (xCO2 > 0) or (xCO > 0) or (xH2O > 0): 
                NAME_PHASE_FUN_GAS ,PHASE_FUN_ID_GAS = getNameAndIdFromGasAndTemp(lut_gas, 'gas', temperature=scene.Temp[i,j,k], xmolar=[xH2O,xCO2,xCO])
                fluid_.append(Fluid(density=1, fluidOpticalPropertyID=PHASE_FUN_ID_GAS, temperatureID=TEMP_FUN_ID))
                nplot_gas_co2 = 1
                xco2Temp.append([scene.Temp[i,j,k],xH2O,xCO2,xCO,NAME_PHASE_FUN_GAS,PHASE_FUN_ID_GAS])
            

            '''
            if xCO > 0: 
                NAME_PHASE_FUN_CO ,PHASE_FUN_ID_CO = getNameAndIdFromGasAndTemp(lut_gas, 'co', temperature=scene.Temp[i,j,k], xmolar=[xH2O,xCO2,xCO])
                fluid.append(Fluid(density=1, fluidOpticalPropertyID=PHASE_FUN_ID_CO, temperatureID=TEMP_FUN_ID))
                loc.append(( (dxy*(i-i0)+dxyD*marge), (dxy*(j-j0)+dxyD*marge), (dz*(k-k0)+dzD*(1.+marge)) ))
                nplot_gas_co = 1

            if xH2O > 0: 
                NAME_PHASE_FUN_H2O ,PHASE_FUN_ID_H2O = getNameAndIdFromGasAndTemp(lut_gas, 'h2o', temperature=scene.Temp[i,j,k], xmolar=[xH2O,xCO2,xCO])
                fluid.append(Fluid(density=1, fluidOpticalPropertyID=PHASE_FUN_ID_H2O, temperatureID=TEMP_FUN_ID))
                loc.append(( (dxy*(i-i0)+dxyD*marge), (dxy*(j-j0)+dxyD*marge), (dz*(k-k0)+dzD*(1.+marge)) ))
                nplot_gas_h2o = 1
            '''
        #except AttributeError as e:
        #    #print "AttributeError raised: {1}".format(e.strerror)
        #    pass
    
        
        if not(flag_removeVeg):

            if lux_vegModel == 'fluid':
                try:
                    #flag_ = (scene.fv[i,j,k] < soot_fv_max * soot_fv_threshold) #if inputConfig.params_DART['useLux'] else True 
                    #if (scene.kappa_veg[i,j,k] > 0) & (flag_): 
                    #if (scene.Temp_veg[i,j,k] > T_ambient) & (k==1):
                    #if (scene.MassDry[i,j,k] > minMassDry): 
                    #if (scene[key_temp_veg][i,j,k] > 0) :
                    if (scene.Temp_veg[i,j,k] > 0):
                        max_height_plot = height_base+height_plot     
                        #print(scene[key_temp_veg][i,j,k], scene.kappa_veg[i,j,k])
                        #print(scene.kappa_veg[i-1:i+2,j-1:j+2,k]) 
                        #print(i,j,k) 
                        #print(scene.kappa_veg[i-1:i+2,j-1:j+2,1]) 
                        # Vegetation air plot
                        kveg = scene.kappa_veg[i,j,k]
                        NAME_PHASE_FUN_VEG, PHASE_FUN_ID_VEG   = getNameAndIdFromGasAndTemp(lut_opticProp_coeff_diff, 'vegetation' ) 
                        fluid_.append(Fluid(density=kveg, fluidOpticalPropertyID=PHASE_FUN_ID_VEG, temperatureID=TEMP_FUN_ID))
                        #loc.append(( (dxy*(i-i0)+dxyD*marge), (dxy*(j-j0)+dxyD*marge), (dz*(k-k0)+dzD*(1.+marge)) ))
                       
                        #pdb.set_trace()
                        nplot_veg += 1
                
                except AttributeError as e:
                    #print "AttributeError raised: {1}".format(e.strerror)
                    pass 
            
            elif lux_vegModel == 'turbid':
              
                # Dry Veg turbid plot
                if (scene.MassDry[i,j,k] > minMassDry): 
                    max_height_plot = height_base+height_plot     
                    NAME_PHASE_FUN_VEG, PHASE_FUN_ID_VEG   = getNameAndIdFromGasAndTemp(lut_veg, 'dryVeg' ) 
                    # MORVAN formulation of LAI. DOI: 10.1007/s10694-010-0160-2
                    LAI =0.5 *  (scene.MassDry[i,j,k]/(dxy**2*dz) / scene.RhoBDry[i,j,k]) * scene.Surf2Vol[i,j,k] * dz  
                    #LAI = (scene.MassDry[i,j,k]/(dxy**2*dz) / scene.RhoBDry[i,j,k]) * scene.Surf2Vol[i,j,k] * dz  
                    turbid_.append(Turbid(lai= LAI, vegetationOpticalPropertyID=PHASE_FUN_ID_VEG, temperatureID=TEMP_FUN_ID))
                    #loc.append(( (dxy*(i-i0)+dxyD*marge), (dxy*(j-j0)+dxyD*marge), (dz*(k-k0)+dzD*(1.+marge)) ))
                    nplot_veg += 1
                    
                # Char turbid plot
                if (scene.MassChar[i,j,k] > minMassChar): 
                    max_height_plot = height_base+height_plot     
                    NAME_PHASE_FUN_VEG, PHASE_FUN_ID_VEG   = getNameAndIdFromGasAndTemp(lut_veg, 'charVeg' ) 
                    if (scene.RhoBChar[i,j,k]== 0) | (scene.Surf2Vol[i,j,k] ==0)  : pdb.set_trace() 
                    LAI = 0.5 * (scene.MassChar[i,j,k]/(dxy**2*dz) / scene.RhoBChar[i,j,k]) * scene.Surf2Vol[i,j,k] * dz
                    turbid_.append(Turbid(lai= LAI, vegetationOpticalPropertyID=PHASE_FUN_ID_VEG, temperatureID=TEMP_FUN_ID))
                    #loc.append(( (dxy*(i-i0)+dxyD*marge), (dxy*(j-j0)+dxyD*marge), (dz*(k-k0)+dzD*(1.+marge)) ))
                    nplot_char += 1

                # Ash turbid plot
                if False: #(scene.MassAsh[i,j,k] > minMassAsh): 
                    max_height_plot = height_base+height_plot     
                    NAME_PHASE_FUN_VEG, PHASE_FUN_ID_VEG   = getNameAndIdFromGasAndTemp(lut_veg, 'ashVeg' ) 
                    #lai = 
                    turbid_.append(Turbid(lai= 3, vegetationOpticalPropertyID=PHASE_FUN_ID_VEG, temperatureID=TEMP_FUN_ID))
                    #loc.append(( (dxy*(i-i0)+dxyD*marge), (dxy*(j-j0)+dxyD*marge), (dz*(k-k0)+dzD*(1.+marge)) ))
                    nplot_ash += 1
                

            else: 
                print ('##################')
                print ('##################')
                print ("you need to define inputConfig.params_DART['lux_vegModel'] ")
                sys.exit()


        if (nplot_gas_co2 ==1 ) or (nplot_gas_soot == 1) :
            
            if flag_useLux:    
                locf.append(( (dxy*(i-i0)+dxyD*marge), (dxy*(j-j0)+dxyD*marge), (dz*(k-k0)+dzD*(1.+marge)) ))
                fluid.append(fluid_)
                
            else: 
                for fluid__ in fluid_:
                    locf.append(( (dxy*(i-i0)+dxyD*marge), (dxy*(j-j0)+dxyD*marge), (dz*(k-k0)+dzD*(1.+marge)) ))
                    fluid.append(fluid__)

        if (nplot_veg ==1 ) or (nplot_char == 1) or (nplot_ash == 1):
    
            if flag_useLux:    
                loct.append(( (dxy*(i-i0)+dxyD*marge), (dxy*(j-j0)+dxyD*marge), (dz*(k-k0)+dzD*(1.+marge)) ))
                turbid.append(turbid_)
            else:
                for turbid__ in turbid_:
                    loct.append(( (dxy*(i-i0)+dxyD*marge), (dxy*(j-j0)+dxyD*marge), (dz*(k-k0)+dzD*(1.+marge)) ))
                    turbid.append(turbid__)


    #
    # plume aerosols
    #
    if flag == 'atm': 
        fluid_ = []
        try: 
            if scene.tracer[i,j,k] > aerosol_tracer_max*aerosol_tracer_threshold:  
                max_height_plot = height_base+height_plot
                
                if (xCO2 > 0) or (xCO > 0) or (xH2O > 0): 
                    NAME_PHASE_FUN_GAS ,PHASE_FUN_ID_GAS = getNameAndIdFromGasAndTemp(lut_gas, 'gas', temperature=scene.temp[i,j,k], xmolar=[xH2O,xCO2,xCO])
                    fluid_.append(Fluid(density=1, fluidOpticalPropertyID=PHASE_FUN_ID_GAS, temperatureID=TEMP_FUN_ID))
                    nplot_plu = 1
                    xco2Temp.append([scene.temp[i,j,k],xH2O,xCO2,xCO])
                
                '''
                if dH2O > 0:  
                    NAME_PHASE_FUN_H2O ,PHASE_FUN_ID_H2O = getNameAndIdFromGasAndTemp(lut_h2o, 'h2o', temperature=scene.temp[i,j,k])
                    fluid.append(Fluid(density=dH2O, fluidOpticalPropertyID=PHASE_FUN_ID_H2O, temperatureID=TEMP_FUN_ID))
                    #loc.append(( (dxy*i+dxyD*0.01), (dxy*j+dxyD*0.01), (dz*k+dzD*1.01) ))
                    locf.append(( (dxy*(i-i0)+dxyD*marge), (dxy*(j-j0)+dxyD*marge), (dz*(k-k0)+dzD*(1.+marge)) ))

                if dCO2 > 0:
                    if PHASE_FUN_ID_CO2 is None:
                        NAME_PHASE_FUN_CO2 ,PHASE_FUN_ID_CO2 = getNameAndIdFromGasAndTemp(lut_co2, 'co2', temperature=scene.Temp[i,j,k])
                        fluid.append(Fluid(density=dCO2, fluidOpticalPropertyID=PHASE_FUN_ID_CO2, temperatureID=TEMP_FUN_ID))
                        loc.append((i,j,k))

                if dCO > 0:
                    if PHASE_FUN_ID_CO is None:
                        NAME_PHASE_FUN_CO ,PHASE_FUN_ID_CO = getNameAndIdFromGasAndTemp(lut_co, 'co', temperature=scene.Temp[i,j,k])
                        fluid.append(Fluid(density=dCO, fluidOpticalPropertyID=PHASE_FUN_ID_CO, temperatureID=TEMP_FUN_ID))
                        loc.append((i,j,k))
                
                if aerosolXMLvalue > 0: 
                    NAME_PHASE_FUN_SOOT, PHASE_FUN_ID_SOOT = getNameAndIdFromGasAndTemp(lut_opticProp_coeff_diff, 'soot' )  
                    NAME_PHASE_FUN_AEROSOL ,PHASE_FUN_ID_AEROSOL = '_',  NAME_PHASE_FUN_SOOT  # TO BE UPDATED
                    fluid.append(Fluid(density=aerosolXMLvalue, fluidOpticalPropertyID=PHASE_FUN_ID_AEROSOL, temperatureID=TEMP_FUN_ID))
                    loc.append((i,j,k))
                '''
                nplot_plu = 1
        
        except AttributeError as e:
            #print "AttributeError raised: {1}".format(e.strerror)
            pdb.set_trace()
            pass
       

        if (nplot_plu ==1 ) :  
            if flag_useLux:    
                locf.append(( (dxy*(i-i0)+dxyD*marge), (dxy*(j-j0)+dxyD*marge), (dz*(k-k0)+dzD*(1.+marge)) ))
                fluid.append(fluid_)
                
            else: 
                for fluid__ in fluid_:
                    locf.append(( (dxy*(i-i0)+dxyD*marge), (dxy*(j-j0)+dxyD*marge), (dz*(k-k0)+dzD*(1.+marge)) ))
                    fluid.append(fluid__)
    


    return fluid, turbid, locf, loct, nplot_veg, nplot_char, nplot_ash, nplot_gas_soot, nplot_gas_co2, nplot_gas_h2o, nplot_gas_co, nplot_plu, max_height_plot


#############################################################
def addAtmScenePlot2Mockup(inputConfig, mockup, fireScene, xs, ys, zs, atmScene, lut, fv_thresholds, T_ambient, dir_current_simulation, onecellDim=None):
   
    if onecellDim is None:
        nx,ny,nz = mockup.getMockupDimension()
        dxy,dz = mockup.getCellSize()
    else: 
        nx,ny,nz = onecellDim[:3]
        dxy,dz   = onecellDim[3:]
    Lx, Ly, Lz = nx*dxy, ny*dxy, nz*dz
    
    temp_threshold = temp_threshold_default
    if 'dao_tempThresh' in inputConfig.params_DART.keys(): 
        temp_threshold = inputConfig.params_DART['dao_tempThresh']
    
    nplot_plu = 0 
    
    flag_removeVeg = False
    if ('removeVeg' in inputConfig.params_model.keys()):
        flag_removeVeg = inputConfig.params_DART['removeVeg']
    
    flag_removeFlame = 'na'
  
    aerosol_tracer_max = atmScene.tracer[:,:,:].max()
    aerosol_tracer_threshold = fv_thresholds[1]

    if aerosol_tracer_threshold is not None: 
        index = np.where( atmScene.tracer > aerosol_tracer_max*aerosol_tracer_threshold)
        print('   max       aerosol tracer = {:2.5e}'.format(aerosol_tracer_max)) 
        print('   threshold aerosol tracer = {:2.5e}'.format(aerosol_tracer_threshold)) 
        print('   nbre cells available in the plume =', len(index[0]))
    else: 
        if atmScene.tracer.max()!=0:  
            print ('stop here, there is aerosol concentration but no aerosol_tracer_threshold defined in config file')
            sys.exit()
    
    dxA, dyA, dzA = atmScene.xc[1,1,1]-atmScene.xc[0,0,0], atmScene.yc[1,1,1]-atmScene.yc[0,0,0], atmScene.zc[1,1,1]-atmScene.zc[0,0,0] 

    #main loop 
    ii = 0
    fluids = []
    turbids = []
    locs   = []
    locts   = []
    idx = np.where(   (atmScene.xc-.5*dxA >= xs[0]) & (atmScene.xc+.5*dxA <= xs[-1]+dxy) \
                    & (atmScene.yc-.5*dyA >= ys[0]) & (atmScene.yc+.5*dyA <= ys[-1]+dxy) \
                    & (atmScene.zc-.5*dzA >= zs[0]) & (atmScene.zc+.5*dzA <= zs[-1]+dz ) )
    try: 
        i00 = idx[0].min()
        j00 = idx[1].min()
        k00 = idx[2].min()
    except: 
        pdb.set_trace()
    idx = np.where((atmScene.tracer > aerosol_tracer_max*aerosol_tracer_threshold) & (atmScene.xc-.5*dxA >= xs[0]) & (atmScene.xc+.5*dxA <= xs[-1]+dxy) \
                                                                                   & (atmScene.yc-.5*dyA >= ys[0]) & (atmScene.yc+.5*dyA <= ys[-1]+dxy) \
                                                                                   & (atmScene.zc-.5*dzA >= zs[0]) & (atmScene.zc+.5*dzA <= zs[-1]+dz ) )
    
    NN = atmScene.shape[0]*atmScene.shape[1]*atmScene.shape[2] 
    for i,j,k in zip(*idx):
    
        #if k == k00 : continue # to avoid overlap
        #add plot
        #-------
        fluids_, turbids_, locs_, locts_, nplot_veg_, nplot_char_, nplot_ash_,                            \
                        nplot_gas_soot_, nplot_gas_co2_, nplot_gas_h2o_, nplot_gas_co_, \
                        nplot_plu_, max_height_plot_ =                                  \
                 addPlotDescription2Mockup ( inputConfig.params_DART['useLux'] ,'fluid', 'na', flag_removeVeg, flag_removeFlame, 'atm',\
                                            i,j,k,i00,j00,k00, 
                                            dxA, dzA, 2*dxy, 2*dz,
                                            atmScene, lut, 
                                            temp_threshold,
                                            None, None, 
                                            aerosol_tracer_max, aerosol_tracer_threshold, 
                                            T_ambient,)
        #set fluid for veg flag as we do not have veg here 

        [fluids.append(fluid_) for fluid_ in fluids_]
        [turbids.append(turbid_) for turbid_ in turbids_]
        [locs.append(loc_)     for loc_   in locs_]
        [locts.append(loc_)     for loc_   in locts_]
        
        nplot_plu+=nplot_plu_
        print (r'{:.2f} %'.format(100.*ii/(NN)),end='\r')
        sys.stdout.flush()
        ii+=1
        #end j


    print('   There are {0} plots in the plume from atmScene.'.format(nplot_plu))

    '''
    #add fluid to mockup
    for fluid_,loc_ in zip(fluids,turbids,locs): 
        #mockup.addPatch(fluid_, loc_, [dxy,dxy,dz])
        if inputConfig.params_DART['useLux']:
            if len(fluid_)> 0: mockup.addPatch(fluid_, loc_, [dxA - 2*marge*dxy ,dxA - 2*marge*dxy, dzA - 2*marge*dz])
            if len(turbid_)> 0: mockup.addPatch(turbid_, loc_, [dxA - 2*marge*dxy ,dxA - 2*marge*dxy, dzA - 2*marge*dz])
        else:
            loc__ = (np.array(loc_[:2])/dxy).tolist() + (np.array(loc_[2])/dz).tolist() 
            if len(fluid_)> 0: mockup.addPatch(fluid_, loc__, [1,1,1])
            if len(turbid_)> 0: mockup.addPatch(turbid_, loc__, [1,1,1])
    '''
    #add fluid to mockup
    for fluid_, loc_ in zip(fluids,locs):
        #mockup.addPatch(fluid_, loc_, [dxy*(1-2*marge),dxy*(1-2*marge),dz*(1-2*marge)])
        if inputConfig.params_DART['useLux']:
            if len(fluid_)>0:  mockup.addPatch(fluid_, loc_, [dxA-2*marge*2*dxy ,dyA-2*marge*2*dxy,dzA-2*marge*2*dz])
        
        else:
            
            #loc__ = (np.array(loc_)/dxy).tolist()
            try:
                loc__ = (np.array(loc_[:2])/dxy).tolist() + (np.array([loc_[2]])/dz).tolist() 
            except: 
                pdb.set_trace()
            if loc__[0] >= nx : pdb.set_trace()
            if loc__[1] >= ny : pdb.set_trace()
            if loc__[2] >= nz : pdb.set_trace()
            mockup.addPatch([fluid_], loc__, [1,1,1])

    for turbid_, loc_ in zip(turbids,locts):
        #mockup.addPatch(fluid_, loc_, [dxy*(1-2*marge),dxy*(1-2*marge),dz*(1-2*marge)])
        if inputConfig.params_DART['useLux']:
            if len(turbid_)>0:  mockup.addPatch(turbid_, loc_, [dxA-2*marge*2*dxy ,dyA-2*marge*2*dxy,dzA-2*marge*2*dz])
        else:
            try:
                loc__ = (np.array(loc_[:2])/dxy).tolist() + (np.array([loc_[2]])/dz).tolist() 
            except: 
                pdb.set_trace()
            
            if loc__[0] >= nx : pdb.set_trace()
            if loc__[1] >= ny : pdb.set_trace()
            if loc__[2] >= nz : pdb.set_trace()
            mockup.addPatch([turbid_], loc__, [1,1,1])




    return mockup


#############################################################
def addFireScenePlot2Mockup(inputConfig, mockup, Firescene, lut, fv_thresholds, T_ambient, dir_current_simulation, onecellDim=None):
    
    if onecellDim is None:
        nx,ny,nz = mockup.getMockupDimension()
        dxy,dz = mockup.getCellSize()
    else: 
        nx,ny,nz = onecellDim[:3]
        dxy,dz   = onecellDim[3:]
    Lx, Ly, Lz = nx*dxy, ny*dxy, nz*dz

    temp_threshold = temp_threshold_default
    if 'dao_tempThresh' in inputConfig.params_DART.keys(): 
        temp_threshold = inputConfig.params_DART['dao_tempThresh']
    
    line_t = ''
    nplot_veg, nplot_char, nplot_ash, nplot_gas_soot, nplot_gas_co2, nplot_gas_h2o, nplot_gas_co, nplot_plu = 0, 0, 0, 0, 0, 0, 0, 0
    
    flag_GrdAmbientT = False
    if ('GrdAmbientT' in inputConfig.params_3DFS.keys()):
        flag_GrdAmbientT = inputConfig.params_3DFS['GrdAmbientT']
    
    #soot_fv_max = Firescene.fv[:,:,:].max()
    #aerosol_fv_max = Firescene.fv_aerosol[:,:,:].max()
    #fv_threshold = 0.01
    #soot_fv_threshold = fv_thresholds[0]
    soot_fv_max = None
    soot_fv_threshold = None
    #index = np.where( Firescene.fv > soot_fv_max * soot_fv_threshold)
    #print('   max       soot fv = {:}'.format(soot_fv_max)) 
    #print('   threshold soot fv = {:}'.format(soot_fv_max * soot_fv_threshold)) 
    #index = np.where(Firescene.hrrpuv >= 1.e-2 )
    
    flag_all_voxel = False
    
    if 'dart_config_bands' in inputConfig.params_DART.keys():
        if 'spectr' in inputConfig.params_DART['dart_config_bands']:
            flag_all_voxel = True
    
    if flag_all_voxel:
        print('select all cell')
        index = np.where((Firescene.Temp >  0))
    else:
        index = np.where((Firescene.Temp >= temp_threshold) & (Firescene.fv > 0))
    
    print('   nbre plot in the flame (T>T_threshold={:.1f}) ='.format(temp_threshold), len(index[0]))
    
    '''
    if aerosol_tracer_threshold is not None: 
        index = np.where( Firescene.fv_aerosol > aerosol_fv_max * aerosol_fv_threshold)
        print('   max       aerosol fv = {:2.5e}'.format(aerosol_fv_max)) 
        print('   threshold aerosol fv = {:2.5e}'.format(aerosol_fv_max * aerosol_fv_threshold)) 
        print('   nbre plot in the plume =', len(index[0]))
    else: 
        if Firescene.fv_aerosol.max()!=0:  
            print ('stop here, there is aerosol concentration but no aerosol_fv_threshold defined in config file')
            sys.exit()
    '''

    #dump ground temperature in the file temperature.txt (first layer)
    '''
    line_t = []
    for i in range(nx):
        for j in range(ny):
            dT = Firescene.Temp_grd[i,j,0]
            #if dT<0.1:    # DART does not accept null temperatures for non empty cells
            #    dT = 0.1
            #ft.write("{0:8.3f} ".format(dT))
            line_t.append("{0:8.3f} ".format(dT))
            #ft.write('\n')
            #line_t.append('\n')
        #ft.write('\n')
        line_t.append('\n')
    line_t.append('\n')
    '''
    line_t = np.zeros([Firescene.Temp.shape[0], Firescene.Temp.shape[1], Firescene.Temp.shape[2]+1]) 
    if not(flag_GrdAmbientT):
        line_t[:,:,0] = Firescene.Temp_grd[:,:,0]
    else:
        line_t[:,:,0] = temp_threshold

    max_height_plot = -999

    #main loop 
    ii = 0
    fluids = []
    turbids = []
    locfs   = []
    locts   = []
    key_temp_veg = 'Temp_veg'
    if 'lux_vegModel' in inputConfig.params_DART.keys():
        if inputConfig.params_DART['lux_vegModel'] == 'turbid': key_temp_veg = 'Temp_veg2'
    #flag_mm = True
    
    i00 = 0 
    j00 = 0
    k00 = 0
    flag_onlyGas = False
    if ('onlyGas' in inputConfig.params_model.keys()):
        flag_onlyGas = inputConfig.params_model['onlyGas']
    
    flag_lux_vegModel = 'fluid'
    if 'lux_vegModel' in inputConfig.params_DART.keys():
        flag_lux_vegModel = inputConfig.params_DART['lux_vegModel']
    
    flag_removeVeg = False
    if ('removeVeg' in inputConfig.params_3DFS.keys()):
        flag_removeVeg = inputConfig.params_3DFS['removeVeg']
    
    flag_removeFlame = False
    if ('removeFlame' in inputConfig.params_3DFS.keys()):
        flag_removeFlame = inputConfig.params_3DFS['removeFlame']

    '''
    print('start scene loop')
    for k in range(nz-1):
        for i in range(nx):
            for j in range(ny):
                
                # Write the temperature of the voxel (here voxel == cell) in temperature.txt
                #-------
                if (Firescene.MassDry[i,j,k] + Firescene.MassChar[i,j,k]) > min([minMassDry,minMassChar]): 
                    dT = Firescene[key_temp_veg][i,j,k]
                else:
                    dT = Firescene.Temp[i,j,k]
               
                if dT < 0 : pdb.set_trace()
                #if dT<0.1:    # DART does not accept null temperatures for non empty cells
                #    dT = 0.1
               
                #ft.write("{0:8.3f} ".format(dT))
                line_t.append("{0:8.3f} ".format(dT))
                # end of writing in temperature.txt

                if (Firescene.Temp[i,j,k] >= temp_threshold):
                    params.append( [flag_lux_vegModel, flag_onlyGas, 'fire',\
                                    i,j,k, i00, j00, k00, dxy, dz, dxy, dz,
                                    Firescene, lut, 
                                    soot_fv_max, soot_fv_threshold, 
                                    None, None, 
                                    T_ambient] )

            line_t.append('\n')
        
        line_t.append('\n')
   
    print('done')
    ''' 

    #line_t[:,:,1:] =np.where( (Firescene.MassDry + Firescene.MassChar) > min([minMassDry,minMassChar]), 
    #                          Firescene[key_temp_veg]  , Firescene.Temp )
    line_t[:,:,1:] =np.where( Firescene[key_temp_veg] > Firescene.Temp,  
                              Firescene[key_temp_veg]  , Firescene.Temp )

    params=[]
    ii = 0
    nn = len(index[0])
    for i,j,k in zip(*(index)): #np.where(Firescene.Temp >= temp_threshold))):

        param =  [inputConfig.params_DART['useLux'], flag_lux_vegModel, flag_onlyGas, flag_removeVeg, flag_removeFlame, 'fire',\
                        i,j,k, i00, j00, k00, dxy, dz, dxy, dz,
                        Firescene, lut, temp_threshold,
                        soot_fv_max, soot_fv_threshold, 
                        None, None, 
                        T_ambient] 
        
        fluids_, turbids_, locfs_, locts_, nplot_veg_, nplot_char_, nplot_ash_, \
                        nplot_gas_soot_, nplot_gas_co2_, nplot_gas_h2o_, nplot_gas_co_, \
                        nplot_plu_, max_height_plot_ = \
                        star_addPlotDescription2Mockup(param)
                 

        [fluids.append(fluid_) for fluid_ in fluids_]
        [turbids.append(turbid_) for turbid_ in turbids_]
        [locfs.append(loc_)     for loc_   in locfs_]
        [locts.append(loc_)     for loc_   in locts_]
        
        
        #    if len(fluids_)>0: 
        #        flag_mm = False

        nplot_veg+=nplot_veg_ ; nplot_char+=nplot_char_ ; nplot_ash+=nplot_ash_ ; 
        nplot_gas_soot+=nplot_gas_soot_ ; nplot_gas_co2+=nplot_gas_co2_ ;  nplot_gas_h2o+=nplot_gas_h2o_  ; nplot_gas_co+=nplot_gas_co_ 
        nplot_plu+=nplot_plu_
        max_height_plot = max([max_height_plot_,max_height_plot])
        print ('{:.2f} %\r'.format(100.*ii/nn),end='')
        
        #print (i,j,k)
        #if nplot_veg == 1: break # MERDE
        
        sys.stdout.flush()
        ii+=1
        
    
    print('   There are {0} plots with vegetation.'.format(nplot_veg))
    print('   There are {0} plots with char.'.format(nplot_char))
    print('   There are {0} plots with ash.'.format(nplot_ash))
    print('   There are {0} plots with soot.'.format(nplot_gas_soot))
    print('   There are {0} plots with co2.'.format(nplot_gas_co2))
    print('   There are {0} plots with h2o.'.format(nplot_gas_h2o))
    print('   There are {0} plots with co.'.format(nplot_gas_co))
    print('   There are {0} plots in the plume.'.format(nplot_plu))
    print('   And max temperature on the ground is {:.2f}'.format(line_t[:,:,0].max()))
    np.save('xco2Temp',xco2Temp)


    #add fluid to mockup
    for fluid_, loc_ in zip(fluids,locfs):
        #mockup.addPatch(fluid_, loc_, [dxy*(1-2*marge),dxy*(1-2*marge),dz*(1-2*marge)])
        if inputConfig.params_DART['useLux']:
            if len(fluid_)>0:  mockup.addPatch(fluid_, loc_, [(1-2*marge)*dxy ,(1-2*marge)*dxy,(1-2*marge)*dz])
        
        else:
            
            #loc__ = (np.array(loc_)/dxy).tolist()
            try:
                loc__ = (np.array(loc_[:2])/dxy).tolist() + (np.array([loc_[2]])/dz).tolist() 
            except: 
                pdb.set_trace()
            if loc__[0] >= nx : pdb.set_trace()
            if loc__[1] >= ny : pdb.set_trace()
            if loc__[2] >= nz : pdb.set_trace()
            mockup.addPatch([fluid_], loc__, [1,1,1])

    for turbid_, loc_ in zip(turbids,locts):
        #mockup.addPatch(fluid_, loc_, [dxy*(1-2*marge),dxy*(1-2*marge),dz*(1-2*marge)])
        if inputConfig.params_DART['useLux']:
            if len(turbid_)>0:  mockup.addPatch(turbid_, loc_, [(1-2*marge)*dxy ,(1-2*marge)*dxy,(1-2*marge)*dz])
        else:
            try:
                loc__ = (np.array(loc_[:2])/dxy).tolist() + (np.array([loc_[2]])/dz).tolist() 
            except: 
                pdb.set_trace()
            
            if loc__[0] >= nx : pdb.set_trace()
            if loc__[1] >= ny : pdb.set_trace()
            if loc__[2] >= nz : pdb.set_trace()
            mockup.addPatch([turbid_], loc__, [1,1,1])


    # write temperature.txt
    print('   write: ' + dir_current_simulation+'input/temperatures.txt')
    with open(dir_current_simulation+'input/temperatures.txt','w') as ft:
    #    ft.writelines(line_t)
        for tmp in np.moveaxis(line_t, -1, 0):
            np.savetxt(ft, tmp, fmt='%.3f')

    return mockup

##############################################################
def addAcetone(mockup, inputConfig, dirdata, time, id_opti=3, obstName='ACETONE POOL 2', onecellDim=None):
    
    if onecellDim is None:
        nx,ny,nz = mockup.getMockupDimension()
        dxy,dz = mockup.getCellSize()
    else: 
        nx,ny,nz = onecellDim[:3]
        dxy,dz   = onecellDim[3:]
    Lx, Ly, Lz = nx*dxy, ny*dxy, nz*dz
   
    #obstacle geometry
    #xb,xe=-0.5, 0.5   # on fds grid reference
    #yb,ye=-0.5, 0.5
    #AcetoneHeight = 0.05
    #read from fds config file
    xb,xe, yb,ye, _, AcetoneHeight = read_fds_obst.load(dirdata, obstName)

    AcetoneHeight_fds = np.floor(AcetoneHeight/dz)*dz + dz
    gridx,gridy,gridz=(xb,xe,int(np.round((xe-xb)/dxy,0))),\
                      (xb,xe,int(np.round((xe-xb)/dxy,0))),\
                      (0,AcetoneHeight_fds,int(AcetoneHeight_fds/dz)) 

    #interpolate temperature
    x, y, z, T  = read_fds_prof.loadProfile(dirdata, time, zZero=AcetoneHeight_fds)
    tempAcetone = read_fds_prof.interpolation2Domain(x, y, z, T, gridx,gridy,gridz )

    #set on DART Domain
    xdart = np.linspace(0,Lx,nx+1)[:-1] - Lx/2  # -Lx/2 as fds domain is centered
    ydart = np.linspace(0,Lx,nx+1)[:-1] - Ly/2
    ib = np.abs(xdart-xb).argmin()
    jb = np.abs(ydart-yb).argmin()

    tempAcetoneDartDomain = np.zeros([nx,ny,nz])
    tempAcetoneDartDomain[ib:ib+tempAcetone.shape[0],jb:jb+tempAcetone.shape[1],0:tempAcetone.shape[2]] = tempAcetone
    
    #plt.switch_backend('Qt5Agg')
    #plt.imshow(tempAcetoneDartDomain[:,:,tempAcetone.shape[2]-1].T, origin='lower'); plt.show()
    #pdb.set_trace()

     # some aliases on dao types
    Fluid = dao.Fluid
    TEMP_FUN_ID = 0

    nplot = 0
    fluids = []
    locs = []
    idx = np.where(tempAcetoneDartDomain>0)
    for i,j,k in zip(idx[0],idx[1],idx[2]):
        m_acetone = 784.e3 #gr/m3
        dAcetone = round(old_div(m_acetone, MAcetone) * NAVOGADRO )#* 1.e-15)     # because densities are in 1e15 m-3 in DART inputs

        fluids.append(Fluid(density=dAcetone, fluidOpticalPropertyID=id_opti, temperatureID=TEMP_FUN_ID))
        locs.append(( (i+marge)*dxy, (j+marge)*dxy, (k+marge)*dz ) )
        nplot += 1
    print('   There are {0} plots in the acetone.'.format(nplot))

    #add fluid to mockup
    for fluid_,loc_ in zip(fluids,locs): 
        #mockup.addPatch(fluid_, loc_, [dxy,dxy,dz])
        
        if inputConfig.params_DART['useLux']:
            mockup.addPatch(fluid_, loc_, [(1-2*marge)*dxy ,(1-2*marge)*dxy,(1-2*marge)*dz])
        
        else:
            #loc__ = (np.array(loc_)/dxy).tolist()
            loc__ = (np.array(loc_[:2])/dxy).tolist() + (np.array([loc_[2]])/dz).tolist() 
            
            mockup.addPatch([fluid_], loc__, [1,1,1])

    return mockup


##############################################################
def setUpDartWithDAO(root_postproc, inputConfig, Firescene, xs, ys, zs, dxy, dz, time, \
                       simulation_name, run_name, dir_simulation, fv_thresholds, T_ambient, curtainLoc,flag_run_phase, flag_extraFDS=False, flag_box=False, flag_debug=False, flag_onecell=False):
  
    if inputConfig.params_DART['flag_run_sensitivity']: run_name += '_sens'
    if flag_box                                       : run_name += '_box'
    dir_current_simulation = dir_simulation+'simulations/'+run_name+'/'
  
    DARTimageflag = '{:s}_'.format(inputConfig.params_DART['dart_config_bands']) \
                    if ('dart_config_bands' in inputConfig.params_DART.keys()) else '' 

    #print ('simulation_name ='+simulation_name)
    #print ('run_name        ='+run_name)
   
    dirFireSceneSrc = '/home/paugam/Src/dart-fire/src/'
    template_dir = 'template'
    if inputConfig.params_DART['useLux']:
        template_dir = template_dir + '_lux'
    #flag_test_geo = True
    #if flag_test_geo:
    #    template_dir = template_dir + '_noAtm_grdBB'
    shutil.copy(dirFireSceneSrc+'../data_static/DART/'+template_dir+'/input/coeff_diff.xml', dir_current_simulation+'input/coeff_diff.xml.bak')
    print('----------')
    print('template = '+template_dir)
    print('----------')
    
    try:
        shutil.copy(dir_current_simulation+'/input/coeff_diff.xml', dir_current_simulation+'input/coeff_diff.xml.bak')
    except: 
        pdb.set_trace()

    flag_removeAtm = False
    if ('removeAtm' in inputConfig.params_3DFS.keys()):
        flag_removeAtm = inputConfig.params_3DFS['removeAtm']
    
    
    #id from boundary file from FDS
    if flag_extraFDS:
        print('   ## Add Acetone ##')
        try: 
            dirdata   = inputConfig.params_model['Acetone_dirData'] #'/mnt/data/FDS/PoolFireAcetone/']
            filebf    = inputConfig.params_model['Acetone_bndfFile'] #'acetone_1_m_ronan_0001_01.bf'
            iplan_arr = inputConfig.params_model['Acetone_bndfID'] #[2,5,6,9]
        except: 
            dirdata   = '/mnt/data/FDS/PoolFireAcetone/'
            filebf    = 'acetone_1_m_ronan_0001_01.bf'
            iplan_arr = [2,5,6,9]

    # some aliases on dao types
    TriangleProperty = dao.TriangleProperty
    Triangle = dao.Triangle
    Turbid = dao.Turbid
    Fluid = dao.Fluid
    Matrix4 = dao.Matrix4

    # disable some warnings (i.e., when trying to add an element outside of the scene)
    dao.Mockup.warnings(False)

    nx,ny,nz = Firescene.shape
    nz += 1
    length = dxy*nx
    width = dxy*ny
    height = dz*nz

    #lambertian
    PHASE_GND_ID = 5  # set to needle instead of dirt = 0
    PHASE_SHEET_ID = 4
    
    #Fluid
    PHASE_ACETONE_ID = 3

    TEMP_FUN_GND_ID = 0

    #load Gas optical Prop
    #-----------------
    #database = 'fluid_Gas_4D_L001cm_noLog.db'
    #database = 'fluid_Gas_4D_L010cm.db'
    #database = 'fluid_Gas_4D_small_05cm_L001cm.db'
    #database = 'fluid_Gas_4D_Erez_L010cm.db'
    database =  inputConfig.params_DART['OPdataBase']
    if os.path.isfile(DART_LOCAL + '/database/' + database):
        databaseName = DART_LOCAL + '/database/' + database
    elif os.path.isfile(DART_HOME + '/database/' + database):
        databaseName = DART_HOME + '/database/' + database
    else:
        print ('missing database: ', database)	
    onlySelectedTemp = None
    print('database = ', database)
    
    if not(flag_debug):

        #print ('################')
        #print ('do not forget to reset removeGas')
        #print ('################')
        try:
            if 'removeGas' in inputConfig.params_model.keys():
                if inputConfig.params_model['removeGas']: 
                    onlySelectedTemp = np.array([1000])
        except: 
            pass
       
        lut_opticProp_coeff_diff = loadOpticProp2CoeffDiff.loadOpticProp2CoeffDiff(dir_current_simulation+'input/', databaseName, onlySelectedTemp= onlySelectedTemp)

        lut_gas = get_lut_gas(lut_opticProp_coeff_diff,)
        #lut_h2o = get_lut_gas(lut_opticProp_coeff_diff, 'h2o')
        #lut_co  = get_lut_gas(lut_opticProp_coeff_diff, 'co' )
        lut_veg  = get_lut_veg(lut_opticProp_coeff_diff)
        lut = [lut_gas, lut_veg, lut_opticProp_coeff_diff]


        #load FDS boundary file
        #-----------------
        if flag_extraFDS:
            bndf_arr, bndf_xyz, NB, IOR, NM, bndf_times = read_fds_bf.load_bndf(dirdata,filebf)


        #set temperature in coeff_diff
        #-----------------
        if flag_extraFDS:
            nExistingTemp = addTempWall2coeffdiff.writeTemp2CoeffDiff(time, iplan_arr, bndf_arr, bndf_xyz, bndf_times, dir_current_simulation)
        #else:
        #    shutil.copy2( dir_current_simulation+'input/coeff_diff.xml.bak', dir_current_simulation+'input/coeff_diff.xml')

        
        #copy xml file to DART simu and run direction and phase
        #-----------------
        if flag_run_phase:
            flag2run='directionPhase'
        else:
            flag2run='dirSetup'
            print('   skip direction and phase')

        rteModel.run_dart(inputConfig, root_postproc, simulation_name, time_requested=round(time,2), flag_set_up_box4_RadFlux=flag_box, flag2run=flag2run,)
    
    else:
        coeff_diff_dir_ = DART_LOCAL + 'simulations/' + simulation_name + run_name.replace('{:s}t_'.format(DARTimageflag),'_') + '/input/'
        lut_opticProp_coeff_diff = loadOpticProp2CoeffDiff.loadOpticProp2CoeffDiff(coeff_diff_dir_, databaseName, onlySelectedTemp= onlySelectedTemp,flag_saving=False)
        lut_gas = get_lut_gas(lut_opticProp_coeff_diff,)
        lut_veg  = get_lut_veg(lut_opticProp_coeff_diff)
        lut = [lut_gas, lut_veg, lut_opticProp_coeff_diff]

    
    # create new scene (or replace existing scene)
    #-----------------
    if flag_onecell: 
        sim = dao.createNewScene(simulation_name+run_name.replace('{:s}t_'.format(DARTimageflag),'_'), DART_HOME, DART_LOCAL, (length, width, height), (length, height), lux_mode=inputConfig.params_DART['useLux'])
        onecellDim = nx,ny,nz,dxy,dz
    else:
        print('(length, width, height) ', length, width, height, end='\n')
        print('(dxy, dz) ', dxy, dz, end='\n')
        sim = dao.createNewScene(simulation_name+run_name.replace('{:s}t_'.format(DARTimageflag),'_'), DART_HOME, DART_LOCAL, (length, width, height), (dxy, dz), lux_mode=inputConfig.params_DART['useLux'])
        onecellDim = None 
    out = sim.getOutputDAO()
    mockup = out.getMockup(dao.Mockup.EMPTY)

    dim = mockup.getMockupDimension()
    size = mockup.getSceneSize()
    cell = mockup.getCellSize()

    print ('   mockup Size = ', size)
    print ('   mockup cellSize = ', cell)

    #set DEM
    #-----------------
    #if flag_onecell: 
    #    create_raster_dem(dir_current_simulation+'input/',1,1,nz,length,length,dz)
    #else:
    create_raster_dem(dir_current_simulation+'input/',nx+1,ny+1,nz,dxy,dxy,dz)
    zscale = 1.
    # load DEM file
    #print(dir_current_simulation+'input/dem_raster.tif')
    dem = dao.DEMloader(dir_current_simulation+'input/dem_raster.tif', size,  cell, repeat=False, scale=True, zscale=1)
    dem.load()
    #dem.load(dir_current_simulation+'input/dem_raster.tif')
    #toto = dao.DEMloader()
    #toto.__size = dem.shape
    #toto.__matrix = dem
    #toto.__minmax = [dem.min(), dem.max()]
    # define an optical property for ground geometry
    groundProperty = TriangleProperty(TriangleProperty.GROUND, False, TriangleProperty.LAMBERTIAN, PHASE_GND_ID, TEMP_FUN_GND_ID)
    # add elevation data to the scen
    dem.addToMockup(mockup, groundProperty),#False, True, zscale, )
    #toto.addToMockup(mockup, groundProperty, False, False, zscale, )



    # add fluid plot from FireScene to mockup and save temperature.txt
    #-----------------
    print('   ----------')
    print('   --Add Fire Scene')
    mockup = addFireScenePlot2Mockup(inputConfig, mockup, Firescene, lut, fv_thresholds, T_ambient, dir_current_simulation, onecellDim=onecellDim)
    shutil.copy( dir_current_simulation+'input/temperatures.txt', DART_LOCAL + 'simulations/' + simulation_name + run_name.replace('{:s}t_'.format(DARTimageflag),'_') + '/input/' )
   

    # add plume plot from AtmScene to mockup. Temperature is saved above
    #-----------------
    dirAtm = inputConfig.params_2DFS['root'] + 'Postproc/' + inputConfig.params_2DFS['dir_postproc'] + '/FromModels/'
    if (os.path.isfile(dirAtm+'timeSeries.npy')) & (not(flag_removeAtm)):
        timeInfo_atmScene = np.load(dirAtm+'timeSeries.npy'); timeInfo_atmScene = timeInfo_atmScene.view(np.recarray)
        if 'filenameRaw' in timeInfo_atmScene.dtype.names:
            print('   ----------')
            print('   --Add Atm Scene')
            atmScene_name2 = timeInfo_atmScene.filenameRaw[np.abs(timeInfo_atmScene.time-time).argmin()]
            AtmsceneFilename = dirAtm + atmScene_name2
            #time__, Atmscene = np.load(AtmsceneFilename,allow_pickle=True)
            time__, Atmscene = myPickle.decompress(AtmsceneFilename+'.pbz2')
            mockup = addAtmScenePlot2Mockup(inputConfig, mockup, Firescene, xs, ys, zs,  Atmscene, lut, fv_thresholds, T_ambient, dir_current_simulation, onecellDim=onecellDim)
            


    #load fds bndf geometry and add to mockup
    #-----------------
    if flag_extraFDS:
        mockup = addBndfAsTriangle2Mockup(mockup, iplan_arr, bndf_arr, bndf_xyz, bndf_times, time, 
                                          id_opti=PHASE_SHEET_ID, id_tempInit=nExistingTemp, onecellDim=onecellDim)


    if curtainLoc is not None:
        print('   add curtain at:', curtainLoc)
        for curtainLoc_ in curtainLoc:
            mockup = addCurtain(mockup, curtainLoc_[0], curtainLoc_[1], curtainLoc_[2], id_opti=2, id_temp=2)


    #load fds profile, create geometry and add to mockup
    #-----------------
    if flag_extraFDS:
        mockup = addAcetone(mockup, inputConfig, dirdata, time, id_opti=PHASE_ACETONE_ID,obstName=inputConfig.params_model['Acetone_surfName'], onecellDim=onecellDim)


    # special simulation for markup spot
    if 'run_backgrd_simulation_with_markup' in inputConfig.params_DART.keys():
        if (time<0) & (inputConfig.params_DART['run_backgrd_simulation_with_markup']):
       
            print('Add markup')
            boxProperties = TriangleProperty(TriangleProperty.WALL, False, TriangleProperty.LAMBERTIAN, 2, 3,) 
            imark,jmark,kmark=[100,100,100,100],[100,100,100,100],np.array([ 19, 29, 39, 49]) -1 +5  #-1cm (estimation of exp acetone level + 5cm to reset on domain origin) 
            bb = 6 # even
            for i_, j_, k_ in zip(imark,jmark,kmark):
                print((i_-bb/2+marge)*dxy, (j_-bb/2+marge)*dxy, (k_-bb/2+marge)*dz, dxy*(bb-2*marge),dxy*(bb-2*marge),dz*(bb-2*marge))
                mesh = createCube([(i_-bb/2+marge)*dxy, (j_-bb/2+marge)*dxy, (k_-bb/2+marge)*dz], [dxy*(bb-2*marge),dxy*(bb-2*marge),dz*(bb-2*marge)])
                mesh.add_to_mockup(mockup, boxProperties, isDEM=False)


    # add box to compute FRP
    #--------------------
    if flag_box: 
        boxProperties = TriangleProperty(TriangleProperty.WALL, True, TriangleProperty.LAMBERTIAN, 3, 1, TriangleProperty.LAMBERTIAN, 2, 1) 
        #boxProperties = TriangleProperty(TriangleProperty.WALL, False, TriangleProperty.LAMBERTIAN, 2, 1) 
        if inputConfig.params_DART['useLux']:
            print ('   add FRPbox: lowerLeft({:},{:},{:}) upperRight({:},{:},{:}) corners'.format(0+marge*dxy, 0+marge*dxy, 0+marge*dz, dxy*(nx-marge),dxy*(ny-marge),dz*(nz-marge)))
            mesh = createCube([0+marge*dxy, 0+marge*dxy, 0+marge*dz], [dxy*(nx-2*marge),dxy*(ny-2*marge),dz*(nz-2*marge)])
            mesh.add_to_mockup(mockup, boxProperties, isDEM=False)
        else:
            print ('   add FRPbox: lowerLeft({:},{:},{:}) upperRight({:},{:},{:}) corners'.format(0+marge*dxy, 0+marge*dxy, 0+marge*dz, dxy*(nx-marge),dxy*(ny-marge),dz*(nz-marge)))
            mesh = createCube([0+marge*dxy, 0+marge*dxy, 0+marge*dz], [dxy*(nx-2*marge),dxy*(ny-2*marge),dz*(nz-2*marge)])
            #mesh = createCube([0, 0, 0], size)
            mesh.addToMockup(mockup, boxProperties)
            #mesh.exportToPLY('toto.ply')
            '''
            planeProperty = TriangleProperty(TriangleProperty.WALL, True, TriangleProperty.LAMBERTIAN, id_opti, idT_)
            #transform = Matrix4().scale(scale*dxfds,scale*dxfds,scale*dxfds).rotate(rotAngle,rotAxe,radians=False)\
            #                     .translate((scale*dxfds)/2,0,(scale*dxfds)/2).translate(xc+x_,yc+y_,zc+z_+0.01)
            
            
            if rotAxe == 'x':
                transform = Matrix4().rotate(rotAngle,rotAxe,radians=False).scale(scale*dxfds,scale*dxfds,scale*dxfds)\
                                    .translate((scale*dxfds)/2,0,(scale*dxfds)/2).translate(xc+x_,yc+y_,zc+z_)
            elif rotAxe == 'y':
                transform = Matrix4().rotate(rotAngle,rotAxe,radians=False).scale(scale*dxfds,scale*dxfds,scale*dxfds)\
                                     .translate(0,(scale*dxfds)/2,(scale*dxfds)/2).translate(xc+x_,yc+y_,zc+z_)
             
            
            obj = dao.OBJloader(os.path.join(DART_HOME, "database", "3D_Objects","Square.obj"))
            obj.load()
            obj.addToMockup(mockup, planeProperty, transform)
            '''
    #save
    #-----------------
    #mockup.enableNewMaketFormat(DART_HOME,False)
    #mockup.save(sceneSize=size, cellSize=cell, centeredCube=False)
    pdb.set_trace()
    mockup.save( centeredCube=False)
    print('   mockup saved')
   
    
    rteModel.run_dart(inputConfig, root_postproc, simulation_name, time_requested=round(time,2), flag_set_up_box4_RadFlux=flag_box, flag2run='atmosphere')
   

    return 'setup dart simu done'

'''
#######################################################
def createCube_FT(mockup, prop, xyz, sizeXYZ):
    vertices = [tuple(xyz)]
    vertices.append( (xyz[0] + sizeXYZ[0], xyz[1], xyz[2]) )
    vertices.append( (xyz[0] + sizeXYZ[0], xyz[1] + sizeXYZ[1], xyz[2]) )
    vertices.append( (xyz[0], xyz[1] + sizeXYZ[1], xyz[2]) )
    vertices.append( (xyz[0], xyz[1], xyz[2] + sizeXYZ[2]) )
    vertices.append( (xyz[0] + sizeXYZ[0], xyz[1], xyz[2] + sizeXYZ[2]) )
    vertices.append( (xyz[0] + sizeXYZ[0], xyz[1] + sizeXYZ[1], xyz[2] + sizeXYZ[2]) )
    vertices.append( (xyz[0], xyz[1] + sizeXYZ[1], xyz[2] + sizeXYZ[2]) )
    vertices = np.array(vertices, dtype=np.float32)
    for v1,v2,v3 in [ (1, 0, 3), (1, 3, 2), (0, 4, 7), (0, 7, 3), (4, 0, 1), (4, 1, 5), (7, 4, 6), (4, 5, 6), (5, 1, 6), (1, 2, 6), (6, 2, 7), (2, 3, 7) ]
        t = Triangle(prop, False, vertex1 = vertices[v1], vertex2=vertices[v2], vertex3=vertices[v3])
        mockup.addTriangle(t) 
'''    


#######################################################
def createCube(xyz, sizeXYZ):
    vertices = [tuple(xyz)]
    vertices.append( (xyz[0] + sizeXYZ[0], xyz[1], xyz[2]) )
    vertices.append( (xyz[0] + sizeXYZ[0], xyz[1] + sizeXYZ[1], xyz[2]) )
    vertices.append( (xyz[0], xyz[1] + sizeXYZ[1], xyz[2]) )
    vertices.append( (xyz[0], xyz[1], xyz[2] + sizeXYZ[2]) )
    vertices.append( (xyz[0] + sizeXYZ[0], xyz[1], xyz[2] + sizeXYZ[2]) )
    vertices.append( (xyz[0] + sizeXYZ[0], xyz[1] + sizeXYZ[1], xyz[2] + sizeXYZ[2]) )
    vertices.append( (xyz[0], xyz[1] + sizeXYZ[1], xyz[2] + sizeXYZ[2]) )
    vertices = np.array(vertices, dtype=np.float32)
    faces = np.array([ (1, 0, 3), (1, 3, 2), (0, 4, 7), (0, 7, 3), (4, 0, 1), (4, 1, 5), (7, 4, 6), (4, 5, 6), (5, 1, 6), (1, 2, 6), (6, 2, 7), (2, 3, 7) ], dtype=np.int32)
    return Mesh(vertices=vertices, faces=faces)


#######################################################
def run_dao(inputConfig, FireName, dxy, dz, time, root_postproc, dir_3DFS, dir_out, fv_thresholds, T_ambient, curtainLoc, flag_extraFDS, flag_box, flag_run_phase, flag_debug=False, flag_onecell=None):
    
    DARTimageflag = '{:s}_'.format(inputConfig.params_DART['dart_config_bands']) \
                    if ('dart_config_bands' in inputConfig.params_DART.keys()) else ''
    
    name_simu = FireName #'fdsExample_multiproc'
    run_name  = '{:s}t_{:03d}_{:02d}_s'.format(DARTimageflag, *np.array(math.modf(round(time,2))*np.array([100,1]),dtype=int)[::-1] ) #'t_039_00_s'#_luxDAO'

    #load firescene
    xs,ys,zs,wind_speed,Firescene = myPickle.decompress(dir_3DFS+ '3DfS_t_{:04d}_{:02d}.pickle.pbz2'.format( \
                                                                      *np.array(math.modf(round(time,2))*np.array([100,1]),dtype=int)[::-1] ) )
    xs-=xs[0]
    ys-=ys[0]
    mesg = setUpDartWithDAO(root_postproc, inputConfig, Firescene, xs,ys,zs, dxy,dz, time, 
                       name_simu, run_name, dir_out.replace('simulations',''), fv_thresholds, T_ambient, curtainLoc, flag_run_phase, flag_extraFDS=flag_extraFDS,flag_box=flag_box, flag_debug=flag_debug, flag_onecell=flag_onecell)

    return mesg


#######################################################
def run_dao_noLoading(inputConfig, FireName, dxy, dz, time, root_postproc,dir_3DFS, dir_out, fv_thresholds, T_ambient, curtainLoc, flag_extraFDS, flag_box, loadedData, flag_run_phase, flag_debug=False, flag_onecell=None):

    DARTimageflag = '{:s}_'.format(inputConfig.params_DART['dart_config_bands']) \
                    if ('dart_config_bands' in inputConfig.params_DART.keys()) else ''

    name_simu = FireName #'fdsExample_multiproc'
    run_name  = '{:s}t_{:03d}_{:02d}_s'.format(DARTimageflag, *np.array(math.modf(round(time,2))*np.array([100,1]),dtype=int)[::-1] ) #'t_039_00_s'#_luxDAO'

    xs,ys,zs,wind_speed,Firescene = loadedData 
   
    xs-=xs[0]
    ys-=ys[0]
    mesg = setUpDartWithDAO(root_postproc, inputConfig, Firescene, xs,ys,zs, dxy,dz, time, 
                       name_simu, run_name, dir_out.replace('simulations',''), fv_thresholds, T_ambient, curtainLoc, flag_run_phase, flag_extraFDS=flag_extraFDS,flag_box=flag_box, flag_debug=flag_debug, flag_onecell=flag_onecell)
    

    return mesg


#############################
if __name__ == '__main__':
#############################

    if True: 

        flag_debug = False
        nx=None
        nz=None
        if False:
            FireName = 'fireFlux'
            dxy = 5.00 
            dz  = 0.2
            #time= -2.00
            time= 46060.0
            #time= -1.00
            #
            path_data = '/mnt/data/' #'/home/paugam/Data/' #'/mnt/data/'
            root_postproc = path_data + 'MNH/FireFlux/Postproc/'
            dir_3DFS      = path_data + 'MNH/FireFlux/Postproc/3DFire/'
            dir_out       = path_data + 'MNH/FireFlux/Postproc/DART_maket/simulations/' 
            #
            fv_thresholds = [1.e-1,  1.e-5]
            T_ambient     = 290.84999999999997
            flag_extraFDS = False
            flag_box = False
            flag_onecell = False
        
        ##--
        ## input param 
        if False:
            FireName = 'fdsExample_multiproc'
            dxy = 0.04 
            dz  = 0.04
            time= -2.00
            #time= 39.00
            #time= -1.00
            #
            path_data = '/mnt/data/' #'/home/paugam/Data/' #'/mnt/data/'
            root_postproc = path_data + 'FDS/test_multiproc/Postproc/'
            dir_3DFS      = path_data + 'FDS/test_multiproc/Postproc/3DFire/'
            dir_out       = path_data + 'FDS/test_multiproc/Postproc/DART_maket/simulations/' 
            #
            fv_thresholds = [1.e-1,  None]
            T_ambient     = 297.15
            flag_extraFDS = False
            flag_box = True
            flag_onecell = False
        
        ##--
        if False: 
            FireName = 'knp14_skukuza6'
            dxy =2.02
            dz  = 8
            time= 599.856
            #
            root_postproc = '/media/paugam/goulven/data/2014_SouthAfrica_LandLord/Postproc/Skukuza6/'
            dir_3DFS      = '/media/paugam/goulven/data/2014_SouthAfrica_LandLord/Postproc/Skukuza6/3DFire/'
            dir_out       = '/media/paugam/goulven/data/2014_SouthAfrica_LandLord/Postproc/Skukuza6/DART_maket/simulations/'
            #
            fv_thresholds = [1.e-1,  0.03]
            T_ambient     = 300.15
            flag_extraFDS = False
            flag_box = True
        
        ##--
        if False: 
            FireName = 'fds_poolFire'
            dxy = 0.01
            dz  = 0.01
            time= .5
            #
            root_postproc = '/mnt/data/FDS/PoolFireAcetone/Postproc/'
            dir_3DFS      = '/mnt/data/FDS/PoolFireAcetone/Postproc/3DFire/'
            dir_out       = '/mnt/data/FDS/PoolFireAcetone/Postproc/DART_maket/simulations/'
            #
            fv_thresholds = [.15,  None]
            T_ambient     = 297.15
            flag_extraFDS = True
            flag_box = False
            flag_onecell = False
        
        ##--
        if False: 
            FireName = 'fds_Case_C064'
            dxy = .5
            dz  = .25
            time= 50.0
            #
            root_postproc = '/mnt/data/FDS/Case_C064_high_2veg/Postproc/'
            dir_3DFS      = '/mnt/data/FDS/Case_C064_high_2veg/Postproc/3DFire/'
            dir_out       = '/mnt/data/FDS/Case_C064_high_2veg/Postproc/DART_maket/simulations/'
            #
            fv_thresholds = [.5,  None]
            T_ambient     = 297.15
            flag_extraFDS = False
            flag_box = False
            flag_onecell = False

        ##--
        if True: 
            FireName = 'fds_burner_test'
            dxy = 0.1
            dz  = 0.03333333333333333
            time= 4
            #
            root_postproc = os.getcwd()+'/Postproc/'
            dir_3DFS      = os.getcwd()+'/Postproc/3DFire/'
            dir_out       = os.getcwd()+'/Postproc/DART_maket/simulations/'
            #
            fv_thresholds = [.1,  None]
            T_ambient     = 293.15
            flag_extraFDS = False
            flag_box = False
            flag_onecell = False
            curtainLoc = None
            flag_debug = False

        ##--
        if False: 
            FireName = 'fds_burnerT3'
            dxy = 0.1
            dz  = 0.03333333333333333
            time= 50
            #
            root_postproc = '/media/paugam/gast/FDS/BurnerT3/Postproc/'
            dir_3DFS      = '/media/paugam/gast/FDS/BurnerT3/Postproc/3DFire/'
            dir_out       = '/media/paugam/gast/FDS/BurnerT3/Postproc/DART_maket/simulations/'
            #
            fv_thresholds = [.1,  None]
            T_ambient     = 293.15
            flag_extraFDS = False
            flag_box = False
            flag_onecell = False
            curtainLoc = None
            flag_debug = False 
    else: 
        parser = argparse.ArgumentParser(description='set dart simulation using DAO.')
        parser.add_a652629rgument('-name','--firename', help='simulation name',      required=True)
        parser.add_argument('-dxy','--dxy'      , help='horizontal resolution',required=True)
        parser.add_argument('-dz','--dz'        , help='vertial resolution',   required=True)
        parser.add_argument('-t','--time'       , help='simulation time',      required=True)
        parser.add_argument('-dirPostproc', '--dirPostproc' , help='dir root postproc', required=True)
        parser.add_argument('-dir3dFS',     '--dir3dFS'     , help='dir 3DFS'         , required=True)
        parser.add_argument('-dirOut',      '--dirOut'      , help='dir out'          , required=True)
        parser.add_argument('-fv',          '--fvthreshold' , help='fv_thresholds'    , required=True)
        parser.add_argument('-tempA',       '--tambient'    , help='T_ambient'        , required=True)
        parser.add_argument('-extraFDS',    '--extraFDS'    , help='flag_extraFDS'    , required=True)
        parser.add_argument('-box',    '--box'    , help='flag_box'    , required=True)
        parser.add_argument('-onecell',    '--onecell'    , help='flag oneCell'    , required=True)
        parser.add_argument('-curtainLoc',    '--curtainLoc'    , help='curtain location'    , required=True)
        args = parser.parse_args()

        FireName = args.firename 
        dxy      = float(args.dxy) 
        dz       = float(args.dz) 
        time     = float(args.time) 
        # 
        root_postproc = args.dirPostproc
        dir_3DFS      = args.dir3dFS
        dir_out       = args.dirOut
        #
        fv_thresholds = [str2float(xx) for xx in args.fvthreshold.split(',')]
        T_ambient     = float(args.tambient)
        flag_extraFDS = str2Bool(args.extraFDS)
        flag_box      = str2Bool(args.box)
        flag_onecell      = str2Bool(args.onecell)
        flag_debug = False

        curtainLoc = args.curtainLoc

    inputConfig =importlib.machinery.SourceFileLoader('input_params_'+FireName,os.getcwd()+'/input_config/input_params_'+FireName+'.py').load_module()

    #load firescene
    loadedData = myPickle.decompress(dir_3DFS+ '3DfS_t_{:04d}_{:02d}.pickle.pbz2'.format( \
                                                                      *np.array(math.modf(round(time,2))*np.array([100,1]),dtype=int)[::-1] ) )

    flag_run_phase = True
    #mesg = run_dao(inputConfig, FireName, dxy, dz, time, root_postproc,dir_3DFS, dir_out, fv_thresholds, T_ambient, curtainLoc, flag_extraFDS, flag_box, flag_run_phase, flag_debug, flag_onecell=flag_onecell)

    mesg = run_dao_noLoading(inputConfig, FireName, dxy, dz, time, root_postproc,dir_3DFS, dir_out, fv_thresholds, T_ambient, curtainLoc, flag_extraFDS, flag_box, loadedData, flag_run_phase, flag_debug=flag_debug, flag_onecell=flag_onecell)

    print(mesg)
    '''
    # define some parameters to shape and arrange the scene
    dxy =0.02 
    dz  = 0.02
    time= 20
    
    dir_3DFS = '/mnt/data/FDS/PoolFireAcetone/Postproc/3DFire/'
    with open(dir_3DFS+ '3DfS_t_{:04d}_{:02d}.pickle'.format( *np.array(math.modf(round(time,2))*np.array([100,1]),dtype=int)[::-1] ), 'rb') as infile:
        xs,ys,zs,wind_speed,Firescene = pickle.load(infile)
    
    simulation_name = 'fds_poolFire_020_00_s_lux'
    dir_simulation  = DART_LOCAL + '/'
    #simulation_name = 't_020_00_s'
    #dir_simulation  = '/mnt/data/FDS/PoolFireAcetone/Postproc/DART_maket/'
    fv_thresholds = [1.5e-1,  None]
    T_ambient     = 24 + 273

    createMaketWithDAO(None, Firescene, dxy,dz, time, simulation_name, dir_simulation, fv_thresholds, T_ambient, flag_extraFDS=True)
    '''

