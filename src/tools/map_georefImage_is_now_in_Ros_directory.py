
path_src = '/home/ronan/Src/'

import sys,os
import numpy as np 
import matplotlib as mpl 
from matplotlib import cm
import matplotlib.pyplot as plt
import glob 
import shapefile
from PIL import Image, ImageDraw
from osgeo import gdal,osr,ogr
from gdalconst import *
import itertools
from scipy import interpolate
sys.path.append('../')
import fireScene2D as fS2D
import fireScene3D as fS3D
import pdb 
import math 
import pandas
import datetime
import matplotlib.path as mpath
from matplotlib.path import Path as mpPath
import matplotlib.patches as mpatches
import itertools
from shapely.geometry import Polygon
import multiprocessing
sys.path.append(path_src+'Factor_number/')
import factor
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import io,ndimage,stats,signal 
import pickle 

#####################################################
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d) 


#######################################################################
def UTMZone(lon, lat):
    return int((lon + 180) / 6) + 1 


######################################################################
def planck_radiance(wavelength,BT):
    # input: wavelength is the wavelength in micron
    #        BT is the brightness Temperature
    # output: the Radiance in W/(m2.sr.um) 
    #
    #      Quantity      Sym.     Value          Unit         Relative
    #                                                      uncertainty (ppm)
    #    -------------------------------------------------------------------
    #     speed of light   c    299792458          m/s          exact
    #      in a vacuum
    #
    #        Planck        h    6.6260755(40)   1.0e-34Js       0.60
    #       constant
    #
    #       Boltzmann      k    1.380658(12)    1.0e-index      8.5
    #       constant

    c = 299792458.e0
    h = 6.6260755e-34 
    k = 1.380658e-23

    # constant
    c1 =  2*h*c*c  # [W.m2]
    c2 =  h*c/k    # [K.m]

    L = np.zeros(wavelength.shape)
    idx___ok = np.where(BT > 0)
    idx_nook = np.where(BT == 0)

    L[idx___ok] = c1 / ( (wavelength[idx___ok]*1.e-6)**5 * ( np.exp(c2/(wavelength[idx___ok]*1.e-6*BT[idx___ok]) ) - 1 ) ) * 1.e-6
    L[idx_nook] = 0.

    return L


######################################################################
def planck_temperature(wavelength,radiance):
    # input: wavelength is the wavelength in micron
    #        Rad is the Radianace W/(m2.sr.um)
    # output: the Brightness Temperature
    #
    #      Quantity      Sym.     Value          Unit         Relative
    #                                                      uncertainty (ppm)
    #    -------------------------------------------------------------------
    #     speed of light   c    299792458          m/s          exact
    #      in a vacuum
    #
    #        Planck        h    6.6260755(40)   1.0e-34Js       0.60
    #       constant
    #
    #       Boltzmann      k    1.380658(12)    1.0e-23J/K      8.5
    #       constant
    
    c = 299792458.e0
    h = 6.6260755e-34 
    k = 1.380658e-23

    # constant
    c1 =  2*h*c*c  # [W.m2]
    c2 =  h*c/k    # [K.m]
    
    BT = np.zeros(wavelength.shape)
    idx___ok = np.where(radiance > 0)
    idx_nook = np.where(radiance == 0)

    BT[idx___ok] = c2 / ( wavelength[idx___ok]*1.e-6 * np.log( c1/( (wavelength[idx___ok]*1.e-6)**5 * radiance[idx___ok]*1.e6 ) + 1 ) )
    BT[idx_nook] = 0.

    return BT


######################################################
def downgrade_resolution(arr, diag_res_cte_shape, flag_interpolation='conservative'):

    '''
    flag_interpolation is conservative, or use max value in the new grid box
    '''
    if flag_interpolation == 'max':
        return shrink_max(arr, diag_res_cte_shape[0], diag_res_cte_shape[1])
   
    elif flag_interpolation == 'min':
        return shrink_min(arr, diag_res_cte_shape[0], diag_res_cte_shape[1])
    
    elif flag_interpolation == 'conservative':
        scaling_ratio = arr.shape[0] / diag_res_cte_shape[0]
        return shrink_sum(arr, diag_res_cte_shape[0], diag_res_cte_shape[1]) / scaling_ratio**2

    elif flag_interpolation == 'average':
        return shrink_average(arr, diag_res_cte_shape[0], diag_res_cte_shape[1])
    
    elif flag_interpolation == 'sum':
        return shrink_sum(arr, diag_res_cte_shape[0], diag_res_cte_shape[1])

    else:
        print 'bad flag'
        pdb.set_trace()

######################################################
def shrink_sum(data, rows, cols):
    return data.reshape(rows, data.shape[0]/rows, cols, data.shape[1]/cols).sum(axis=1).sum(axis=2)


######################################################
def shrink_max(data, rows, cols):
    return data.reshape(rows, data.shape[0]/rows, cols, data.shape[1]/cols).max(axis=1).max(axis=2)


######################################################
def shrink_min(data, rows, cols):
    return data.reshape(rows, data.shape[0]/rows, cols, data.shape[1]/cols).min(axis=1).min(axis=2)


######################################################
def shrink_average(data, rows, cols, nodata=-999):
    out = np.zeros([rows,cols]) + nodata
    id_cell = ndimage.zoom(np.arange(rows*cols).reshape([rows,cols]), (data.shape[0]/rows, data.shape[1]/cols), order=0)
    for id_cell_val in np.unique(id_cell):
        idx=np.where( (id_cell==id_cell_val) & (data!=nodata) )
        if len(idx[0])==0: 
            continue
        out[np.unravel_index(id_cell_val, (rows,cols))] = data[idx].mean()
    return out
    #return ((data.reshape(rows, data.shape[0]/rows, cols, data.shape[1]/cols).sum(axis=1)/(data.shape[0]/rows)).sum(axis=2)/(data.shape[1]/cols))


#######################################
def out_plotMask(grid_e,grid_n,pts_utm):
    
    nx, ny = grid_e.shape
    resolution = grid_e[1,1]-grid_e[0,0]
    pt_sw = [grid_e[0,0],grid_n[0,0]]
    plotmask = np.zeros_like(grid_e)
    # polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
    width, height = plotmask.shape
    polygon =[ tuple( np.array(np.round((np.array(pt)-np.array(pt_sw))/resolution,0),dtype=np.int))  \
               for pt in pts_utm ]
    '''
    polygon_c = []
    nx,ny = plotmask.shape[0]-1,  plotmask.shape[1]-1
    for pt in  polygon:
        if (pt[0]>=0) & (pt[0]<=nx) & (pt[1]>=0) & (pt[1]<=ny):
            polygon_c.append(pt)
    ''' 

    img = Image.new('L', (width, height), 0)
    ImageDraw.Draw(img).polygon(polygon, outline=1, fill=1)
    mask = np.array(img).T
    idx=np.where(mask != 0)
    plotmask[idx] = 2
    return plotmask,polygon

########################################
def poly_area2D(poly):
    total = 0.0
    N = len(poly)
    for i in range(N):
        v1 = poly[i]
        v2 = poly[(i+1) % N]
        total += v1[0]*v2[1] - v1[1]*v2[0]
    return abs(total/2)


########################################
def func_star_getValij(args):
    return getValij(*args)
   

########################################
def getValij(ii, jj, gridpt_path_coord, full_tile_path_coord, tile_pixel_map, reso_geo,tile_nx,tile_ny,tile_data):
    
    valij            = 0 
    valij_tile_cover = 0 

    if (full_tile_path_coord.contains_path(gridpt_path_coord)) | (full_tile_path_coord.intersects_path(gridpt_path_coord)): 
        
        gridpt_extent = gridpt_path_coord.get_extents()
        idx = np.where( (tile_pixel_map[:,:,0] > gridpt_extent.extents[0]-reso_geo) & (tile_pixel_map[:,:,0] < gridpt_extent.extents[2]+reso_geo) &\
                        (tile_pixel_map[:,:,1] > gridpt_extent.extents[1]-reso_geo) & (tile_pixel_map[:,:,1] < gridpt_extent.extents[3]+reso_geo)  ) 
        i_min = max([idx[0].min(),0])
        i_max = min([idx[0].max(),tile_nx])
        j_min = max([idx[1].min(),0])
        j_max = min([idx[1].max(),tile_ny])
        
        if i_min==i_max: 
            i_range = [i_min]    
        else:
            i_range = range(i_min,i_max)
        
        if j_min==j_max: 
            j_range = [j_min]    
        else:
            j_range = range(j_min,j_max)
        
        sub_range = np.dstack(itertools.product(i_range,j_range))[0]
        flag_ = []
        tile_path_coord = []
        for ii,jj in zip(sub_range[0],sub_range[1]):
            pts = [ tile_pixel_map[ii,  jj  ], \
                    tile_pixel_map[ii+1,jj  ], \
                    tile_pixel_map[ii+1,jj+1], \
                    tile_pixel_map[ii  ,jj+1], \
                  ]
            tile_path_coord.append(create_linear_path(pts))

            flag_.append( ( (gridpt_path_coord.contains_path(tile_path_coord[-1])) | (gridpt_path_coord.intersects_path(tile_path_coord[-1])) ) )

        idx = np.where(flag_)

        for iii in idx[0]:

            ii, jj =  zip(sub_range[0],sub_range[1])[iii]

            tile_path = tile_path_coord[iii]
            tile_polygon = Polygon(tile_path.to_polygons()[0])
            
            grid_polygon = Polygon(gridpt_path_coord.to_polygons()[0])

            tile_area_in_grid = grid_polygon.intersection(tile_polygon).area
            grid_area         = grid_polygon.area

            valij            += tile_area_in_grid/grid_area * tile_data[ii,jj]
            valij_tile_cover += tile_area_in_grid/grid_area
                    
    return valij, valij_tile_cover


########################################
def georef(tile_data, tile_pixel_map, full_tile_path_coord, maps_fire, radiance_bckgrd,flag_parallel=False):
   
    reso_geo = np.sqrt(maps_fire.grid_e[1,1]-maps_fire.grid_e[0,0]) * (maps_fire.grid_n[1,1]-maps_fire.grid_n[0,0])
    geo_data = np.zeros_like(maps_fire.grid_e ) + radiance_bckgrd
    geo_data_tile_cover = np.zeros_like(maps_fire.grid_e )
    idx_grid = np.where(geo_data>=0)
    
    tile_nx = tile_data.shape[0]-1
    tile_ny = tile_data.shape[1]-1
            
    args_here = []
    for i_loop, (ii,jj) in enumerate(zip(idx_grid[0],idx_grid[1])):
        
        if (ii == geo_data.shape[0]-1) | (jj == geo_data.shape[1]-1) : 
            continue
        pts = [ [ maps_fire.grid_e[ii,  jj  ],maps_fire.grid_n[ii,  jj  ] ], \
                [ maps_fire.grid_e[ii+1,jj  ],maps_fire.grid_n[ii+1,jj  ] ], \
                [ maps_fire.grid_e[ii+1,jj+1],maps_fire.grid_n[ii+1,jj+1] ], \
                [ maps_fire.grid_e[ii  ,jj+1],maps_fire.grid_n[ii  ,jj+1] ], \
                [ maps_fire.grid_e[ii,  jj  ],maps_fire.grid_n[ii,  jj  ] ]  \
              ]
        gridpt_path_coord = create_linear_path(pts)
        args_here.append([ii, jj, gridpt_path_coord, full_tile_path_coord, tile_pixel_map, reso_geo, tile_nx, tile_ny, tile_data])

    #loop over the grid
    if flag_parallel: 
        
        # set up a pool to run the parallel processing
        cpus = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cpus)

        # then the map method of pool actually does the parallelisation  
        results = pool.map(func_star_getValij, args_here)
        pool.close()
        pool.join()
        
        for i_loop, arg in enumerate(args_here):
            ii = arg[0]
            jj = arg[1]
            geo_data[ii,jj]            = results[i_loop][0]
            geo_data_tile_cover[ii,jj] = results[i_loop][1]    
    
    else:
        for i_loop, arg in enumerate(args_here):
            print '{:.03f}\r'.format(100.*i_loop/len(args_here)),
            sys.stdout.flush()
            geo_data[arg[0],arg[1]], geo_data_tile_cover[arg[0],arg[1]] = getValij(*arg)
            
    return geo_data


########################################
def create_linear_path(pts):
    Path = mpath.Path
    path_data = [ (Path.MOVETO,tuple( pts[0] )) ]
    for pt in pts[1:]:
        path_data.append( (Path.LINETO,tuple( pt )) )
    path_data.append( (Path.CLOSEPOLY,tuple( pts[0] )) ) 
    codes, verts = zip(*path_data)
    return mpath.Path(verts, codes)


########################################
def create_path(pts):
    Path = mpath.Path
    path_data = [ (Path.MOVETO,tuple( pts[0] )) ]
    for pt in pts[1:]:
        path_data.append( (Path.CURVE3,tuple( pt )) )
    path_data.append( (Path.CLOSEPOLY,tuple( pts[0] )) ) 
    codes, verts = zip(*path_data)
    return mpath.Path(verts, codes)


########################################
def map_tile(filename, maps_fire, pts_utm, data_bckgrd = 0., time_info=None, time_ignition=None, data_type='lwir',flag_parallel=False):

    cell_size_geo =  (maps_fire.grid_e[1,1]-maps_fire.grid_e[0,0]) * (maps_fire.grid_n[1,1]-maps_fire.grid_n[0,0])
    
    #open tif
    gdal_data = gdal.Open(filename, GA_ReadOnly)
   
    #set no data to 0
    band = gdal_data.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    tile_data = gdal_data.ReadAsArray()[::-1].T
    idx = np.where(tile_data == nodata)
    tile_data[idx]=0 
    
    #set geotransform
    geoT = gdal_data.GetGeoTransform() # Get The geotransform
    igeo=gdal.InvGeoTransform(geoT)[1] # calculate the inverse geotransform
    
    #get pixel tile location
    tile_grid_e = np.zeros_like(tile_data)
    tile_grid_n = np.zeros_like(tile_data)
    utm_1 = gdal.ApplyGeoTransform(geoT, 0                 ,                  0)
    utm_2 = gdal.ApplyGeoTransform(geoT, tile_data.shape[0]-1,                  0)
    utm_3 = gdal.ApplyGeoTransform(geoT, 0                 , tile_data.shape[1]-1)
    utm_4 = gdal.ApplyGeoTransform(geoT, tile_data.shape[0]-1, tile_data.shape[1]-1)
    utm_5 = gdal.ApplyGeoTransform(geoT, 1                 ,                  1)
    dx = (utm_5[0]-utm_1[0])
    dy = (utm_1[1]-utm_5[1])
    tile_grid_n, tile_grid_e = np.meshgrid(np.arange(utm_3[1],utm_1[1],dy),np.arange(utm_1[0],utm_2[0],dx))
    tile_plotMask,tile_polygon = out_plotMask(tile_grid_e,tile_grid_n,pts_utm)
  
    if data_type == 'lwir':
        #get time
        idx_time = np.where(time_info.Name == os.path.basename(filename).split('.')[0])
        time_beg = pandas.to_datetime(time_info.TimeBegin)[idx_time[0][0]]
        time_end = pandas.to_datetime(time_info.TimeEnd)[idx_time[0][0]]
        time_beg = datetime.datetime(time_beg.year,time_beg.month,time_beg.day,time_beg.hour,time_beg.minute,time_beg.second,time_beg.microsecond)
        time_end = datetime.datetime(time_end.year,time_end.month,time_end.day,time_end.hour,time_end.minute,time_end.second,time_end.microsecond)
        half_seconds = (time_end - time_beg).total_seconds() / 2
        tile_time = (time_beg + datetime.timedelta(seconds=half_seconds) - time_ignition).total_seconds()
  

    #crop tile
    crop_i_l = max([np.where(tile_plotMask==2)[0].min()-80,0])
    crop_i_r = min([np.where(tile_plotMask==2)[0].max()+80,tile_data.shape[0]-1])
    crop_j_l = max([np.where(tile_plotMask==2)[1].min()-80,0])
    crop_j_r = min([np.where(tile_plotMask==2)[1].max()+80,tile_data.shape[1]-1])
    tile_grid_e = tile_grid_e[crop_i_l:crop_i_r,crop_j_l:crop_j_r]
    tile_grid_n = tile_grid_n[crop_i_l:crop_i_r,crop_j_l:crop_j_r]
    tile_data = tile_data[crop_i_l:crop_i_r,crop_j_l:crop_j_r]
    tile_plotMask = tile_plotMask[crop_i_l:crop_i_r,crop_j_l:crop_j_r]    
 
    '''
    #resize tile data if too big  Something else should be done, may be splitting the input array
    if max(tile_data.shape) > 2000:
        if 2 not in  factor.get_factor(tile_data.shape[0]):
            tile_grid_e   = tile_grid_e[:-1,:]
            tile_grid_n   = tile_grid_n[:-1,:]
            tile_plotMask = tile_plotMask[:-1,:]
            tile_data     = tile_data[:-1,:]
        if 2 not in  factor.get_factor(tile_data.shape[1]):
            tile_grid_e   = tile_grid_e[:,:-1]
            tile_grid_n   = tile_grid_n[:,:-1]
            tile_plotMask = tile_plotMask[:,:-1]
            tile_data     = tile_data[:,:-1]

        tile_grid_e   = downgrade_resolution(tile_grid_e,   np.array(tile_data.shape)/2, flag_interpolation='min')
        tile_grid_n   = downgrade_resolution(tile_grid_n,   np.array(tile_data.shape)/2, flag_interpolation='min')
        tile_plotMask = downgrade_resolution(tile_plotMask, np.array(tile_data.shape)/2, flag_interpolation='max')
        tile_data     = downgrade_resolution(tile_data,     np.array(tile_data.shape)/2, flag_interpolation='conservative')
    '''

    #get path of all tile pixel
    tile_pixel_map = np.zeros([tile_grid_e.shape[0]+1,tile_grid_e.shape[1]+1,2])
    tile_pixel_map[:-1,:-1,0] = tile_grid_e
    tile_pixel_map[:-1,:-1,1] = tile_grid_n
    tile_pixel_map[-1 ,:-1,0] = tile_pixel_map[-2,:-1,0] + dx
    tile_pixel_map[-1 ,:-1,1] = tile_pixel_map[-2,:-1,1] 
    tile_pixel_map[:-1,-1 ,0] = tile_pixel_map[:-1,-2,0] 
    tile_pixel_map[:-1,-1 ,1] = tile_pixel_map[:-1,-2,1] + dy
    tile_pixel_map[-1,-1 ,0] = tile_pixel_map[-2,-2 ,0] + dx
    tile_pixel_map[-1,-1 ,1] = tile_pixel_map[-2,-2 ,1] + dy

    #tile_path_coord  = []
    #idx = np.where(tile_data>=0)
    #for ii,jj in zip(idx[0],idx[1]):
    #    pts = [ tile_pixel_map[ii,  jj  ], \
    #            tile_pixel_map[ii+1,jj  ], \
    #            tile_pixel_map[ii+1,jj+1], \
    #            tile_pixel_map[ii  ,jj+1], \
    #          ]
    #    tile_path_coord.append(create_linear_path(pts))
    #tile_path_coord = np.array(tile_path_coord).reshape(tile_data.shape)
    
    #and the contour of the full tile
    pts = [tile_pixel_map[ 0, 0],tile_pixel_map[-1, 0],tile_pixel_map[-1,-1],tile_pixel_map[ 0,-1]]
    full_tile_path_coord = create_linear_path(pts)


    #grid data
    tile_geo_data  = georef(tile_data, tile_pixel_map, full_tile_path_coord, maps_fire, data_bckgrd, flag_parallel=flag_parallel)

    if data_type == 'lwir':
        #radiance_raw_plot = 
        idx = np.where((tile_plotMask==2) & (tile_data>0))
        radiance_raw_plot = tile_data[idx].sum()*dx*dy
        tile_plot_size = len(np.where(tile_plotMask==2)[0])*dy*dx *1.e-4
        #radiance_geo_plot = 
        idx = np.where((maps_fire.plotMask==2) & (tile_geo_data>0))
        radiance_geo_plot = tile_geo_data[idx].sum()*cell_size_geo
        geo_plot_size = len(np.where(maps_fire.plotMask==2)[0])*cell_size_geo*1.e-4

        return tile_geo_data, radiance_geo_plot, radiance_raw_plot, tile_time, [tile_plotMask,tile_grid_e,tile_grid_n,tile_data,tile_polygon]


    if data_type == 'lidar':
        return tile_geo_data, [tile_plotMask, tile_grid_e, tile_grid_n, tile_data, tile_polygon]


#####################################################
def get_residenceTime(data_x_in,data_y_in,T_residTime, T_arrivalTime, dx=5.):

    #smooth scatter point over 5s time interval
    #dx = 5.s # is default value
    data_x = np.arange(data_x_in.min(),data_x_in.max()+2*dx,dx)
    data_y = np.zeros(data_x.shape)
    for i, x_ in enumerate(data_x[:-1]):
        idx = np.where( (data_x_in>=x_) & (data_x_in< data_x[i+1]) )
        if len(idx[0])>0:
            data_y[i] = data_y_in[idx].max()
        else:
            data_y[i] = -999.
    idx  = np.where( data_y > 0 )
    data_x = data_x[idx]
    data_y = data_y[idx]

    #smooth scatter point: remove singular extreme value from local mean over window of 5 pt
    for i, y in enumerate(data_y):
        
        if (i >= 2) & (i< data_x.shape[0]-2): 
            idx = np.arange(i-2,i+2 +1)
        elif i < 2: 
            idx = np.arange(0,5)
        elif i >= data_x.shape[0]-2:
            idx = np.arange(-5,0)

        local_mean = data_y[idx].mean()
        local_std  = data_y[idx].std()
        #print local_mean, local_std
        if np.abs(y - local_mean) > 1.5*local_std: 
            data_x[i] = -1000
    idx  = np.where( data_x > 0 )
    data_x = data_x[idx]
    data_y = data_y[idx]
    
    idx = np.where(data_y > T_residTime) # run opt only when 5 points are above T_residTime
    
    if len(idx[0]) < 2: 
        #plt.scatter(data_x,data_y,c='k',s=80)
        #plt.show()
        #pdb.set_trace()
        return -999, 'na'
        
    #res = np.histogram(data_y,bins=np.arange(data_y.min(),data_y.max()+20,20))
    #hist = res[0]
    #hist_edges = res[1]
    #if (hist[-2] == 0) & (hist[-1] == 1): # remove the upper point
    #    idx = np.where(data_y < hist_edges[-2])
    #    data_x = data_x[idx]
    #    data_y = data_y[idx]

    #plt.scatter(data_x,data_y,c='w')
    #plt.show()
    #pdb.set_trace()
    
    #fit fct
    def lamda_fct(x,params):
        y = np.zeros(x.shape)
        for i,x_ in enumerate(x):
            a, b, c, d, f = params
            e = ambientT
            #d = 1
            if x_ < f : 
                y[i] = e
            else:
                y[i] =  e + a**b * (x_-f)**b * math.exp(-c**d * (x_-f)**d)
        return y
    
    #residual
    def residual(params, data_y, data_x):
        y_model = lamda_fct(data_x,params)
        idx = np.where(data_y > .95*data_y.max())
        if False: #(y_model.max() < max([600,data_y[idx].mean()]) )                | \
           #(np.abs(y_model[0] - data_y[0]) > 50) :
            residual = 1.e12
        else:
            #model error
            error_model = data_y-y_model
            
            '''
            #integral error
            #for the model
            dx_new = (data_x.max()-data_x.min())/500
            x_new = np.arange(data_x.min(),data_x.max()+dx_new,dx_new)
            y_model2 = lamda_fct(x_new,params)
            int_model = y_model2.sum()*dx_new

            #for the data 
            dx_data = data_x[1] - data_x[0]  # not sorted, all tc are stored sequentially
            int_data = 0
            for i_, (x,y) in enumerate(zip(data_x[:-1],data_y[:-1])):
                int_data += y * (data_x[i_+1]-x)
            
            erro_int = int_data - int_model

            index_residual = np.where(data_x> params[-1])
            '''
            #residual = np.sum(error_model[index_residual]**2) + erro_int**2
            residual = np.sum(error_model**2)

            #print '{0:6.3f} {1:6.3f} {2:6.3f} {3:6.3f} {4:6.3f} -- residual = {5:}'.\
            #format(params[0],params[1],params[2],params[3],params[4], residual)
            #format(params[0],params[1],params[2],params[3], residual,np.sum(error_model[index_residual]**2),erro_int**2)
            #sys.stdout.flush()
        
        return residual
    
    #set up param for the fit
    x_peak = data_x[data_y.argmax()]
    ranges_param = [(1.,5.), (1.,5.), (1.e-1,1.e0),(.1,2),(x_peak-50,x_peak+20)]
    ambientT = data_y.min() # T_threshold
   
    #call brute force
    res = scipy.optimize.brute( residual, ranges_param, Ns=5, args=(data_y,data_x))#, finish=None)
    
    x_plot = np.arange(data_x.min(),data_x.max(),.1)
    T_model = lamda_fct(x_plot,res)
    idx = np.where(T_model > T_residTime)
    if (len(idx[0]) == 0) | (T_model.max() < T_residTime ): 
        return -1001, 'na'
    time_flame = x_plot[idx]
    try :
        residenceTime = time_flame.max() - time_flame.min()
    except ValueError:
        pdb.set_trace()

    nbre_zeros = 0
    for i_, x_ in enumerate(x_plot[:-1]):
        if (T_model[i_] < T_residTime) & (T_model[i_+1] >= T_residTime):
            #print '--', x_
            nbre_zeros += 1
        if (T_model[i_] > T_residTime) & (T_model[i_+1] <= T_residTime):
            #print '--', x_
            nbre_zeros += 1


    if ( nbre_zeros < 2) : 
        #plt.scatter(data_x,data_y)
        #plt.plot(x_plot,lamda_fct(x_plot,res),c='r')
        #plt.axhline(y=T_residTime)
        #plt.title('{:6.3f} -- {:6.3f}'.format(residenceTime, residual(res, data_y, data_x) ))
        #print ''
        #print T_model[0], T_residTime, nbre_zeros
        #plt.show()
        
        return -1002, 'na'

        #pdb.set_trace()
    #if residenceTime > 20:
    #    plt.scatter(data_x,data_y)
    #    plt.plot(x_plot,lamda_fct(x_plot,res),c='r')
    #    plt.axhline(y=T_residTime)
    #    plt.title('{:6.3f}'.format(residenceTime))
    #    plt.show()
    #pdb.set_trace()


    return residenceTime, res


###################################################
#fit fct
def lamda_fct__(x,params,ambientT):
    y = np.zeros(x.shape)
    for i,x_ in enumerate(x):
        a, b, c, d, f = params
        e = ambientT
        #d = 1
        if x_ < f : 
            y[i] = e
        else:
            y[i] =  e + a**b * (x_-f)**b * math.exp(-c**d * (x_-f)**d)
    return y


######################################################
def func_star_residenceTime(args__):
    """Function to split input arguments,    
    """
    res = get_residenceTime(*args__) 
    return res


##################################################
def compute_residenceTime(time_saved, temp_saved, maps_fire,    \
                          T_arrivalTime,  T_residTime_max, T_residTime_min, \
                          dt_residTime):

    plotMask    = maps_fire.plotMask 
    arrivalTime = maps_fire.arrivalTime

    #compute residence time
    #######################
    residenceTime = np.zeros_like(plotMask)

    #clean time series
    idx_time = np.where(time_saved >= 0)
    x_all = time_saved[idx_time]
    idx_plot = np.where(plotMask > 0)
    ii = 0; ii_len = len(idx_plot[0])

    args_here = []
    indices_pool = []
    for i,j in zip(idx_plot[0],idx_plot[1]):
        y = temp_saved[i,j,:] # T series
        y = y[idx_time] # clean time 
        x = x_all
        
        #remove pt where we do not have temperature
        idx_data_ok = np.where(y>0)
        y = y[idx_data_ok]
        x = x[idx_data_ok]

        #if y.min() < 300:
        #    pdb.set_trace()

        #idx = np.where(y==0) # clean time before fire arrived
        #y[idx] = T_threshold

        if (len(y) < 10)  : 
            continue 
        if (y.max() < 520):
            continue

        T_residTime = min([T_residTime_max,max([.8*y.max(),T_residTime_min])])

        indices_pool.append([i,j])
        args_here.append([x,y,T_residTime,T_arrivalTime,dt_residTime])

    flag_parallel = False
    if flag_parallel: 
   
        # set up a pool to run the parallel processing
        cpus = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cpus)

        # then the map method of pool actually does the parallelisation  
        results = pool.map(func_star_residenceTime, args_here)
        pool.close()
        pool.join()
        for k, [i,j] in enumerate(indices_pool):
            residenceTime[i,j] = results[k][0]
    
    else:
        k_len = len(indices_pool)
        for k, [i,j] in enumerate(indices_pool):
           
            #if (i != 217) | (j != 260):    
            #if i<40 :
            #    continue
                
            #residenceTime[i,j], arrivalTime[i,j], res = get_residenceTime(*args_here[k])
            residenceTime[i,j], res = get_residenceTime(*args_here[k])
            print residenceTime[i,j], '... {:d}/{:d} ({:3.1f}%)\r'.format(k,k_len,100.*k/k_len),
            sys.stdout.flush()
            
            if (residenceTime[i,j] > 1000):
                x_plot = np.arange(args_here[k][0].min(),args_here[k][0].max(),.1)
                
                mpl.rcdefaults()
                mpl.rcParams['text.usetex'] = True
                mpl.rcParams['font.family'] = 'Comic Sans MS'
                mpl.rcParams['axes.linewidth'] = 1
                mpl.rcParams['axes.labelsize'] = 16.
                mpl.rcParams['legend.fontsize'] = 'small'
                mpl.rcParams['legend.fancybox'] = True
                mpl.rcParams['font.size'] = 14.
                mpl.rcParams['xtick.labelsize'] = 16.
                mpl.rcParams['ytick.labelsize'] = 16.
                mpl.rcParams['figure.subplot.left'] = .05
                mpl.rcParams['figure.subplot.right'] = .92
                mpl.rcParams['figure.subplot.top'] = .93
                mpl.rcParams['figure.subplot.bottom'] = .1
                mpl.rcParams['figure.subplot.hspace'] = 0.1
                mpl.rcParams['figure.subplot.wspace'] = 0.2


                fig = plt.figure(figsize=(14,6))
                ax = plt.subplot(121)
                im = ax.imshow(np.ma.masked_where(plotMask!=2,arrivalTime).T,origin='lower',interpolation='nearest')
                divider = make_axes_locatable(ax)
                cbaxes = divider.append_axes("bottom", size="5%", pad=0.05)
                cbar = fig.colorbar(im ,cax = cbaxes,orientation='horizontal')
                cbar.set_label(r'$T_{a}$ (s)',labelpad=10)

                ax.set_title(r'Arrival Time $T_{a}$ (s)')
                ax.scatter(i,j,c='w',s=100)
                idx = np.where(plotMask==2)
                ax.set_xlim(idx[0].min()-3,idx[0].max()+3)
                ax.set_ylim(idx[1].min()-3,idx[1].max()+3)
                ax.axes.set_aspect('equal')
                ax.set_axis_off()

                ax = plt.subplot(122)
                ax.scatter(args_here[k][0],args_here[k][1],c='k',label='$T_{pixel}$')
                ax.plot(x_plot, lamda_fct__(x_plot,res,args_here[k][1].min()),c='r',label='$T_{pixel}$ modelled')
                ax.axhline(y=args_here[k][2], c='k',linestyle='--',label=r'$T_{resi}$ threshold')
                ax.axhline(y=T_arrivalTime, c='k',linestyle=':',label=r'$T_{a}$ threshold')

                ax.set_xlabel('time (s)')
                ax.set_xlim(0,x_plot.max())
                ax.set_ylabel(r'T_{pixel} (K)')
                
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles[::-1], labels[::-1],loc="upper right", shadow=False)

                #fig.savefig('/home/paugam/Desktop/residenceTime_example_rose172.png')
                #plt.close(fig)
                #print 'stop here'
                #sys.exit()

                plt.show()
                pdb.set_trace()

    plt.imshow(np.ma.masked_where(residenceTime<=0, residenceTime).T,origin='lower',interpolation='nearest')
    plt.imshow(np.ma.masked_where(plotMask!=2     ,residenceTime).T,origin='lower',interpolation='nearest')
    plt.show()

    mask = residenceTime > 0
    residenceTime = interpolate_and_blur(residenceTime, mask, 0, maps_fire, method='nearest', kernel=2)

    plt.imshow(np.ma.masked_where(residenceTime_all_blur <=0., residenceTime_all_blur).T,origin='lower',interpolation='nearest')
    plt.colorbar()
    plt.show()

    return residenceTime

###########################################################
def interpolate_and_blur(data, mask, fill_val, diag_res_cte, method='nearest', kernel=3):
    
    out = np.zeros(data.shape) + fill_val

    #interpolated
    
    # data in plot
    idx = np.where( (data >= 0) & (diag_res_cte.plotMask == 2) )
    coord_pts_in = np.vstack((diag_res_cte.grid_e[idx], diag_res_cte.grid_n[idx])).T
    data_in   = data[idx]

    #outside plot
    idx =  np.where(diag_res_cte.plotMask != 2)
    coord_pts_out = np.vstack((diag_res_cte.grid_e[idx], diag_res_cte.grid_n[idx])).T    
    data_out = np.zeros(coord_pts_out.shape[0]) + fill_val

    #concatenate array
    coord_pts_conc = np.concatenate((coord_pts_in, coord_pts_out), axis=0)
    data_conc      = np.concatenate((data_in,data_out))
        
    data_all = interpolate.griddata(coord_pts_conc , data_conc, (diag_res_cte.grid_e, diag_res_cte.grid_n), \
                                             fill_value=fill_val, method=method)
    
    #blur with a kernel of 3 pixel on the plot only
    idx = np.where(diag_res_cte.plotMask ==2)
    i_s = idx[0].min(); i_e = idx[0].max()+1
    j_s = idx[1].min(); j_e = idx[1].max()+1
    data_plot = np.array(data_all[i_s:i_e,j_s:j_e])
    data_plot_out = np.copy(data_plot)
    
    data_plot_out = ndimage.gaussian_filter(data_plot, sigma=(kernel, kernel), order=0)
    out[i_s:i_e,j_s:j_e] = data_plot_out

    idx = np.where(diag_res_cte.plotMask !=2)
    out[idx] = fill_val
    
    return out


#########################################################
def plot_frp(filename,tile_time,tile_geo_frp,maps_fire,maps_fire_extent,frp_min,frp_max,i_file,tile_time_arr):

    #save image
    mpl.rcdefaults()
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'Comic Sans MS'
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['axes.labelsize'] = 16.
    mpl.rcParams['legend.fontsize'] = 'small'
    mpl.rcParams['legend.fancybox'] = True
    mpl.rcParams['font.size'] = 14.
    mpl.rcParams['xtick.labelsize'] = 16.
    mpl.rcParams['ytick.labelsize'] = 16.
    mpl.rcParams['figure.subplot.left'] = .02
    mpl.rcParams['figure.subplot.right'] = .98
    mpl.rcParams['figure.subplot.top'] = .96
    mpl.rcParams['figure.subplot.bottom'] = .03
    mpl.rcParams['figure.subplot.hspace'] = 0.1
    mpl.rcParams['figure.subplot.wspace'] = 0.2
    fig = plt.figure()
    ax = plt.subplot(111)
    #lwir_min = 1.
    #lwir_max = 700.
    im = ax.imshow(np.ma.masked_where( (tile_geo_frp<=0) | (maps_fire.plotMask!=2) ,tile_geo_frp).T,\
                   origin='lower',interpolation='nearest',                                            \
                   extent=maps_fire_extent,norm=mpl.colors.LogNorm(vmin=frp_min,vmax=frp_max)                 )

    im_bckgrd = ax.imshow(np.ma.masked_where( (tile_geo_frp>0) | (maps_fire.plotMask!=2) ,tile_geo_frp).T,\
                   origin='lower',interpolation='nearest',                                            \
                   extent=maps_fire_extent,cmap=cm.Greys_r)
    
    cbaxes = fig.add_axes([0.2, 0.12, 0.6, 0.03])
    cbar = fig.colorbar(im,orientation='horizontal',cax=cbaxes,ticks=np.linspace(frp_min,frp_max,6),format='$%.0f$')
    cbar.set_label('lwir ($W/sr/m^2$)',labelpad=10)
 
    if i_file == 0: 
        fig.text(.1,.95,'t={:.2f}s'.format(tile_time))
    else:
        fig.text(.1,.95,'t={:.2f}s; dt ={:.2f}s'.format(tile_time,tile_time-tile_time_arr[-2]))

    ax.set_axis_off()

    fig.savefig(out_dir_png+os.path.basename(filename).split('.')[0]+'.png')
    plt.clf()
    plt.close(fig)


#########################################
if __name__ == '__main__':
#########################################

    reload(fS2D)
    # register all of the drivers
    gdal.AllRegister()


    data_path = '/mnt/data/RxCadre/'
    #data_path = '/media/paugam/FIREDATA/RxCadre/'
    plot_name = '703C'
    gps_igintion_name = '703C_Research'
    good_rec = [1]
    contour_file_plot = data_path + 'gps_plot/AllBlocks/AllBlocks'
    contour_file_igni = data_path + 'gps_ignition/ignition_lines/'+ gps_igintion_name
    contour_file_patch = data_path + 'gps_plot/AllBlocks/patch_'+plot_name
    b_dickinson = 5.216
    M_dickinson = 1.374
    domain_size = 4500. # meter
    resolution = 2.8     # meter
    file_time_info = '703c_allimages_w_time.csv'#'703CBeginandEndTimes.xlsx'
    
    ########################
    flag_parallel    = True
    #
    flag_create_maps = True
    flag_lidar       = False  
    # 
    flag_load_tile   = False # load and georef
    flag_restart     = True
    ########################

    
    excitance_threshold = .350 # kW/m2
    time_limit = 2724.0   #=1.e12 no time limit
    dtm_file = 'dtm_eglin_703c.img'
    csm_file = 'csm_eglin_703c.img'
    chm_file = 'chm_eglin_703c.img'

    dir_tif   = data_path + 'RDS-2016-0007/2011_703C/'
    dir_lidar = data_path + 'LIDAR/703C/'

    out_dir = data_path + 'Postproc/' map_tile+ plot_name + '/'
    out_dir_png = out_dir+'png/'
    out_dir_npy = out_dir+'npy/'
    ensure_dir(out_dir_png)
    ensure_dir(out_dir_npy)

    sigma_SB = 5.67e-8 # W/m2/k4


    if flag_create_maps:
    #------------------
        #load contour from shapefile in pts_utm
        ctr = shapefile.Reader(contour_file_plot)
        fields_name = np.array([ctr.fields[i][0] for i in range(len(ctr.fields))])
        idx_name = np.where(fields_name == 'Block_Name')[0] -1 
        records = ctr.shapeRecords() #will store the geometry separately
        for i, record in enumerate(records):
            pts        = record.shape.points #will show you the points of the polygon
            attributes = record.record       #will show you the attributes
            if attributes[idx_name[0]] != plot_name:
                continue
            pts_utm_plot = []
            for ii, pt in enumerate(pts):
                pts_utm_plot.append(pt)
            break


        #define grid
        x,y=zip(*pts_utm_plot)
        center=(max(x)+min(x))/2., (max(y)+min(y))/2.
        resolution, grid_e, grid_n = fS2D.defineGrid(center[0],center[1],box_size=domain_size,res = resolution)


        #define maps array
        maps_fire  = np.zeros(grid_e.shape, dtype=np.dtype([('grid_e',float),('grid_n',float),      \
                                                            ('plotMask',float), ('ignition',float),\
                                                            ('fuel_load',float),('dtm',float),('csm',float),('chm',float)]))
        maps_fire = maps_fire.view(np.recarray)
        maps_fire.grid_e = grid_e
        maps_fire.grid_n = grid_n

        #plotmask
        maps_fire.plotMask[:,:], polygon = out_plotMask(grid_e,grid_n,pts_utm_plot)

        
        #plot ignition
        #load contour from shapefile in pts_utm
        ctr = shapefile.Reader(contour_file_igni)
        fields_name = np.array([ctr.fields[i][0] for i in range(len(ctr.fields))])
        idx_alt = np.where(fields_name == 'ALTITUDE')[0] -1 
        idx_time = np.where(fields_name == 'TIME')[0] -1 
        records = ctr.shapeRecords() #will store the geometry separately
        pts_utm_igni  = []
        pts_alt  = []
        pts_time = []
        pts_reccord = []
        for i, record in enumerate(records):
            if i in good_rec:
                pts        = record.shape.points #will show you the points of the polygon
                attributes = record.record       #will show you the attributes
                for ii, pt in enumerate(pts):
                    pts_utm_igni.append(pt)
                    pts_alt.append(attributes[idx_alt[0]])
                    pts_time.append(attributes[idx_time[0]])
                    pts_reccord.append(i)
        
        #save ignition line info
        ignition_line = create_linear_path(pts_utm_igni)
        ignition_info = {'path': ignition_line, 'pts': pts_utm_igni, 'altitude':pts_alt, 'time':pts_time }
        pickle.dump( ignition_info, open( out_dir+"ignition_info.p", "wb" ) )
       
        #grid the ignition line
        img = Image.new('L', (maps_fire.grid_e.shape), 0)
        pt_sw = [maps_fire.grid_e[0,0]+resolution/2,maps_fire.grid_n[0,0]+resolution/2]
        polygon =[ tuple( np.array(np.round((np.array(pt)-np.array(pt_sw))/resolution,0),dtype=np.int))  \
                 for pt in pts_utm_igni ]
        ImageDraw.Draw(img).polygon(polygon, outline=1, fill=0)
        maps_fire.ignition[:,:] = np.array(img,dtype=float).T


        #load dtm, csm, and chm
        if flag_lidar:
            print 'interpolate lidar data'
            maps_fire.dtm[:,:], tmp = map_tile(dir_lidar+dtm_file, maps_fire, pts_utm_plot, data_bckgrd = 0., data_type='lidar', flag_parallel=flag_parallel)
            maps_fire.csm[:,:], tmp = map_tile(dir_lidar+csm_file, maps_fire, pts_utm_plot, data_bckgrd = 0., data_type='lidar', flag_parallel=flag_parallel)
            maps_fire.chm[:,:], tmp = map_tile(dir_lidar+chm_file, maps_fire, pts_utm_plot, data_bckgrd = 0., data_type='lidar', flag_parallel=flag_parallel)
      
       

        #save maps
        np.save(out_dir_npy+'maps_fire_high_res',maps_fire)
 
        sys.exit()
    else:
        maps_fire = np.load(out_dir_npy+'maps_fire_high_res.npy')
        resolution = maps_fire.grid_e[1,1]- maps_fire.grid_e[0,0]

    maps_fire_extent=(maps_fire.grid_e.min(),maps_fire.grid_e.max()+resolution,              \
                       maps_fire.grid_n.min(),maps_fire.grid_n.max()+resolution)  
    print 'plot size (ha) =', len(np.where(maps_fire.plotMask==2)[0]) * resolution**2 * 1.e-4
    
    #ax = plt.subplot(111)
    #im1 = ax.imshow( maps_fire.dtm.T,origin='lower',interpolation='nearest',extent=maps_fire_extent)
    #plt.colorbar()
    #plt.show() 
    #sys.exit()


    #load time info
    #------------------
    if '13' in pandas.__version__:
        time_info = pandas.read_excel(dir_tif + file_time_info,'Export_Output')
    else:
        time_info = pandas.read_excel(dir_tif + file_time_info)
    
    time_ignition = pandas.to_datetime(time_info.TimeBegin)[0]
    time_ignition = datetime.datetime(time_ignition.year,time_ignition.month,time_ignition.day,\
                                      time_ignition.hour,time_ignition.minute,time_ignition.second,time_ignition.microsecond)
    print 'ignition time = ', time_ignition
    print ''  


    #loop over tile to create npy and png
    #------------------
    if flag_load_tile : 
        for i_file, filename in enumerate(sorted(glob.glob(dir_tif+'*.tif'))):
           
            if (flag_restart) & (os.path.isfile(out_dir_npy+os.path.basename(filename).split('.')[0]+'.npy')): 
                print os.path.basename(filename).split('.')[0], 'already done' 
                continue

            tile_geo_frp, radiance_geo_plot, radiance_raw_plot, tile_time, tile_map = map_tile(filename,maps_fire,         \
                                                                                               time_info=time_info,        \
                                                                                               time_ignition=time_ignition,\
                                                                                               flag_parallel=flag_parallel )
            print os.path.basename(filename).split('.')[0], tile_time, 
            print radiance_geo_plot/radiance_raw_plot
           
            tile_plotMask = tile_map[0]
            tile_grid_e   = tile_map[1]
            tile_grid_n   = tile_map[2]
            tile_lwir     = tile_map[3]
            tile_polygon  = tile_map[4]

            #save npy format
            np.save(out_dir_npy+os.path.basename(filename).split('.')[0],{'time':tile_time,'tile':tile_geo_frp})


