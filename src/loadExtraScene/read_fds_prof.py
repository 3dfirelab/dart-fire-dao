import numpy as np 
import matplotlib.pyplot as plt
import glob 
import pandas 
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pdb 
from skimage import measure 
import os 

#####################################################
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d) 

#############################################
def loadProfile(dirdata, input_time, zZero=0.04 ):
    configfile = glob.glob(dirdata+'*.fds')[0]
    with open(configfile,'r') as f: 
        lines = f.readlines()

    #load config
    prof_info = []
    for line in lines:
        if line[:5] == '&PROF':
            profid  = line.split("ID='")[1].split("'")[0]
            profloc = [float(xx) for xx in line.split("XYZ=")[1].split(",")[:3]]
            profior = int(line.split("IOR=")[1].split(r"/")[0].strip())
            prof_info.append([profid,profloc,profior])
        if line[:5] == '&HEAD':
            namesimu = line.split("CHID='")[1].split("'")[0]
    

    nprof = len(prof_info)
    profid_arr = np.array([xx[0] for xx in prof_info])

    #load prof
    profFiles = sorted(glob.glob('{:s}{:s}_prof*.csv'.format(dirdata,namesimu)))
    profTimes  = []
    times     = []
    for iprof, profFile in enumerate(profFiles):
        
        csv = pandas.read_csv(profFile,sep=',',skiprows=[0,1], header=None)
        with open(profFile,'r') as f: 
            lines = f.readlines()
        profid_ = lines[0].rstrip()

        iiprof =np.where(profid_arr == profid_)[0][0]
        
        for irow in range(len(csv)): 
            if iprof == 0: 
                profTimes.append([])
                [profTimes[irow].append([]) for xx in range(nprof)]
                times.append(csv.loc[irow][0])
            numptprof = int(csv.loc[irow][1])
            profTimes[irow][iiprof].append([np.array(csv.loc[irow][2          :2+numptprof  ]), 
                                           np.array(csv.loc[irow][2+numptprof:2+2*numptprof])])

    times = np.array(times)
    it = np.abs(times-input_time).argmin()
     
    profTime = profTimes[it]
    x = []
    y = []
    z = []
    T = []
    for iprof, profData in enumerate(profTime):
        for zz,tt in zip(*profData[0]):
            if zZero-zz < 0: continue
            x.append(prof_info[iprof][1][0] )
            y.append(prof_info[iprof][1][1] )
            z.append(zZero-zz)
            T.append(tt)

    return x, y, z, T


################################
def interpolation2Domain(x, y, z, T, gridx, gridy, gridz, flag_interp='griddata'):
    
    (x0,Lx,nx) = gridx
    (y0,Ly,ny) = gridy
    (z0,Lz,nz) = gridz

    dxy = (Lx-x0)/nx
    dz  = Lz/nz
    nzZoomFactor = 100
    dz2 = Lz/(nzZoomFactor*nz)

    xi = np.linspace(x0,Lx,nx+1)[:-1] + dxy/2 
    yi = np.linspace(y0,Ly,ny+1)[:-1] + dxy/2
    zi = np.linspace(z0,Lz,nzZoomFactor*nz+1)[:-1] + dz2/2
    
    if flag_interp == 'rbf':
        print('rbf')
        interp = interpolate.Rbf(x, y, z, T, function="thin_plate")
        tempAcetone = np.zeros([xi.shape[0],yi.shape[0],zi.shape[0]])
        for k in range(zi.shape[0]):
            for j in range(yi.shape[0]):
                for i in range(xi.shape[0]):
                    tempAcetone[i,j,k] = interp(xi[i],yi[j],zi[k])
    
    elif flag_interp == 'griddata':
        print('griddata')
        xv,yv,zv = np.meshgrid(xi, yi, zi,  sparse=False, indexing='ij')
        X = xv.flatten(); Y = yv.flatten(); Z = zv.flatten() 
        
        tempAcetone_nearest = interpolate.griddata((x,y,z), T, (X,Y,Z), method='nearest').reshape(xv.shape)
        tempAcetone = interpolate.griddata((x,y,z), T, (X,Y,Z), method='linear').reshape(xv.shape)

        tempAcetone = np.where(np.isnan(tempAcetone), tempAcetone_nearest, tempAcetone)

    return measure.block_reduce(tempAcetone, block_size=(1,1,nzZoomFactor), func=np.average)


################################
if __name__ == '__main__':
################################

    dirdata = '/mnt/data/FDS/PoolFireAcetone_v12/'
    ensure_dir(dirdata + 'Postproc/ProfilesPlot/')
    input_times = [20,40,60,80]
    
    #out grid
    gridx,gridy,gridz=(-.5,0.5,50),(-0.5,0.5,50),(0,0.06,3) 

    for input_time in input_times:
        x, y, z, T = loadProfile(dirdata, input_time, zZero=0.05) 
        tempAcetone = interpolation2Domain(x, y, z, T, gridx,gridy,gridz)

        print(tempAcetone.shape)
        
        fig = plt.figure()
        ax = plt.subplot(111, projection='3d')
        ax.view_init(6, -79)
        im = ax.scatter(x,y,z,c=T,vmin=20,vmax=100)
        plt.colorbar(im)#, cax=cax, orientation='horizontal');
        
        ax.set_title('t={:.2f}s'.format(input_time))
        #ax = plt.subplot(222)
        #ax.imshow(tempAcetone[:,:,0].T, origin='lower')
        
        #ax = plt.subplot(224)
        #ax.imshow(tempAcetone[:,:,1].T, origin='lower')
        

        fig.savefig(dirdata + 'Postproc/ProfilesPlot/{:03d}.png'.format(input_time))
        plt.close(fig)        
