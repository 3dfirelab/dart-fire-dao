import numpy as np 
#from fortio import FortranFile
from scipy.io import FortranFile
import pdb
import matplotlib.pyplot as plt

########################
def load_bndf(dirdata,filebf):
    time = []
    bndf = []
    with FortranFile(dirdata+filebf) as f:
        quantity   = ''.join([xx.decode() for xx in f.read_record('c')])
        #print(quantity)
        short_name = ''.join([xx.decode() for xx in f.read_record('c')])
        units      = ''.join([xx.decode() for xx in f.read_record('c')])
        npatch     = f.read_ints()[0]
        bndf = [[] for x in range(npatch)]
        I1,I2,J1,J2,K1,K2,IOR,NB,NM = [],[],[],[],[],[],[],[],[]
        for ii in range(npatch):
            I1_,I2_,J1_,J2_,K1_,K2_,IOR_,NB_,NM_ = f.read_ints()
            I1.append(I1_);I2.append(I2_);J1.append(J1_);J2.append(J2_)
            K1.append(K1_);K2.append(K2_);IOR.append(IOR_);NB.append(NB_);NM.append(NM_)
       
        while True:
            try:
                time.append( f.read_reals(dtype='f4')[0] )
                for ii in range(npatch):
                    bndf[ii].append(f.read_reals(dtype='f4'))
            
            except: 
                break
    

    #load grid
    meshfile = '_'.join((dirdata+filebf).split('_')[:-1])+'.xyz'
    with FortranFile(meshfile) as f:
        nx,ny,nz= f.read_ints()
        mm = f.read_reals(dtype='f4')
        dd = nx*ny*nz
        x = mm[:dd].reshape([nx,ny,nz],order='F')
        y = mm[dd:2*dd].reshape([nx,ny,nz],order='F')
        z= mm[2*dd:3*dd].reshape([nx,ny,nz],order='F')


    bndf_arr = [[] for x in range(npatch)]
    bndf_xyz = [[] for x in range(npatch)]
    for ii in range(npatch):
        sizei = I2[ii]-I1[ii]+1
        sizej = J2[ii]-J1[ii]+1
        sizek = K2[ii]-K1[ii]+1
        sizet = len(time)
        arr_ = np.zeros([sizei,sizej,sizek,sizet])
        for it, bndf_ in enumerate(bndf[ii]):
            arr_[:,:,:,it] = np.array(bndf_).reshape([sizei,sizej,sizek],order='F')
        bndf_arr[ii] = arr_
        bndf_xyz[ii] = [ x[I1[ii]:I2[ii]+1,J1[ii]:J2[ii]+1,K1[ii]:K2[ii]+1], 
                         y[I1[ii]:I2[ii]+1,J1[ii]:J2[ii]+1,K1[ii]:K2[ii]+1],
                         z[I1[ii]:I2[ii]+1,J1[ii]:J2[ii]+1,K1[ii]:K2[ii]+1] ]

    return bndf_arr,bndf_xyz,NB,IOR,NM, np.array(time) 

if __name__ == '__main__':

    dirdata = '/mnt/data/FDS/PoolFireAcetone/'
    filebf = 'acetone_1_m_ronan_0001_01.bf'

    bndf_arr, bndf_xyz, NB, IOR, NM, time = load_bndf(dirdata,filebf)

    plt.imshow(bndf_arr[1][:,:,0,-1].T, origin='lower')
    plt.show()
