from __future__ import print_function
from __future__ import division
from builtins import zip
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
import numpy as np 
from scipy import interpolate
import multiprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pdb 
from scipy.signal import savgol_filter

#warnings.filterwarnings('ignore', '*umber of iterations maxit (set t*',)

###################################
def ApplyUnivariateSpline(time_fq_in, temp_cam_fq_in, dt_average=3):

        length_m = dt_average
        '''
        if np.mod(time_fq.shape[0], length_m) != 0:
            new_l = length_m * (old_div(time_fq.shape[0],length_m) + 1)
            new_o = time_fq.shape[0]
            
            time_fq_      = np.zeros(new_l)
            temp_cam_fq_ = np.zeros(new_l)
            
            time_fq_[:new_o]      = time_fq
            temp_cam_fq_[:new_o] = temp_cam_fq
            
            time_fq_[new_o:]      = np.nan 
            temp_cam_fq_[new_o:] = np.nan 
            
            time_fq      = time_fq_
            temp_cam_fq = temp_cam_fq_
        '''

        duration = (time_fq_in.max() - time_fq_in.min())
        nn = int(duration / dt_average) + 1
        time_all = np.linspace(time_fq_in.min(), time_fq_in.max(), nn)
        
        bt = np.zeros_like(time_all[:-1])
        t = np.zeros_like(time_all[:-1])
        for i, (tb,te) in enumerate(zip(time_all[:-1], time_all[1:])):
            idx = np.where( (time_fq_in>=tb) & (time_fq_in<te) )
            if len(idx[0])>0:
                bt[i] = temp_cam_fq_in[idx].mean()
                t[i] = time_fq_in[idx].mean()
            else:
                bt[i] = np.nan
                t[i] = np.nan

        #t = np.nanmean(time_fq.reshape(-1,length_m),axis=1)
        #c = np.nanmean(temp_cam_fq.reshape(-1,length_m),axis=1)

        idx_noNan = ~np.isnan(bt)
        time_fq_     = t[idx_noNan] 
        temp_cam_fq_ = bt[idx_noNan]

        err = []
        ss  = []
        s = 1.e6
        while s > 2.e-3:
            try:
                spl_cam = interpolate.UnivariateSpline(time_fq_, temp_cam_fq_, ext=3, s=s,)
                
                idx_ = np.where( temp_cam_fq_ > 600) 
                if len(idx[0])==0:
                    idx = np.where((time_fq_<time_fq_[0]+dt_average) | (time_fq_>time_fq_[-1]-10*dt_average))
                err_ = np.sum(((temp_cam_fq_[idx_]-spl_cam(time_fq_[idx_])))**2, axis=0)
                err.append(err_)
                ss.append(s)
            except UserWarning:
                pass
            s /= 2
        
        idx_ = np.array(err).argmin()
        s = ss[idx_]
        spl_cam = interpolate.UnivariateSpline(time_fq_, temp_cam_fq_, ext=3, s=s,)

        return spl_cam, time_fq_, temp_cam_fq_


############################################
class tempts(object):
    
    def __init__(self,): 
        self.cameras   = 'optrisP400, agema550'

    def init(self,input_):
        self.mir_var    = input_[0]
        self.mir_fitmax = input_[1]
        self.mir_p80    = input_[2]
        self.spl_mir    = input_[3]
        #self.lwir_var    = input_[4]
        #self.lwir_fitmax = input_[5]
        #self.lwir_p80    = input_[6]
        #self.spl_lwir    = input_[7]
        

############################################
def start_process():
    print('Starting', multiprocessing.current_process().name)

##############################################
def star_interpolateTemperatureTimeSeries(args):
    return interpolateTemperatureTimeSeries(*args)

##############################################
def interpolateTemperatureTimeSeries(ij, time_data, lwir_temp_data, mir_temp_data, frp_proxy, flag_plot, dir_out):
   
    idx_same_lwirtime = []
    idx_sametime = []
    ij_ = [300,300] # any point would work
    diff_lwir = np.diff(lwir_temp_data[:,ij_[0],ij_[1]])
    for idx_, diff_ in enumerate(diff_lwir):
        idx_sametime.append(idx_)
        if diff_ != 0: 
            idx_same_lwirtime.append(idx_sametime)
            idx_sametime = []

    temp_lwir = lwir_temp_data[:,ij[0],ij[1]]
    temp_mir  = mir_temp_data[ :,ij[0],ij[1]]

    if temp_mir.max() < 550:
        return 1, None, None, None, None

    if (np.std(np.sort(temp_mir)[-3:]) > 100) : return 2, None, None, None, None

    '''
    temp_lwir_fq = []
    temp_mir_fq  = []
    time_fq      = []
    for idxs_ in idx_same_lwirtime:
        temp_lwir_fq.append(temp_lwir[idxs_[0]])
        temp_mir_fq.append(temp_mir[idxs_[0]])
        time_fq.append(time_data[idxs_[0]])
    ''' 
    temp_mir_fq = temp_mir
    time_fq     = time_data

    if np.sort(temp_mir_fq)[-3:].min() < 550: 
        return 3, None, None, None , None
    ''' 
    if temp_lwir_fq[0] > 350: 
        #pdb.set_trace()
        return 4, None, None 
    '''

    
    idx = np.where(np.array(temp_mir_fq)>=475)
    temp_mir_fq  = np.array(temp_mir_fq)[idx]
    #temp_lwir_fq = np.array(temp_lwir_fq)[idx]
    time_fq      = np.array(time_fq)[idx] #- time_data[0]
    
    


    #keep only points around max with dt > 5.s
    '''
    dts = np.diff(time_fq)
    #temp_mir_fq
    idx_to_keep = []
    for ii, dt in enumerate(dts): 
        if (dt>10) & (time_fq[ii]<time_fq[temp_mir_fq.argmax()]):
            idx_to_keep = []
            continue
        elif (dt>10) & (time_fq[ii]>time_fq[temp_mir_fq.argmax()]):
            break
        else:
            idx_to_keep.append(ii)
    
    temp_mir_fq  = temp_mir_fq[idx_to_keep]
    temp_lwir_fq = temp_lwir_fq[idx_to_keep]
    time_fq      = time_fq[idx_to_keep]
    '''
    
    #spl_lwir, tlwir, clwir = ApplyUnivariateSpline(time_fq, temp_lwir_fq, dt_average=3 )
    #if spl_lwir is None: return 3, None, None 
    
    try:
        spl_mir, tmir, cmir = ApplyUnivariateSpline(time_fq, temp_mir_fq, dt_average=2 )
    except: 
        spl_mir = None 

    if temp_mir_fq.shape[0] > 60:
        temp_mir_fq2  = savgol_filter(temp_mir_fq, 21, 3)
        spl_mir2 = interpolate.UnivariateSpline(time_fq, temp_mir_fq2, ext=3, )
    else:
        spl_mir2 = None

    if spl_mir is None: return 5, None, spl_mir2, time_fq, temp_mir_fq


    idx_varts_ = np.where(temp_mir_fq>650)
    if len(idx_varts_[0]) > 0: 
        idx_varts = np.where( (time_fq >= time_fq[idx_varts_].min()) & (time_fq <= time_fq[idx_varts_].max()) )
        spl_mir_max =   spl_mir(time_fq[idx_varts]).max()
        spl_mir_per =   np.percentile(temp_mir_fq[idx_varts],80)
    else: 
        idx_varts = (np.array([], dtype=np.int64),)
        spl_mir_max = -99
        spl_mir_per = -99

    result = tempts()
    result.init([  \
                    #mir
                    ((spl_mir(time_fq[idx_varts])-temp_mir_fq[idx_varts])**2).sum(),
                    spl_mir_max,  
                    spl_mir_per,
                    spl_mir,
                    # lwir
                    #((spl_lwir(time_fq[idx_varts])-temp_lwir_fq[idx_varts])**2).sum(), 
                    #spl_lwir(time_fq[idx_varts]).max(), 
                    #np.percentile(temp_lwir_fq[idx_varts],80),
                    #spl_lwir
                ])

    '''
    #var
    varts_mir  = ((spl_mir(time_fq[idx_varts])-temp_mir_fq[idx_varts])**2).sum()
    varts_lwir = ((spl_lwir(time_fq[idx_varts])-temp_lwir_fq[idx_varts])**2).sum()
    
    #max
    p80ts_mir  = np.percentile(temp_mir_fq[idx_varts],80)
    p80ts_lwir = np.percentile(temp_lwir_fq[idx_varts],80)
    
    maxfit_mir  = spl_mir(time_fq[idx_varts]).max()
    maxfit_lwir = spl_lwir(time_fq[idx_varts]).max()
    '''

    if flag_plot:
        print('*')
        fig = plt.figure()
        
        ax = plt.subplot(121)
        if len(idx_varts[0]) > 0: 
            ax.axvspan(time_fq[idx_varts].min(), time_fq[idx_varts].max(), color='0.5', alpha=0.5)
        
        #ax.scatter(time_fq, temp_lwir_fq, c='b')
        #ax.scatter(tlwir,clwir, c='k',marker='x')
        #ax.plot(time_fq, spl_lwir(time_fq), 'b-', lw=2, alpha=0.7, )
        
        ax.scatter(time_fq, temp_mir_fq, c='orange')
        ax.scatter(tmir,cmir, c='k')
        ax.plot(time_fq, spl_mir(time_fq), c='orange', lw=2, alpha=0.7,)

        
        ax = plt.subplot(122)
        ax.imshow(frp_proxy.T, origin='lower')
        ax.scatter(ij[0],ij[1],c='k')

        fig.savefig('{:s}/{:03d}x{:03d}.png'.format(dir_out,ij[0],ij[1]))
        plt.close(fig)


    #if (ij[0] == 117) & (ij[1]==265): pdb.set_trace()

    return 0, result, spl_mir2, time_fq, temp_mir_fq

