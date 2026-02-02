import numpy as np 
import f90nml 
import glob
import pdb

##########################3
def load(dirData, obstName):
    try: 
        fileConfig = glob.glob(dirData+'*.fds')[0]
    except: 
        pdb.set_trace()
    fds_nml = f90nml.read(fileConfig)

    dimObst = []
    for obst in fds_nml['OBST']:
        try:
            for surfId in obst['surf_ids']:
                if surfId == obstName:
                    dimObst.append(obst['XB'])    
        except KeyError:
            for surfId in obst['surf_id']:
                if surfId == obstName:
                    dimObst.append(obst['XB'])    

    if len(dimObst)!=1: 
        print('pb in selection of obstacle dimension')
        pdb.set_trace()


    return dimObst[0]


if __name__ == '__main__':


    dimension = load('/mnt/data/FDS/PoolFireAcetone/', 'ACETONE POOL 2')

