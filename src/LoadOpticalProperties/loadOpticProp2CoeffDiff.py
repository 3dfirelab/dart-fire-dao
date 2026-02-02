import sys, os
import numpy as np 
import argparse
import sqlite3
import pdb

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



###############################
def load_coeff_template(dir_input):
    
    #load coeff
    coeffxml_template_file = dir_input + '/coeff_diff.xml'
    f = open(coeffxml_template_file,'r')
    coeffxml_template = f.readlines()
    f.close()
    

    # extract fluid and veg
    ########################
    coeffxml_template_old = coeffxml_template
    checks = ['AirMultiFunctions', 'UnderstoryMultiFunctions']
    fluid_def = []; turbi_def = []; idx_to_keep = []
    flag_checks=[False,False]
    for i_line, line in enumerate(coeffxml_template):

        flag_to_keep = True
        
        if '</'+checks[0] in line:
            flag_checks[0]= False
        if '</'+checks[1] in line:
            flag_checks[1]= False
        
        if flag_checks[0]:  
            fluid_def.append(line)
            flag_to_keep = False
       
        if flag_checks[1]:  
            turbi_def.append(line)
            flag_to_keep = False
        
        if flag_to_keep:
            idx_to_keep.append(i_line)
        
        if '<'+checks[0] in line:
            flag_checks[0] = True
        elif '<'+checks[1] in line:
            flag_checks[1] = True

    coeffxml_template = [coeffxml_template_old[idx] for idx in idx_to_keep]
    

    return coeffxml_template, fluid_def, turbi_def


############################################
def dump_coeff_template(dir_input, coeff_templates, fluid_template, turbi_template):


    #load coeff
    coeffxml_template_file = dir_input + '/coeff_diff.xml'
    print ('   update coeff_diff.xml with fluid_Gas.db in DART_maket')
    os.rename(coeffxml_template_file, coeffxml_template_file+'.old')
    
    checks = ['AirMultiFunctions', 'UnderstoryMultiFunctions']
    line2write = []
    for line in coeff_templates:
        line2write.append(line)
        if '<'+checks[0] in line:
            for line_ in fluid_template:
                line2write.append(line_)
        if '<'+checks[1] in line:
            for line_ in turbi_template:
                line2write.append(line_)
                

    f = open(coeffxml_template_file,'w')
    for line in line2write:
        f.write(line)
    f.close()
   

    return 0


###############################
def loadOpticProp2CoeffDiff(dir_input, databaseName, onlySelectedTemp=None, flag_saving=True):
    
    #load existing coeff_diff
    coeff_templates, fluid_template, turbi_template = load_coeff_template(dir_input)
   
    flag_print = False
    if flag_print:
        print ('present fluid')
        print ('-------------')
        for xx in fluid_template:
            print (xx)

    if databaseName is not None:
        #load available fluid optical properties avaible
        connection = sqlite3.connect(databaseName)
        cursor = connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        fluidNames = [xx[0].lower() for xx in cursor.fetchall()]
        fluidNames.remove('_comments')
        fluidNames.remove('sqlite_sequence')
        fluidNames.remove('_gas_type')
        
        if flag_print:
            print ('new fluid')
            print ('-------------')
            print (sorted(fluidNames))

    else: 
        fluidNames = []

    template_fluid_entry = '            <AirFunction useMultiplicativeFactorForLUT="0" ModelName="{:s}" databaseName="{:s}" ident="{:s}"/>\n'


    #loop over exising fluid prop and remove those that are present in the database
    #to add all of them at the end
    check = 'AirFunction '
    fluid_template_new = []
    fluid_template = fluid_template + [' ']
    lut = []
    id_ = 0
    for iline, (line, lineNext) in enumerate(zip(fluid_template[:-1],fluid_template[1:])):
        #if ('<'+check +' ' in line) & (('<'+check +' ' in lineNext) | (lineNext == ' ')):
        flag_check = False
        if ('<'+check in line) & ('/>' in line) : 
            line_to_check = line
            flag_check = True
        elif ('<'+check in line) & ('/>' in lineNext):
            line_to_check = line.strip()+' '+lineNext.strip()+'\n'
            flag_check = True

        if flag_check:
            fluidName = line_to_check.split('ModelName')[1].split('"')[1].lower()
            indentName = line_to_check.split('ident')[1].split('"')[1]
            if fluidName in fluidNames:
                continue
            else:
                lut.append(['fluid',id_,indentName,fluidName])
                id_+=1
        fluid_template_new.append(line_to_check)
  
    #add database
    for fluidName in sorted(fluidNames):
        indentName = fluidName.upper()
        temp = float(fluidName.split('_')[0][2:-1])
        if onlySelectedTemp is not None:
            if abs((temp-onlySelectedTemp).min()) > 0.1: continue
        fluid_template_new.append(template_fluid_entry.format(fluidName.lower(), os.path.basename(databaseName), indentName ))
        lut.append(['fluid',id_,indentName,fluidName.lower()])
        id_+=1
   

    #add Turbid to lut
    check = 'UnderstoryMulti'
    turbi_template_new = []
    turbi_template = turbi_template + [' ']
    oneTurbi = []
    id_ = 0
    flag_log = False
    flag_logged = False
    for iline, line  in enumerate(turbi_template):
        if ('<'+check+' ' in line): 
            flag_log = True
        elif ('</'+check+'>' in line) :
            oneTurbi.append(line)
            flag_log = False
            flag_logged = True 
       
        if flag_log :  
            oneTurbi.append(line)

        if flag_logged:
            
            for line_ in oneTurbi: 
                if 'ModelName' in line_: turbiName  = line_.split('ModelName')[1].split('"')[1].lower()
                if 'ident'     in line_: indentName = line_.split('ident')[1].split('"')[1]
                turbi_template_new.append(line_)
            
            lut.append(['turbid',id_,indentName,turbiName])

            id_+=1
            flag_logged = False
            oneTurbi = []
   

    lut = np.array(lut)


    #write final coeff_diff.xml
    if flag_saving:
        dump_coeff_template(dir_input, coeff_templates, fluid_template_new, turbi_template_new)

    lut_array = np.array( [('mm',0,'','')]*len(lut), dtype=np.dtype([('type', '<U100'),('id', int), ('ident','<U100'), ('name', '<U100') ]) )
    lut_array = lut_array.view(np.recarray)
    lut_array.type = lut[:,0]
    lut_array.id = lut[:,1]
    lut_array.ident = lut[:,2]
    lut_array.name = lut[:,3]
 

    return lut_array

###############################
if __name__ == '__main__':
###############################

    parser = argparse.ArgumentParser(description='import full database in coeff.xml and create a LookUpTable')
    parser.add_argument('-s','--simulation', help='simulation name',required=True)
    parser.add_argument('-d','--database', help='database name',required=True)

    args = parser.parse_args()

    simuName = args.simulation
    if os.path.isfile(DART_LOCAL + '/database/' + args.database):
        databaseName = DART_LOCAL + '/database/' + args.database

    elif os.path.isfile(DART_HOME + '/database/' + args.database):
        databaseName = DART_HOME + '/database/' + args.database

    else: 
        print ('missing database: ', args.database)
        print ('stop')
        sys.exit()
    
    dir_input = DART_LOCAL+'/simulations/'+simuName+'/input/'

    lut = loadOpticProp2CoeffDiff(dir_input, databaseName)
