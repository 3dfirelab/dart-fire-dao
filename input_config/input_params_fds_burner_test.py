params_run ={# flag that control the driver  
            'flag_process_vis_image'  : False,
            'flag_cornerFire_tracker':  False,  # not needed here
            'flag_2DfireScene':         True,
            'flag_AtmScene':            True,
            'flag_3DfireScene':         True,
            'flag_RTE_model':           'DART', # = DART | SRTE39
            'flag_run_RTE_model':       True,
            'flag_process_DART_output': False,
            'flag_plot_DART_output':    False,
            #
            'flag_only_Georef':         False,
            }



params_cte = {'frp_to_HRR_ratio': 0.1,
              'chf_to_HRR_ratio': 0.8,
              'FI_correction_factor' : 3,
              'heatOfCombustion': 20e3, # kJ/kg
              }

params_model = {'Atm_model_name'         : 'FDS', # = MesoNH| FDS
                'onlyAtm'                : False,
                'input_dir'              : '/data/paugam/FDS/Burner_test/'         ,
                'output_dir'             : '/data/paugam/FDS/Burner_test/Postproc/',
                'input_fds_config_file'  : 'input_fixed_burner.fds' ,
                'flag_out_3DFS_from_FDS' : True,  # if False 2D info are extracted from the fds scene
                'max_bckgrd_H'           : 2.     , # max background sensible heat kW/m2
                'flag_plot_2dfs_diag'    : True,
                'onlyGas'  : False,  #There are 15289 plots with gas or soot.
                'onlyGrd'  : False,
                'onlyVeg'  : False,
                'onlySoot' : False,      
                #'curtainLoc': [[0., 60., 5.], [120., 60., 5.], [60., 0., 5.] ,[60., 120., 5.] ],

                } 

params_2DFS = {#misc
              'dataInputType': 'model',
              'fire_name'      : 'Burnert_test',
              'fire_date'      : 'na',
              #dir
              'root'      : 'na', # if na will use params_model['input_dir']
              'root_data_in'   : '',
              'dir_postproc'   : '',
              'dir_MIR'        : 'FromModels',
              'dir_gps'        : 'na',
              'dir_vis'        : 'na',
              #location file and grid info
              'flag_cf_format' : 'na',   # or textFile_latlong
              'cf_file'        : 'na', # corner plot here, missing corner fire.
              'contour_file'   : 'na', # use Point in cfFile
              'grid_resolution': -999,  # if na estimated from the distance of the camera to the plot
              'grid_size'      : -999.,  # m -- need to remove the plotMask in dir_postproc to apply change
              #visible image
              'vis_image_videoName'  : 'na',
              'flag_extract_image_from_video': False,
              'ask_for_pixel_Plotcorner': 'na',
              'flag_geo_vis_restart': False, #if True do not remove previous file
              'vis_image_ref'        : 'na', #skukuza6 'GOPR2875.JPG'
              'vis_mir_timeShift'    :  'na', # in second >0 if the nadir camera in advance
              #camera info
              'cameraLocation_file': 'na',
              'lens'               : -999, 
              'cameraLocation'     : 'na',
              'flag_use_old_list_frame_ok' : False,
              'filter_on'          : False, 
              'filter_first_last_frame'   : ['na','na'], #if all ['all','na'], otherwise [first frame, last frame ] number 
              #diag parameters
              'T_arrivalTime' : 600,    
              'T_residenseTime_min_max': [650,700], #[700,750],
              'dt_residenseTime_computation': 1. , #default is 5s,

              'flame_depth_method': 'na', #Visible or MIRcam
              'T_flameDepth': 'na' ,
              'roi_window'    : -999,    # 20
              #diag product characteristic
              'scaling_ratio_res': -999,  #scaling ratio resolution (integer, dx/f): multiple of 16, 8, 4, or 2
              'time_resolution'  : -999,
              #for plotting:
              'levels_Temp' : [600, 700, 800, 900],
              'ros_legend'  : 2.0, 
              'ros_color'  : 'w', 
              #flag
              'flag_plot_2dfs_diag'            : False,
              'flag_do_georef'                 : False,
              'flag_create_mir_animation'      : False,
              'flag_compute_resid_Arri_Time'   : False,
              'flag_extract_diag'              : False,
              'flag_compute_flame_depth_FI_vis': False,
              'flag_timeSeries'                : False,
              'flag_overlay_mir_vis'           : False
              }


params_3DFS = {#output
               'flag_netcdf'    : True,    
               # misc
               #'time_analysis': [30,40,50,60,70,80,90], # ['all'],#['all'], #[46060,46200,46340,46580],
               'time_analysis': [0,2,4], # ['all'],#['all'], #[46060,46200,46340,46580],
               #'time_analysis': [50], # ['all'],#['all'], #[46060,46200,46340,46580],
               'scaling_ratio_3Dfs'          : 1,   #2,     
               'dz'                          : -99, #.25,   
               #supplementary dir
               'dir_output'      : '3DFire/',
               'dir_wind_measure': None,
               #wind data
               'wind_files' : None,
               'wind_var'   : None,
               'cte_wind'   : [ 'na', 'na'],
               #test flag in dao
               'GrdAmbientT': False,  
               'removeFlame': False,  
               'removeVeg': False, 
               'removeAtm': False, 
               }

params_veg =  {#vegetation
               'nbre_species':1,                  # only works for one species here, to improve later
               'sv'         :[9.e3], # m-1
               'bulkD'      :[12.89], 
               'D'          :[512.], # kg/m3
               }

params_DART = {#dir
              #'root_dart_simulation': '/home/paugam/DART_user_data/simulations/',
              #'root_dart_tools':      '/home/paugam/DART585_1225/tools/linux/',
              #dart config
              'useDAO': True,                
              'useLux': True,                
              'dao_tempThresh':270,
              'lux_max_time': 200,
              'lux_rayDensity': 10000,
              'lux_radiativeBudget_meshgrid': 5.,
              'lux_vegModel': 'turbid', # 'turbid' or 'fluid'
              'nproc': 96,                
              'fv_threshold': 1.e-1 , # low value here because of the 1.19 kg/m3 at z=0 for the pool fire set up#ratio of the max fv to clip. keep plot with fv > fv_threshold * fv.max()
              'flag_update_DART_maket': True,                
              'flag_run_radiometerFlux'  : False,
              'flag_compute_FRP': True, #required pinhole image
              'radiometer_plot_SpectralqrRange'  : [0,1.0],
              #           
              'OPdataBase': 'fluid_Gas_4D_small_05cm_L001cm.db',
              #'OPdataBase': 'fluid_Gas_4D_Erez_L010cm.db',
              #
              'flag_run_sensitivity'  : False,
              'run_sensitivity_nbre_view'       : 40, 
              'run_sensitivity_camType'            : 'pinhole', # 'pinhole' or 'InsideSensor'
              'run_sensitivity_pinholeDistToCenter': 200., 
              'run_sensitivity_extraZenithAngle'   : 20., # in degree (+ val are substracted to zenith angle)
              'flag_model_image': True,
              #
              'spectralbandConfigName': 'Agema550MIR', #'largeIR', 'Agema550MIR', 'dualIR'
                                                   # if flag_run_sensitivity, then hard force to Agema550MIR
              #'image_info_type'          : ['pinhole', 'insideScenePinHole', ],  # 0,0 is the SW corner
              #'image_info_loc_x'         : [60, 80.0, ],  # 0,0 is the SW corner
              #'image_info_loc_y'         : [60, 60.0, ],
              #'image_info_loc_z'         : [200, 0.751,  ],
              #'image_info_angle_phi'     : [0  , 180, ],     # azimuth is pointing south and clockwise
              #'image_info_angle_theta'   : [0  , 90, ],
              #'image_info_angle_rot_intr': [0,  0, ],
              #'image_info_conf_file'     : [['AGEMA_va_40'], ['AGEMA_va_40'] ], #radiometer
              'image_info_type'          : ['pinhole', 'insideScenePinHole', 'insideScenePinHole'],  # 0,0 is the SW corner
              'image_info_loc_x'         : [ 3.01,   0.01,  3.01, ],  # 0,0 is the SW corner
              'image_info_loc_y'         : [ 3.01,   3.01,  0.01, ],
              'image_info_loc_z'         : [ 6.00,   1.01,  1.02, ],
              'image_info_angle_phi'     : [  0,       90,   180,   ],     # azimuth=0 is pointing south and anti-clockwise
              'image_info_angle_theta'   : [  0,       90,    90,    ],
              'image_info_angle_rot_intr': [  0,        0,     0,    ],
              'image_info_conf_file'     : [['AGEMA_va_40'], ['AGEMA_va_40'], ['AGEMA_va_40'] ], #radiometer
              'radiometer'               : {'nbrPix': 1001, 'zenithMin':0, 'zenithMax':90, 'radius':0.01, 'theta':0 },
}

params_sensAna = {
                 'run_backgrd_simulation' :True, # only needed for sens analysis
                 'flag_test_isotropie_on_homogene_plot': False,
                 'PSF_sigma': .98,  #.86,#(TET) # pixels
                 'limitNbre_shift': 10, 
                 #plotting possible image obs
                 'selection_time_for_possible_image': [50], # need to be a in params_3DFS['time_analysis']
                 #for the sens, select reso
                 'selection_reso_2_run': [0.5, 2],  # in meter. we ll get the closest one from ratio_reso_sens
                 'selection_reso_2_plot_in_angular_dist': [1,2], # subsample of above array starting a 1
                 }
