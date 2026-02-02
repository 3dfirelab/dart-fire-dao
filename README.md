# Inputs:
- DART simulation in dirDART=`/home/paugam/DART_user_data/simulations/fds_burner_test_004_00_s`. coeffdiff and phase do not need to be run before.
- This script run then the DAO using the fire scene set in dir3dFS=`../data_test/Postproc/3DFire/`.
- The command lines is the followin:
```
python dart_tools_dao.py -name 'fds_burner_test'                                                  \
                      -dxy 0.1 -dz 0.033 -t 4                                                     \
                      -dir3dFS '../data_test/Postproc/3DFire/'                  \
                      -dirDART '/home/paugam/DART_user_data/simulations/fds_burner_test_004_00_s' \
                      -tempA 293.15
```
the input apart from the directroy already mentioned above are:
- `dxy` the horizontal resolution of the dao scene. keep the same as the fire scene.
- '`dz` same overt the vertical. 
- `name` is the generic name of the fire simulation
- `t` is the time of the scene that is passed to DART.
- `tempA` is the ambient temperature. 

Several parameters could be read from the fire scene. 
This would be considered in the future.

## To setup the env:
- In `data_static`, the template of the lux simulation should be updated to the `DART` version. In the current version it is build with `DART 1449`.
- `data_test/fds_burner_test_004_00_s.tar.gz` need to be untar in your `DART` simulation directory.
- a working python env is provided in `pyhtonEnv-dart.yml`.
```
mamba env create -f pyhtonEnv-dart.yml
```
- `copy Lambertian_vegetation.db` in the `DART` system database.

# Outputs: 
it includes the dao scene in the `DART` simulation configuration.

