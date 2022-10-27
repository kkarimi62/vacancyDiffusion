#!/bin/bash
#--- set natom and box size (output of KartInput.py)
set natom=`cat .natom.txt`
set lx=`cat .lx.txt`
set temp=`cat .temp.txt`

#--- set directories for lmp scripts
export INCLUDE=${INC}

export PathEAM=${PathEam}

export DATAFILE=${DataFile}


###################################### Main Input/Output files ########################################################

export INI_FILE_NAME='init_xyz.conf'   # The file name containing the intial configuration 

 
###################################### Simulation Details ##################################################

export NBRE_KMC_STEPS=100       # The max number of KMC steps to be executed

export TEMPERATURE=${temp} #500.0    # The simulated temperature in kelvin


export NUMBER_ATOMS=${natom} #864     # The total number of atoms 

export SIMULATION_BOX=${lx} #21.47   # The size of the simulation box (x, y and z)

export NSPECIES=5     # The number of different atom types (default: 2)

export ATOMIC_SYMBOLS='Ni Co Cr Fe Mn' #"Ni Co Cr"

#setenv NTRAVAILLEUR                     3     # The number of cores associated with forces calculations per ARTnouveau search (default:1)
###################################### Restart options #####################################################

export RESTART_KMC=False
export RESTART_FILE="this_conf" # The file name used to continue a simulation from where it was last stopped

export RESTART_IMPORT=False
export NEW_CATALOGUE=False


#################################### Basin parameters #################################
export OSCILL_TREAT=BMRM       # choose between BMRM, TABU or NON

export BASIN_LOCAL=False
export MIN_SIG_BARRIER=0.1    # Max height of barrier and inv. barrier for an event to be considered inside a basin


#################################### Topology Params ##################################

export TOPO_RADIUS=4.0     # radius for topology cluster > CRYST_TOPO_RADIUS 

export MAX_TOPO_CUTOFF=3.0     # length-cutoff used by default to link two atoms

export MIN_TOPO_CUTOFF=2.2     # minimal length cutoff used when looking at secondary topologies 

export CRYST_TOPOID=1079097  # topo id of the crystalline-like topologies

export CRYST_TOPO_RADIUS=3.0     # radius for crystal-like topologies (default: 4.0 A) 

export TOPOLOGY_FILE='Topologies' # Store info about topologies

export TOPO_STAT_FILE='topos.list' # Store statistics about topologies

#setenv MAX_NODES_GRAPH				  22 #kam
export UNIQUE_CRYST_STRUCT=True

#################################### Force calculations ###############################

export ENERGY_CALC=LAM      # choose between EDP or SWP or SWA or LAM (Lammps)

export FORCE_CALC=TTL      # choose TTL for total force calculation and PAR for partial force


export INPUT_LAMMPS_FILE='in_kart.lammps' # LAMMPS input file when using ENERGY_TYPE = LAM to calculate the forces

  
export UNITS_CONVERSION='metal'      # Converts the energy and distance units provided by the force code

                                          # into a desired value for ART - affects only the parameters in ART and kART
                                          # units available are real , electron and si
                                          # be very careful when choosing other units, you must change parameters of energy 
                                          # and distance (if it's not in Angstrom) in input files for LAMMPS 
 
export UPDATE_VER_NEIB=TTL      # choose TTL for total force calculation and PAR for partial force

export NEIB_CALC=ALL      # choose ALL or VER

export UPDATE_TOPO=TTL       # choose TTL or PAR

export PAR_DISP_THRESH2=0.00001     # max displacement squared which triggers an update of the neighbor 

                                           # list when using VER lists (default: 0.00001)                                                              
                                            
################ FIRE MINIMIZATION ###############
# setenv DTMAX_FIRE           0.15        # default value
export DTMAX_FIRE=0.25        # default value

export DTMAX_FIRE_MIN=0.05        # default value=DTMAX_FIRE

# setenv MAX_ITER_FIRE        1500        # default value
# setenv NORM_CRITERIUM       0.0005      # default value
# setenv FMAX_CRITERIUM       0.0005      # default value

################ FIRE PERP. HYPERPLANE MINIMIZATION ############
# setenv MAX_ITER_FIRE_PERP    15          # default value
# setenv FOR_LEAVING_BASIN_USE_FIRE .true. # When relaxing perp for leaving harmonic basin if .true. choose FIRE algorithm else SD

################# ART PARAMETERS ######################################################
export SADDLE_PUSH_PARAM=0.1     # The fraction of the initial-saddle distance used to push saddle config. away from initial minimum (default: 0.1)

export TYPE_OF_EVENTS=local   # Initial move for events - global or local

export RADIUS_INITIAL_DEFORMATION=3.0     # Cutoff for local-move (in angstroems)

export EIGENVALUE_THRESHOLD=-1.0     # Eigenvalue threshold for leaving basin


export EXIT_FORCE_THRESHOLD=0.1    # Threshold for convergence at saddle point

export FORCE_THRESHOLD_PERP_REL=0.05    # Threshold for perpendicular relaxation


export FINE_EXIT_FORCE_THRESHOLD=0.05    # finner Threshold for convergence at saddle point 

export FINE_FORCE_THRESHOLD_PERP_REL=0.01    # finner Threshold for perpendicular relaxation


#kam setenv MIN_NUMBER_KSTEPS          2       # Min. number of ksteps before calling lanczos
export INCREMENT_SIZE=0.1     # Overall scale for the increment moves in activation


export INITIAL_STEP_SIZE=1.00    # Size of initial displacement, in A

export BASIN_FACTOR=3.00

export MIN_NUMBER_KSTEPS=3       # Min. number of ksteps before calling lanczos

export MAX_PERP_MOVES_BASIN=6       # Maximum number of perpendicular steps leaving basin

export MAX_PERP_MOVES_ACTIV=20       # Maximum number of perpendicular steps during activation

export MAX_ITER_BASIN=40      # Maximum number of iteraction for leaving the basin (kter)




export MAX_ITER_ACTIVATION=30      # Maximum number of iteraction during activation (iter)


export NUMBER_LANCZOS_VECTORS=15      # Number of vectors included in lanczos procedure

export LANCZOS_STEP=1e-3    #1e-2    # Size of the step for the numerical derivative (def: 0.001)





# CHECK_LANCZOS_STAB overwrite LANCZOS_STEP and loop for 1e-1 to 1e-7 to check stability and then stop the code. 
# setenv CHECK_LANCZOS_STAB  .true.         # Check lanczos stability over 200 steps, each iteration uses previous lanczos vector
export NBRE_POINTS_LANCZOS=2       # number of extra points for numerical derivative in lanczos default 1 supported 1 or 2 for now.



#################################### GENERIC events parameters ########################
export SEARCH_FREQUENCY=10           # Minimum number of attempts to find a GENERIC event per new topology encountered

export THRES_INCREASE_FREQ=25           # Number of failed attempts encountered because increasing the EIGEN_THRESH

export TYPE_EVENT_UPDATE=SPEC         # choose between SPEC or GENE 

export USE_LOG_SEARCH=True

export CHECK_INI_SAD_CONNECTIVITY=True
                                          #  and minimizes.
                                          # If minimized config. not the same as the initial one, the saddle is rejected.

############### Printing details ######################################################################

export ALLCONF_WITH_SADDLE=True
export PRINT_DETAILS=True
export MINSAD_DETAILS=False
export USE_TXT_EVENTFILE=True
export STATISTICS=True
export OUTPUT_CONFIG_EVENTS=True
export OUTPUT_SPECIFIC=True
#setenv OUTPUT_NEB_GEN_EVENT        .true.    # Can be useful

 

############### Run the simulation ######################################################################
unlimit stacksize
ln -s ../../lmpScripts/${INPUT_LAMMPS_FILE} . #--- lmp script 
