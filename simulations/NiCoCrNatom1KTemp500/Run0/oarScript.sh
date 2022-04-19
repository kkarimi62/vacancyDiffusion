#!/bin/bash

EXEC_DIR=/home/kamran.karimi1/Project/git/vacancyDiffusion/simulations/NiCoCrNatom1KTemp500

MEAM_library_DIR=/home/kamran.karimi1/Project/git/lammps2nd/lammps/potentials

module load openmpi/4.0.2-gnu730
module load lib/openblas/0.3.13-gnu

#mpirun -np 1 $EXEC_DIR/lmp_mpi < in.minimization -echo screen -var OUT_PATH . -var PathEam /home/kamran.karimi1/Project/git/lammps2nd/lammps/potentials -var INC /home/kamran.karimi1/Project/git/vacancyDiffusion/simulations/lmpScripts  -var buff 0.0 -var nevery 1000 -var ParseData 0 -var natoms 1000 -var cutoff 3.52  -var DumpFile dumpMin.xyz -var WriteData data_minimized.txt
#mpirun -np 1 $EXEC_DIR/lmp_mpi < in.Thermalization -echo screen -var OUT_PATH . -var PathEam /home/kamran.karimi1/Project/git/lammps2nd/lammps/potentials -var INC /home/kamran.karimi1/Project/git/vacancyDiffusion/simulations/lmpScripts  -var buff 0.0 -var T 500.0 -var P 0.0 -var nevery 1000 -var ParseData 1 -var DataFile data_minimized.txt -var DumpFile dumpThermalized.xyz -var WriteData Equilibrated_500.dat
python3 KartInput.py  Equilibrated_500.dat init_xyz.conf /home/kamran.karimi1/Project/git/vacancyDiffusion/simulations/lmpScripts


mpirun -x PathEam=/home/kamran.karimi1/Project/git/lammps2nd/lammps/potentials -x INC=/home/kamran.karimi1/Project/git/vacancyDiffusion/simulations/lmpScripts  -x DataFile=Equilibrated_500.dat -np 4 KMC.sh
