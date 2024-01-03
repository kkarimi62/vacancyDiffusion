import configparser
def makeOAR( EXEC_DIR, node, core, tpartitionime, PYFIL, argv, argvv):
    #--- parse conf. file
    confParser = configparser.ConfigParser()
    confParser.read('configuration.ini')
    #--- set parameters
    confParser.set('input files','lib_path',os.getcwd()+'/../../HeaDef/postprocess')
    confParser.set('input files','input_path',argv)
    confParser.set('Vacancy Dynamics','input_path',argvv)
    #--- write
    confParser.write(open('configuration.ini','w'))	
    #--- set environment variables

    someFile = open( 'oarScript.sh', 'w' )
    print('#!/bin/bash\n',file=someFile)
    print('EXEC_DIR=%s\n source /mnt/opt/spack-0.17/share/spack/setup-env.sh\n\nspack load python@3.8.12%%gcc@8.3.0\n\n'%( EXEC_DIR ),file=someFile)
    if convert_to_py:
        print('time ipython3 py_script.py\n',file=someFile)

    else:
        print('jupyter nbconvert --execute $EXEC_DIR/%s --to html --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;ls output.html'%(PYFIL), file=someFile)
    someFile.close() 
#
if __name__ == '__main__':
    import os

    runs     = range(8)
    nNode    = 1
    nThreads = 1
    jobname  = {
                '4':'flickers/cantorNatom1K/multipleTemp/temp0',
                '5':'energyTimeseries/nicocr/kmc/NiCoCrNatom1KTemp1000K',
                '6':'sro/cantor/kmc/cantorNatom1KTemp1000K',
                '3':'msd/cantor/kmc/cantorNatom1KTemp1000K',
                '7':'msdPerType/nicocr/kmc/NiCoCrNatom1KTemp1000K',
                '8':'energyBarrierPerType/nicocr/kmc/NiCoCrNatom1KTemp1000K',
                '9':'nicocr/md/nicocrNatom1K/md/temp0',
                '11':'ni/mlmc/latest_void5th',
                '12':'msd_definition/ni/temp0_6th',
                '13':'ni/koreanPotential/size0',
                '14':'ni/shengPotential/temp0_vac',
                '15':'vacancy/shengPotential/temp0_vac',
                }['14']
    DeleteExistingFolder = True
    readPath = os.getcwd() + {
                                '3':'/../simulations/NiCoCrNatom1KTemp1000K',
                                '4':'/../simulations/nicocrTemp1000K/n0',
                                '6':'/../simulations/cantorNatom1K/multipleTemp/temp0', 
                                '5':'/../simulations/nicocr/kmc/NiCoCrNatom1KTemp1000K',
                                '9':'/../simulations/nicocr/md/nicocrNatom1K/md/temp0',
                                '11':'/../../crystalDefect/test/mlmc/ni/latest_void5th', 
                                '12':'/../simulations/ni/koreanPotential/NiNatom1KTemp1000K', 
                                '13':'/../simulations/ni/koreanPotential/size0',
                                '14':'/../simulations/ni/shengPotential/temp0',
                                '15':'/ni/shengPotential/temp0_vac',
                            }['14'] #--- source
    EXEC_DIR = '.'     #--- path for executable file
    durtn = '23:59:59'
    mem = '32gb'
    partition = ['INTEL_PHI','INTEL_CASCADE','INTEL_SKYLAKE','INTEL_IVY','INTEL_HASWELL'][-1]
    argv = "%s"%(readPath) #--- don't change! 
    PYFILdic = { 
        0:'postproc.ipynb',
        1:'vacancyDynamics.ipynb',
        2:'test.ipynb',
        }
    keyno = 0
    convert_to_py = True
#---
#---
    PYFIL = PYFILdic[ keyno ]
    if convert_to_py:
        os.system('jupyter nbconvert --to script %s --output py_script\n'%PYFIL)
        PYFIL = 'py_script.py'
    #--- update argV
    #---
    if DeleteExistingFolder:
        print('rm %s'%jobname)
        os.system( 'rm -rf %s' % jobname ) # --- rm existing
    # --- loop for submitting multiple jobs
    for counter in runs:
        print(' i = %s' % counter)
        writPath = os.getcwd() + '/%s/Run%s' % ( jobname, counter ) # --- curr. dir
        os.system( 'mkdir -p %s' % ( writPath ) ) # --- create folder
#		os.system( 'cp utility.py LammpsPostProcess2nd.py OvitosCna.py %s' % ( writPath ) ) #--- cp python module
        makeOAR( writPath, 1, 1, durtn, PYFIL, argv+"/Run%s"%counter, argv) # --- make oar script
        os.system( 'chmod +x oarScript.sh; mv oarScript.sh %s; cp configuration.ini %s;cp %s/%s %s' % ( writPath, writPath, EXEC_DIR, PYFIL, writPath ) ) # --- create folder & mv oar scrip & cp executable
        jobname0 = jobname.replace('/','_')
        os.system( 'sbatch --partition=%s --mem=%s --time=%s --job-name %s.%s --output %s.%s.out --error %s.%s.err \
                                 --chdir %s --ntasks-per-node=%s --nodes=%s %s/oarScript.sh >> jobID.txt'\
                            % ( partition, mem, durtn, jobname0, counter, jobname0, counter, jobname0, counter \
                                , writPath, nThreads, nNode, writPath ) ) # --- runs oarScript.sh! 

