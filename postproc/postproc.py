from backports import configparser
def makeOAR( EXEC_DIR, node, core, tpartitionime, PYFIL, argv):
	#--- parse conf. file
	confParser = configparser.ConfigParser()
	confParser.read('configuration.ini')
	#--- set parameters
#	confParser.set('parameters','itime0','5')
	confParser.set('input files','lib_path',os.getcwd()+'/../../HeaDef/postprocess')
	confParser.set('input files','input_path',argv)
	#--- write
	confParser.write(open('configuration.ini','w'))	
	#--- set environment variables

	someFile = open( 'oarScript.sh', 'w' )
	print('#!/bin/bash\n',file=someFile)
	print('EXEC_DIR=%s\n'%( EXEC_DIR ),file=someFile)
#	print >> someFile, 'papermill --prepare-only %s/%s ./output.ipynb %s %s'%(EXEC_DIR,PYFIL,argv,argv2nd) #--- write notebook with a list of passed params
	print('jupyter nbconvert --execute $EXEC_DIR/%s --to html --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.allow_errors=True;ls output.html'%(PYFIL), file=someFile)
	someFile.close()										  
#
if __name__ == '__main__':
	import os
#
	runs	 = range(1)#6)
	jobname  = {
				'3':'CantorNatom16KTemp1000K_ensemble', 
				'4':'CantorNatom16KTemp1400K_ensemble_min', 
				}['4']
	DeleteExistingFolder = True
	readPath = os.getcwd() + {
								'3':'/../simulations/CantorNatom16KTemp1000K_ensemble',
								'4':'/../simulations/CantorNatom16KTemp1400K_ensemble_min',
 							}['4'] #--- source
	EXEC_DIR = '.'     #--- path for executable file
	durtn = '23:59:59'
	mem = '8gb'
	partition = ['parallel','cpu2019','bigmem','single'][3] 
	argv = "%s"%(readPath) #--- don't change! 
	PYFILdic = { 
		0:'postproc.ipynb',
		}
	keyno = 0
#---
#---
	PYFIL = PYFILdic[ keyno ] 
	#--- update argV
	#---
	if DeleteExistingFolder:
		os.system( 'rm -rf %s' % jobname ) # --- rm existing
	# --- loop for submitting multiple jobs
	for counter in runs:
		print(' i = %s' % counter)
		writPath = os.getcwd() + '/%s/Run%s' % ( jobname, counter ) # --- curr. dir
		os.system( 'mkdir -p %s' % ( writPath ) ) # --- create folder
		os.system( 'cp configuration.ini %s' % ( writPath ) ) #--- cp python module
		makeOAR( writPath, 1, 1, durtn, PYFIL, argv+"/Run%s"%counter) # --- make oar script
		os.system( 'chmod +x oarScript.sh; cp oarScript.sh configuration.ini %s; cp %s/%s %s' % ( writPath, EXEC_DIR, PYFIL, writPath ) ) # --- create folder & mv oar scrip & cp executable
		os.system( 'sbatch --partition=%s --mem=%s --time=%s --job-name %s.%s --output %s.%s.out --error %s.%s.err \
						    --chdir %s -c %s -n %s %s/oarScript.sh'\
						   % ( partition, mem, durtn, jobname[:4], counter, jobname[:4], counter, jobname[:4], counter \
						       , writPath, 1, 1, writPath ) ) # --- runs oarScript.sh!
											 

