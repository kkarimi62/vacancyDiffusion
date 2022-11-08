if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
	lnums = [ 27, 32 ]
	string=open('postproc.py').readlines() #--- python script
	#---
#	PHI  = dict(zip(range(6),np.linspace(1000,2000,6)))
	PHI  = dict(zip(range(11),np.arange(1000,1440,40)))
#		{ 
#            '0':'FeNi',
#            '1':'CoNiFe',
#           '2':'CoNiCrFe',
#           '3' :'CoCrFeMn',
#            '4':'CoNiCrFeMn',
#            '5':'Co5Cr2Fe40Mn27Ni26'
#			'6':'cuzr',
#		}

	nphi = len(PHI)
	#---
	count = 0
	for key in PHI:
			#---	
				inums = lnums[ 0 ] - 1
				string[ inums ] = "\t\'4\':\'NiNatom16KTemp%sK\',\n" % (int(PHI[key])) #--- change job name
		#---	densities
				inums = lnums[ 1 ] - 1
				string[ inums ] = "\t\'4\':\'/../simulations/NiNatom16KTemp%sK\',\n"%(int(PHI[key]))
		#
				sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
				os.system( 'python3 junk%s.py'%count )
				os.system( 'rm junk%s.py'%count )
				count += 1
