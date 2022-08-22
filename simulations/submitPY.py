if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
	lnums = [ 37, 102   ]
	string=open('simulations.py').readlines() #--- python script
	#---
	PHI = {1: 1040.0, 2: 1080.0, 3: 1120.0, 4: 1160.0, 5: 1200.0, 6: 1240.0, 7: 1280.0, 8: 1320.0, 9: 1360.0, 10: 1400.0}
#dict(zip(range(11),np.linspace(1000,1400,11)))
#		{
#             '0':1,
#             '1':2**3,
#            '2':3**3,
#         }
	nphi = len(PHI)
	#---
	#---
	#--- 
	count = 0
	keyss= list(PHI.keys())
	keyss.sort()
	for iphi in keyss:
			#---	
			#---	densities
				inums = lnums[ 0 ] - 1
				string[ inums ] = "\t3:\'CantorNatom16KTemp%sK\',\n"%(int(PHI[iphi])) #--- change job name
			#---
				inums = lnums[ 1 ] - 1
				string[ inums ] = "\t\'p3\':\' data_minimized.txt init_xyz.conf"+" %"+"s"+" %s\'%%(os.getcwd()+\'/lmpScripts\'),\n"%(PHI[iphi])
				#---

				sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
				os.system( 'python junk%s.py'%count )
				os.system( 'rm junk%s.py'%count )
				count += 1
