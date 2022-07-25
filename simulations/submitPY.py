if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
	lnums = [ 30, 37, 93   ]
	string=open('simulations.py').readlines() #--- python script
	#---
	PHI ={
             '0':1,
             '1':4,
            '2':9,
            '3' :16,
             '4':25,
             '5':36,
         }
	nphi = len(PHI)
	#---
	#---
	EPS={'0':0}
	times=[0]
	#--- 
	count = 0
	keyss= list(PHI.keys())
	keyss.sort()
	for iphi in keyss:
		for epsi in EPS:
			for itime in times:
			#---	
			#---	densities
				inums = lnums[ 0 ] - 1
				string[ inums ] = "\tnThreads = %s\n"%(PHI[iphi]) #--- change job name
			#---	densities
				inums = lnums[ 1 ] - 1
				string[ inums ] = "\t3:\'NiCoCrNatom50Kannealed%s\',\n"%(count) #--- change job name
			#---
				inums = lnums[ 2 ] - 1
				string[ inums ] = "\t5:' -var buff 0.0 -var nevery 100 -var ParseData 0 -var natoms %s -var cutoff 3.54  -var DumpFile dumpMin.xyz -var WriteData data_minimized.txt',\n"%(PHI[iphi]*2000)
				#---

				sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
				os.system( 'python junk%s.py'%count )
				os.system( 'rm junk%s.py'%count )
				count += 1
