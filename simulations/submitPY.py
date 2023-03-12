if __name__ == '__main__':
	import sys
	import os
	import numpy as np
	#---
#	lnums = [ 33, 96   ]
	lnums = [ 36, 96   ]
#	lnums = [ 37, 103   ]
	string=open('simulations-ncbj.py').readlines() #--- python script
#	string=open('simulations.py').readlines() #--- python script
	#---
#	PHI = dict(zip(range(6),np.linspace(1000,2000,6)))
	PHI = dict(zip(range(16),np.linspace(1000,2000,16)))
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
				temp = int(PHI[iphi])
			#---	
			#---	densities
				inums = lnums[ 0 ] - 1
#				string[ inums ] = "\t1:\'NiNatom16KTemp%sK\',\n"%(int(PHI[iphi])) #--- change job name
#				string[ inums ] = "\t3:\'CantorNatom16KTemp%sKEnsemble8\',\n"%(int(PHI[iphi])) #--- change job name
#				string[ inums ] = "\t0:\'NiMultTemp/Temp%sK\',\n"%(int(PHI[iphi])) #--- change job name
				string[ inums ] = "\t3:\'NiCoCrMultTemp/Temp%sK\',\n"%(int(PHI[iphi])) #--- change job name
			#---
				inums = lnums[ 1 ] - 1
#				string[ inums ] = "\t\'p3\':\' data_minimized.txt init_xyz.conf"+" %"+"s"+" %s\'%%(os.getcwd()+\'/lmpScripts\'),\n"%(PHI[iphi])
				string[ inums ] = "\t7:\' -var buff 0.0 -var T %s -var P 0.0 -var nevery 100 -var ParseData 1 -var DataFile data_minimized.txt -var DumpFile dumpThermalized.xyz -var WriteData Equilibrated_%s.dat\',\n"%(temp,temp)
				#---

				sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
				os.system( 'python junk%s.py'%count )
				os.system( 'rm junk%s.py'%count )
				count += 1
