if __name__ == '__main__':
    import sys
    import os
    import numpy as np
    #---
    lnums = [ 30, 35 ]
    script = 'postproc.py postproc_ncbj_slurm.py'.split()[1]
    PHI  = dict(zip(range(6),np.linspace(1000,2000,6,dtype=int)))

    string=open(script).readlines() #--- python script
    #---
#	PHI  = dict(zip(range(11),np.arange(1000,1440,40)))
#		{ 
#             '0':'FeNi',
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
                temp = PHI[key]
            #---	
                inums = lnums[ 0 ] - 1
                string[ inums ] = "\t\'3\':\'sro/NiCoCrNatom1KTemp%sK\',\n" % (temp) #--- change job name
        #---	densities
                inums = lnums[ 1 ] - 1
                string[ inums ] = "\t\'3\':\'/../simulations/NiCoCrNatom1KTemp%sK\',\n"%(temp)
        #
                sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
                os.system( 'python3 junk%s.py'%count )
                os.system( 'rm junk%s.py'%count )
                count += 1
