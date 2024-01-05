if __name__ == '__main__':
    import sys
    import os
    import numpy as np
    #---
    lnums = [ 44, 59 ]
    script = 'postproc.py postproc_ncbj_slurm.py '.split()[1]
    PHI  = dict(zip(range(6),np.linspace(1000,2000,6,dtype=int)))

    string=open(script).readlines() #--- python script
    #---
    nphi = len(PHI)
    #---
    count = 0
    for key in PHI:
                temp = PHI[key]
            #---	
                inums = lnums[ 0 ] - 1
                string[ inums ] = "\t\'14\':\'vacancy/FoilesPotential/temp%s_total\',\n" % (key) #--- change job name
        #---	densities
                inums = lnums[ 1 ] - 1
                string[ inums ] = "\t\'14\':\'/ni/FoilesPotential/temp%s\',\n"%(key)
        #
                sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
                os.system( 'python3 junk%s.py'%count )
                os.system( 'rm junk%s.py'%count )
                count += 1
