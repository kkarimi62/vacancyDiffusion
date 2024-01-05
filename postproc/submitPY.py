if __name__ == '__main__':
    import sys
    import os
    import numpy as np
    #---
    lnums = [ 43, 58 ]
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
                string[ inums ] = "\t\'6\':\'nicocr/kmc/NiCoCrNatom1KTemp%sK/msd\',\n" % (temp) #--- change job name
        #---	densities
                inums = lnums[ 1 ] - 1
                string[ inums ] = "\t\'6\':\'/nicocr/kmc/NiCoCrNatom1KTemp%sK\',\n"%(temp)
        #
                sfile=open('junk%s.py'%count,'w');sfile.writelines(string);sfile.close()
                os.system( 'python3 junk%s.py'%count )
                os.system( 'rm junk%s.py'%count )
                count += 1
