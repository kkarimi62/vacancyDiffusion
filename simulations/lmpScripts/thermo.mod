#--- output thermodynamic variables
variable varStep equal step
variable varPress equal press
variable varTemp equal temp
variable varTime equal v_varStep*${dt}
variable varXy	 equal	xy
variable varLy	 equal	ly
variable varVol	 equal	vol
variable varPe	 equal	pe
variable varPxx	 equal	pxx
variable varPyy	 equal	pyy
variable varPzz	 equal	pzz
variable varPxy	 equal	pxy
variable varSxy	 equal	-v_varPxy*${cfac}
variable varSzz	 equal	-v_varPzz*${cfac}
variable varExy	 equal	v_varXy/v_varLy		
<<<<<<< HEAD
variable varEzz	 equal	0 #v_varTime*${GammaDot}		
variable ntherm  equal	10 #${nevery} #ceil(${Nstep}/${nthermo})
variable varn	 equal	v_ntherm
fix extra all print ${varn} "${varStep} ${varTime} ${varEzz} ${varTemp} ${varPe} ${varPxx} ${varPyy} ${varSzz} ${varVol}" screen no title "step time ezz temp pe pxx pyy szz vol" file thermo.txt
=======
variable varEzz	 equal	v_varTime*${GammaDot}		
fix extra all print 100 "${varStep} ${varTime} ${varEzz} ${varTemp} ${varPe} ${varPxx} ${varPyy} ${varSzz} ${varVol}" screen no title "step time ezz temp pe pxx pyy szz vol" file thermo.txt
>>>>>>> 4816dd8ec44d64b66ec694967107ad97753563d2

