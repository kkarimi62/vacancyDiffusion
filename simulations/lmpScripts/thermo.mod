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
variable varEzz	 equal	v_varTime*${GammaDot}		
fix extra all print 100 "${varStep} ${varTime} ${varEzz} ${varTemp} ${varPe} ${varPxx} ${varPyy} ${varSzz} ${varVol}" screen no title "step time ezz temp pe pxx pyy szz vol" file thermo.txt

