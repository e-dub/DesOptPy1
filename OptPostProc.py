# -*- coding: utf-8 -*-
'''
Title:    OptPostProc.py
Units:    -
Author:   E. J. Wehrle
Date:     November 22, 2014
---------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------
Description:

Postprocessing of optimization problem via optimality check and shadow prices

---------------------------------------------------------------------------------------------------
'''

import numpy as np

def OptPostProc(fGradOpt, gc, gOptActiveIndex, g_xLU_GradOptActive, c_xLU_OptActive, c_xLU_ActiveType, DesVarNorm):
#--------------------------------------------------------------------------------------------------
#   § Calculate Lagrangian multipliers
#--------------------------------------------------------------------------------------------------
    if np.size( g_xLU_GradOptActive)>0:
        lambda_c,blah0,blah1,blah2=np.linalg.lstsq(g_xLU_GradOptActive,-fGradOpt)  #lambda_c=np.linalg.solve(cGradOptActive,fGradOpt)

#--------------------------------------------------------------------------------------------------
#   § Calculate shadow prices - Denormalization of Lagrangian multipliers
#--------------------------------------------------------------------------------------------------
        SPg = np.zeros(np.shape(lambda_c))
        for ii in range(np.size(lambda_c)):
            if c_xLU_OptActive[ii] == 0.0:  # Assume 0 values not normalized
                SPg[ii] = lambda_c[ii]
            else:
                if c_xLU_ActiveType[ii] == "Bound" and DesVarNorm in [None, "None", False]:
                    SPg[ii]=lambda_c[ii]
                else:
                    SPg[ii] = lambda_c[ii]/c_xLU_OptActive[ii]
# here further thoughts are needed to normalized bounds for the shadow prices, this is not yet correct!


#        if np.size(gc)==1 and gOptActiveIndex.all==True: #correct what about xL and xU in gOptActiveIndex?
#            SPg=-lambda_c/c_xLU_OptActive
#        else:
#            try: SPg=-lambda_c/c_xLU_OptActive
#            except: SPg=[]

#--------------------------------------------------------------------------------------------------
#   §     Caclulate optimality after Karush, Kuhn and Tucker, as well as first-order optimality
#--------------------------------------------------------------------------------------------------
        OptRes=fGradOpt-np.dot(lambda_c,g_xLU_GradOptActive.T)
        Opt1Order=np.linalg.norm(OptRes)
        KKTmax=max(abs(OptRes))
    else:
        lambda_c    =[]
        lambda_g    =[]
        SPg         =[]
        OptRes      =[]
        Opt1Order   =[]
        KKTmax      =[]
    return lambda_c, SPg, OptRes, Opt1Order, KKTmax
