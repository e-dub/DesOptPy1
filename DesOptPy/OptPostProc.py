"""
Title:    OptPostProc.py
Author:   E. J. Wehrle
Date:     April 12, 2019
-------------------------------------------------------------------------------
Description:
Postprocessing of optimization problem via optimality check and shadow prices
-------------------------------------------------------------------------------
"""
import numpy as np
from numpy.linalg import pinv, norm


def CalcLagrangeMult(fNabla, gNabla):
    return(-pinv(gNabla)@fNabla)


def CheckKKT(lam, fNabla, gNabla, g, kkteps=1e-6):
    if len(lam)==1:
        OptResidual = fNabla-lam*gNabla
    else:
        OptResidual = fNabla-lam@gNabla
    Opt1Order = norm(OptResidual)
    kktMax = np.max(np.abs(OptResidual))
    PrimalFeas = (np.max(g) < kkteps)
    DualFeas = (np.min(lam) > -kkteps)
    ComplSlack = (np.abs(g@lam) < kkteps)
    kktOpt = bool(PrimalFeas*DualFeas*ComplSlack)
    return(kktOpt, Opt1Order, OptResidual, kktMax)


def CalcShadowPrice(lam, gc, gcType, DesVarNorm):
    SP = np.zeros(len(lam))
    for ii in range(len(lam)):
        if gc == 0.0 or (gcType[ii]=="Bound" and DesVarNorm in [None, "None", False]):
            SP[ii] = lam[ii]
        else:
            SP[ii] = lam[ii]/gc[ii]
    return SP
