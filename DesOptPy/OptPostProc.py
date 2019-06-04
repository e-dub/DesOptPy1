"""
Title:    OptPostProc.py
Author:   E. J. Wehrle
Date:     June 4, 2019
-------------------------------------------------------------------------------
Description:
Postprocessing of optimization problem via optimality check and shadow prices
-------------------------------------------------------------------------------
"""
import numpy as np
from numpy.linalg import pinv, norm


def CalcLagrangeMult(fNabla, gNabla):
    lam = -(fNabla@pinv(gNabla)).T
    return(lam)


def CheckKKT(lam, fNabla, gNabla, g, kkteps=1e-3):
    if np.size(g) == 0:
        OptResidual = fNabla
        kktOpt = (np.abs(norm(fNabla)) < kkteps)
    else:
        if np.size(lam)==1:
            OptResidual = fNabla+float(lam)*gNabla
        else:
            OptResidual = fNabla+lam.T@gNabla.T
        PrimalFeas = (np.max(g) < kkteps)
        ComplSlack = (np.abs(g@lam) < kkteps)
        DualFeas = (np.min(lam) > -kkteps)
        kktOpt = bool(PrimalFeas*DualFeas*ComplSlack)
    Opt1Order = norm(OptResidual)
    kktMax = np.max(np.abs(OptResidual))
    return(kktOpt, Opt1Order, OptResidual, kktMax)


def CalcShadowPrice(lam, gc, gcType, DesVarNorm):
    SP = np.zeros(len(lam))
    for ii, lamii in enumerate(lam):
        if gc[ii] == 0.0 or (gcType[ii]=="Bound" and DesVarNorm in [None, "None", False]):
            SP[ii] = lamii
        else:
            SP[ii] = lamii/gc[ii]
    return SP
