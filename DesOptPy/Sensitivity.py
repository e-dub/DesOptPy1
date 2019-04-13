#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from DesOptPy.Normalize import(normalize, denormalize, normalizeSens,
                               denormalizeSens)
import numpy as np


def FiniteDiff(SysEq, x0, xU=[], xL=[], gc=[], hc=[], SensEq=[],
               DesVarNorm=True, deltax=1e-3):

    def OptSysEq(x):
        f, g = SysEq(x, gc)
        return f, g

    def OptSysEqNorm(xNorm):
        x = denormalize(xNorm, x0, xL, xU, DesVarNorm)
        f, g = OptSysEq(x)
        return f, g

    dx = deltax
    df = np.zeros((len(x0), 1))
    dg = np.zeros((len(x0), len(gc)))
    for ii in range(len(x0)+1):
        if ii == 0:
#            if DesVarNorm in ["None", None, False]:
#                f_x0, g_x0 = OptSysEq(x0)
#            else:
#                f_x0, g_x0 = OptSysEqNorm(xNorm)
            fx0, gx0 = OptSysEq(x0)
            gx0 = np.array(gx0).reshape(len(gc))
        else:
            x = x0.copy()
            x[ii-1] += dx
            fx, gx = OptSysEq(x)
            df[ii-1, 0] = fx-fx0
            dg[ii-1, :] = np.array(gx).reshape(len(gc))-gx0
    dfdx = df/dx
    dgdx = dg/dx
    return(dfdx, dgdx)

#add plots, from Veit
def PlotSens3D(dfdx, dgdx, labels=[]):
    return()


def PlotSens2D(dfdx, dgdx, labels=[]):
    return()
