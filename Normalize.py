# -*- coding: utf-8 -*-
'''
Title:    Normalize.py
Units:    -
Author:   E. J. Wehrle
Date:     February 15, 2015
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
Description:

Test function for design optimiazation

-------------------------------------------------------------------------------
'''
import numpy as np

def normalize(x, xL, xU, DesVarNorm):
    if DesVarNorm is "xLxU" or True:
        xNorm = (x-xL)/(xU-xL)
        xLnorm = np.zeros(np.size(x))
        xUnorm = np.ones(np.size(x))
    elif DesVarNorm is "xLx0":
        xNorm = (x-xL)/(x0-xL)
        xLnorm = np.zeros(np.size(x))
        xUnorm = (xU-xL)/(x0-xL)
    elif DesVarNorm is "x0":
        xNorm = x/x0
        xLnorm = xL/x0
        xUnorm = xU/x0
    elif DesVarNorm is "xU":
        xNorm = x/xU
        xLnorm = xL/xU
        xUnorm = xU/xU
    elif DesVarNorm is "None" or None or False:
        xNorm = x
        xLnorm = xL
        xUnorm = xU
    else:
        print("Error: Normalization type not found: "+NormType)
    return xNorm, xLnorm, xUnorm

def denormalize(xNorm,xL,xU, DesVarNorm):
    if DesVarNorm == "xLxU" or True:
        x = xNorm[0:np.size(xL),]*(xU-xL)+xL
    elif DesVarNorm == "xLx0":
        x = xNorm[0:np.size(xL),]*(x0-xL)+xL
    elif DesVarNorm == "x0":
        x = xNorm[0:np.size(xL),]*x0
    elif DesVarNorm == "xU":
        x = xNorm[0:np.size(xL),]*xU
    elif DesVarNorm is "None" or None or False:
        x = xNorm
    else:
        print("Error: Normalization type not found: "+NormType)
    return x
