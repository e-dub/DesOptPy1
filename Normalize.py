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


TODOs

TODO needs x0!
-------------------------------------------------------------------------------
'''

import numpy as np


def normalize(x, x0, xL, xU, DesVarNorm):
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
        print("Error: Normalization type not found: " + DesVarNorm)
    return xNorm, xLnorm, xUnorm


def denormalize(xNorm, x0, xL, xU, DesVarNorm):
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
        print("Error: Normalization type not found: " + DesVarNorm)
    return x

def normalizeSens(drdx, x0, xL, xU, DesVarNorm):
    if drdx == []:
        return drdx
    if DesVarNorm is "xLxU" or True:
        drdxNorm = (drdx)*np.tile((xU-xL), [np.shape(drdx)[0], 1])
    elif DesVarNorm is "xLx0":
        drdxNorm = (drdx)*np.tile((xL-x0), [np.shape(drdx)[0], 1])
    elif DesVarNorm is "x0":
        drdxNorm = drdx/np.tile(x0, [np.shape(drdx)[0], 1])
    elif DesVarNorm is "xU":
        drdxNorm = drdx/np.tile(xU, [np.shape(drdx)[0], 1])
    elif DesVarNorm is "None" or None or False:
        drdxNorm = drdx
    else:
        print("Error: Normalization type not found: " + DesVarNorm)
    return drdxNorm
    #   drdxNorm = drdx * (xU - xL)
    #   dgdx = dgxdx * (np.tile((xU - xL), [len(g), 1]))
