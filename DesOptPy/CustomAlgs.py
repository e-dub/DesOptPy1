# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 23:33:57 2015

@author: wehrle
"""
import numpy as np
import copy

global nEval
nEval = 0
def SystemEquations(DefOptSysEq, x, gamma):
    global nEval
    nEval += 1
    f, g, fail = DefOptSysEq(x)
    if g is not []:
        f = f - gamma*sum(1./np.array(g))
        g = []
    return(f)


def GradJacobian(DefOptSysEq, x, f, dx, gamma):
    df = np.zeros([np.size(x),])
    for ix in range(np.size(x)):
        xnew = copy.deepcopy(x)
        xnew[ix] += (dx)
        df[ix] = SystemEquations(DefOptSysEq, xnew, gamma) - f
    dfdx = df/dx
    return(dfdx)


def CalcHessian(DefOptSysEq, x, f, dx, gamma):
    d2f = np.zeros([np.size(x), np.size(x)])
    for ix in range(np.size(x)):
        for jx in range(np.size(x)):
            xnew = copy.deepcopy(x)
            xnew[ix] += (dx)
            xnew[jx] += (dx)
            d2f[ix, jx] = SystemEquations(DefOptSysEq, xnew, gamma) - f
    d2fdxdx = d2f/dx/dx
    return(d2fdxdx)


def SteepestDescentSUMT(DefOptSysEq, x0, xL, xU):
    gamma = 0.1
    dx = 1e-6
    xNext = copy.deepcopy(x0)
    x = copy.deepcopy(x0)
    alpha = 1e-3
    eps = 1e-5
    nIter = 0
    while sum(abs(x - xNext)) > eps or nIter < 20:
        x = copy.deepcopy(xNext)
        f = SystemEquations(DefOptSysEq, x, gamma)
        dfdx = GradJacobian(DefOptSysEq, x, f, dx, gamma)
        xNext = x - alpha * dfdx
        nIter += 1
    return(f, x, nIter, nEval)


def NewtonSUMT(DefOptSysEq, x0, xL, xU):
    gamma = 0.1
    dx = 1e-6
    xNext = copy.deepcopy(x0)
    x = copy.deepcopy(x0)
    alpha = 1e-3
    eps = 1e-5
    nIter = 0
    while sum(abs(x - xNext)) > eps or nIter < 20:
        x = copy.deepcopy(xNext)
        f = SystemEquations(DefOptSysEq, x, gamma)
        dfdx = GradJacobian(DefOptSysEq, x, f, dx, gamma)
        d2fdxdx = CalcHessian(DefOptSysEq, x, f, dx, gamma)
        xNext = x - np.linalg.inv(d2fdxdx) * dfdx #*d2fdxdx*dfdx
        nIter += 1
        print xNext
    return(f, x, nIter, nEval)