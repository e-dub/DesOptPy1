# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 19:05:47 2016

@author: wehrle
"""
import pyOpt
import numpy as np
from Normalize import normalize, denormalize

def OptReadHis(OptName, Alg, x0, xL, xU, DesVarNorm):
    OptHist = pyOpt.History(OptName, "r")
    fAll = OptHist.read([0, -1], ["obj"])[0]["obj"]
    xAll = OptHist.read([0, -1], ["x"])[0]["x"]
    gAll = OptHist.read([0, -1], ["con"])[0]["con"]
    if Alg == "NLPQLP":
        gAll = [x * -1 for x in gAll]
    gGradIter = OptHist.read([0, -1], ["grad_con"])[0]["grad_con"]
    fGradIter = OptHist.read([0, -1], ["grad_obj"])[0]["grad_obj"]
    failIter = OptHist.read([0, -1], ["fail"])[0]["fail"]
    if Alg == "COBYLA" or Alg == "NSGA2" or Alg[:5] == "PyGMO":
        fIter = fAll
        xIter = xAll
        gIter = gAll
    else:
        fIter = [[]] * len(fGradIter)
        xIter = [[]] * len(fGradIter)
        gIter = [[]] * len(fGradIter)
        # SuIter = [[]] * len(fGradIter)
        for ii in range(len(fGradIter)):
            Posdg = OptHist.cues["grad_con"][ii][0]
            Posf = OptHist.cues["obj"][ii][0]
            iii = 0
            while Posdg > Posf:
                iii = iii + 1
                try:
                    Posf = OptHist.cues["obj"][iii][0]
                except:
                    Posf = Posdg + 1
            iii = iii - 1
            fIter[ii] = fAll[iii]
            xIter[ii] = xAll[iii]
            gIter[ii] = gAll[iii]
    OptHist.close()

# -----------------------------------------------------------------------------
#       Convert all data to numpy arrays
# -----------------------------------------------------------------------------
    fIter = np.asarray(fIter)
    xIter = np.asarray(xIter)
    gIter = np.asarray(gIter)
    gGradIter = np.asarray(gGradIter)
    fGradIter = np.asarray(fGradIter)
    return fIter, xIter, gIter, gGradIter, fGradIter