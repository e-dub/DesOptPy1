# -*- coding: utf-8 -*-
'''
Title:    OptReadHis.py
Units:    -
Author:   E. J. Wehrle
Date:     July 9, 2016
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
Description:

ToDos:
Change to be callable a posteri

-------------------------------------------------------------------------------
'''
from __future__ import absolute_import, division, print_function
import pyOpt
import numpy as np
from Normalize import normalize, denormalize

def OptReadHis(OptName, Alg, AlgOptions, x0, xL, xU, DesVarNorm):
    OptHist = pyOpt.History(OptName, "r")
    inform = " "
    fAll = OptHist.read([0, -1], ["obj"])[0]["obj"]
    xAll = OptHist.read([0, -1], ["x"])[0]["x"]
    gAll = OptHist.read([0, -1], ["con"])[0]["con"]
    if Alg == "NLPQLP":
        gAll = [x * -1 for x in gAll]
    gGradIter = OptHist.read([0, -1], ["grad_con"])[0]["grad_con"]
    fGradIter = OptHist.read([0, -1], ["grad_obj"])[0]["grad_obj"]
    failIter = OptHist.read([0, -1], ["fail"])[0]["fail"]

    if Alg in ["COBYLA", "NSGA2", "SDPEN", "ALPSO", "MIDACO", "ALGENCAN", "ALHSO"] or Alg[:5] == "PyGMO":
        fIter = fAll
        xIter = xAll
        gIter = gAll
    elif Alg == "NSGA-II" and np.size(gAll)>0:
        Iteration = 'Generation'
        if inform == 0:
            inform = 'Optimization terminated successfully'
        PopSize = AlgOptions['PopSize'][1]
        for i in range(0, fAll.__len__() / PopSize):  # Iteration trough the Populations
            best_fitness = 9999999
            max_violation_of_all_g = np.empty(PopSize)
            max_violation_of_all_g.fill(99999999)
            for u in range(0, PopSize):  # Iteration trough the Individuals of the actual population
                if np.max(gAll[i * PopSize + u]) < max_violation_of_all_g[u]:
                    max_violation_of_all_g[u] = np.max(gAll[i * PopSize + u])
            pos_smallest_violation = np.argmin(max_violation_of_all_g)
            # only not feasible designs, so choose the less violated one as best
            if max_violation_of_all_g[pos_smallest_violation] > 0:
                fIter.append(fAll[i * PopSize + pos_smallest_violation])
                xIter.append(xAll[i * PopSize + pos_smallest_violation])
                gIter.append(gAll[i * PopSize + pos_smallest_violation])
            else:  # find the best feasible one
                # Iteration trough the Individuals of the actual population
                for u in range(0, PopSize):
                    if np.max(fAll[i * PopSize + u]) < best_fitness:
                        if np.max(gAll[i * PopSize + u]) <= 0:
                            best_fitness = fAll[i * PopSize + u]
                            pos_of_best_ind = i * PopSize + u
                fIter.append(fAll[pos_of_best_ind])
                xIter.append(xAll[pos_of_best_ind])
                gIter.append(gAll[pos_of_best_ind])
    else:

        inform = 'Optimization terminated successfully'
        fIter = [[]] * len(fGradIter)
        xIter = [[]] * len(fGradIter)
        gIter = [[]] * len(fGradIter)
        for ii in range(len(fIter)):
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
#    if Alg != "NSGA2":
#        if len(fGradIter) == 0:  # first calculation
#            fIter = fAll
#            xIter = xAll
#            gIter = gAll
# -----------------------------------------------------------------------------
#       Convert all data to numpy arrays
# -----------------------------------------------------------------------------
    fIter = np.asarray(fIter)
    xIter = np.asarray(xIter)
    gIter = np.asarray(gIter)
    gGradIter = np.asarray(gGradIter)
    fGradIter = np.asarray(fGradIter)
    return fIter, xIter, gIter, gGradIter, fGradIter, inform