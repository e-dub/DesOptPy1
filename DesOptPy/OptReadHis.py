"""
Title:    OptReadHis.py
Units:    -
Author:   E. J. Wehrle
Date:     June 4, 2019
-------------------------------------------------------------------------------
Description:

-------------------------------------------------------------------------------
ToDos:
Change to be callable a posteri
-------------------------------------------------------------------------------
"""
import pyOpt
import numpy as np
#from DesOptPy.Normalize import normalize, denormalize


def OptReadHis(OptName, Alg, AlgOptions, x0, xL, xU, gc, DesVarNorm):
    nx = len(x0)
    ng = len(gc)
    OptHist = pyOpt.History(OptName, "r")
    inform = " "
    fAll = OptHist.read([0, -1], ["obj"])[0]["obj"]
    xAll = OptHist.read([0, -1], ["x"])[0]["x"]
    gAll = OptHist.read([0, -1], ["con"])[0]["con"]
    if Alg == "NLPQLP":
        gAll = [x * -1 for x in gAll]
    gNablaIt = OptHist.read([0, -1], ["grad_con"])[0]["grad_con"]
    fNablaIt = OptHist.read([0, -1], ["grad_obj"])[0]["grad_obj"]
    failIt = OptHist.read([0, -1], ["fail"])[0]["fail"]
    nIt = len(fNablaIt)
    if Alg in ["COBYLA", "NSGA2", "SDPEN", "ALPSO", "MIDACO", "ALGENCAN",
               "ALHSO"] or Alg[:5] == "PyGMO":
        nIt = len(fAll)
        fIt = fAll
        xIt = xAll
        gIt = gAll
    elif Alg == "NSGA-II" and np.size(gAll) > 0:
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
                fIt.append(fAll[i * PopSize + pos_smallest_violation])
                xIt.append(xAll[i * PopSize + pos_smallest_violation])
                gIt.append(gAll[i * PopSize + pos_smallest_violation])
            else:  # find the best feasible one
                # Iteration trough the Individuals of the actual population
                for u in range(0, PopSize):
                    if np.max(fAll[i * PopSize + u]) < best_fitness:
                        if np.max(gAll[i * PopSize + u]) <= 0:
                            best_fitness = fAll[i * PopSize + u]
                            pos_of_best_ind = i * PopSize + u
                fIt.append(fAll[pos_of_best_ind])
                xIt.append(xAll[pos_of_best_ind])
                gIt.append(gAll[pos_of_best_ind])
        nIt = len(fIt)
    elif Alg == "IPOPT":
        inform = 'Optimization terminated successfully'
        nIt = len(fNablaIt)
        fIt = [[]] * int(len(fNablaIt)-2)
        xIt = [[]] * int(len(fNablaIt)-2)
        gIt = [[]] * int(len(fNablaIt)-2)
        for ii in range(len(fIt)):
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
            fIt[ii] = fAll[iii]
            xIt[ii] = xAll[iii]
            gIt[ii] = gAll[iii]
        nIt = len(fIt)
    else:
        inform = 'Optimization terminated successfully'
        fIt = [[]] * len(fNablaIt)
        xIt = [[]] * len(fNablaIt)
        gIt = [[]] * len(fNablaIt)
        for ii in range(len(fIt)):
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
            fIt[ii] = fAll[iii]
            xIt[ii] = xAll[iii]
            gIt[ii] = gAll[iii]
        nIt = len(fIt)
    OptHist.close()
#    if Alg != "NSGA2":
#        if len(fNablaIt) == 0:  # first calculation
#            fIt = fAll
#            xIt = xAll
#            gIt = gAll
# -----------------------------------------------------------------------------
#       Convert all data to numpy arrays
# -----------------------------------------------------------------------------
    xIt = np.asarray(xIt).T
    fIt = np.asarray(fIt[:]).reshape((nIt,))
    gIt = np.asarray(gIt).T
    fNablaIt = np.asarray(fNablaIt).T
    gNablaItTemp = np.zeros((ng, nx, nIt))
    if ng>0:
        for i in range(nIt):
            gNablaIt[i].resize(ng, nx)
            gNablaItTemp[:, :, i] = gNablaIt[i]
        gNablaIt = gNablaItTemp
    else:
        gNablaIt = np.array([])
    return fIt, xIt, gIt, gNablaIt, fNablaIt, inform
