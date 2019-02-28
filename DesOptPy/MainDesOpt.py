# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Title:          MainDesOpt.py
Version:        1.4α
Units:          Unitless
Author:         E. J. Wehrle
Contributors:   S. Rudolph (<α0.5), F. Wachter (α0.5-1.2), M. Richter (α0.5)
Date:           August 14, 2018
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
Description
-------------------------------------------------------------------------------
DesOptPy - DESign OPTimization in PYthon - is an optimization toolbox in Python

-------------------------------------------------------------------------------
Change log
-------------------------------------------------------------------------------
1.4 (Future)
    General clean-up
    NumPy problems
    PyGMO working again
    Sensitivity analysis

1.3 (July 30, 2016)
    Removed support for surrogate model based optimization
    PEP8

1.2 June 29, 2016
    New status report backend
    Renaming of __init__ in MainDesOpt
    General clean-up

1.1 November 18, 2015
    General clean-up
    Support for multiobjective optimization with pyOpt's NSGAII
    DesOpt returns fOpt, xOpt, Results.  Results replaces SP and includes
        "everything"
    More approximation options!

1.02 November 16, 2015
    Again improved status reports

1.01 November 10, 2015
    Reworked status reports

1.0 October 18, 2015 -- Initial public release

α2.0 April 6, 2015 -- Preparation for release as open source
    Main file renamed from DesOpt.py to __init__.pyCMA
    Position of file (and rest of package) changed from ~/DesOpt/.DesOptPy
        to /usr/local/lib/python2.7/dist-packages/DesOptPy/

α1.3 April 3, 2015
    Algorithm options added to call, only for pyOpt algorithms
    Discrete variables added analogous to APSIS (Schatz & Wehrle & Baier 2014),
        though without the reduction of design variables by 1, example in
        AxialBar

α1.2 February 15, 2015
    PyGMO solver
    New setup of optimization problem to ease integration of new algorithms
    Presentation in alpha version---not finished
    Removed algorithm support for pyCMA as interface to CMA-ES in PyGMO

α1.1:
    Seconds in OptName
    gc in SysEq and SensEq

α1.0:
    Changed configuration so that SysEq.py imports and calls DesOpt!
    LLB -> FGCM

α0.5:
    SBDO functions with Latin hypercube and Gaussian process (i. e. Kriging)
        but not well (E. J. Wehrle)
    Parallelized finite differencing capabilities on LLB clusters Flettner and
        Dornier (M. Richter)
    Fixed negative times (M. Richter)
    Now using CPickle instead of pickle for speed-up (M. Richter)
    Implementation of three types of design variable normalization
        (E. J. Wehrle)

-------------------------------------------------------------------------------
To do and ideas
-------------------------------------------------------------------------------
TODO: change private (inner) functions to _$FunctionName
TODO: Return lagrangian multipliers for orderly: df/d[g, xU, xL] where non-
    active components are shown as zero!
TODO: Constraint handling in PyGMO
TODO every iteration to output window (file)
    to output text file as well as status report.
TODO normalize deltax?
TODO extend to use with other solvers: IPOPT!, CVXOPT (http://cvxopt.org/),
    pyCOIN (http://www.ime.usp.br/~pjssilva/software.html#pycoin)
TODO extend to use with  fmincon!!!!!
TODO extend to use with OpenOpt
TODO Evolutionary strategies with PyEvolve, inspyred, DEAP, pybrain
TODO Discrete optimization with python-zibopt?
TODO Multiobjective
    http://www.midaco-solver.com/index.php/more/multi-objective
    http://openopt.org/interalg
    single objective for pareto front fhat = k*f1 + (k-1)*f2, where k = 0...1
TODO Lagrangian multiplier for one-dimensional optimization, line 423
TODO gGradIter is forced into
TODO sens_mode='pgc'
TODO Output data formats
        pyTables?
        readable in Excel?
TODO excel report with "from xlwt import Workbook"
TODO range to xrange (xrange is faster)?
TODO Examples
   SIMP with ground structure
   Multimaterial design (SIMP)
   Shape and sizing optimization of a space truss
   Robustness
   Reliability
   Analytical sensitivities,
   Analytical sensitivities for nonlinear dynamic problems
   Discrete problems
   gGradIter in dgdxIter
   fGradIter in dfdxIter etc
-------------------------------------------------------------------------------
"""


# -----------------------------------------------------------------------------
# Import necessary Python packages and toolboxes
# -----------------------------------------------------------------------------
from __future__ import absolute_import, division, print_function
import os
import shutil
import sys
import inspect
import pyOpt
from DesOptPy import OptAlgOptions
from DesOptPy import OptHis2HTML
from DesOptPy import OptVideo
from DesOptPy import OptResultReport
import numpy as np
try:
    import _pickle as pickle
except:
    try:
        import cPickle as pickle
    except:
        import pickle
import scipy.io as spio
import time
import copy
import datetime
import getpass
import multiprocessing
import platform
from DesOptPy.Normalize import(normalize, denormalize, normalizeSens,
                               denormalizeSens)
from DesOptPy.OptPostProc import OptPostProc
from DesOptPy.OptReadHis import OptReadHis


# -----------------------------------------------------------------------------
# Package details
# -----------------------------------------------------------------------------
__title__ = "DESign OPTimization in PYthon"
__shorttitle__ = "DesOptPy"
__version__ = "1.3pre"
__all__ = ['DesOpt']
__author__ = "E. J. Wehrle"
__copyright__ = "Copyright 2015, 2016, 2017, E. J. Wehrle"
__email__ = "Erich.Wehrle(a)unibz.it"
__license__ = "GNU Lesser General Public License"
__url__ = 'www.DesOptPy.org'


# -----------------------------------------------------------------------------
# Print details of DesOptPy
# -----------------------------------------------------------------------------
def PrintDesOptPy():
    print(__title__+" - "+__shorttitle__)
    print("Version:                 "+__version__)
    print("Internet:                "+__url__)
    print("License:                 "+__license__)
    print("Copyright:               "+__copyright__)
    print("\n")

#global nEval
nEval = 0

# -----------------------------------------------------------------------------
# Main function DesOpt
# -----------------------------------------------------------------------------
def DesOpt(SysEq, x0, xU, xL, xDis=[], gc=[], hc=[], SensEq=[], Alg="SLSQP",
           SensCalc="FD", DesVarNorm=True, nf=1, deltax=1e-3,
           StatusReport=False, ResultReport=False, Video=False, nDoE=0,
           DoE="LHS+Corners", SBDO=False, Approx=[], Debug=False,
           PrintOut=True, OptNameAdd="", AlgOptions=[], KeepEval=False,
           Alarm=True):

# -----------------------------------------------------------------------------
# Define optimization problem and optimization options
# -----------------------------------------------------------------------------
    """
    :type OptNode: object
    """
    global nEval
    #nEval = 0
    if Debug:
        StatusReport = False
        if StatusReport:
            print("Debug is set to True; overriding StatusReport setting it \
                  to False")
            StatusReport = False
        if ResultReport:
            print("Debug is set to True; overriding ResultReport setting it \
                  to False")
            ResultReport = False
    computerName = platform.uname()[1]
    operatingSystem = platform.uname()[0]
    architecture = platform.uname()[4]
    nProcessors = str(multiprocessing.cpu_count())
    userName = getpass.getuser()
    OptTime0 = time.time()
    OptNodes = "all"
    MainDir = os.getcwd()
    OptModel = os.getcwd().split(os.sep)[-1]
    try:
        OptAlg = eval("pyOpt." + Alg + '()')
        pyOptAlg = True
    except:
        OptAlg = Alg
        pyOptAlg = False
    if hasattr(SensEq, '__call__'):
        SensCalc = "OptSensEq"
        print("Function for sensitivity analysis has been provided, \
               overriding SensCalc to use function")
    else:
        pass
    StartTime = datetime.datetime.now()
    loctime = time.localtime()
    today = time.strftime("%B", time.localtime()) + ' ' + str(loctime[2]) + \
                          ', ' + str(loctime[0])
    OptName = OptModel + OptNameAdd + "_" + Alg + "_" + \
              StartTime.strftime("%Y%m%d%H%M%S")
    LocalRun = True         #left over from cluster parallelization...remove?
    ModelDir = os.getcwd()[:-(len(OptModel) + 1)]
    ModelFolder = ModelDir.split(os.sep)[-1]
    DesOptDir = ModelDir[:-(len(ModelFolder) + 1)]
    ResultsDir = DesOptDir + os.sep + "Results"
    RunDir = DesOptDir + os.sep + "Run"
    try:
        inform
    except NameError:
        inform = ["Running"]
    if LocalRun and Debug is False:
        try:
            os.mkdir(ResultsDir)
        except:
            pass
        os.mkdir(ResultsDir + os.sep + OptName)
        os.mkdir(ResultsDir + os.sep + OptName + os.sep + "ResultReport" +
                 os.sep)
        shutil.copytree(os.getcwd(), RunDir + os.sep + OptName)
    if LocalRun and Debug is False:
        os.chdir("../../Run/" + OptName + "/")
    sys.path.append(os.getcwd())


# -----------------------------------------------------------------------------
#       Print start-up splash to output screen
# -----------------------------------------------------------------------------
    if PrintOut:
        print("-"*80)
        PrintDesOptPy()
        print("Optimization model:      " + OptModel)
        try:
            print("Optimization algorithm:  " + Alg)
        except:
            pass
        print("Optimization start:      " + StartTime.strftime("%Y%m%d%H%M"))
        print("Optimization name:       " + OptName)
        print("-"*80)


# -----------------------------------------------------------------------------
#       Optimization problem
# -----------------------------------------------------------------------------
    import glob

# -----------------------------------------------------------------------------
#       Define functions: system equation, normalization, etc.
# -----------------------------------------------------------------------------
    def OptSysEq(x):
        global nEval
        x = np.array(x)  # NSGA2 gives a list back, this makes a float! TODO Inquire why it does this!
        if KeepEval:
            os.mkdir(str(nEval))
            for filename in glob.glob('*.*'):
                if filename[0:len(OptName)] != OptName \
                   and filename not in ["desVars.csv", "desVarsNorm.csv",
                                        "constraints.csv", "objFct_maxCon.csv",
                                        "initial1.html"]:
                    # [filename[-3:] != "csv" and filename != "inital1.html":
                    shutil.copy(filename, str(nEval))
            os.chdir(str(nEval))
        f, g = SysEq(x, gc)
        if KeepEval:
            os.chdir("..")
        fail = 0
        nEval += 1
        if StatusReport:
            OptHis2HTML.OptHis2HTML(OptName, Alg, AlgOptions, DesOptDir, x0,
                                    xL, xU, DesVarNorm, inform[0], OptTime0)
        if len(xDis) > 0:
            nD = len(xDis)
            gDis = [[]]*2*nD
            for ii in range(nD):
                gDis[ii+0] = np.sum(x[-1*xDis[ii]:])-1
                gDis[ii+1] = 1-np.sum(x[-1*xDis[ii]:])
            gNew = np.concatenate((g, gDis), 0)
            g = copy.copy(gNew)
        # TODO add print out for optimization development!!
        return f, g, fail

    def OptSysEqNorm(xNorm):
        xNorm = np.array(xNorm)  # NSGA2 gives a list back, this makes a float! TODO Inquire why it does this!
        x = denormalize(xNorm, x0, xL, xU, DesVarNorm)
        f, g, fail = OptSysEq(x)
        return f, g, fail

    def OptPenSysEq(x):
        f, g, fail = OptSysEq(x)
        fpen = f
        return fpen

    def OptSensEq(x, f, g):
        if KeepEval:
            os.chdir(str(nEval-1))
        dfdx, dgdx = SensEq(x, f, g, gc)
        if KeepEval:
            os.chdir("..")
        dfdx = dfdx.reshape(1, len(x))
        if np.size(dgdx) > 0:
            dgdx = dgdx.reshape(len(g), len(x))
        fail = 0
        return dfdx, dgdx, fail

    def OptSensEqNorm(xNorm, f, g):
        x = denormalize(xNorm, x0, xL, xU, DesVarNorm)
        dfxdx, dgxdx, fail = OptSensEq(x, f, g)
        dfdx = dfxdx * (xU - xL)
        dgdx = normalizeSens(dgxdx, x0, xL, xU, DesVarNorm)
        # TODO not general for all normalizations! needs to be rewritten; done: check if correct
        # dfdx = normalizeSens(dfxdx, xL, xU, DesVarNorm)
        # if dgxdx != []:
        #    dgdx = dgxdx * np.tile((xU - xL), [len(g), 1])
        # else:
        #    dgdx = []
        return dfdx, dgdx, fail

    def OptSensEqParaFD(x, f, g):
        #global nEval
        dfdx, dgdx, nb = OptSensParaFD.Para(x, f, g, deltax, OptName, OptNodes)
        nEval += nb
        fail = 0
        return dfdx, dgdx, fail

    def VectOptSysEq(x):
        f, g, fail = OptSysEq(x)
        np.array(f)
        if len(g) > 0:
            r = np.concatenate((f, g))
        else:
            r = f
        return r

    def OptSensEqAD(x, f, g):
        import autograd
        OptSysEq_dx = autograd.jacobian(VectOptSysEq)
        drdx = OptSysEq_dx(x)
        if g > 0:
            dfdx = np.array([drdx[0, :]]).T
            dgdx = drdx[1:, :].T
        else:
            dfdx = drdx.reshape(1, len(x))
            dgdx = []
        fail = 0
        return dfdx, dgdx, fail

    def VectOptSysEqNorm(x):
        f, g, fail = OptSysEqNorm(x)
        np.array(f)
        if len(g) > 0:
            r = np.concatenate((f, g))
        else:
            r = f
        return r

    def OptSensEqNormAD(x, f, g):
        import autograd
        OptSysEq_dx = autograd.jacobian(VectOptSysEqNorm)
        drdx = OptSysEq_dx(x)
        if g > 0:
            dfdx = np.array([drdx[0, :]])
            dgdx = drdx[1:, :]
        else:
            dfdx = drdx.reshape(1, len(x))
            dgdx = []
        fail = 0
        return dfdx, dgdx, fail

    def OptSensEqParaFDNorm(xNorm, f, g):
        x = denormalize(xNorm, xL, xU, DesVarNorm)
        dfxdx, dgxdx, fail = OptSensEqParaFD(x, f, g)
        dfdx = normalizeSens(dfxdx, x0, xL, xU, DesVarNorm)
        dgdx = normalizeSens(dgxdx, x0, xL, xU, DesVarNorm)
        # TODO not general for all normalizations! needs to be rewritten, done: check if correct
        # dfdx = dfxdx * (xU - xL)
        # dgdx = dgxdx * (np.tile((xU - xL), [len(g), 1]))
        return dfdx, dgdx, fail


# -----------------------------------------------------------------------------
# Removed Surrogate-based optimization, use SuPy!
# -----------------------------------------------------------------------------
    if SBDO is not False:
        print("Use SuPy!")
    if xDis is not []:
        for ii in range(np.size(xDis, 0)):
            xExpand0 = np.ones(xDis[ii]) * 1./xDis[ii]   # Start at uniform of all materials etc.
            xNew0 = np.concatenate((x0, xExpand0), 0)
            xExpandL = np.ones(xDis[ii]) * 0.0001
            xNewL = np.concatenate((xL, xExpandL), 0)
            xExpandU = np.ones(xDis[ii])
            xNewU = np.concatenate((xU, xExpandU), 0)
            x0 = copy.copy(xNew0)
            xL = copy.copy(xNewL)
            xU = copy.copy(xNewU)
            gcNew = np.concatenate((gc, np.ones(2,)), 0)
            gc = copy.copy(gcNew)
    if DesVarNorm in ["None", None, False]:
        x0norm = x0
        xLnorm = xL
        xUnorm = xU
        DefOptSysEq = OptSysEq
    else:
        [x0norm, xLnorm, xUnorm] = normalize(x0, x0, xL, xU, DesVarNorm)
        DefOptSysEq = OptSysEqNorm
    nx = np.size(x0)
    ng = np.size(gc)

# -----------------------------------------------------------------------------
#       pyOpt optimization
# -----------------------------------------------------------------------------
    if pyOptAlg:
        OptProb = pyOpt.Optimization(OptModel, DefOptSysEq)
        if np.size(x0) == 1:
            OptProb.addVar('x', 'c', value=x0norm, lower=xLnorm, upper=xUnorm)
        elif np.size(x0) > 1:
            for ii in range(np.size(x0)):
                OptProb.addVar('x' + str(ii + 1), 'c', value=x0norm[ii],
                               lower=xLnorm[ii], upper=xUnorm[ii])
        if nf == 1:
            OptProb.addObj('f')
        elif nf > 1:
            for ii in range(nf):
                OptProb.addObj('f' + str(ii + 1))
        if np.size(gc) == 1:
            OptProb.addCon('g', 'i')
            # ng = 1
        elif np.size(gc) > 1:
            for ii in range(len(gc)):
                OptProb.addCon('g' + str(ii + 1), 'i')
            # ng = ii + 1
        if np.size(hc) == 1:
            OptProb.addCon('h', 'i')
        elif np.size(hc) > 1:
            for ii in range(ng):
                OptProb.addCon('h' + str(ii + 1), 'i')
        if not AlgOptions:
            AlgOptions = OptAlgOptions.setDefault(Alg)
        OptAlg = OptAlgOptions.setUserOptions(AlgOptions, Alg, OptName, OptAlg)
        # if AlgOptions == []:
        #    OptAlg = OptAlgOptions.setDefaultOptions(Alg, OptName, OptAlg)
        # else:
        #    OptAlg = OptAlgOptions.setUserOptions(AlgOptions, Alg, OptName, OptAlg)
        if PrintOut:
            print(OptProb)
        if Alg in ["MMA", "FSQP", "GCMMA", "CONMIN", "SLSQP", "PSQP",
                   "KSOPT", "ALGENCAN", "NLPQLP", "IPOPT"]:
            if SensCalc == "OptSensEq":
                if DesVarNorm not in ["None", None, False]:
                    [fOpt, xOpt, inform] = OptAlg(OptProb,
                                                  sens_type=OptSensEqNorm,
                                                  store_hst=OptName)
                else:
                    [fOpt, xOpt, inform] = OptAlg(OptProb, sens_type=OptSensEq,
                                                  store_hst=OptName)
            elif SensCalc == "ParaFD":  # Michi Richter
                if DesVarNorm not in ["None", None, False]:
                    [fOpt, xOpt, inform] = OptAlg(OptProb,
                                                  sens_type=OptSensEqParaFDNorm,
                                                  store_hst=OptName)
                else:
                    [fOpt, xOpt, inform] = OptAlg(OptProb,
                                                  sens_type=OptSensEqParaFD,
                                                  store_hst=OptName)
            elif SensCalc == "AD":
                if DesVarNorm not in ["None", None, False]:
                    [fOpt, xOpt, inform] = OptAlg(OptProb,
                                                  sens_type=OptSensEqNormAD,
                                                  store_hst=OptName)
                else:
                    [fOpt, xOpt, inform] = OptAlg(OptProb,
                                                  sens_type=OptSensEqAD,
                                                  store_hst=OptName)
            else:  # Here FD (finite differencing)
                [fOpt, xOpt, inform] = OptAlg(OptProb, sens_type=SensCalc,
                                              sens_step=deltax,
                                              store_hst=OptName)

        elif Alg in ["SDPEN", "SOLVOPT"]:
            [fOpt, xOpt, inform] = OptAlg(OptProb, store_hst=OptName)
        else:                                               # NSGA2
            [fOpt, xOpt, inform] = OptAlg(OptProb, store_hst=OptName)
            print(fOpt)
        if PrintOut:
            try:
                print(OptProb.solution(0))
            except:
                pass
        if Alg not in ["PSQP", "SOLVOPT", "MIDACO", "SDPEN", "ralg"] \
           and PrintOut:
            print(OptAlg.getInform(0))

# -----------------------------------------------------------------------------
#       OpenOpt optimization -- not fully implemented in this framework and not yet working...
# -----------------------------------------------------------------------------
    elif Alg == "ralg":
        from openopt import NLP
        f, g = lambda x: OptSysEq(x)
        # g = lambda x: OptSysEq(x)[1][0]
        p = NLP(f, x0, c=g, lb=xL, ub=xU, iprint=50, maxIter=10000,
                maxFunEvals=1e7, name='NLP_1')
        r = p.solve(Alg, plot=0)
        print(OptAlg.getInform(1))

# -----------------------------------------------------------------------------
#       pyCMAES
# -----------------------------------------------------------------------------
    elif Alg == "pycmaes":
        print("CMA-ES == not fully implemented in this framework")
        print("    no constraints")
        import cma

        def CMA_ES_ObjFn(x):
            f, g, fail = OptSysEq(x)
            return f
        OptRes = cma.fmin(CMA_ES_ObjFn, x0, sigma0=1)
        xOpt = OptRes[0]
        fOpt = OptRes[1]
        nEval = OptRes[4]
        nIter = OptRes[5]
# -----------------------------------------------------------------------------
#       MATLAB fmincon optimization -- not fully implemented in this framework and not yet working...
# -----------------------------------------------------------------------------
    elif Alg == "fmincon":  # not fully implemented in this framework
        def ObjFn(x):
            f, g, fail = OptSysEqNorm(xNorm)
            return f, []
        from mlabwrap import mlab
        mlab._get(ObjFn)
        mlab.fmincon(mlab._get("ObjFn"), x)
        # g,h, dgdx = mlab.fmincon(x.T,cg,ch, nout=3)

# -----------------------------------------------------------------------------
#       PyGMO optimization
# -----------------------------------------------------------------------------
    elif Alg[:5] == "PyGMO":
        import pygmo as pg
        if not AlgOptions:
            AlgOptions = OptAlgOptions.setDefault(Alg)
        #OptAlg = OptAlgOptions.setUserOptions(AlgOptions, Alg, OptName, OptAlg)
        class OptProbPyGMO():
            def fitness(self, x):
                if DesVarNorm == "None":
                    f, g, fail = OptSysEq(x)
                else:
                    f, g, fail = OptSysEqNorm(x)
                try:
                    g = g.tolist()
                except:
                    pass
                global HistData
                if nEval == 1:
                    AlgInst = pyOpt.Optimizer(Alg)
                    HistData = pyOpt.History(OptName, 'w', optimizer=AlgInst,
                                             opt_prob=OptName)
                HistData.write(x, "x")
                HistData.write(f, "obj")
                HistData.write(g, "con")
                if StatusReport == 1:
                    OptHis2HTML.OptHis2HTML(OptName, Alg, AlgOptions,
                                            DesOptDir, x0, xL, xU, DesVarNorm,
                                            inform[0], OptTime0)
                fg = g
                fg.insert(0, f)
                return(fg)
            def get_bounds(self):
                if DesVarNorm == "None":
                    return (xL, xU)
                else:
                    return (xLnorm, xUnorm)
            def get_nic(self):
                return(len(gc))
            def get_nec(self):
                return 0
            def gradient(self, x):
                return pg.estimate_gradient_h(lambda x: self.fitness(x), x)
#        prob = pg.problem(OptProbPyGMO(xL.tolist(), xU.tolist(),
#                                       xLnorm.tolist(), xUnorm.tolist(),
#                                       gc.tolist(), DesVarNorm))
        prob = pg.problem(OptProbPyGMO())
        #algo = pg.algorithm(uda = pg.nlopt('auglag'))
        #algo.extract(pg.nlopt).local_optimizer = pg.nlopt('var2')
        #pop = pg.population(prob=prob, size=1)

        pop = pg.population(prob=prob, size=AlgOptions.nIndiv)
        if Alg[6:] == "monte_carlo":
            algo = pg.algorithm.monte_carlo(iters=AlgOptions.iter)
        elif len(gc) > 0:
            algo = pg.algorithm(pg.cstrs_self_adaptive(iters=AlgOptions.gen,
                                                       algo=eval("pg." +
                                                                 Alg[6:] +
                                                                 '(' +
                                                                 str(AlgOptions.nIndiv)+
                                                                 ')')))
            pop.problem.c_tol = [1E-6] * len(gc)
        else:
            algo = eval("pg.algorithm(pg." + Alg[6:] + '(' + str(AlgOptions.gen)+'))')
        pop = algo.evolve(pop)
        xOpt = pop.champion_x
        fOpt = pop.champion_f[0]
        gOpt = pop.champion_f[1:]

# -----------------------------------------------------------------------------
#        SciPy optimization
# -----------------------------------------------------------------------------
    elif Alg[:5] == "scipy":
        import scipy.optimize as sciopt
        bounds = [[]]*len(x0)
        for ii in range(len(x0)):
            bounds[ii] = (xL[ii], xU[ii])
        print(bounds)
        if Alg[6:] == "de":
            sciopt.differential_evolution(DefOptSysEq, bounds,
                                          strategy='best1bin', maxiter=None,
                                          popsize=15, tol=0.01,
                                          mutation=(0.5, 1), recombination=0.7,
                                          seed=None, callback=None, disp=False,
                                          polish=True, init='latinhypercube')

# -----------------------------------------------------------------------------
#        Simple optimization algorithms to demonstrate use of custom algorithms
# -----------------------------------------------------------------------------
    # TODO: add history to these
    elif Alg == "SteepestDescentSUMT":
        from CustomAlgs import SteepestDescentSUMT
        fOpt, xOpt, nIter, nEval = SteepestDescentSUMT(DefOptSysEq, x0, xL, xU)
    elif Alg == "NewtonSUMT":
        from CustomAlgs import NewtonSUMT
        fOpt, xOpt, nIter, nEval = NewtonSUMT(DefOptSysEq, x0, xL, xU)

# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
    else:
        raise Exception("Error on line " +
                        str(inspect.currentframe().f_lineno) +
                        " of file " + __file__ +
                        ": algorithm misspelled or not supported")
        #sys.exit("Error on line " + str(inspect.currentframe().f_lineno) +
        #         " of file " + __file__ +
        #         ": algorithm misspelled or not supported")

# -----------------------------------------------------------------------------
#       Optimization post-processing
# -----------------------------------------------------------------------------
    if StatusReport == 1:
        OptHis2HTML.OptHis2HTML(OptName, Alg, AlgOptions, DesOptDir, x0, xL,
                                xU, DesVarNorm, inform["text"], OptTime0)
    OptTime1 = time.time()
    loctime0 = time.localtime(OptTime0)
    hhmmss0 = time.strftime("%H", loctime0) + ' : ' + \
                            time.strftime("%M", loctime0) + ' : ' + \
                            time.strftime("%S", loctime0)
    loctime1 = time.localtime(OptTime1)
    hhmmss1 = time.strftime("%H", loctime1) + ' : ' + \
                            time.strftime("%M", loctime1) + ' : ' + \
                            time.strftime("%S", loctime1)
    diff = OptTime1 - OptTime0
    h0, m0, s0 = (diff // 3600), int((diff / 60) -
                 (diff // 3600) * 60), diff % 60
    OptTime = "%02d" % (h0) + " : " + "%02d" % (m0) + " : " + "%02d" % (s0)

    NewRead = True
    if NewRead:
        fIter, xIter, gIter, gGradIter, fGradIter, inform = OptReadHis(OptName,
                                                                       Alg,
                                                                       AlgOptions,
                                                                       x0, xL,
                                                                       xU,
                                                                       DesVarNorm)
        xOpt = np.resize(xOpt[0:np.size(xL)], np.size(xL))
        if DesVarNorm in ["None", None, False]:
            x0norm = []
            xIterNorm = []
            xOptNorm = []
        else:
            xOpt = np.resize(xOpt, [np.size(xL), ])
            xOptNorm = xOpt
            xOpt = denormalize(xOptNorm.T, x0, xL, xU, DesVarNorm)
            try:
                xIterNorm = xIter[:, 0:np.size(xL)]
                xIter = np.zeros(np.shape(xIterNorm))
                for ii in range(len(xIterNorm)):
                    xIter[ii] = denormalize(xIterNorm[ii], x0, xL, xU,
                                            DesVarNorm)
            except:
                x0norm = []
                xIterNorm = []
                xOptNorm = []
        nIter = np.size(fIter, 0)
        if np.size(fIter) > 0:
            if len(fIter[0]) > 0:
                fIterNorm = fIter / fIter[0]
                # fIterNorm=(fIter-fIter[nEval-1])/(fIter[0]-fIter[nEval-1])
            else:
                fIterNorm = fIter
        else:
            fIterNorm = []

# -----------------------------------------------------------------------------
#       Read in results from history files
# -----------------------------------------------------------------------------
    if NewRead is False:
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

# -----------------------------------------------------------------------------
# Denormalization of design variables
# -----------------------------------------------------------------------------
        xOpt = np.resize(xOpt[0:np.size(xL)], np.size(xL))
        if DesVarNorm in ["None", None, False]:
            x0norm = []
            xIterNorm = []
            xOptNorm = []
        else:
            xOpt = np.resize(xOpt, [np.size(xL), ])
            xOptNorm = xOpt
            xOpt = denormalize(xOptNorm.T, x0, xL, xU, DesVarNorm)
            try:
                xIterNorm = xIter[:, 0:np.size(xL)]
                xIter = np.zeros(np.shape(xIterNorm))
                for ii in range(len(xIterNorm)):
                    xIter[ii] = denormalize(xIterNorm[ii], x0, xL, xU,
                                            DesVarNorm)
            except:
                x0norm = []
                xIterNorm = []
                xOptNorm = []
        nIter = np.size(fIter, 0)
        if np.size(fIter) > 0:
            if len(fIter[0]) > 0:
                fIterNorm = fIter / fIter[0]
                # fIterNorm=(fIter-fIter[nEval-1])/(fIter[0]-fIter[nEval-1])
            else:
                fIterNorm = fIter
        else:
            fIterNorm = []

# -----------------------------------------------------------------------------
#  Active constraints for use in the calculation of the Lagrangian multipliers and optimality criterion
# -----------------------------------------------------------------------------
    epsActive = 1e-3
    xL_ActiveIndex = (xOpt - xL) / xU < epsActive
    xU_ActiveIndex = (xU - xOpt) / xU < epsActive
    xL_Grad = -np.eye(nx)
    xU_Grad = np.eye(nx)
    xL_GradActive = xL_Grad[:, xL_ActiveIndex]
    xU_GradActive = xU_Grad[:, xU_ActiveIndex]  # or the other way around!
    xGradActive = np.concatenate((xL_GradActive, xU_GradActive), axis=1)
    # TODO change so that 1D optimization works!
    try:
        xL_Active = xL[xL_ActiveIndex]
    except:
        xL_Active = np.array([])
    try:
        xU_Active = xU[xU_ActiveIndex]
    except:
        xU_Active = np.array([])
    if len(xL_Active) == 0:
        xActive = xU_Active
    elif len(xU_Active) == 0:
        xActive = xL_Active
    else:
        xActive = np.concatenate((xL_Active, xU_Active))
    if np.size(xL) == 1:
        if xL_ActiveIndex == False:
            xL_Active = np.array([])
        else:
            xL_Active = xL
        if xU_ActiveIndex == False:
            xU_Active = np.array([])
        else:
            xU_Active = xU
    else:
        xL_Active = xL[xL_ActiveIndex]
        xU_Active = np.array(xU[xU_ActiveIndex])
    if len(xL_Active) == 0:
        xLU_Active = xU_Active
    elif len(xU_Active) == 0:
        xLU_Active = xL_Active
    else:
        xLU_Active = np.concatenate((xL_Active, xU_Active))
    # TODO needs to be investigated for PyGMO!
    # are there nonlinear constraints active, in case equality constraints are
    # added later, this must also be added
    if np.size(gc) > 0:  # and Alg[:5] != "PyGMO":
        gMaxIter = np.zeros([nIter])
        for ii in range(len(gIter)):
            gMaxIter[ii] = max(gIter[ii])
        gOpt = gIter[nIter - 1]
        gOptActiveIndex = gOpt > -epsActive
        gOptActive = gOpt[gOpt > -epsActive]
    elif np.size(gc) == 0:
        gOptActiveIndex = [[False]] * len(gc)
        gOptActive = np.array([])
        gMaxIter = np.array([] * nIter)
        gOpt = np.array([])
    else:
        gMaxIter = np.zeros([nIter])
        for ii in range(len(gIter)):
            gMaxIter[ii] = max(gIter[ii])
        gOptActiveIndex = gOpt > -epsActive
        gOptActive = gOpt[gOpt > -epsActive]
    if len(xLU_Active) == 0:
        g_xLU_OptActive = gOptActive
    elif len(gOptActive) == 0:
        g_xLU_OptActive = xLU_Active
    else:
        if np.size(xLU_Active) == 1 and np.size(gOptActive) == 1:
            g_xLU_OptActive = np.array([xLU_Active, gOptActive])
        else:
            g_xLU_OptActive = np.concatenate((xLU_Active, gOptActive))
    if np.size(fGradIter) > 0:  # Iteration data present
        # fGradOpt = fGradIter[nIter - 1]
        fGradOpt = fGradIter[-1]
        if np.size(gc) > 0:  # Constrained problem
            # Which is better? Both the same?
            try:
                gGradOpt = gGradIter[nIter-1]
            except:
                gGradOpt = gGradIter[-1]
            gGradOpt = gGradOpt.reshape([ng, nx]).T
            gGradOptActive = gGradOpt[:, gOptActiveIndex == True]
            #gGradOpt = []
            try:
                cOptActive = gc[gOptActiveIndex == True]
                cActiveType = ["Constraint"]*np.size(cOptActive)
            except:
                cOptActive = []
                cActiveType = []
            if np.size(xGradActive) == 0:
                g_xLU_GradOptActive = gGradOptActive
                c_xLU_OptActive = cOptActive
                c_xLU_ActiveType = cActiveType
            elif np.size(gGradOptActive) == 0:
                g_xLU_GradOptActive = xGradActive
                c_xLU_OptActive = xActive
                c_xLU_ActiveType = ["Bound"]*np.size(xActive)
            else:
                g_xLU_GradOptActive = np.concatenate((gGradOptActive,
                                                      xGradActive), axis=1)
                c_xLU_OptActive = np.concatenate((cOptActive, xActive))
                xActiveType = ["Bound"]*np.size(xActive)
                c_xLU_ActiveType = np.concatenate((cActiveType, xActiveType))
        else:
            g_xLU_GradOptActive = xGradActive
            gGradOpt = np.array([])
            c_xLU_OptActive = np.array([])
            g_xLU_GradOptActive = np.array([])
            c_xLU_ActiveType = np.array([])

    else:
        fGradOpt = np.array([])
        gGradOpt = np.array([])
        g_xLU_GradOptActive = np.array([])
        c_xLU_OptActive = np.array([])
        c_xLU_ActiveType = np.array([])

# -----------------------------------------------------------------------------
#   §      Post-processing of optimization solution
# -----------------------------------------------------------------------------
    lambda_c, SPg, OptRes, Opt1Order, KKTmax = OptPostProc(fGradOpt, gc,
                                                           gOptActiveIndex,
                                                           g_xLU_GradOptActive,
                                                           c_xLU_OptActive,
                                                           c_xLU_ActiveType,
                                                           DesVarNorm)

# -----------------------------------------------------------------------------
#   §      Save optimization solution to file
# -----------------------------------------------------------------------------
#    if sys.version_info>(3,6):
#        pass
#    else:
    OptSolData = {}
    OptSolData['x0'] = x0
    OptSolData['xOpt'] = xOpt
    OptSolData['xOptNorm'] = xOptNorm
    OptSolData['xIter'] = xIter
    OptSolData['xIterNorm'] = xIterNorm
    OptSolData['fOpt'] = fOpt
    OptSolData['fIter'] = fIter
    OptSolData['fIterNorm'] = fIterNorm
    OptSolData['gIter'] = gIter
    OptSolData['gMaxIter'] = gMaxIter
    OptSolData['gOpt'] = gOpt
    OptSolData['fGradIter'] = fGradIter
    OptSolData['gGradIter'] = gGradIter
    OptSolData['gGradOpt'] = gGradOpt
    OptSolData['gGradOptDenorm'] =denormalizeSens(gGradOpt, x0, xL, xU,
                                                  DesVarNorm)
    OptSolData['fGradOpt'] = fGradOpt
    OptSolData['fGradOptDenorm'] =denormalizeSens(fGradOpt, x0, xL, xU,
                                                  DesVarNorm)
    OptSolData['OptName'] = OptName
    OptSolData['OptModel'] = OptModel
    OptSolData['OptTime'] = OptTime
    #OptSolData['loctime'] = loctime
    OptSolData['today'] = today
    OptSolData['computerName'] = computerName
    OptSolData['operatingSystem'] = operatingSystem
    OptSolData['architecture'] = architecture
    OptSolData['nProcessors'] = nProcessors
    OptSolData['userName'] = userName
    OptSolData['Alg'] = Alg
    OptSolData['DesVarNorm'] = DesVarNorm
    OptSolData['KKTmax'] = KKTmax
    OptSolData['lambda_c'] = lambda_c
    OptSolData['nEval'] = nEval
    OptSolData['nIter'] = nIter
    OptSolData['SPg'] = SPg
    OptSolData['gc'] = gc
    OptSolData['SensCalc'] = SensCalc
    OptSolData['xIterNorm'] = xIterNorm
    OptSolData['x0norm'] = x0norm
    OptSolData['xL'] = xL
    OptSolData['xU'] = xU
    OptSolData['ng'] = ng
    OptSolData['nx'] = nx
    OptSolData['nf'] = nf
    OptSolData['Opt1Order'] = Opt1Order
    OptSolData['hhmmss0'] = hhmmss0
    OptSolData['hhmmss1'] = hhmmss1


# -----------------------------------------------------------------------------
#   §    Save in Python format
# -----------------------------------------------------------------------------
    output = open(OptName + "_OptSol.pkl", 'wb')
    pickle.dump(OptSolData, output)
    output.close()
    np.savez(OptName + "_OptSol", x0, xOpt, xOptNorm, xIter, xIterNorm, xIter,
             xIterNorm, fOpt, fIter, fIterNorm, gIter, gMaxIter, gOpt,
             fGradIter, gGradIter, fGradOpt, gGradOpt, OptName, OptModel,
             OptTime, loctime, today, computerName, operatingSystem,
             architecture, nProcessors, userName, Alg, DesVarNorm, KKTmax)

# -----------------------------------------------------------------------------
#   §5.2    Save in MATLAB format
# -----------------------------------------------------------------------------
    # OptSolData['OptAlg'] = []
    spio.savemat(OptName + '_OptSol.mat', OptSolData, oned_as='row')

# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
    os.chdir(MainDir)
    if LocalRun and Debug is False:
        try:
            shutil.move(RunDir + os.sep + OptName,
                        ResultsDir + os.sep + OptName + os.sep + "RunFiles" +
                        os.sep)
        # except WindowsError:
        except:
            print("Run files not deleted from " + RunDir + os.sep + OptName)
            shutil.copytree(RunDir + os.sep + OptName,
                            ResultsDir + os.sep + OptName + os.sep +
                            "RunFiles" + os.sep)

# -----------------------------------------------------------------------------
#   §    Graphical post-processing
# -----------------------------------------------------------------------------
    if ResultReport:
        print("Entering preprocessing mode")
        OptResultReport.OptResultReport(OptName, OptAlg, DesOptDir, diagrams=1,
                                        tables=1, lyx=1)
        # try: OptResultReport.OptResultReport(OptName, diagrams=1, tables=1, lyx=1)
        # except: print("Problem with generation of Result Report. Check if all prerequisites are installed")
    if Video:
        OptVideo.OptVideo(OptName)


# -----------------------------------------------------------------------------
#   § Print out
# -----------------------------------------------------------------------------
    if PrintOut:
        print("")
        print("-"*80)
        print("Optimization results - DesOptPy")
        print("-"*80)
        print("Optimization with " + Alg)
        print("f* = " + str(fOpt))
        print("g* = " + str(gOpt))
        print("x* = " + str(xOpt.T))
        print("Lagrangian multipliers = " + str(lambda_c))
        print("Shadow prices = " + str(SPg))
        print("Time of optimization [hh:mm:ss] = " + OptTime)
        try:
            print("nGen = " + str(nGen))
        except:
            print("nIter = " + str(nIter))
        print("nEval = " + str(nEval))
        if Debug is False:
            print("See results directory: " + ResultsDir + os.sep + OptName)
        else:
            print("Local run, no results saved to results directory")
        print("-"*80)
        if operatingSystem == "Linux" and Alarm:
            t = 1
            freq = 350
            os.system('play --no-show-progress --null \
                      --channels 1 synth %s sine %f' % (t, freq))
    nEval = 0
    return xOpt, fOpt, OptSolData




# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    PrintDesOptPy()
    print("Start DesOptPy from file containing system equations!")
    print("See documentation for further help.")
