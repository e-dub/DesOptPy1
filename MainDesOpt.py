# -*-  coding: utf-8 -*-
"""
------------------------------------------------------------------------------------------------------------------------
Title:          MainDesOpt.py
Version:        1.2pre
Units:          Unitless
Author:         E. J. Wehrle
Contributors:   S. Rudolph, F. Wachter, M. Richter
Date:           May 21, 2016
------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------------------------------------------------------
Description
------------------------------------------------------------------------------------------------------------------------
DesOptPy -- DESign OPTimization in PYthon -- is an optimization toolbox in Python

------------------------------------------------------------------------------------------------------------------------
Change log
------------------------------------------------------------------------------------------------------------------------
1.2     expected: June 1, 2016
        New status report backend
        Renaming of __init__ in MainDesOpt
        General clean-up

1.1     November 18, 2015
        General clean-up
        Support for multiobjective optimization with pyOpt's NSGAII
        DesOpt returns fOpt, xOpt, Results.  Results replaces SP and includes "everything"
        More approximation options!

1.02    November 16, 2015
        Again improved status reports

1.01    November 10, 2015
        Reworked status reports

1.0     October 18, 2015 -- Initial public release

α2.0    April 6, 2015 -- Preparation for release as open source
        Main file renamed from DesOpt.py to __init__.pyCMA
        Position of file (and rest of package) changed to /usr/local/lib/python2.7/dist-packages/DesOptPy/ from ~/DesOpt/.DesOptPy

α1.3    April 3, 2015
        Algorithm options added to call, only for pyOpt algorithms
        Discrete variables added analogous to APSIS (Schatz & Wehrle & Baier 2014), though without the
        reduction of design variables by 1, example in AxialBar

α1.2    February 15, 2015
        PyGMO solver
        New setup of optimization problem to ease integration of new algorithms
        Presentation in alpha version---not finished
        Removed algorithm support for pyCMA as two interfaces to CMA-ES in PyGMO

α1.1:
        Seconds in OptName
        gc in SysEq and SensEq

α1.0:
        Changed configuration so that SysEq.py imports and calls DesOpt!
        LLB -> FGCM

α0.5:
        SBDO functions with Latin hypercube and Gaussian process (i. e. Kriging) but not well (E. J. Wehrle)
        Parallelized finite differencing capabilities on LLB clusters Flettner and Dornier (M. Richter)
        Fixed negative times (M. Richter)
        Now using CPickle instead of pickle for speed-up (M. Richter)
        Implementation of three type of design variable normalization (E. J. Wehrle)

------------------------------------------------------------------------------------------------------------------------
To do and ideas
------------------------------------------------------------------------------------------------------------------------
TODO: Constraint handling in PyGMO
TODO  max line length = 79? (PEP8)
TODO every iteration to output window (file)
    to output text file as well as status report.
TODO normalize deltax?
TODO extend to use with other solvers: IPOPT!, CVXOPT (http://cvxopt.org/), pyCOIN (http://www.ime.usp.br/~pjssilva/software.html#pycoin)
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
TODO SBDO
   Adaptive surrogating!
   DoE methods
   Approximation methods
       Polynomial
       Radial basis: scipy.interpolate.Rbf
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
---------------------------------------------------------------------------------------------------
"""


#-----------------------------------------------------------------------------------------------------------------------
# Import necessary Python packages and toolboxes
#-----------------------------------------------------------------------------------------------------------------------
from __future__ import division
import os
import shutil
import sys
import inspect
import pyOpt
import OptAlgOptions
import OptHis2HTML
import OptVideo
import OptResultReport
import numpy as np
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
from Normalize import normalize, denormalize, normalizeSens
from OptPostProc import OptPostProc
try:
    import PyGMO
    from PyGMO.problem import base
    IsPyGMO = True
except:
    IsPyGMO = False

#-----------------------------------------------------------------------------------------------------------------------
# Package details
#-----------------------------------------------------------------------------------------------------------------------
__title__ = "DESign OPTimization in PYthon"
__shorttitle__ = "DesOptPy"
__version__ = "1.2"
__all__ = ['DesOpt']
__author__ = "E. J. Wehrle"
__copyright__ = "Copyright 2015, 2016, E. J. Wehrle"
__email__ = "wehrle(a)tum.de"
__license__ = "GNU Lesser General Public License"
__url__ = 'www.DesOptPy.org'


#-----------------------------------------------------------------------------------------------------------------------
# Print details of DesOptPy
#-----------------------------------------------------------------------------------------------------------------------
def PrintDesOptPy():
    print(__title__+" - "+__shorttitle__)
    print("Version:                 "+__version__)
    print("Internet:                "+__url__)
    print("License:                 "+__license__)
    print("Copyright:               "+__copyright__)
    print("")


global nEval
nEval = 0
#-----------------------------------------------------------------------------------------------------------------------
# PyGMO call (must be outside of main function)
#-----------------------------------------------------------------------------------------------------------------------
HistDat = []
def CombSysEq(SysEq, x, gc):
    f, g = SysEq(np.array(x), gc)
    global nEval
    nEval += 1
    return f, g

#global xLast
xLast = []
def ObjFnEq(SysEq, x, gc):
    global xLast
    if x == xLast:
        global fIt
    else:
        global fIt
        global gIt
        fIt, gIt = CombSysEq(SysEq, x, gc)
        xLast = x
    return fIt

def ConFnEq(SysEq, x, gc):
    global xLast
    if x == xLast:
        global gIt
    else:
        global fIt
        global gIt
        fIt, gIt = CombSysEq(SysEq, x, gc)
        xLast = x
    return gIt

if IsPyGMO:
    class OptSysEqPyGMO(base):

        def __init__(self, SysEq=None, xL=0.0, xU=1.0, gc=[], OptName="OptName", Alg="Alg", DesOptDir="DesOptDir",
                     DesVarNorm="DesVarNorm", StatusReport=False, dim=1, nEval=0, inform=[], OptTime0=[]):
            super(OptSysEqPyGMO, self).__init__(dim)
            self.set_bounds(xL, xU)
            self.__dim = dim
            self.gc = gc
            self.SysEq = SysEq
            self.nEval = nEval
            self.OptName = OptName
            self.Alg = Alg
            self.xL = xL
            self.xU = xU
            self.DesOptDir = DesOptDir
            self.DesVarNorm = DesVarNorm
            self.StatusReport = StatusReport
            self.AlgInst = pyOpt.Optimizer(self.Alg)
            self.inform = inform
            self.OptTime0 = OptTime0

        def _objfun_impl(self, x):
            x = np.array(x)
            f, g = CombSysEq(self.SysEq, x, self.gc)
            gnew = np.zeros(np.shape(g))
            global HistData
            global nEval
            self.nEval = nEval
            if nEval == 1:
                HistData = pyOpt.History(self.OptName, 'w', optimizer=self.AlgInst, opt_prob=self.OptName)
            HistData.write(x, "x")
            HistData.write(f, "obj")
            HistData.write(g, "con")
            if self.StatusReport == 1:
                try:
                    OptHis2HTML.OptHis2HTML(self.OptName, self.AlgInst, self.DesOptDir, self.x0, self.xL, self.xU, self.DesVarNorm, self.inform[0], self.OptTime0)
                except:
                    sys.exit("Error on line "+ str(inspect.currentframe().f_lineno) + " of file "+ __file__ + ": Problem in OptSysEqPyGMO __init__")
            if g is not []:
                for ii in range(np.size(g)):
                    if g[ii] > 0.0:
                        gnew[ii] = 1e4
                fpen = f + sum(gnew)
            else:
                fpen = f
            return(fpen,)


    class OptSysEqConPyGMO(base):
        def __init__(self, SysEq=None, xL=0.0, xU=1.0, gc=[], OptName="OptName", Alg="Alg", DesOptDir="DesOptDir",
                     DesVarNorm="DesVarNorm", StatusReport=False, dim=1, nEval=0, inform=[], OptTime0=[]):
            #                                  (nx,       nxdis, nf,      ng,      ng, tolerance on con violation)
            super(OptSysEqConPyGMO, self).__init__(dim, 0, 1, 2, 0, 1e-4)
            self.set_bounds(xL, xU)
            self.__dim = dim
            self.gc = gc
            self.SysEq = SysEq
            self.nEval = nEval
            self.OptName = OptName
            self.Alg = Alg
            self.xL = xL
            self.xU = xU
            self.DesOptDir = DesOptDir
            self.DesVarNorm = DesVarNorm
            self.StatusReport = StatusReport
            self.AlgInst = pyOpt.Optimizer(self.Alg)
            self.inform = inform
            self.OptTime0 = OptTime0
            self.f = np.zeros(1)
            self.f = np.zeros(np.shape(gc))
        def _objfun_impl(self, x):
            f = ObjFnEq(self.SysEq, x, self.gc)
            self.f = f
            global HistData
            HistData.write(self.f, "obj")
            return(f,)
        def _compute_constraints_impl(self, x):
            #self.nEval += 1
            #print self.nEval
            g = ConFnEq(self.SysEq, x, self.gc)
            #f, g = self.SysEq(np.array(x), self.gc)
            self.g = g
            global HistData
            global nEval
            self.nEval = nEval
            if nEval == 1:
                HistData = pyOpt.History(self.OptName, 'w', optimizer=self.AlgInst, opt_prob=self.OptName)
            HistData.write(x, "x")
            HistData.write(self.g, "con")
            if self.StatusReport == 1:
                try:
                    OptHis2HTML.OptHis2HTML(self.OptName, self.AlgInst, self.DesOptDir, self.x0, self.xL, self.xU, self.DesVarNorm, self.inform[0], self.OptTime0)
                except:
                    sys.exit("Error on line "+ str(inspect.currentframe().f_lineno) + " of file "+ __file__ + ": Problem in OptSysEqPyGMO __init__ with status report")

            return g
'''
Constrained like this...does not work yet...

class OptSysEqPyGMO(base):
    def __init__(self, SysEq=None, xL=0.0, xU=2.0,  gc=[], dim=1, nEval=0, c_dim_=[], c_ineq_dim_=[], c_tol_=0):
        super(OptSysEqPyGMO, self).__init__(dim, 0, c_dim_, c_ineq_dim_, c_tol_)
        self.set_bounds(xL, xU)
        self.__dim = dim
        self.gc = gc
        self.SysEq = SysEq
        self.nEval = nEval

    def _objfun_impl(self, x):
        self.nEval += 1
        f, g = self.SysEq(x, self.gc)
        Data['g'] = g
        output = open("ConPyGMO.pkl", 'wb')
        pickle.dump(Data, output)
        output.close()
        return(f,)

    def _compute_constraints_impl(self,params):
        Data = pickle.load(open("ConPyGMO.pkl"))
        g = Data["g"]
        return g
'''
#-----------------------------------------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------------------------------------
def DesOpt(SysEq, x0, xU, xL, xDis=[], gc=[], hc=[], SensEq=[], Alg="SLSQP", SensCalc="FD", DesVarNorm=True, nf=1,
           deltax=1e-3, StatusReport=False, ResultReport=False, Video=False, nDoE=0, DoE="LHS+Corners", SBDO=False,
           Approx=[], Debug=False, PrintOut=True, OptNameAdd="", AlgOptions=[], Alarm=True):

#-----------------------------------------------------------------------------------------------------------------------
# Define optimization problem and optimization options
#-----------------------------------------------------------------------------------------------------------------------
    """
    :type OptNode: object
    """
    if Debug:
        StatusReport = False
        if StatusReport:
            print("Debug is set to True; overriding StatusReport setting it to False")
            StatusReport = False
        if ResultReport:
            print("Debug is set to True; overriding ResultReport setting it to False")
            ResultReport = False
    computerName = platform.uname()[1]
    operatingSystem = platform.uname()[0]
    architecture = platform.uname()[4]
    nProcessors = str(multiprocessing.cpu_count())
    userName = getpass.getuser()
    OptTime0 = time.time()
    OptNodes = "all"
    MainDir = os.getcwd()
    if operatingSystem  != 'Windows':
        DirSplit = "/"
        homeDir = "/home/"
    else:
        DirSplit = "\\"
        homeDir = "c:\\Users\\"
    OptModel = os.getcwd().split(DirSplit)[-1]
    try:
        OptAlg = eval("pyOpt." + Alg + '()')
        pyOptAlg = True
    except:
        OptAlg = Alg
        pyOptAlg = False
    if hasattr(SensEq, '__call__'):
        SensCalc = "OptSensEq"
        print("Function for sensitivity analysis has been provided, overriding SensCalc to use function")
    else:
        pass
    StartTime = datetime.datetime.now()
    loctime = time.localtime()
    today = time.strftime("%B", time.localtime()) + ' ' + str(loctime[2]) + ', ' + str(loctime[0])
    if SBDO:
        OptNameAdd += "_SBDO"
    OptName = OptModel + OptNameAdd + "_" + Alg + "_" + StartTime.strftime("%Y%m%d%H%M%S")
    LocalRun = True
    ModelDir = os.getcwd()[:-(len(OptModel) + 1)]
    ModelFolder = ModelDir.split(DirSplit)[-1]
    DesOptDir = ModelDir[:-(len(ModelFolder) + 1)]
    ResultsDir = DesOptDir + os.sep + "Results"
    RunDir = DesOptDir + os.sep + "Run"
    try:
        inform
    except NameError:
        inform = ["Running"]
    if LocalRun and Debug is False:
        try: os.mkdir(ResultsDir)
        except: pass
        os.mkdir(ResultsDir + DirSplit + OptName)
        os.mkdir(ResultsDir + os.sep + OptName + os.sep + "ResultReport" + os.sep)
        shutil.copytree(os.getcwd(), RunDir + os.sep + OptName)
    #if SensCalc == "ParaFD":
    #    import OptSensParaFD
    #    os.system("cp -r ParaPythonFn " + homeDir + userName + "/DesOptRun/" + OptName)
    if LocalRun and Debug is False:
        os.chdir("../../Run/" + OptName + "/")
    sys.path.append(os.getcwd())


#-----------------------------------------------------------------------------------------------------------------------
#       Print start-up splash to output screen
#-----------------------------------------------------------------------------------------------------------------------
    if PrintOut:
        print("--------------------------------------------------------------------------------")
        PrintDesOptPy()
        print("Optimization model:      " + OptModel)
        try: print("Optimization algorithm:  " + Alg)
        except: pass
        print("Optimization start:      " + StartTime.strftime("%Y%m%d%H%M"))
        print("Optimization name:       " + OptName)
        print("--------------------------------------------------------------------------------")


#-----------------------------------------------------------------------------------------------------------------------
#       Optimization problem
#-----------------------------------------------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------------------------------------------
#       Define functions: system equation, normalization, etc.
#-----------------------------------------------------------------------------------------------------------------------
    def OptSysEq(x):
        x = np.array(x)  # NSGA2 gives a list back, this makes a float! TODO Inquire why it does this!
        f, g = SysEq(x, gc)
        fail = 0
        global nEval
        nEval += 1
        if StatusReport:
            OptHis2HTML.OptHis2HTML(OptName, OptAlg, DesOptDir, x0, xL, xU, DesVarNorm, inform[0], OptTime0)
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
        dfdx, dgdx = SensEq(x, f, g, gc)
        dfdx = dfdx.reshape(1, len(x))
        fail = 0
        return dfdx, dgdx, fail

    def OptSensEqNorm(xNorm, f, g):
        x = denormalize(xNorm, x0, xL, xU, DesVarNorm)
        dfxdx, dgxdx, fail = OptSensEq(x, f, g)
        dfdx = dfxdx * (xU - xL)
        dgdx = normalizeSens(dgxdx, x0, xL, xU, DesVarNorm)
        # TODO not general for all normalizations! needs to be rewritten; done: check if correct
        # dfdx = normalizeSens(dfxdx, xL, xU, DesVarNorm)
        #if dgxdx != []:
        #    dgdx = dgxdx * np.tile((xU - xL), [len(g), 1])
        #else:
        #    dgdx = []
        return dfdx, dgdx, fail

    def OptSensEqParaFD(x, f, g):
        global nEval
        dfdx, dgdx, nb = OptSensParaFD.Para(x, f, g, deltax, OptName, OptNodes)
        nEval += nb
        fail = 0
        return dfdx, dgdx, fail

    def VectOptSysEq(x):
        f, g, fail = OptSysEq(x)
        np.array(f)
        r = np.concatenate((f, g))
        return r

    def OptSensEqAD(x, f, g):
        import autograd
        OptSysEq_dx = autograd.jacobian(VectOptSysEq)
        drdx = OptSysEq_dx(x)
        dfdx = np.array([drdx[0, :]]).T
        dgdx = drdx[1:,:].T
        fail = 0
        return dfdx, dgdx, fail

    def VectOptSysEqNorm(x):
        f, g, fail = OptSysEqNorm(x)
        np.array(f)
        r = np.concatenate((f, g))
        return r

    def OptSensEqNormAD(x, f, g):
        import autograd
        OptSysEq_dx = autograd.jacobian(VectOptSysEqNorm)
        drdx = OptSysEq_dx(x)
        dfdx = np.array([drdx[0, :]])
        dgdx = drdx[1:,:]
        fail = 0
        return dfdx, dgdx, fail


    def OptSensEqParaFDNorm(xNorm, f, g):
        x = denormalize(xNorm, xL, xU, DesVarNorm)
        dfxdx, dgxdx, fail = OptSensEqParaFD(x, f, g)
        dfdx = normalizeSens(dfxdx, x0, xL, xU, DesVarNorm)
        dgdx = normalizeSens(dgxdx, x0, xL, xU, DesVarNorm)
        # TODO not general for all normalizations! needs to be rewritten, done: check if correct
        # dfdx = dfxdx * (xU - xL)
        #dgdx = dgxdx * (np.tile((xU - xL), [len(g), 1]))
        return dfdx, dgdx, fail

#-----------------------------------------------------------------------------------------------------------------------
#       Surrogate-based optimization (not fully functioning yet!!!!) replace with supy!
#-----------------------------------------------------------------------------------------------------------------------
    if SBDO is not False:
        if nDoE > 0:
            import pyDOE
            try:
                n_gc = len(gc)
            except:
                n_gc = 1
            if DoE == "LHS+Corners":
                xTemp = np.ones(np.size(xL)) * 2
                xSampFF = pyDOE.fullfact(np.array(xTemp, dtype=int))  # Kriging needs boundaries too!!
                xSampLH = pyDOE.lhs(np.size(xL), nDoE)
                xDoE_Norm = np.concatenate((xSampFF, xSampLH), axis=0)
            elif DoE == "LHS":
                xDoE_Norm = pyDOE.lhs(np.size(xL), nDoE)
            else:
                sys.exit("Error on line "+ str(inspect.currentframe().f_lineno) + " of file "+ __file__ + ": DoE misspelled or not supported")
            xDoE = np.zeros(np.shape(xDoE_Norm))
            fDoE = np.zeros([np.size(xDoE_Norm, 0), 1])
            gDoE = np.zeros([np.size(xDoE_Norm, 0), n_gc])
            for ii in range(np.size(xDoE_Norm, 0)):
                xDoE[ii] = denormalize(xDoE_Norm[ii], x0, xL, xU, DesVarNorm)
                fDoEii, gDoEii, fail = OptSysEqNorm(xDoE_Norm[ii])
                fDoE[ii] = fDoEii
                gDoE[ii, :] = gDoEii
            #n_theta = np.size(x0) + 1

            # Approximation
            if Approx == []:
                ApproxType=[[]]*(1+ng)
                for ii in range(len(ApproxType)):
                    ApproxType[ii] = "GaussianProcess"
            elif np.size(Approx)==1:
                ApproxType=[[]]*(1+ng)
                for ii in range(len(ApproxType)):
                    ApproxType[ii] = Approx
            else:
                ApproxType = Approx
            if "GaussianProcess" in ApproxType:
                from sklearn.gaussian_process import GaussianProcess
            if "Reg" in ApproxType:
                #from PolyReg import *
                import PolyReg
            # Approximation of objective
            if ApproxType[0] == "GaussianProcess":
                approx_f = GaussianProcess(regr='quadratic', corr='squared_exponential',
                                           normalize=True, theta0=0.1, thetaL=1e-4, thetaU=1e+1,
                                           optimizer='fmin_cobyla')
            elif ApproxType[0] == "QuadReg":
                approx_f = PolyReg()
            else:
                sys.exit("Error on line "+ str(inspect.currentframe().f_lineno) + " of file "+ __file__ + ": Approximation type misspelled or not supported")
            approx_f.fit(xDoE, fDoE)

            # Approximation of constraints
            gDoEr = np.zeros(np.size(xDoE_Norm, 0))
            approx_g = [[]] * n_gc
            gpRegr = ["quadratic"] * n_gc
            gpCorr = ["squared_exponential"] * n_gc
            for ii in range(n_gc):
                for iii in range(np.size(xDoE_Norm, 0)):
                    gDoEii = gDoE[iii]
                    gDoEr[iii] = gDoEii[ii]
                if ApproxType[ii+1] == "GaussianProcess":
                    approx_g[ii] = GaussianProcess(regr=gpRegr[ii], corr=gpCorr[ii], theta0=0.01,
                                                   thetaL=0.0001, thetaU=10., optimizer='fmin_cobyla')
                elif ApproxType[0] == "QuadReg":
                    approx_g[ii]  = PolyReg()
                else:
                    sys.exit("Error on line "+ str(inspect.currentframe().f_lineno) + " of file "+ __file__ + ": Approximation type misspelled or not supported")
                approx_g[ii].fit(xDoE, gDoEr)
            DoE_Data = {}
            DoE_Data['xDoE_Norm'] = xDoE_Norm
            DoE_Data['gDoE'] = gDoE
            DoE_Data['fDoE'] = fDoE
            output = open(OptName + "_DoE.pkl", 'wb')
            pickle.dump(DoE_Data, output)
            output.close()
        else:
            Data = pickle.load(open("Approx.pkl"))
            approx_f = Data["approx_f"]
            approx_g = Data["approx_g"]

    def ApproxOptSysEq(x):
        f = approx_f.predict(x)
        g = np.zeros(len(gc))
        for ii in range(len(gc)):
            # exec("g[ii], MSE = gp_g"+str(ii)+".predict(x, eval_MSE=True)")
            g[ii] = approx_g[ii].predict(x)
        # sigma = np.sqrt(MSE)
        fail = 0
        return f, g, fail

    def ApproxOptSysEqNorm(xNorm):
        xNorm = xNorm[0:np.size(xL), ]
        x = denormalize(xNorm, x0, xL, xU, DesVarNorm)
        f = approx_f.predict(x)
        g = np.zeros(len(gc))
        for ii in range(len(gc)):
            # exec("g[ii], MSE = gp_g"+str(ii)+".predict(x, eval_MSE=True)")
            g[ii] = approx_g[ii].predict(x)
        # sigma = np.sqrt(MSE)
        fail = 0
        return f, g, fail
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

#-----------------------------------------------------------------------------------------------------------------------
#       pyOpt optimization
#-----------------------------------------------------------------------------------------------------------------------
    if pyOptAlg:
        if SBDO is not False and DesVarNorm in ["xLxU", True, "xLx0", "x0", "xU"]:  # in ["None", None, False]:
            OptProb = pyOpt.Optimization(OptModel, ApproxOptSysEqNorm, obj_set=None)
        elif SBDO is not False and DesVarNorm in ["None", None, False]:
            OptProb = pyOpt.Optimization(OptModel, ApproxOptSysEq, obj_set=None)
        else:
            OptProb = pyOpt.Optimization(OptModel, DefOptSysEq)
        if np.size(x0) == 1:
            OptProb.addVar('x', 'c', value=x0norm, lower=xLnorm, upper=xUnorm)
        elif np.size(x0) > 1:
            for ii in range(np.size(x0)):
                OptProb.addVar('x' + str(ii + 1), 'c', value=x0norm[ii], lower=xLnorm[ii], upper=xUnorm[ii])
        if nf == 1:
            OptProb.addObj('f')
        elif nf > 1:
            for ii in range(nf):
                OptProb.addObj('f' + str(ii + 1))
        if np.size(gc) == 1:
            OptProb.addCon('g', 'i')
            #ng = 1
        elif np.size(gc) > 1:
            for ii in range(len(gc)):
                OptProb.addCon('g' + str(ii + 1), 'i')
            #ng = ii + 1
        if np.size(hc) == 1:
            OptProb.addCon('h', 'i')
        elif np.size(hc) > 1:
            for ii in range(ng):
                OptProb.addCon('h' + str(ii + 1), 'i')
        if AlgOptions == []:
            AlgOptions = OptAlgOptions.setDefault(Alg)
        OptAlg = OptAlgOptions.setUserOptions(AlgOptions, Alg, OptName, OptAlg)
        #if AlgOptions == []:
        #    OptAlg = OptAlgOptions.setDefaultOptions(Alg, OptName, OptAlg)
        #else:
        #    OptAlg = OptAlgOptions.setUserOptions(AlgOptions, Alg, OptName, OptAlg)
        if PrintOut:
            print(OptProb)
        if Alg in ["MMA", "FFSQP", "FSQP", "GCMMA", "CONMIN", "SLSQP", "PSQP", "KSOPT", "ALGENCAN", "NLPQLP", "IPOPT"]:
            if SensCalc == "OptSensEq":
                if DesVarNorm  not in ["None", None, False]:
                    [fOpt, xOpt, inform] = OptAlg(OptProb, sens_type=OptSensEqNorm, store_hst=OptName)
                else:
                    [fOpt, xOpt, inform] = OptAlg(OptProb, sens_type=OptSensEq, store_hst=OptName)
            elif SensCalc == "ParaFD":  # Michi Richter
                if DesVarNorm  not in ["None", None, False]:
                    [fOpt, xOpt, inform] = OptAlg(OptProb, sens_type=OptSensEqParaFDNorm, store_hst=OptName)
                else:
                    [fOpt, xOpt, inform] = OptAlg(OptProb, sens_type=OptSensEqParaFD, store_hst=OptName)
            elif SensCalc == "AD":
                if DesVarNorm  not in ["None", None, False]:
                    [fOpt, xOpt, inform] = OptAlg(OptProb, sens_type=OptSensEqNormAD, store_hst=OptName)
                else:
                    [fOpt, xOpt, inform] = OptAlg(OptProb, sens_type=OptSensEqAD, store_hst=OptName)
            else:  # Here FD (finite differencing)
                [fOpt, xOpt, inform] = OptAlg(OptProb, sens_type=SensCalc, sens_step=deltax, store_hst=OptName)

        elif Alg in ["SDPEN", "SOLVOPT"]:
            [fOpt, xOpt, inform] = OptAlg(OptProb)
        else:
            [fOpt, xOpt, inform] = OptAlg(OptProb, store_hst=OptName)
        if PrintOut:
            try: print(OptProb.solution(0))
            except: pass
        if Alg not in ["PSQP", "SOLVOPT", "MIDACO", "SDPEN", "ralg"] and PrintOut:
            print(OptAlg.getInform(0))

#-----------------------------------------------------------------------------------------------------------------------
#       OpenOpt optimization -- not fully implemented in this framework and not yet working...
#-----------------------------------------------------------------------------------------------------------------------
    elif Alg == "ralg":
        from openopt import NLP
        f, g = lambda x: OptSysEq(x)
        # g = lambda x: OptSysEq(x)[1][0]
        p = NLP(f, x0, c=g, lb=xL, ub=xU, iprint=50, maxIter=10000, maxFunEvals=1e7, name='NLP_1')
        r = p.solve(Alg, plot=0)
        print(OptAlg.getInform(1))

#-----------------------------------------------------------------------------------------------------------------------
#       pyCMAES
#-----------------------------------------------------------------------------------------------------------------------
    elif Alg == "pycmaes":
        print "CMA-ES == not fully implemented in this framework"
        print "    no constraints"
        import cma
        def CMA_ES_ObjFn(x):
            f, g, fail = OptSysEq(x)
            return f
        OptRes = cma.fmin(CMA_ES_ObjFn, x0, sigma0=1)
        xOpt = OptRes[0]
        fOpt = OptRes[1]
        nEval = OptRes[4]
        nIter = OptRes[5]
#-----------------------------------------------------------------------------------------------------------------------
#       MATLAB fmincon optimization -- not fully implemented in this framework and not yet working...
#-----------------------------------------------------------------------------------------------------------------------
    elif Alg == "fmincon":  # not fully implemented in this framework
        def ObjFn(x):
            f, g, fail = OptSysEqNorm(xNorm)
            return f, []
        from mlabwrap import mlab
        mlab._get(ObjFn)
        mlab.fmincon(mlab._get("ObjFn"), x)      # g,h, dgdx = mlab.fmincon(x.T,cg,ch, nout=3)

#-----------------------------------------------------------------------------------------------------------------------
#       PyGMO optimization
#-----------------------------------------------------------------------------------------------------------------------
    elif Alg[:5] == "PyGMO":
        DesVarNorm = "None"
        #print nindiv
        dim = np.size(x0)
        # prob = OptSysEqPyGMO(dim=dim)
        if AlgOptions == []:
            AlgOptions = OptAlgOptions.setDefault(Alg)
        OptAlg = OptAlgOptions.setUserOptions(AlgOptions, Alg, OptName, OptAlg)
        if ng == 0 or AlgOptions.ConstraintHandling == "SimplePenalty":
            prob = OptSysEqPyGMO(SysEq=SysEq, xL=xL, xU=xU, gc=gc, dim=dim, OptName=OptName, Alg=Alg, DesOptDir=DesOptDir,
                                 DesVarNorm=DesVarNorm, StatusReport=StatusReport, inform=inform, OptTime0=OptTime0)
            # prob = problem.death_penalty(prob_old, problem.death_penalty.method.KURI)
            #algo = eval("PyGMO.algorithm." + Alg[6:]+"()")
            #de (gen=100, f=0.8, cr=0.9, variant=2, ftol=1e-06, xtol=1e-06, screen_output=False)
            #NSGAII (gen=100, cr=0.95, eta_c=10, m=0.01, eta_m=10)
            #sga_gray.__init__(gen=1, cr=0.95, m=0.02, elitism=1, mutation=PyGMO.algorithm._algorithm._gray_mutation_type.UNIFORM, selection=PyGMO.algorithm._algorithm._gray_selection_type.ROULETTE, crossover=PyGMO.algorithm._algorithm._gray_crossover_type.SINGLE_POINT)
            #nsga_II.__init__(gen=100, cr=0.95, eta_c=10, m=0.01, eta_m=10)
            #emoa  (hv_algorithm=None, gen=100, sel_m=2, cr=0.95, eta_c=10, m=0.01, eta_m=10)
            #pade  (gen=10, max_parallelism=1, decomposition=PyGMO.problem._problem._decomposition_method.BI, solver=None, T=8, weights=PyGMO.algorithm._algorithm._weight_generation.LOW_DISCREPANCY, z=[])
            #nspso (gen=100, minW=0.4, maxW=1.0, C1=2.0, C2=2.0, CHI=1.0, v_coeff=0.5, leader_selection_range=5, diversity_mechanism=PyGMO.algorithm._algorithm._diversity_mechanism.CROWDING_DISTANCE)
            #corana: (iter=10000, Ts=10, Tf=0.1, steps=1, bin_size=20, range=1)
            #if Alg[6:] in ["de", "bee_colony", "nsga_II", "pso", "pso_gen", "cmaes", "py_cmaes",
            #               "spea2", "nspso", "pade", "sea", "vega", "sga", "sga_gray", "de_1220",
            #               "mde_pbx", "jde"]:
            #    algo.gen = ngen
            #elif Alg[6:] in ["ihs", "monte_carlo", "sa_corana"]:
            #    algo.iter = ngen
            #elif Alg[6:] == "sms_emoa":
            #    print "sms_emoa not working"
            #else:
            #    sys.exit("improper PyGMO algorithm chosen")
            #algo.f = 1
            #algo.cr=1
            #algo.ftol = 1e-3
            #algo.xtol = 1e-3
            #algo.variant = 2
            #algo.screen_output = False
            #if Alg == "PyGMO_de":
            #    algo = PyGMO.algorithm.de(gen=ngen, f=1, cr=1, variant=2,
            #                              ftol=1e-3, xtol=1e-3, screen_output=False)
            #else:
            #    algo = PyGMO.algorithm.de(gen=ngen, f=1, cr=1, variant=2,
            #                              ftol=1e-3, xtol=1e-3, screen_output=False)
            #pop = PyGMO.population(prob, nIndiv)
            #pop = PyGMO.population(prob, nIndiv, seed=13598)  # Seed fixed for random generation of first individuals
            #algo.evolve(pop)
            isl = PyGMO.island(OptAlg, prob, AlgOptions.nIndiv)
            isl.evolve(1)
            isl.join()
            xOpt = isl.population.champion.x
            # fOpt = isl.population.champion.f[0]
            nEval = isl.population.problem.fevals
            nGen = int(nEval/AlgOptions.nIndiv)  # currently being overwritten and therefore not being used
            StatusReport = False  # turn off status report, so not remade (and destroyed) in following call!
            fOpt, gOpt, fail = OptSysEq(xOpt)  # verification of optimal solution as values above are based on penalty!
        elif AlgOptions.ConstraintHandling == "CoevolutionPenalty":
            prob = OptSysEqConPyGMO(SysEq=SysEq, xL=xL, xU=xU, gc=gc, dim=dim, OptName=OptName, Alg=Alg, DesOptDir=DesOptDir,
                                 DesVarNorm=DesVarNorm, StatusReport=StatusReport, inform=inform, OptTime0=OptTime0)
            algo_self_adaptive = PyGMO.algorithm.cstrs_self_adaptive(OptAlg, AlgOptions.gen)
            pop = PyGMO.population(prob, AlgOptions.nIndiv)
            pop = algo_self_adaptive.evolve(pop)
            xOpt = pop.champion.x
            fOpt = pop.champion.f
            global nEval
            #nEval = pop.problem.fevals
            nGen = int(nEval/AlgOptions.nIndiv)
        elif AlgOptions.ConstraintHandling == "MultiobjTrans":
            Prob = OptSysEqConPyGMO(SysEq=SysEq, xL=xL, xU=xU, gc=gc, dim=dim, OptName=OptName, Alg=Alg, DesOptDir=DesOptDir,
                                    DesVarNorm=DesVarNorm, StatusReport=StatusReport, inform=inform, OptTime0=OptTime0)
            ProbNew = (Prob, problem.con2mo.method.OBJ_CSTRS)
            pop = population(prob_mo, pop_size)
            pop = algo.evolve(pop)
        elif AlgOptions.ConstraintHandling == "DeathPenalty":
            Prob = OptSysEqConPyGMO(SysEq=SysEq, xL=xL, xU=xU, gc=gc, dim=dim, OptName=OptName, Alg=Alg, DesOptDir=DesOptDir,
                                    DesVarNorm=DesVarNorm, StatusReport=StatusReport, inform=inform, OptTime0=OptTime0)
            ProbNew = problem.death_penalty(Prob, problem.death_penalty.method.SIMPLE)
            isl = island(algo, ProbNew, AlgOptions.nIndiv)
            isl.evolve(1)
            isl.join()
            xOpt = isl.population.champion.x
            fOpt = isl.population.champion.f
            nEval = isl.population.problem.fevals
        elif AlgOptions.ConstraintHandling == "SelfAdaptivePenalty":
            algo_self_adaptive = algorithm.cstrs_self_adaptive(algo, n_gen)
            pop = population(Prob, pop_size)
            pop = algo_self_adaptive.evolve(pop)
            xOpt = pop.champion.x
            fOpt = pop.champion.f
            nEval = pop.problem.fevals
        elif AlgOptions.ConstraintHandling == "Immune":
            sys.exit("Error on line "+ str(inspect.currentframe().f_lineno) + " of file "+ __file__ + ": Constraint handling immune for PyGMO not yet implemented in DesOptPy")
        elif AlgOptions.ConstraintHandling == "Repair":
            sys.exit("Error on line "+ str(inspect.currentframe().f_lineno) + " of file "+ __file__ + ": Constraint handling repair for PyGMO not yet implemented in DesOptPy")
        else:
            sys.exit("Error on line "+ str(inspect.currentframe().f_lineno) + " of file "+ __file__ + ": Constraint handling for PyGMO not recognized")


#-----------------------------------------------------------------------------------------------------------------------
#        SciPy optimization
#-----------------------------------------------------------------------------------------------------------------------
    elif Alg[:5] == "scipy":
        import scipy.optimize as sciopt
        bounds = [[]]*len(x0)
        for ii in range(len(x0)):
            bounds[ii] = (xL[ii], xU[ii])
        print bounds
        if Alg[6:] == "de":
            sciopt.differential_evolution(DefOptSysEq, bounds, strategy='best1bin',
                                          maxiter=None, popsize=15, tol=0.01, mutation=(0.5, 1),
                                          recombination=0.7, seed=None, callback=None, disp=False,
                                          polish=True, init='latinhypercube')

#-----------------------------------------------------------------------------------------------------------------------
#        Simple optimization algorithms to demonstrate use of custom algorithms
#-----------------------------------------------------------------------------------------------------------------------
    #TODO: add history to these
    elif Alg == "SteepestDescentSUMT":
        from CustomAlgs import SteepestDescentSUMT
        fOpt, xOpt, nIter, nEval = SteepestDescentSUMT(DefOptSysEq, x0, xL, xU)
    elif Alg == "NewtonSUMT":
        from CustomAlgs import NewtonSUMT
        fOpt, xOpt, nIter, nEval = NewtonSUMT(DefOptSysEq, x0, xL, xU)

#-----------------------------------------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------------------------------------
    else:
        sys.exit("Error on line "+ str(inspect.currentframe().f_lineno) + " of file "+ __file__ + ": algorithm misspelled or not supported")

#-----------------------------------------------------------------------------------------------------------------------
#       Optimization post-processing
#-----------------------------------------------------------------------------------------------------------------------
    if StatusReport == 1:
        OptHis2HTML.OptHis2HTML(OptName, OptAlg, DesOptDir, x0, xL, xU, DesVarNorm, inform.values()[0], OptTime0)
    OptTime1 = time.time()
    loctime0 = time.localtime(OptTime0)
    hhmmss0 = time.strftime("%H", loctime0)+' : '+time.strftime("%M", loctime0)+' : '+time.strftime("%S", loctime0)
    loctime1 = time.localtime(OptTime1)
    hhmmss1 = time.strftime("%H", loctime1)+' : '+time.strftime("%M", loctime1)+' : '+time.strftime("%S", loctime1)
    diff = OptTime1 - OptTime0
    h0, m0, s0 = (diff // 3600), int((diff / 60) - (diff // 3600) * 60), diff % 60
    OptTime = "%02d" % (h0) + " : " + "%02d" % (m0) + " : " + "%02d" % (s0)

#-----------------------------------------------------------------------------------------------------------------------
#       Read in results from history files
#-----------------------------------------------------------------------------------------------------------------------
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

#-----------------------------------------------------------------------------------------------------------------------
#       Convert all data to numpy arrays
#-----------------------------------------------------------------------------------------------------------------------
    fIter = np.asarray(fIter)
    xIter = np.asarray(xIter)
    gIter = np.asarray(gIter)
    gGradIter = np.asarray(gGradIter)
    fGradIter = np.asarray(fGradIter)

#-----------------------------------------------------------------------------------------------------------------------
# Denormalization of design variables
#-----------------------------------------------------------------------------------------------------------------------
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
                xIter[ii] = denormalize(xIterNorm[ii], x0, xL, xU, DesVarNorm)
        except:
            x0norm = []
            xIterNorm = []
            xOptNorm = []
    nIter = np.size(fIter,0)
    if np.size(fIter) > 0:
        if len(fIter[0]) > 0:
            fIterNorm = fIter / fIter[0]  # fIterNorm=(fIter-fIter[nEval-1])/(fIter[0]-fIter[nEval-1])
        else:
            fIterNorm = fIter
    else:
        fIterNorm = []

#-----------------------------------------------------------------------------------------------------------------------
#  Active constraints for use in the calculation of the Lagrangian multipliers and optimality criterion
#-----------------------------------------------------------------------------------------------------------------------
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
    if len(xL_Active)==0:
        xActive = xU_Active
    elif len(xU_Active)==0:
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
    if len(xL_Active)==0:
        xLU_Active = xU_Active
    elif len(xU_Active)==0:
        xLU_Active = xL_Active
    else:
        xLU_Active = np.concatenate((xL_Active, xU_Active))
    #TODO needs to be investigated for PyGMO!
    # are there nonlinear constraints active, in case equality constraints are added later, this must also be added
    if np.size(gc) > 0: #and Alg[:5] != "PyGMO":
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
    if len(xLU_Active)==0:
        g_xLU_OptActive = gOptActive
    elif len(gOptActive)==0:
        g_xLU_OptActive = xLU_Active
    else:
        if np.size(xLU_Active) == 1 and np.size(gOptActive) == 1:
            g_xLU_OptActive = np.array([xLU_Active, gOptActive])
        else:
            g_xLU_OptActive = np.concatenate((xLU_Active, gOptActive))
    if np.size(fGradIter) > 0:  # Iteration data present
        #fGradOpt = fGradIter[nIter - 1]
        fGradOpt = fGradIter[-1]
        if np.size(gc) > 0: # Constrained problem
            gGradOpt = gGradIter[nIter - 1]
            gGradOpt = gGradOpt.reshape([ng, nx]).T
            gGradOptActive = gGradOpt[:, gOptActiveIndex == True]
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
                g_xLU_GradOptActive = np.concatenate((gGradOptActive, xGradActive), axis=1)
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

#-----------------------------------------------------------------------------------------------------------------------
#   §      Post-processing of optimization solution
#-----------------------------------------------------------------------------------------------------------------------
    lambda_c, SPg, OptRes, Opt1Order, KKTmax = OptPostProc(fGradOpt, gc, gOptActiveIndex, g_xLU_GradOptActive,
                                                           c_xLU_OptActive, c_xLU_ActiveType, DesVarNorm)

#-----------------------------------------------------------------------------------------------------------------------
#   §      Save optimization solution to file
#-----------------------------------------------------------------------------------------------------------------------
    global nEval
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
    OptSolData['fGradOpt'] = fGradOpt
    OptSolData['gGradOpt'] = gGradOpt
    OptSolData['OptName'] = OptName
    OptSolData['OptModel'] = OptModel
    OptSolData['OptTime'] = OptTime
    OptSolData['loctime'] = loctime
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


#-----------------------------------------------------------------------------------------------------------------------
#   §    Save in Python format
#-----------------------------------------------------------------------------------------------------------------------
    output = open(OptName + "_OptSol.pkl", 'wb')
    pickle.dump(OptSolData, output)
    output.close()
    np.savez(OptName + "_OptSol", x0, xOpt, xOptNorm, xIter, xIterNorm, xIter, xIterNorm, fOpt, fIter, fIterNorm, gIter,
             gMaxIter, gOpt, fGradIter, gGradIter,
             fGradOpt, gGradOpt, OptName, OptModel, OptTime, loctime, today, computerName, operatingSystem,
             architecture, nProcessors, userName, Alg, DesVarNorm, KKTmax)

#-----------------------------------------------------------------------------------------------------------------------
#   §5.2    Save in MATLAB format
#-----------------------------------------------------------------------------------------------------------------------
    #OptSolData['OptAlg'] = []
    spio.savemat(OptName + '_OptSol.mat', OptSolData, oned_as='row')



#-----------------------------------------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------------------------------------
    os.chdir(MainDir)
    if LocalRun and Debug is False:
        try:
            shutil.move(RunDir + os.sep + OptName,
                        ResultsDir + os.sep + OptName + os.sep + "RunFiles" + os.sep)
        # except WindowsError:
        except:
            print "Run files not deleted from " + RunDir + os.sep + OptName
            shutil.copytree(RunDir + os.sep + OptName,
                            ResultsDir + os.sep + OptName + os.sep + "RunFiles" + os.sep)

#-----------------------------------------------------------------------------------------------------------------------
#   §    Graphical post-processing
#-----------------------------------------------------------------------------------------------------------------------
    if ResultReport:
        print("Entering preprocessing mode")
        OptResultReport.OptResultReport(OptName, OptAlg, DesOptDir, diagrams=1, tables=1, lyx=1)
        # try: OptResultReport.OptResultReport(OptName, diagrams=1, tables=1, lyx=1)
        # except: print("Problem with generation of Result Report. Check if all prerequisites are installed")
    if Video:
        OptVideo.OptVideo(OptName)


#-----------------------------------------------------------------------------------------------------------------------
#   § Print out
#-----------------------------------------------------------------------------------------------------------------------
    if PrintOut:
        print("")
        print("--------------------------------------------------------------------------------")
        print("Optimization results - DesOptPy")
        print("--------------------------------------------------------------------------------")
        print("Optimization with " + Alg)
        print("f* = " + str(fOpt))
        print("g* = " + str(gOpt))
        print("x* = " + str(xOpt.T))
        print("Lagrangian multipliers = " + str(lambda_c))
        print("Shadow prices = " + str(SPg))
        print("Time of optimization [h:m:s] = " + OptTime)
        try:
            print("nGen = " + str(nGen))
        except:
            print("nIter = " + str(nIter))
        print("nEval = " + str(nEval))
        if Debug is False:
            print("See results directory: " + ResultsDir + os.sep + OptName + os.sep)
        else:
            print("Local run, no results saved to results directory")
        print("--------------------------------------------------------------------------------")
        if operatingSystem == "Linux" and Alarm:
            t = 1
            freq = 350
            os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (t, freq))
    return xOpt, fOpt, OptSolData

#-----------------------------------------------------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    PrintDesOptPy()
    print("Start DesOptPy from file containing system equations!")
    print("See documentation for further help.")
