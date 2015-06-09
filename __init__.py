# -*-  coding: utf-8 -*-
'''
---------------------------------------------------------------------------------------------------
Title:          __init__.py
Date:           April 6, 2015
Revision:       2.0
Units:          Unitless
Author:         E. J. Wehrle
Contributors:   S. Rudolph, F. Wachter, M. Richter
---------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------
Description
---------------------------------------------------------------------------------------------------
DesOptPy -- DESign OPTimization for PYthon -- is an optimization toolbox for Python

---------------------------------------------------------------------------------------------------
Change log
---------------------------------------------------------------------------------------------------
2.0 alpha -- April 6, 2015 -- Preperation for release as open source
    Main file renamed from DesOpt.py to __init__.pyCMA
    Position of file (and rest of package) changed to /usr/local/lib/python2.7/dist-packages/DesOptPy/ from ~/DesOpt/.DesOptPy
1.3 -- April 3, 2015
    Algorithm options added to call, only for pyOpt algorithms
    Discrete variables added analoguous to APSIS (Schatz & Wehrle & Baier 2014), though without the
        reduction of design variables by 1, example in AxialBar

1.2 -- February 15, 2015
    PyGMO solver
    New setup of optimization problem to ease integration of new algorithms
    Presentation in alpha version---not finished
    Removed algorithm support for pyCMA as two interfaces to CMA-ES in PyGMO

1.1:
    Seconds in OptName
    gc in SysEq and SensEq

1.0:
    Changed configuration so that SysEq.py imports and calls DesOpt!
    LLB -> FGCM

0.5:
    SBDO functions with Latin hypercube and Gaussian process (i. e. Kriging) but not well (E. J. Wehrle)
    Parallelized finite differencing capabilities on LLB clusters Flettner and Dornier (M. Richter)
    Fixed negative times (M. Richter)
    Now using CPickle instead of pickle for speed-up (M. Richter)
    Implementation of three type of design variable normalization (E. J. Wehrle)

---------------------------------------------------------------------------------------------------
To do and ideas
---------------------------------------------------------------------------------------------------
Need to do immediately
X TODO give optimization parameters and optimizations in SysEq?
X TODO remove gc? needed for proper lagrangians!!!
X    IDEA: if not give = 0, check size of g, possible without calculation?
TDOO: Nightly automatic benchmark run to make sure everything working?
TODO  max line lenght = 79? (PEP8)
?done? TODO non-normalized design vector NLPQLP
TODO every iteration to output window (file)
    to output text file as well as status report.
TODO normalize deltax?
TODO extend to other solvers: IPOPT, CVXOPT (http://cvxopt.org/), pyCOIN (http://www.ime.usp.br/~pjssilva/software.html#pycoin) pyGMO, openopt, etc
TODO extend to fmincon!!!!!
TODO Langragian multiplier for one-dimensional optimization, line 423
TODO gGradIter is forced into
TODO sens_mode='pgc'
TODO pyTables? for outputs, readable in Excel
TODO range to xrange (xrange is faster)
TODO Evolutaionary strategies
    TODO add PyEvolve
    TODO add inspyred
    TODO add DEAP
    TODO add pybrain
X    TODO add PyGMO
TODO Disrete optimization:
    TODO python-zibopt
X    TODO APSIS
TODO Multiobjective
    http://www.midaco-solver.com/index.php/more/multi-objective
    http://openopt.org/interalg
x    http://esa.github.io/pygmo/
TODO excel report with "from xlwt import Workbook"
TODO SBDO
   DoE methods:
x       LHS, S...
   Approximation methods
x       Kriging
       Polynomial
       Radial basis: scipy.interpolate.Rbf
   Change optimization name: $Model_SBDO_$Alg_
TODO Examples
   SIMP with ground structure
   Multimaterial design (SIMP)
   Shape and sizing optimization of a space truss
   Shape and sizing optimization of a space truss
   Robustness
   Reliability
   Analytical sensitivities,
   Analytical sensitivities for nonlinear dynamic problems
   Discrete problems
   gGradIter in dgdxIter
   fGradIter in dfdxIter etc
TODO His2HTML
    TODO His2HTML: after first evaluation, make html!
    TODO His2HTML: last iteration not shown?
    TODO His2HTML: target -> objective!
    TODO His2HTML: line for g=0
    TODO His2HTML: des variable and constraint number +1 (not pythonish 0)
    TODO His2HTML: all other algorithms do not result in error!
    TODO His2HTML: for zeroth order (non-gradient-based) algorithms (specifically COBYLA, NSGA2)
    TODO His2HTML: integrate NSGA2, out files in DesOptRun, read in as text etc.
    TODO His2HTML: Normalized AND Demormalized design variables!
    TODO His2HTML: values not always on
TODO OptResultReport: Finish presentation!
TODO OptResultReport: Files in /.DesOptPy/
-TODO OptResultReport: Gradients gGradOpt plot in Lyx
-TODO OptResultReport: Gradients gGradOpt plot only integers
-TODO OptResultReport: smart sizing of diagrams with legends
-TODO OptResultReport: postprocessing for unconstrained problems
TODO OptResultReport: with DesVarNorm = "None"!!!!!
TODO OptResultReport: Speed-up possible?
TODO OptResultReport: change file repository to DesOpt/.DesOptPy/
'''
__title__ = "DesOptPy"
__version__ = "2.0 alpha"
__all__ = ['DesOpt']
__author__ = "E. J. Wehrle"
__copyright__ = "Copyright 2015, E. J. Wehrle"
__email__ = "wehrle(a)tum.de"
__license__ = "GNU Lesser General Public License"
__url__ = 'http://inspyred.github.com'

# -------------------------------------------------------------------------------------------------
# Import necessary Python packages and toolboxes
# -------------------------------------------------------------------------------------------------
import os
import shutil
import sys
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
from Normalize import normalize, denormalize
from OptPostProc import OptPostProc
try:
    import PyGMO
    from PyGMO.problem import base
    IsPyGMO = True
except:
    IsPyGMO = False


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
def DesOpt(SysEq, x0, xU, xL, gc=[], hc=[], SensEq=[], Alg="SLSQP", SensCalc="FD", DesVarNorm=True,
           deltax=1e-3, OptStatus=False, OptPostProcess=False, Video=False, DoE=False, SBDO=False,
           Debug=False, PrintOut=True, OptNameAdd=""):
'''

if IsPyGMO is True:
    class OptSysEqPyGMO(base):
        def __init__(self, SysEq=None, xL=0.0, xU=2.0,  gc=[], dim=1, nEval=0):
            super(OptSysEqPyGMO, self).__init__(dim)
            self.set_bounds(xL, xU)
            self.__dim = dim
            self.gc = gc
            self.SysEq = SysEq
            self.nEval = nEval

        def _objfun_impl(self, x):
            self.nEval += 1
            f, g = self.SysEq(x, self.gc)
            gnew = np.zeros(np.shape(g))
            if g is not []:
                for ii in range(np.size(g)):
                    if g[ii] > 0.0:
                        gnew[ii] = 1e4
                # gnew[g<0] = 0.0
                fpen = f + sum(gnew)
                # print fpen
                # gamma = 1e4
                # f = f - gamma*sum(1./np.array(g))
                # g = []
            # fNew = [f,g]
            else:
                fpen = f
            return(fpen,)


def DesOpt(SysEq, x0, xU, xL, xDis=[], gc=[], hc=[], SensEq=[], Alg="SLSQP", SensCalc="FD", DesVarNorm=True,
           deltax=1e-3, StatusReport=False, ResultReport=False, Video=False, DoE=False, SBDO=False,
           Debug=False, PrintOut=True, OptNameAdd="", AlgOptions=[]):

# -------------------------------------------------------------------------------------------------
# Define optimization problem and optimization options
# -------------------------------------------------------------------------------------------------
    """
    :type OptNode: object
    """
    if Debug is True:
        print "Debug is set to False; overriding ResultReport and StatusReport"
        StatusReport = False
        ResultReport = False
    computerName = platform.uname()[1]
    operatingSystem = platform.uname()[0]
    architecture = platform.uname()[4]
    nProcessors = str(multiprocessing.cpu_count())
    userName = getpass.getuser()
    OptTime0 = time.time()
    OptNodes = "all"
    MainDir = os.getcwd()
    if operatingSystem == "Linux":
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
    StartTime = datetime.datetime.now()
    loctime = time.localtime()
    today = time.strftime("%B", time.localtime()) + ' ' + str(loctime[2]) + ', ' + str(loctime[0])
    OptName = OptModel + OptNameAdd + "_" + Alg + "_" + StartTime.strftime("%Y%m%d%H%M%S")
    global nEval
    nEval = 0
    LocalRun = True
    ModelDir = os.getcwd()[:-(len(OptModel)+1)]
    ModelFolder = ModelDir.split(DirSplit)[-1]
    DesOptDir = ModelDir[:-(len(ModelFolder)+1)]
    ResultsDir = DesOptDir + os.sep + "Results"
    RunDir = DesOptDir + os.sep + "Run"
    if LocalRun is True and Debug is False:
        try: os.mkdir(ResultsDir)
        except: pass
        os.mkdir(ResultsDir + DirSplit + OptName)
        os.mkdir(ResultsDir + os.sep + OptName + os.sep + "ResultReport" + os.sep)
        shutil.copytree(os.getcwd(),RunDir + os.sep + OptName)
    if SensCalc == "ParaFD":
        import OptSensParaFD
        os.system("cp -r ParaPythonFn " + homeDir + userName + "/DesOptRun/" + OptName)
    if LocalRun is True and Debug is False:
        os.chdir("../../Run/" + OptName + "/")
    sys.path.append(os.getcwd())


# -------------------------------------------------------------------------------------------------
#       Print start-up splash to output screen
# -------------------------------------------------------------------------------------------------
    print("------------------------------------------------------------------------------")
    print("E. J. Wehrle")
    print("Fachgebiet Computational Mechanics")
    print("Technische Universität München")
    print("wehrle@tum.de")
    print("")
    print("DESign OPTimization in PYthon")
    print("Version: DesOptPy 1.2")
    print("")
    print("Optimization model:      " + OptModel)
    try: print("Optimization algorithm:  " + "pyOpt." + OptAlg.name)
    except: pass
    print("Optimization start:      " + StartTime.strftime("%Y%m%d%H%M"))
    print("Optimization name:       " + OptName)
    print("------------------------------------------------------------------------------")

# -------------------------------------------------------------------------------------------------
#       Optimization problem
# -------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------
#       Define functions: system equation, normalization, etc.
# -------------------------------------------------------------------------------------------------
    def OptSysEq(x):
        x = np.array(x)  # NSGA2 gives a list back, this makes a float! TODO Iquire why it does this!
        f, g = SysEq(x, gc)
        fail = 0
        global nEval
        nEval += 1
        if StatusReport == 1:
            OptHis2HTML.OptHis2HTML(OptName, OptAlg,DesOptDir )
        if xDis is not []:
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
        xNorm = np.array(xNorm)  # NSGA2 gives a list back, this makes a float! TODO Iquire why it does this!
        x = denormalize(xNorm, xL, xU, DesVarNorm)
        f, g, fail = OptSysEq(x)
        return f, g, fail

    def OptPenSysEq(x):
        f, g, fail = OptSysEq(x)
        fpen = f
        return fpen

    def OptSensEq(x, f, g):
        dfdx, dgdx = SensEq(x, f, g, gc)
        fail = 0
        return dfdx, dgdx, fail

    def OptSensEqNorm(xNorm, f, g):
        x = denormalize(xNorm, xL, xU, DesVarNorm)
        dfxdx, dgxdx, fail = OptSensEq(x, f, g)
        dfdx = dfxdx * (xU - xL)
        # TODO not general for all normalizations! needs to be rewritten
        if dgxdx != []:
            dgdx = dgxdx * np.tile((xU - xL), [len(g), 1])
            # TODO not general for all normalizations! needs to be rewritten
        else:
            dgdx = []
        return dfdx, dgdx, fail

    def OptSensEqParaFD(x, f, g):
        global nEval
        dfdx, dgdx, nb = OptSensParaFD.Para(x, f, g, deltax, OptName, OptNodes)
        nEval += nb
        fail = 0
        return dfdx, dgdx, fail

    def OptSensEqParaFDNorm(xNorm, f, g):
        x = denormalize(xNorm, xL, xU, DesVarNorm)
        dfxdx, dgxdx, fail = OptSensEqParaFD(x, f, g)
        dfdx = dfxdx * (xU - xL)
        # TODO not general for all normalizations! needs to be rewritten
        dgdx = dgxdx * (np.tile((xU - xL), [len(g), 1]))
        # TODO not general for all normalizations! needs to be rewritten
        return dfdx, dgdx, fail

# -------------------------------------------------------------------------------------------------
#       Surrogate-based optimization (not fully funtioning yet!!!!)
# -------------------------------------------------------------------------------------------------
    # TODO SBDO in a seperate file???
    if SBDO is not False:
        if DoE > 0:
            import pyDOE
            try:
                n_gc = len(gc)
            except:
                n_gc = 1
            SampleCorners = True
            if SampleCorners is True:
                xTemp = np.ones(np.size(xL)) * 2
                xSampFF = pyDOE.fullfact(np.array(xTemp, dtype=int))  # Kriging needs boundaries too!!
                xSampLH = pyDOE.lhs(np.size(xL), DoE)
                xDoE_Norm = np.concatenate((xSampFF, xSampLH), axis=0)
            else:
                xDoE_Norm = pyDOE.lhs(np.size(xL), DoE)
            xDoE = np.zeros(np.shape(xDoE_Norm))
            fDoE = np.zeros([np.size(xDoE_Norm, 0), 1])
            gDoE = np.zeros([np.size(xDoE_Norm, 0), n_gc])
            for ii in range(np.size(xDoE_Norm, 0)):
                xDoE[ii] = denormalize(xDoE_Norm[ii], xL, xU, DesVarNorm)
                fDoEii, gDoEii, fail = OptSysEqNorm(xDoE_Norm[ii])
                fDoE[ii] = fDoEii
                gDoE[ii, :] = gDoEii
            n_theta = np.size(x0) + 1
            ApproxObj = "QuadReg"
            ApproxObj = "GaussianProcess"
            if ApproxObj == "GaussianProcess":
                from sklearn.gaussian_process import GaussianProcess

                approx_f = GaussianProcess(regr='quadratic', corr='squared_exponential',
                                           normalize=True, theta0=0.1, thetaL=1e-4, thetaU=1e+1,
                                           optimizer='fmin_cobyla')
            elif ApproxObj == "QuadReg":
                #                from PolyReg import *
                approx_f = PolyReg()
            approx_f.fit(xDoE, fDoE)
            from sklearn.gaussian_process import GaussianProcess

            gDoEr = np.zeros(np.size(xDoE_Norm, 0))
            approx_g = [[]] * n_gc
            gpRegr = ["quadratic"] * n_gc
            gpCorr = ["squared_exponential"] * n_gc
            for ii in range(n_gc):
                for iii in range(np.size(xDoE_Norm, 0)):
                    gDoEii = gDoE[iii]
                    gDoEr[iii] = gDoEii[ii]
                approx_g[ii] = GaussianProcess(regr=gpRegr[ii], corr=gpCorr[ii], theta0=0.01,
                                               thetaL=0.0001, thetaU=10., optimizer='fmin_cobyla')
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
        x = denormalize(xNorm, xL, xU, DesVarNorm)
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
        [x0norm, xLnorm, xUnorm] = normalize(x0, xL, xU, DesVarNorm)
        DefOptSysEq = OptSysEqNorm
    nx = np.size(x0)
    ng = np.size(gc)

# -------------------------------------------------------------------------------------------------
#       pyOpt optimization
# -------------------------------------------------------------------------------------------------
    if pyOptAlg is True:
        if SBDO is not False and DesVarNorm  in ["xLxU", True, "xLx0", "x0", "xU"]: #in ["None", None, False]:
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
            OptProb.addObj('f')
        if np.size(gc) == 1:
            OptProb.addCon('g', 'i')
            ng = 1
        elif np.size(gc) > 1:
            for ii in range(len(gc)):
                OptProb.addCon('g' + str(ii + 1), 'i')
            ng = ii + 1
        if np.size(hc) == 1:
            OptProb.addCon('h', 'i')
        elif np.size(hc) > 1:
            for ii in range(ng):
                OptProb.addCon('h' + str(ii + 1), 'i')
        if AlgOptions == []:
            OptAlg = OptAlgOptions.setDefaultOptions(Alg, OptName, OptAlg)
        else:
            OptAlg = OptAlgOptions.setUserOptions(AlgOptions, Alg, OptName, OptAlg)
        print(OptProb)
        if Alg in ["MMA", "IPOPT", "GCMMA", "CONMIN", "SLSQP", "PSQP", "KSOPT", "ALGENCAN", "NLPQLP"]:
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
            else:
                [fOpt, xOpt, inform] = OptAlg(OptProb, sens_type=SensCalc, sens_step=deltax, store_hst=OptName)

        elif Alg in ["SDPEN", "SOLVOPT"]:
            [fOpt, xOpt, inform] = OptAlg(OptProb)
        else:
            [fOpt, xOpt, inform] = OptAlg(OptProb, store_hst=OptName)
        try: print(OptProb.solution(0))
        except: pass
        if Alg not in ["PSQP", "SOLVOPT", "MIDACO", "SDPEN", "ralg"]:
            print(OptAlg.getInform(0))
    elif Alg == "ralg":  # not fully implemented in this framework
        from openopt import NLP
        f, g = lambda x: OptSysEq(x)
        # g = lambda x: OptSysEq(x)[1][0]
        p = NLP(f, x0, c=g, lb=xL, ub=xU, iprint=50, maxIter=10000, maxFunEvals=1e7, name='NLP_1')
        r = p.solve(Alg, plot=0)
        print(OptAlg.getInform(1))
    elif Alg == "fmincon":  # not fully implemented in this framework
        def ObjFn(x):
            f, g, fail = OptSysEqNorm(xNorm)
            return f, []
        from mlabwrap import mlab
        mlab._get(ObjFn)
        mlab.fmincon(mlab._get("ObjFn"), x)      # g,h, dgdx = mlab.fmincon(x.T,cg,ch, nout=3)
#    elif Alg == "CMA-ES":
#        print "CMA-ES == not fully implemented in this framework"
#        print "    no constraints"
#        import cma
#
#        def CMA_ES_ObjFn(x):
#            f, g, fail = OptSysEq(x)
#            return f
#        OptRes = cma.fmin(CMA_ES_ObjFn, x0, sigma0=1)
#        xOpt = OptRes[0]
#        fOpt = OptRes[1]
#        nEval = OptRes[4]
#        nIter = OptRes[5]
    elif Alg[:5] == "PyGMO":
        DesVarNorm = "None"
        ngen = 500
        nindiv = max(nx*3, 7)
        print nindiv
        dim = np.size(x0)
        # prob = OptSysEqPyGMO(dim=dim)
        prob = OptSysEqPyGMO(SysEq=SysEq, xL=xL, xU=xU, gc=gc, dim=dim)
        # prob = problem.death_penalty(prob_old, problem.death_penalty.method.KURI)
        if Alg[6:] in ["de", "bee_colony", "nsga_II", "pso", "pso_gen", "cmaes", "py_cmaes",
                       "spea2", "nspso", "pade", "sea", "vega", "sga", "sga_gray", "de_1220",
                       "mde_pbx", "jde"]:
            algo = eval("PyGMO.algorithm." + Alg[6:] + "(gen=ngen)")
        elif Alg[6:] in ["ihs", "monte_carlo", "sa_corana"]:
            algo = eval("PyGMO.algorithm." + Alg[6:] + "(iter=ngen)")
        elif Alg[6:] == "sms_emoa":
            print "sms_emoa not working"

        if Alg == "PyGMO_de":
            algo = PyGMO.algorithm.de(gen=ngen, f=1, cr=1, variant=2,
                                      ftol=1e-3, xtol=1e-3, screen_output=False)
        else:
            algo = PyGMO.algorithm.de(gen=ngen, f=1, cr=1, variant=2,
                                      ftol=1e-3, xtol=1e-3, screen_output=False)
        pop = PyGMO.population(prob, nindiv, seed=13598)  # Seed fixed for random generation of first individuals
        pop = algo.evolve(pop)

        isl = PyGMO.island(algo, prob, nindiv)
        isl.evolve(1)
        isl.join()
        xOpt = isl.population.champion.x
        fOpt = isl.population.champion.f[0]
        nEval = isl.population.problem.fevals
        fOpt, gOpt, fail = OptSysEq(xOpt)
        xIter = []
        fIter = []
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
    elif Alg == "SteepestDescentSUMT":
        from CustomAlgs import SteepestDescentSUMT
        fOpt, xOpt, nIter, nEval = SteepestDescentSUMT(DefOptSysEq, x0, xL, xU)
        print nEval
        print nIter
        print fOpt
        print xOpt
    elif Alg == "NewtonSUMT":
        from CustomAlgs import NewtonSUMT
        fOpt, xOpt, nIter, nEval = NewtonSUMT(DefOptSysEq, x0, xL, xU)
        print nEval
        print nIter
        print fOpt
        print xOpt
    else:
        print "algorithm spelled wrong or not supported"

# -------------------------------------------------------------------------------------------------
#       Optimization post-processing
# -------------------------------------------------------------------------------------------------
    if StatusReport == 1:
            OptHis2HTML.OptHis2HTML(OptName, OptAlg,DesOptDir )
    OptTime1 = time.time()
    loctime0 = time.localtime(OptTime0)
    hhmmss0 = time.strftime("%H", loctime0) + ' : ' + time.strftime("%M", loctime0) + ' : ' + time.strftime("%S", loctime0)
    loctime1 = time.localtime(OptTime1)
    hhmmss1 = time.strftime("%H", loctime1) + ' : ' + time.strftime("%M", loctime1) + ' : ' + time.strftime("%S", loctime1)
    diff = OptTime1 - OptTime0
    h0, m0, s0 = (diff // 3600), int((diff / 60) - (diff // 3600) * 60), diff % 60
    OptTime = "%02d" % (h0) + " : " + "%02d" % (m0) + " : " + "%02d" % (s0)

# -------------------------------------------------------------------------------------------------
#       Optimization post-processing
# -------------------------------------------------------------------------------------------------
#    try:
#        SuAll = np.load("ShadowUncertainty.npy")
#        Fuzzy = True
#    except:
#        Fuzzy = False

# -------------------------------------------------------------------------------------------------
#       Read in results from history files
# -------------------------------------------------------------------------------------------------
    if pyOptAlg == True:
        OptHist = pyOpt.History(OptName, "r")
        fAll = OptHist.read([0, -1], ["obj"])[0]["obj"]
        xAll = OptHist.read([0, -1], ["x"])[0]["x"]
        gAll = OptHist.read([0, -1], ["con"])[0]["con"]
        if Alg == "NLPQLP":
            gAll = [x * -1 for x in gAll]
        gGradIter = OptHist.read([0, -1], ["grad_con"])[0]["grad_con"]
        fGradIter = OptHist.read([0, -1], ["grad_obj"])[0]["grad_obj"]
        failIter = OptHist.read([0, -1], ["fail"])[0]["fail"]
        if Alg == "COBYLA" or Alg == "NSGA2":
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
#                if Fuzzy is True:
#                    try:
#                        SuIter[ii] = SuAll[iii]
#                    except:
#                        pass
        OptHist.close()
    else:
        fAll = []
        xAll = []
        gAll = []
        gGradIter = []
        fGradIter = []
        failIter = []
        fIter = fAll
        xIter = xAll
        gIter = gAll
        # SuIter = []

# -------------------------------------------------------------------------------------------------
#       Convert all data to numpy arrays
# -------------------------------------------------------------------------------------------------
    fIter = np.asarray(fIter)
    xIter = np.asarray(xIter)
    gIter = np.asarray(gIter)
    gGradIter = np.asarray(gGradIter)
    fGradIter = np.asarray(fGradIter)

# -------------------------------------------------------------------------------------------------
# Denormalization of design variables
# -------------------------------------------------------------------------------------------------
    xOpt = np.resize(xOpt[0:np.size(xL)], np.size(xL))
    if DesVarNorm in ["None", None, False]:
        x0norm = []
        xIterNorm = []
        xOptNorm = []
    else:
        xOpt = np.resize(xOpt, [np.size(xL), ])
        xOptNorm = xOpt
        xOpt = denormalize(xOptNorm.T, xL, xU, DesVarNorm)
        try:
            xIterNorm = xIter[:, 0:np.size(xL)]
            xIter = np.zeros(np.shape(xIterNorm))
            for ii in range(len(xIterNorm)):
                xIter[ii] = denormalize(xIterNorm[ii], xL, xU, DesVarNorm)
        except:
            x0norm = []
            xIterNorm = []
            xOptNorm = []
    nIter = np.size(fIter)
    if np.size(fIter) > 0:
        fIterNorm = fIter / fIter[0]  # fIterNorm=(fIter-fIter[nEval-1])/(fIter[0]-fIter[nEval-1])
    else:
        fIterNorm = []

# -------------------------------------------------------------------------------------------------
#  Active constraints for use in the calcualtion of the Lagrangian multipliers and optimality criterium
# -------------------------------------------------------------------------------------------------
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
        xL_Active = []
    try:
        xU_Active = xU[xU_ActiveIndex]
    except:
        xU_Active = []
    xActive = np.concatenate((xL_Active, xU_Active), axis=1)
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
    if xL_Active == []:
        xLU_Active = xU_Active
    elif xU_Active == []:
        xLU_Active = xL_Active
    else:
        xLU_Active = np.concatenate((xL_Active, xU_Active), axis=1)
    if np.size(gc) > 0 and Alg[:5] != "PyGMO":  # are there nonlinear constraints active, in case equaility constraints are added later, this must also be added
        gMaxIter = np.zeros([nIter])
        for ii in range(len(gIter)):
            gMaxIter[ii] = max(gIter[ii])
        gOpt = gIter[nIter - 1]
        gOptActiveIndex = gOpt > -epsActive
        gOptActive = gOpt[gOpt > -epsActive]
    else:
        gOptActiveIndex = [[False]] * len(gc)
        gOptActive = np.array([])
        gMaxIter = np.array([] * nIter)
        gOpt = np.array([])
    if xLU_Active == []:
        g_xLU_OptActive = gOptActive
    elif gOptActive == []:
        g_xLU_OptActive = xLU_Active
    else:
        if np.size(xLU_Active) == 1 and np.size(gOptActive) == 1:
            g_xLU_OptActive = np.array([xLU_Active, gOptActive])
        else:
            g_xLU_OptActive = np.concatenate((xLU_Active, gOptActive), axis=1)
    if np.size(fGradIter) > 0:  # are there gradients available from the hist file?
        fGradOpt = fGradIter[nIter - 1]
        if np.size(gc) > 0:
            gGradOpt = gGradIter[nIter - 1]
            gGradOpt = gGradOpt.reshape([ng, nx]).T
            gGradOptActive = gGradOpt[:, gOptActiveIndex == True]
            try:
                cOptActive = gc[gOptActiveIndex == True]
            except:
                cOptActive = []
            if np.size(xGradActive) == 0:
                g_xLU_GradOptActive = gGradOptActive
                c_xLU_OptActive = cOptActive
            elif np.size(gGradOptActive) == 0:
                g_xLU_GradOptActive = xGradActive
                c_xLU_OptActive = xActive
            else:
                g_xLU_GradOptActive = np.concatenate((gGradOptActive, xGradActive), axis=1)
                c_xLU_OptActive = np.concatenate((cOptActive, xActive), axis=1)
        else:
            g_xLU_GradOptActive = xGradActive
            gGradOpt = np.array([])
            c_xLU_OptActive = np.array([])
            g_xLU_GradOptActive = np.array([])

    else:
        fGradOpt = np.array([])
        gGradOpt = np.array([])
        g_xLU_GradOptActive = np.array([])
        c_xLU_OptActive = np.array([])

# -------------------------------------------------------------------------------------------------
#   §      Post-processing of optimization solution
# -------------------------------------------------------------------------------------------------
    lambda_c, SPg, OptRes, Opt1Order, KKTmax = OptPostProc(fGradOpt, gc, gOptActiveIndex, g_xLU_GradOptActive,
                                                           c_xLU_OptActive)

# -------------------------------------------------------------------------------------------------
#   §      Save optimizaiton solution to file
# -------------------------------------------------------------------------------------------------
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
    OptSolData['OptAlg'] = OptAlg
    OptSolData['SensCalc'] = SensCalc
    OptSolData['xIterNorm'] = xIterNorm
    OptSolData['x0norm'] = x0norm
    OptSolData['xL'] = xL
    OptSolData['xU'] = xU
    OptSolData['ng'] = ng
    OptSolData['nx'] = nx
    OptSolData['Opt1Order'] = Opt1Order
    OptSolData['hhmmss0'] = hhmmss0
    OptSolData['hhmmss1'] = hhmmss1


# -------------------------------------------------------------------------------------------------
#   §    Save in Python format
# -------------------------------------------------------------------------------------------------
    output = open(OptName + "_OptSol.pkl", 'wb')
    pickle.dump(OptSolData, output)
    output.close()
    np.savez(OptName + "_OptSol", x0, xOpt, xOptNorm, xIter, xIterNorm, xIter, xIterNorm, fOpt, fIter, fIterNorm, gIter,
             gMaxIter, gOpt, fGradIter, gGradIter,
             fGradOpt, gGradOpt, OptName, OptModel, OptTime, loctime, today, computerName, operatingSystem,
             architecture, nProcessors, userName, Alg, DesVarNorm, KKTmax)

# -------------------------------------------------------------------------------------------------
#   §5.2    Save in MATLAB format
# -------------------------------------------------------------------------------------------------
    OptSolData['OptAlg'] = []
    spio.savemat(OptName + '_OptSol.mat', OptSolData, oned_as='row')

# -------------------------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------------------------
    os.chdir(DesOptDir)
    if LocalRun is True and Debug is False:
        try:
            shutil.move(RunDir + os.sep + OptName,
                        ResultsDir + os.sep + OptName + os.sep + "RunFiles" + os.sep)
        # except WindowsError:
        except:
            print "Run files not deleted from " + RunDir + os.sep + OptName
            shutil.copytree(RunDir + os.sep + OptName,
                            ResultsDir + os.sep + OptName + os.sep + "RunFiles" + os.sep)

# -------------------------------------------------------------------------------------------------
#   §    Graphical post-processing
# -------------------------------------------------------------------------------------------------
    if ResultReport is True:
        print("Entering preprocessing mode")
        OptResultReport.OptResultReport(OptName, diagrams=1, tables=1, lyx=1)
    if Video is True:
        OptVideo.OptVideo(OptName)

# -------------------------------------------------------------------------------------------------
#   § Print out
# -------------------------------------------------------------------------------------------------
    if PrintOut is True:
        print("g* =")
        print(gOpt)
        print("x* =")
        print(xOpt.T)
        print("f* =")
        print(fOpt)
        print("nEval =")
        print(nEval)
        print("nIter =")
        print(nIter)
        print(OptTime)
        print(lambda_c)
        print(SPg)
        # print("$n_{eval}$    $n_{it}$      $f^*$")
        if operatingSystem == "Linux":
            t = 1
            freq = 350
            os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (t, freq))
    #os.chdir(MainDir)
    return (xOpt, fOpt, SPg)

if __name__ == "__main__":
    print("E. J. Wehrle")
    print("Fachgebiet Computational Mechanics")
    print("Technische Universität München")
    print("wehrle@tum.de")
    print("\n")
    print("DESign OPTimization in PYthon")
    print("Version: DesOptPy 1.2")
    print("\n")
    print("Start DesOptPy from file containing system equations!")
    print("See documentation for further help.")
