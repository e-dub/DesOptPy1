# -*- coding: utf-8 -*-
'''
Title:    OptAlgOptions.py
Units:    -
Author:   Originally by S. Rudolph, fully reworked by E. J. Wehrle
Date:     March 4, 2015
---------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------
Description:
---------------------------------------------------------------------------------------------------
'''

class setDefault():
    def __init__(self, Alg):
        self.Alg = Alg
        if Alg == "MMA":
            self.GEPS = 1.0e-3
            self.DABOBJ = 1.0e-3
            self.DELOBJ = 1.0e-3
            self.ITRM = 1
            self.MAXIT = 60
            self.IFILE = True                            # +"_Outfile.out"
        elif Alg == "GCMMA":
            self.GEPS = 1.0e-3
            self.DABOBJ = 1.0e-3
            self.DELOBJ = 1.0e-3
            self.ITRM = 1
            self.MAXIT = 60
            self.INNMAX = 5
            self.IFILE = True                            # +"_Outfile.out"
        elif Alg == "NLPQLP":
            self.ACC = 1.0e-6      #Convergence Accurancy
            self.ACCQP = 1.0e-6    #QP Solver Convergence Accurancy
            self.STPMIN = 1.0e-6   #Minimum Step Length
            self.MAXFUN = 10       #Maximum Number of Function Calls During Line Search
            self.MAXIT = 50        #Maximum Number of Outer Iterations
            self.RHOB = 0.         #BFGS-Update Matrix Initialization Parameter
            self.MODE = 0          #NLPQL Mode (0 - Normal Execution, 1 to 18 - See Manual)
            self.LQL = True        #QP Solver (True - Quasi-Newton, False - Cholesky)
            self.IFILE = True
        elif Alg == "IPOPT":
            self.tol = 1.0e-3
            self.print_level = 5
            self.print_user_options = "yes"
            self.linear_system_scaling = "none"
            self.max_iter = 60
            self.IFILE = True
            #self.output_file = OptName+"_Outfile.out"
        elif Alg == "SLSQP":
            self.ACC = 1.0e-3
            self.MAXIT = 100
            self.IFILE = True
        elif Alg == "PSQP":
            self.XMAX = 1000.
            self.TOLX = 1.0e-3
            self.TOLC = 1.0e-3
            self.TOLG = 1.0e-3
            self.RPF = 1.0e-3
            self.MIT = 50
            self.MFV = 3000
            self.MET = 2
            self.MEC = 2
            self.IFILE = True
        elif Alg == "COBYLA":
            self.RHOBEG = 0.5
            self.RHOEND = 1e-6
            self.MAXFUN = 15000
            self.IFILE = True
        elif Alg == "CONMIN":
            self.ITMAX = 500
            self.DELFUN = 1e-3
            self.DABFUN = 1e-3
            self.ITRM = 2
            self.NFEASCT = 20
            self.IFILE = True
        elif Alg == "KSOPT":
            self.RDFUN = 1e-3
            self.RHOMIN = 5.
            self.RHOMAX = 100.
            self.ITMAX = 30
            self.IFILE = True
        elif Alg == "SOLVOPT":
            self.xtol = 1e-3
            self.ftol = 1e-3
            self.gtol = 1e-3
            self.maxit = 30
            self.spcdil = 25.
            self.IFILE = True
        elif Alg == "ALGENCAN":
            self.epsfeas = 1e-3
            self.epsopt = 1e-8
            self.IFILE = True
        elif Alg == "NSGA2":
            self.PopSize = 200
            self.maxGen = 10
            #self.pCross_real = 0.6
            #self.pMut_real = 0.2
            #self.eta_c = 10.
            #self.eta_m = 20.
            #self.pCross_bin = 0.
            #self.pMut_bin = 0.
            #self.seed = 0.
        elif Alg == "MIDACO":
            self.ACC = 1e-3
            self.ISEED = 1
            #self.QSTART = 0
            self.AUTOSTOP = 0
            #self.ORACLE = 0
            self.ANTS = 0
            self.KERNEL = 0
            self.CHARACTER = 0
            #self.MAXEVAL = 1e2
            #self.MAXTIME = 1e3
            #self.IFILE = True

    def setSimple(self, stopTol=[], maxIter=[], maxEval=[]):
        if stopTol != []:
            if self.Alg == "MMA":
                self.GEPS = stopTol
                self.DABOBJ = stopTol
                self.DELOBJ = stopTol
            elif self.Alg == "GCMMA":
                self.GEPS = stopTol
                self.DABOBJ = stopTol
                self.DELOBJ = stopTol
            elif self.Alg == "NLPQLP":
                self.ACC = stopTol
                self.ACCQP = stopTol
            elif self.Alg == "SLSQP":
                self.ACC = stopTol
            elif self.Alg == "PSQP":
                self.TOLX = stopTol
                self.TOLC = stopTol
                self.TOLG = stopTol
            elif self.Alg == "COBYLA":
                self.RHOEND = stopTol
            elif self.Alg == "CONMIN":
                self.DELFUN = stopTol
                self.DABFUN = stopTol
            elif self.Alg == "KSOPT":
                self.RDFUN = stopTol
            elif self.Alg == "SOLVOPT":
                self.xtol = stopTol
                self.ftol = stopTol
                self.gtol = stopTol
            elif self.Alg == "ALGENCAN":
                self.epsfeas = stopTol
                self.epsopt = stopTol
            elif self.Alg == "MIDACO":
                self.ACC = stopTol
        if maxIter != []:
            if self.Alg == "MMA":
                self.MAXIT = maxIter
            elif self.Alg == "GCMMA":
                self.MAXIT = maxIter
            elif self.Alg == "NLPQLP":
                self.MAXIT = maxIter
            elif self.Alg == "SLSQP":
                self.MAXIT = maxIter
            elif self.Alg == "PSQP":
                self.MIT = maxIter
            elif self.Alg == "CONMIN":
                self.ITMAX = maxIter
            elif self.Alg == "KSOPT":
                self.ITMAX = maxIter
            elif self.Alg == "SOLVOPT":
                self.maxit = maxIter
        if maxEval != []:
            if self.Alg == "NLPQLP":
                self.MAXFUN = maxEval
            elif self.Alg == "PSQP":
                self.MFV = maxEval
                self.MAXFUN = maxEval
            elif self.Alg == "MIDACO":
                self.MAXEVAL = maxEval


def setDefaultOptions(Alg, OptName, OptAlg):
    if Alg == "MMA":
        OptAlg.setOption("GEPS", 1.0e-3)
        OptAlg.setOption("DABOBJ", 1.0e-3)
        OptAlg.setOption("DELOBJ", 1.0e-3)
        OptAlg.setOption("ITRM", 1)
        OptAlg.setOption("MAXIT", 60)
        OptAlg.setOption("IFILE", OptName+"_Outfile.out")
    elif Alg == "GCMMA":
        OptAlg.setOption("GEPS", 1.0e-3)
        OptAlg.setOption("DABOBJ", 1.0e-3)
        OptAlg.setOption("DELOBJ", 1.0e-3)
        OptAlg.setOption("ITRM", 1)
        OptAlg.setOption("MAXIT", 60)
        OptAlg.setOption("INNMAX", 5)
        OptAlg.setOption("IFILE", OptName+"_Outfile.out")
    elif Alg == "NLPQLP":
        OptAlg.setOption("ACC", 1.0e-6)      #Convergence Accurancy
        OptAlg.setOption("ACCQP", 1.0e-6)    #QP Solver Convergence Accurancy
        OptAlg.setOption("STPMIN", 1.0e-6)   #Minimum Step Length
        OptAlg.setOption("MAXFUN", 10)       #Maximum Number of Function Calls During Line Search
        OptAlg.setOption("MAXIT",50)        #Maximum Number of Outer Iterations
        OptAlg.setOption("RHOB",0.)         #BFGS-Update Matrix Initialization Parameter
        OptAlg.setOption("MODE",0)          #NLPQL Mode (0 - Normal Execution, 1 to 18 - See Manual)
        OptAlg.setOption("LQL",True)        #QP Solver (True - Quasi-Newton, False - Cholesky)
        OptAlg.setOption("IFILE", OptName+"_Outfile.out")
    elif Alg == "IPOPT":
        OptAlg.setOption("tol", 1.0e-3)
        OptAlg.setOption("print_level", 5)
        OptAlg.setOption("print_user_options", "yes")
        OptAlg.setOption('linear_system_scaling', "none")
        OptAlg.setOption("max_iter", 60)
        OptAlg.setOption("output_file", OptName+"_Outfile.out")
    elif Alg == "SLSQP":
        OptAlg.setOption("ACC", 1.0e-3)
        OptAlg.setOption("MAXIT", 100)
        OptAlg.setOption("IFILE", OptName+"_Outfile.out")
    elif Alg == "PSQP":
        OptAlg.setOption("XMAX", 1000.)
        OptAlg.setOption("TOLX", 1.0e-3)
        OptAlg.setOption("TOLC", 1.0e-3)
        OptAlg.setOption("TOLG", 1.0e-3)
        OptAlg.setOption("RPF", 1.0e-3)
        OptAlg.setOption("MIT", 50)
        OptAlg.setOption("MFV", 3000)
        OptAlg.setOption("MET", 2)
        OptAlg.setOption("MEC", 2)
        OptAlg.setOption("IFILE", OptName+"_Outfile.out")
    elif Alg == "COBYLA":
        OptAlg.setOption("RHOBEG", 0.5)
        OptAlg.setOption("RHOEND", 1e-6)
        OptAlg.setOption("MAXFUN", 15000)
        OptAlg.setOption("IFILE", OptName+"_Outfile.out")
    elif Alg == "CONMIN":
        OptAlg.setOption("ITMAX", 500)
        OptAlg.setOption("DELFUN", 1e-3)
        OptAlg.setOption("DABFUN", 1e-3)
        OptAlg.setOption("ITRM", 2)
        OptAlg.setOption("NFEASCT", 20)
        OptAlg.setOption("IFILE", OptName+"_Outfile.out")
    elif Alg == "KSOPT":
        OptAlg.setOption("RDFUN", 1e-3)
        OptAlg.setOption("RHOMIN", 5.)
        OptAlg.setOption("RHOMAX", 100.)
        OptAlg.setOption("ITMAX", 30)
        OptAlg.setOption("IFILE", OptName+"_Outfile.out")
    elif Alg == "SOLVOPT":
        OptAlg.setOption("xtol", 1e-3)
        OptAlg.setOption("ftol", 1e-3)
        OptAlg.setOption("gtol", 1e-3)
        OptAlg.setOption("maxit", 30)
        OptAlg.setOption("spcdil", 25.)
        #OptAlg.setOption("IFILE",OptName+"_Outfile.out")
    elif Alg == "ALGENCAN":
        OptAlg.setOption("epsfeas", 1e-3)
        OptAlg.setOption("epsopt", 1e-8)
        #OptAlg.setOption("IFILE",OptName+"_Outfile.out")
    elif Alg == "NSGA2":
        OptAlg.setOption("PopSize", 200)
        OptAlg.setOption("maxGen", 10)
        #OptAlg.setOption("pCross_real",0.6)
        #OptAlg.setOption("pMut_real",0.2)
        #OptAlg.setOption("eta_c", 10.)
        #OptAlg.setOption("eta_m",20.)
        #OptAlg.setOption("pCross_bin",0.)
        #OptAlg.setOption("pMut_bin",0.)
        #OptAlg.setOption("seed",0.)
    elif Alg == "MIDACO":
        OptAlg.setOption("ACC", 1e-3)
        OptAlg.setOption("ISEED", 1)
        #OptAlg.setOption("QSTART",0)
        OptAlg.setOption("AUTOSTOP", 0)
        #OptAlg.setOption("ORACLE",0)
        OptAlg.setOption("ANTS", 0)
        OptAlg.setOption("KERNEL", 0)
        OptAlg.setOption("CHARACTER", 0)
        #OptAlg.setOption("MAXEVAL", 1e2)
        #OptAlg.setOption("MAXTIME", 1e3)
        #OptAlg.setOption("IFILE",OptName+"_Outfile.out")
    return OptAlg


def setUserOptions(UserOpt, Alg, OptName, OptAlg):
    if UserOpt.IFILE is True:
        OptAlg.setOption("IFILE", OptName+"_Outfile.out")
    if Alg == "MMA":
        OptAlg.setOption("GEPS", UserOpt.GEPS)
        OptAlg.setOption("DABOBJ", UserOpt.DABOBJ)
        OptAlg.setOption("DELOBJ", UserOpt.DELOBJ)
        OptAlg.setOption("ITRM", UserOpt.ITRM)
        OptAlg.setOption("MAXIT", UserOpt.MAXIT)
    elif Alg == "GCMMA":
        OptAlg.setOption("GEPS", UserOpt.GEPS)
        OptAlg.setOption("DABOBJ", UserOpt.DABOBJ)
        OptAlg.setOption("DELOBJ", UserOpt.DELOBJ)
        OptAlg.setOption("ITRM", UserOpt.ITRM)
        OptAlg.setOption("MAXIT", UserOpt.MAXIT)
        OptAlg.setOption("INNMAX", UserOpt.INNMAX)
    elif Alg == "NLPQLP":
        OptAlg.setOption("ACC", UserOpt.ACC)      #Convergence Accurancy
        OptAlg.setOption("ACCQP", UserOpt.ACCQP)    #QP Solver Convergence Accurancy
        OptAlg.setOption("STPMIN", UserOpt.STPMIN)   #Minimum Step Length
        OptAlg.setOption("MAXFUN", UserOpt.MAXFUN)       #Maximum Number of Function Calls During Line Search
        OptAlg.setOption("MAXIT", UserOpt.MAXIT)        #Maximum Number of Outer Iterations
        OptAlg.setOption("RHOB", UserOpt.RHOB)         #BFGS-Update Matrix Initialization Parameter
        OptAlg.setOption("MODE", UserOpt.MODE)          #NLPQL Mode (0 - Normal Execution, 1 to 18 - See Manual)
        OptAlg.setOption("LQL", UserOpt.LQL)        #QP Solver (True - Quasi-Newton, False - Cholesky)
    elif Alg == "IPOPT":
        OptAlg.setOption("tol", UserOpt.tol)
        OptAlg.setOption("print_level", UserOpt.print_level)
        OptAlg.setOption("print_user_options", UserOpt.print_user_options)
        OptAlg.setOption('linear_system_scaling', UserOpt.linear_system_scaling)
        OptAlg.setOption("max_iter", UserOpt.max_iter)
    elif Alg == "SLSQP":
        OptAlg.setOption("ACC", UserOpt.ACC)
        OptAlg.setOption("MAXIT", UserOpt.MAXIT)
    elif Alg == "PSQP":
        OptAlg.setOption("XMAX", UserOpt.XMAX)
        OptAlg.setOption("TOLX", UserOpt.TOLX)
        OptAlg.setOption("TOLC", UserOpt.TOLC)
        OptAlg.setOption("TOLG", UserOpt.TOLG)
        OptAlg.setOption("RPF", UserOpt.RPF)
        OptAlg.setOption("MIT", UserOpt.MIT)
        OptAlg.setOption("MFV", UserOpt.MFV)
        OptAlg.setOption("MET", UserOpt.MET)
        OptAlg.setOption("MEC", UserOpt.MEC)
    elif Alg == "COBYLA":
        OptAlg.setOption("RHOBEG", UserOpt.RHOBEG)
        OptAlg.setOption("RHOEND", UserOpt.RHOEND)
        OptAlg.setOption("MAXFUN", UserOpt.MAXFUN)
    elif Alg == "CONMIN":
        OptAlg.setOption("ITMAX", UserOpt.ITMAX)
        OptAlg.setOption("DELFUN", UserOpt.DELFUN)
        OptAlg.setOption("DABFUN", UserOpt.DABFUN)
        OptAlg.setOption("ITRM", UserOpt.ITRM)
        OptAlg.setOption("NFEASCT", UserOpt.NFEASCT)
    elif Alg == "KSOPT":
        OptAlg.setOption("RDFUN", UserOpt.RDFUN)
        OptAlg.setOption("RHOMIN", UserOpt.RHOMIN)
        OptAlg.setOption("RHOMAX", UserOpt.RHOMAX)
        OptAlg.setOption("ITMAX", UserOpt.ITMAX)
    elif Alg == "SOLVOPT":
        OptAlg.setOption("xtol", UserOpt.xtol)
        OptAlg.setOption("ftol", UserOpt.ftol)
        OptAlg.setOption("gtol", UserOpt.gtol)
        OptAlg.setOption("maxit", UserOpt.maxit)
        OptAlg.setOption("spcdil", UserOpt.spcdil)
    elif Alg == "ALGENCAN":
        OptAlg.setOption("epsfeas", UserOpt.epsfeas)
        OptAlg.setOption("epsopt", UserOpt.epsopt)
    elif Alg == "NSGA2":
        OptAlg.setOption("PopSize", UserOpt.PopSize)
        OptAlg.setOption("maxGen", UserOpt.maxGen)
        #OptAlg.setOption("pCross_real",0.6)
        #OptAlg.setOption("pMut_real",0.2)
        #OptAlg.setOption("eta_c", 10.)
        #OptAlg.setOption("eta_m",20.)
        #OptAlg.setOption("pCross_bin",0.)
        #OptAlg.setOption("pMut_bin",0.)
        #OptAlg.setOption("seed",0.)
    elif Alg == "MIDACO":
        OptAlg.setOption("ACC", UserOpt.ACC)
        OptAlg.setOption("ISEED", UserOpt.ISEED)
        #OptAlg.setOption("QSTART",0)
        OptAlg.setOption("AUTOSTOP", UserOpt.AUTOSTOP)
        #OptAlg.setOption("ORACLE",0)
        OptAlg.setOption("ANTS", UserOpt.ANTS)
        OptAlg.setOption("KERNEL", UserOpt.KERNEL)
        OptAlg.setOption("CHARACTER",)
        #OptAlg.setOption("MAXEVAL", 1e2)
        #OptAlg.setOption("MAXTIME", 1e3)
    return OptAlg
