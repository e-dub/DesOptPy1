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
import sys
import inspect

class setDefault():
    def __init__(self, Alg):
        self.Alg = Alg
        if Alg[:5] == "PyGMO":
            import PyGMO
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
            self.print_level = 1
            self.print_user_options = "no"
            self.linear_system_scaling = "none"
            self.max_iter = 10
            self.IFILE = False
            #self.IFILE = True
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
            #self.IFILE = True
            #self.pCross_real = 0.6
            #self.pMut_real = 0.2
            #self.eta_c = 10.
            #self.eta_m = 20.
            #self.pCross_bin = 0.
            #self.pMut_bin = 0.
            #self.seed = 0.
        elif Alg == "ALPSO":
            self.SwarmSize = 40
            self.maxOuterIter = 200
            self.maxInnerIter = 6
            self.vcrazy = 1e4
            self.HoodSize = 40
            self.HoodModel = "gbest"
            self.HoodSelf = 1
            self.Scaling = 1
            self.vmax = 1.0
            self.vinit = 2.0
            self.IFILE = False
        elif Alg == "MIDACO":
            self.ACC = 1e-3
            self.ISEED = 1
            #self.QSTART = 0
            self.FSTOP = 0
            self.AUTOSTOP = 0
            self.ORACLE = 0.0
            self.FOCUS = 0
            self.ANTS = 0
            self.KERNEL = 0
            self.CHARACTER = 0
            self.MAXEVAL = 10000
            self.MAXTIME = 86400
            self.IFILE = False
        #elif Alg[:5] == "PyGMO":
        #    import PyGMO
        #    self = eval("PyGMO.algorithm." + Alg[6:]+"()")
        elif Alg == "PyGMO_de":
            self.gen=100
            self.f=0.8
            self.cr=0.9
            self.variant=2
            self.ftol=1e-3
            self.xtol=1e-3
            self.screen_output=False
        elif Alg == "PyGMO_jde":
            self.gen=100
            self.variant=2
            self.variant_adptv=1
            self.ftol=1e-3
            self.xtol=1e-3
            self.memory=False
            self.screen_output=False
        elif Alg == "PyGMO_mde_pbx":
            self.gen=100
            self.qperc=0.15
            self.nexp=1.5
            self.ftol=1e-06
            self.xtol=1e-06
            self.screen_output=False
        elif Alg == "PyGMO_de_1220":
            self.gen=100
            self.variant_adptv=1
            self.allowed_variants=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            self.memory=False
            self.ftol=1e-06
            self.xtol=1e-06
            self.screen_output=False
        elif Alg == "PyGMO_pso":
            self.gen=1
            self.omega=0.7298
            self.eta1=2.05
            self.eta2=2.05
            self.vcoeff=0.5
            self.variant=5
            self.neighb_type=2
            self.neighb_param=4
        elif Alg == "PyGMO_pso_gen":
             self.gen=10
             self.omega=0.7298
             self.eta1=2.05
             self.eta2=2.05
             self.vcoeff=0.5
             self.variant=5
             self.neighb_type=2
             self.neighb_param=4
        elif Alg == "PyGMO_sea":
            self.gen=100
            self.limit=20
        elif Alg == "PyGMO_sga":
            self.gen=10
            self.cr=0.95
            self.m=0.02
            self.elitism=1
            self.mutation=PyGMO.algorithm._algorithm._sga_mutation_type.GAUSSIAN
            self.width=0.1
            self.selection=PyGMO.algorithm._algorithm._sga_selection_type.ROULETTE
            self.crossover=PyGMO.algorithm._algorithm._sga_crossover_type.EXPONENTIAL
        elif Alg == "PyGMO_vega":
            self.gen=1
            self.cr=0.95
            self.m=0.02
            self.elitism=1
            self.mutation=PyGMO.algorithm._algorithm._vega_mutation_type.GAUSSIAN
            self.width=0.1
            self.crossover=PyGMO.algorithm._algorithm._vega_crossover_type.EXPONENTIAL
        elif Alg == "PyGMO_sga_gray":
            self.gen=1
            self.cr=0.95
            self.m=0.02
            self.elitism=1
            self.mutation=PyGMO.algorithm._algorithm._gray_mutation_type.UNIFORM
            self.selection=PyGMO.algorithm._algorithm._gray_selection_type.ROULETTE
            self.crossover=PyGMO.algorithm._algorithm._gray_crossover_type.SINGLE_POINT
        elif Alg == "PyGMO_nsga_II":
            self.gen=100
            self.cr=0.95
            self.eta_c=10
            self.m=0.01
            self.eta_m=10
        elif Alg == "PyGMO_sms_emoa":
            self.hv_algorithm=None
            self.gen=100
            self.sel_m=2
            self.cr=0.95
            self.eta_c=10
            self.m=0.01
            self.eta_m=10
        elif Alg == "PyGMO_pade":
            self.gen=10
            self.decomposition='tchebycheff'
            self.weights='grid'
            self.solver=None
            self.threads=8
            self.T=8
            self.z=[]
        elif Alg == "PyGMO_nspso":
            self.gen=100
            self.minW=0.4
            self.maxW=1.0
            self.C1=2.0
            self.C2=2.0
            self.CHI=1.0
            self.v_coeff=0.5
            self.leader_selection_range=5
            #self.diversity_mechanism='crowding distance'
        elif Alg == "PyGMO_spea2":
            self.gen=100
            self.cr=0.95
            self.eta_c=10
            self.m=0.01
            self.eta_m=50
            self.archive_size=0
        elif Alg == "PyGMO_sa_corana":
            self.iter=10000
            self.Ts=10
            self.Tf=0.1
            self.steps=1
            self.bin_size=20
            self.range=1
        elif Alg == "PyGMO_bee_colony":
            self.gen=100
            self.limit=20
        elif Alg == "PyGMO_ms":
            self.algorithm=None
            self.iter=1
        elif Alg == "PyGMO_mbh":
            self.algorithm=None
            self.stop=5
            self.perturb=0.05
            self.screen_output=False
        elif Alg == "PyGMO_cstrs_co_evolution":
            self.original_algo=None
            self.original_algo_penalties=None
            self.pop_penalties_size=30
            self.gen=20
            #self.method=PyGMO.algorithm._algorithm._co_evo_method_type.SIMPLE
            self.pen_lower_bound=0.0
            self.pen_upper_bound=100000.0
            self.f_tol=1e-15
            self.x_tol=1e-15
        elif Alg == "PyGMO_cstrs_immune_system":
            self.algorithm=None
            self.algorithm_immune=None
            self.gen=1
            self.select_method=PyGMO.algorithm._algorithm._immune_select_method_type.BEST_ANTIBODY
            self.inject_method=PyGMO.algorithm._algorithm._immune_inject_method_type.CHAMPION
            self.distance_method=PyGMO.algorithm._algorithm._immune_distance_method_type.EUCLIDEAN
            self.phi=0.5
            self.gamma=0.5
            self.sigma=0.3333333333333333
            self.f_tol=1e-15
            self.x_tol=1e-15
        elif Alg == "PyGMO_cstrs_core":
            self.algorithm=None
            self.repair_algorithm=None
            self.gen=1
            self.repair_frequency=10
            self.repair_ratio=1.0
            self.f_tol=1e-15
            self.x_tol=1e-15
        elif Alg == "PyGMO_cs":
            self.max_eval=1
            self.stop_range=0.01
            self.start_range=0.1
            self.reduction_coeff=0.5
        elif Alg == "PyGMO_ihs":
            self.iter=100
            self.hmcr=0.85
            self.par_min=0.35
            self.par_max=0.99
            self.bw_min=1e-05
            self.bw_max=1
        elif Alg == "PyGMO_monte_carlo":
            self.iter=10000
        elif Alg == "PyGMO_py_example":
            self.iter=10
        elif Alg == "PyGMO_py_cmaes":
            self.gen=500
            self.cc=-1
            self.cs=-1
            self.c1=-1
            self.cmu=-1
            #self.sigma0=0.5
            self.ftol=1e-06
            self.xtol=1e-06
            self.memory=False
            self.screen_output=False
        elif Alg == "PyGMO_cmaes":
            self.gen=500
            self.cc=-1
            self.cs=-1
            self.c1=-1
            self.cmu=-1
            #self.sigma0=0.5
            self.ftol=1e-06
            self.xtol=1e-06
            self.memory=False
            self.screen_output=False
        elif Alg == "PyGMO_scipy_fmin":
            self.maxiter=1
            self.xtol=0.0001
            self.ftol=0.0001
            self.maxfun=None
            self.disp=False
        elif Alg == "PyGMO_scipy_l_bfgs_b":
            self.maxfun=1
            self.m=10
            self.factr=10000000.0
            self.pgtol=1e-05
            self.epsilon=1e-08
            self.screen_output=False
        elif Alg == "PyGMO_scipy_slsqp":
            self.max_iter=100
            self.acc=1e-08
            self.epsilon=1.4901161193847656e-08
            self.screen_output=False
        elif Alg == "PyGMO_scipy_tnc":
            self.maxfun=15000
            self.xtol=-1
            self.ftol=-1
            self.pgtol=1e-05
            self.epsilon=1e-08
            self.screen_output=False
        elif Alg == "PyGMO_scipy_cobyla":
            self.max_fun=1
            self.rho_end=1e-05
            self.screen_output=False
        elif Alg == "PyGMO_nlopt_cobyla":
            self.max_iter=100
            self.ftol=1e-06
            self.xtol=1e-06
        elif Alg == "PyGMO_nlopt_bobyqa":
            self.max_iter=100
            self.ftol=1e-06
            self.xtol=1e-06
        elif Alg == "PyGMO_nlopt_sbplx":
            self.max_iter=100
            self.ftol=1e-06
            self.xtol=1e-06
        elif Alg == "nlopt_mma":
            self.max_iter=100
            self.ftol=1e-06
            self.xtol=1e-06
        elif Alg == "PyGMO_nlopt_auglag":
            self.aux_algo_id=1
            self.max_iter=100
            self.ftol=1e-06
            self.xtol=1e-06
            self.aux_max_iter=100
            self.aux_ftol=1e-06
            self.aux_xtol=1e-06
        elif Alg == "PyGMO_nlopt_auglag_eq":
            self.aux_algo_id=1
            self.max_iter=100
            self.ftol=1e-06
            self.xtol=1e-06
            self.aux_max_iter=100
            self.aux_ftol=1e-06
            self.aux_xtol=1e-06
        elif Alg == "PyGMO_nlopt_slsqp":
            self.max_iter=100
            self.ftol=1e-06
            self.xtol=1e-06
        elif Alg == "PyGMO_gsl_nm2rand":
            self.max_iter=100
            self.ftol=1e-06
            self.xtol=1e-06
        elif Alg == "PyGMO_gsl_nm2":
            self.max_iter=100
            self.ftol=1e-06
            self.xtol=1e-06
        elif Alg == "PyGMO_gsl_nm":
            self.max_iter=100
            self.ftol=1e-06
            self.xtol=1e-06
        elif Alg == "PyGMO_gsl_pr":
            self.max_iter=100
            self.step_size=1e-08
            self.tol=1e-08
            self.grad_step_size=0.01
            self.grad_tol=0.0001
        elif Alg == "PyGMO_gsl_fr":
            self.max_iter=100
            self.step_size=1e-08
            self.tol=1e-08
            self.grad_step_size=0.01
            self.grad_tol=0.0001
        elif Alg == "PyGMO_gsl_bfgs2":
            self.max_iter=100
            self.step_size=1e-08
            self.tol=1e-08
            self.grad_step_size=0.01
            self.grad_tol=0.0001
        elif Alg == "PyGMO_gsl_bfgs":
            self.max_iter=100
            self.step_size=1e-08
            self.tol=1e-08
            self.grad_step_size=0.01
            self.grad_tol=0.0001
        elif Alg == "PyGMO_snopt":
            self.major_iter=100
            self.feas_tol=1e-08
            self.opt_tol=1e-06
            self.screen_output=False
        elif Alg == "PyGMO_ipopt":
            self.max_iter=100
            self.constr_viol_tol=1e-08
            self.dual_inf_tol=1e-08
            self.compl_inf_tol=1e-08
            self.nlp_scaling_method=True
            self.obj_scaling_factor=1.0
            self.mu_init=0.1
            self.screen_output=False
        else:
            sys.exit("Error on line "+ str(inspect.currentframe().f_lineno) + " of file "+ __file__ + ": Algorithm misspelled or not supported")
        if Alg[:5] == "PyGMO":
            self.nIndiv=8
            self.ConstraintHandling = "CoevolutionPenalty"


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


#def setDefaultOptions(Alg, OptName, OptAlg):
#    if Alg == "MMA":
#        OptAlg.setOption("GEPS", 1.0e-3)
#        OptAlg.setOption("DABOBJ", 1.0e-3)
#        OptAlg.setOption("DELOBJ", 1.0e-3)
#        OptAlg.setOption("ITRM", 1)
#        OptAlg.setOption("MAXIT", 60)
#        OptAlg.setOption("IFILE", OptName+"_Outfile.out")
#    elif Alg == "GCMMA":
#        OptAlg.setOption("GEPS", 1.0e-3)
#        OptAlg.setOption("DABOBJ", 1.0e-3)
#        OptAlg.setOption("DELOBJ", 1.0e-3)
#        OptAlg.setOption("ITRM", 1)
#        OptAlg.setOption("MAXIT", 60)
#        OptAlg.setOption("INNMAX", 5)
#        OptAlg.setOption("IFILE", OptName+"_Outfile.out")
#    elif Alg == "NLPQLP":
#        OptAlg.setOption("ACC", 1.0e-6)      #Convergence Accurancy
#        OptAlg.setOption("ACCQP", 1.0e-6)    #QP Solver Convergence Accurancy
#        OptAlg.setOption("STPMIN", 1.0e-6)   #Minimum Step Length
#        OptAlg.setOption("MAXFUN", 10)       #Maximum Number of Function Calls During Line Search
#        OptAlg.setOption("MAXIT",50)        #Maximum Number of Outer Iterations
#        OptAlg.setOption("RHOB",0.)         #BFGS-Update Matrix Initialization Parameter
#        OptAlg.setOption("MODE",0)          #NLPQL Mode (0 - Normal Execution, 1 to 18 - See Manual)
#        OptAlg.setOption("LQL",True)        #QP Solver (True - Quasi-Newton, False - Cholesky)
#        OptAlg.setOption("IFILE", OptName+"_Outfile.out")
#    elif Alg == "IPOPT":
#        OptAlg.setOption("tol", 1.0e-3)
#        OptAlg.setOption("print_level", 5)
#        OptAlg.setOption("print_user_options", "yes")
#        OptAlg.setOption('linear_system_scaling', "none")
#        OptAlg.setOption("max_iter", 60)
#        OptAlg.setOption("output_file", OptName+"_Outfile.out")
#    elif Alg == "SLSQP":
#        OptAlg.setOption("ACC", 1.0e-3)
#        OptAlg.setOption("MAXIT", 100)
#        OptAlg.setOption("IFILE", OptName+"_Outfile.out")
#    elif Alg == "PSQP":
#        OptAlg.setOption("XMAX", 1000.)
#        OptAlg.setOption("TOLX", 1.0e-3)
#        OptAlg.setOption("TOLC", 1.0e-3)
#        OptAlg.setOption("TOLG", 1.0e-3)
#        OptAlg.setOption("RPF", 1.0e-3)
#        OptAlg.setOption("MIT", 50)
#        OptAlg.setOption("MFV", 3000)
#        OptAlg.setOption("MET", 2)
#        OptAlg.setOption("MEC", 2)
#        OptAlg.setOption("IFILE", OptName+"_Outfile.out")
#    elif Alg == "COBYLA":
#        OptAlg.setOption("RHOBEG", 0.5)
#        OptAlg.setOption("RHOEND", 1e-6)
#        OptAlg.setOption("MAXFUN", 15000)
#        OptAlg.setOption("IFILE", OptName+"_Outfile.out")
#    elif Alg == "CONMIN":
#        OptAlg.setOption("ITMAX", 500)
#        OptAlg.setOption("DELFUN", 1e-3)
#        OptAlg.setOption("DABFUN", 1e-3)
#        OptAlg.setOption("ITRM", 2)
#        OptAlg.setOption("NFEASCT", 20)
#        OptAlg.setOption("IFILE", OptName+"_Outfile.out")
#    elif Alg == "KSOPT":
#        OptAlg.setOption("RDFUN", 1e-3)
#        OptAlg.setOption("RHOMIN", 5.)
#        OptAlg.setOption("RHOMAX", 100.)
#        OptAlg.setOption("ITMAX", 30)
#        OptAlg.setOption("IFILE", OptName+"_Outfile.out")
#    elif Alg == "SOLVOPT":
#        OptAlg.setOption("xtol", 1e-3)
#        OptAlg.setOption("ftol", 1e-3)
#        OptAlg.setOption("gtol", 1e-3)
#        OptAlg.setOption("maxit", 30)
#        OptAlg.setOption("spcdil", 25.)
#        #OptAlg.setOption("IFILE",OptName+"_Outfile.out")
#    elif Alg == "ALGENCAN":
#        OptAlg.setOption("epsfeas", 1e-3)
#        OptAlg.setOption("epsopt", 1e-8)
#        #OptAlg.setOption("IFILE",OptName+"_Outfile.out")
#    elif Alg == "NSGA2":
#        OptAlg.setOption("PopSize", 20)
#        OptAlg.setOption("maxGen", 10)
#        #OptAlg.setOption("pCross_real",0.6)
#        #OptAlg.setOption("pMut_real",0.2)
#        #OptAlg.setOption("eta_c", 10.)
#        #OptAlg.setOption("eta_m",20.)
#        #OptAlg.setOption("pCross_bin",0.)
#        #OptAlg.setOption("pMut_bin",0.)
#        #OptAlg.setOption("seed",0.)
#    elif Alg == "MIDACO":
#        OptAlg.setOption("ACC", 1e-3)
#        OptAlg.setOption("ISEED", 1)
#        #OptAlg.setOption("QSTART",0)
#        OptAlg.setOption("AUTOSTOP", 0)
#        #OptAlg.setOption("ORACLE",0)
#        OptAlg.setOption("ANTS", 0)
#        OptAlg.setOption("KERNEL", 0)
#        OptAlg.setOption("CHARACTER", 0)
#        #OptAlg.setOption("MAXEVAL", 1e2)
#        #OptAlg.setOption("MAXTIME", 1e3)
#        #OptAlg.setOption("IFILE",OptName+"_Outfile.out")
#    return OptAlg


def setUserOptions(UserOpt, Alg, OptName, OptAlg):
    if Alg[:5] == "PyGMO":
        import PyGMO
    elif Alg == "NSGA2":
        pass
    elif UserOpt.IFILE is True:     # needs to be changed as all non-pygmo algorithms land here! only for pyOpt!
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
    elif Alg == "ALPSO":
        OptAlg.setOption("SwarmSize", UserOpt.SwarmSize)
        OptAlg.setOption("maxOuterIter", UserOpt.maxOuterIter)
        OptAlg.setOption("maxInnerIter", UserOpt.maxInnerIter)
        OptAlg.setOption("vcrazy", UserOpt.vcrazy)
        OptAlg.setOption("HoodSize", UserOpt.HoodSize)
        OptAlg.setOption("vmax", UserOpt.vmax)
        OptAlg.setOption("vinit", UserOpt.vinit)
        OptAlg.setOption("HoodModel", UserOpt.HoodModel)
        OptAlg.setOption("HoodSelf", UserOpt.HoodSelf)
        OptAlg.setOption("Scaling", UserOpt.Scaling)
    elif Alg == "MIDACO":
        OptAlg.setOption("ACC", UserOpt.ACC)
        OptAlg.setOption("ISEED", UserOpt.ISEED)
        #OptAlg.setOption("QSTART",0)
        OptAlg.setOption("FSTOP", UserOpt.FSTOP)
        OptAlg.setOption("AUTOSTOP", UserOpt.AUTOSTOP)
        OptAlg.setOption("ORACLE", float(UserOpt.ORACLE))
        OptAlg.setOption("FOCUS",UserOpt.FOCUS)
        OptAlg.setOption("ANTS", UserOpt.ANTS)
        OptAlg.setOption("KERNEL", UserOpt.KERNEL)
        OptAlg.setOption("CHARACTER", UserOpt.CHARACTER)
        OptAlg.setOption("MAXEVAL",  int(UserOpt.MAXEVAL))
        OptAlg.setOption("MAXTIME", int(UserOpt.MAXTIME))
    elif Alg == "PyGMO_de":
        OptAlg = PyGMO.algorithm.de(gen=UserOpt.gen, f=UserOpt.f, cr=UserOpt.cr,
                                   variant=UserOpt.variant, ftol=UserOpt.ftol, xtol=UserOpt.xtol,
                                   screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_jde":
        OptAlg = PyGMO.algorithm.jde(gen=UserOpt.gen, variant=UserOpt.variant, variant_adptv=UserOpt.variant_adptv,
                                    ftol=UserOpt.ftol, xtol=UserOpt.xtol, memory=UserOpt.memory,
                                    screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_mde_pbx":
        OptAlg = PyGMO.algorithm.mde_pbx(gen=UserOpt.gen, qperc=UserOpt.qperc, nexp=UserOpt.nexp, ftol=UserOpt.ftol,
                                     xtol=UserOpt.xtol, screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_de_1220":
        OptAlg = PyGMO.algorithm.de_1220(gen=UserOpt.gen, variant_adptv=UserOpt.variant_adptv,
                                   allowed_variants=UserOpt.allowed_variants,
                                   ftol=UserOpt.ftol, xtol=UserOpt.xtol,
                                   screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_pso":
        OptAlg = PyGMO.algorithm.pso(gen=UserOpt.gen, omega=UserOpt.omega, eta1=UserOpt.eta1, eta2=UserOpt.eta2,
                                     vcoeff=UserOpt.vcoeff, variant=UserOpt.variant, neighb_type=UserOpt.neighb_type,
                                     neighb_param=UserOpt.neighb_param)
    elif Alg == "PyGMO_pso_gen":
        OptAlg = PyGMO.algorithm.pso_gen(gen=UserOpt.gen, omega=UserOpt.omega, eta1=UserOpt.eta1, eta2=UserOpt.eta2,
                                         vcoeff=UserOpt.vcoeff, variant=UserOpt.variant, neighb_type=UserOpt.neighb_type,
                                         neighb_param=UserOpt.neighb_param)
    elif Alg == "PyGMO_sea":
        OptAlg = PyGMO.algorithm.sea(gen=UserOpt.gen, limit=UserOpt.limit)
    elif Alg == "PyGMO_sga":
        OptAlg = PyGMO.algorithm.sga(gen=UserOpt.gen, cr=UserOpt.cr, m=UserOpt.m, elitism=UserOpt.elitism,
                                      mutation=UserOpt.mutation, width=UserOpt.width, selection=UserOpt.selection,
                                      crossover=UserOpt.crossover)
    elif Alg == "PyGMO_vega":
        OptAlg = PyGMO.algorithm.vega(gen=UserOpt.gen, cr=UserOpt.cr, m=UserOpt.m, elitism=UserOpt.elitism,
                                       mutation=UserOpt.mutation, width=UserOpt.width,
                                       crossover=UserOpt.crossover)
    elif Alg == "PyGMO_sga_gray":
        OptAlg = PyGMO.algorithm.sga_gray(gen=UserOpt.gen, cr=UserOpt.cr, m=UserOpt.m, elitism=UserOpt.elitism,
                                           mutation=UserOpt.mutation, selection=UserOpt.selection, crossover=UserOpt.crossover)
    elif Alg == "PyGMO_nsga_II":
        OptAlg = PyGMO.algorithm.nsga_II(gen=UserOpt.gen, cr=UserOpt.cr, eta_c=UserOpt.eta_c, m=UserOpt.m, eta_m=UserOpt.eta_m)
    elif Alg == "PyGMO_sms_emoa":
        OptAlg = PyGMO.algorithm.sms_emoa(hv_algorithm=UserOpt.hv_algorithm, gen=UserOpt.gen, cr=UserOpt.cr, eta_c=UserOpt.eta_c,
                                          m=UserOpt.m, eta_m=UserOpt.eta_m)
    elif Alg == "PyGMO_pade":
        OptAlg = PyGMO.algorithm.pade(gen=UserOpt.gen, decomposition=UserOpt.decomposition, weights=UserOpt.weights,
                                      solver=UserOpt.weights, threads=UserOpt.threads, T=UserOpt.T, z=UserOpt.z)
    elif Alg == "PyGMO_nspso":
        OptAlg = PyGMO.algorithm.nspso(gen=UserOpt.gen, minW=UserOpt.minW, maxW=UserOpt.maxW, C1=UserOpt.C1, C2=UserOpt.C2,
                                       CHI=UserOpt.CHI, v_coeff=UserOpt.v_coeff, leader_selection_range=UserOpt.leader_selection_range)
    elif Alg == "PyGMO_spea2":
        OptAlg = PyGMO.algorithm.spea2(gen=UserOpt.gen, cr=UserOpt.cr, eta_c=UserOpt.eta_c, m=UserOpt.m, eta_m=UserOpt.eta_m,
                                       archive_size=UserOpt.archive_size)
    elif Alg == "PyGMO_sa_corana":
        OptAlg = PyGMO.algorithm.sa_corana(iter=UserOpt.iter, Ts=UserOpt.Ts, Tf=UserOpt.Tf, steps=UserOpt.steps,
                                           bin_size=UserOpt.bin_size, range=UserOpt.range)
    elif Alg == "PyGMO_bee_colony":
        OptAlg= PyGMO.algorithm.bee_colony(gen=UserOpt.gen, limit=UserOpt.limit)
    elif Alg == "PyGMO_ms":
        OptAlg= PyGMO.algorithm.ms(algorithm=UserOpt.algorithm, iter=UserOpt.iter)
    elif Alg == "PyGMO_mbh":
        OptAlg= PyGMO.algorithm.mbh(algorithm=UserOpt.algorithm, stop=UserOpt.stop, perturb=UserOpt.perturb, screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_cstrs_co_evolution":
        OptAlg= PyGMO.algorithm.cstrs_co_evolution(original_algo=UserOpt.original_algo, original_algo_penalties=UserOpt.original_algo_penalties,
                                                   pop_penalties_size=UserOpt.pop_penalties_size, gen=UserOpt.gen,
                                                   pen_lower_bound=UserOpt.pen_lower_bound, pen_upper_bound=UserOpt.pen_upper_bound,
                                                   f_tol=UserOpt.f_tol, x_tol=UserOpt.x_tol)
    elif Alg == "PyGMO_cstrs_immune_system":
        OptAlg= PyGMO.algorithm.cstrs_immune_system(algorithm=UserOpt.algorithm, algorithm_immune=UserOpt.algorithm_immune, gen=UserOpt.gen,
                                                    select_method=UserOpt.select_method, inject_method=UserOpt.inject_method,
                                                    distance_method=UserOpt.distance_method, phi=UserOpt.phi, gamma=UserOpt.gamma,
                                                    sigma=UserOpt.sigma, f_tol=UserOpt.f_tol, x_tol=UserOpt.x_tol)
    elif Alg == "PyGMO_cstrs_core":
        OptAlg= PyGMO.algorithm.cstrs_core(algorithm=UserOpt.algorithm, repair_algorithm=UserOpt.repair_algorithm, gen=UserOpt.gen,
                                            repair_frequency=UserOpt.repair_frequency, repair_ratio=UserOpt.repair_ratio, f_tol=UserOpt.f_tol,
                                            x_tol=UserOpt.x_tol)
    elif Alg == "PyGMO_cs":
        OptAlg= PyGMO.algorithm.cs(max_eval=UserOpt.max_eval, stop_range=UserOpt.stop_range, start_range=UserOpt.start_range,
                                   reduction_coeff=UserOpt.reduction_coeff)
    elif Alg == "PyGMO_ihs":
        OptAlg= PyGMO.algorithm.ihs(iter=UserOpt.iter, hmcr=UserOpt.hmcr, par_min=UserOpt.par_min, par_max=UserOpt.par_max, bw_min=UserOpt.bw_min, bw_max=UserOpt.bw_max)
    elif Alg == "PyGMO_monte_carlo":
        OptAlg= PyGMO.algorithm.monte_carlo(iter=UserOpt.iter)
    elif Alg == "PyGMO_py_example":
        OptAlg= PyGMO.algorithm.py_example(iter=UserOpt.iter)
    elif Alg == "PyGMO_py_cmaes":
        OptAlg = PyGMO.algorithm.py_cmaes(gen=UserOpt.gen, cc=UserOpt.cc, cs=UserOpt.cs, c1=UserOpt.c1,
                                          cmu=UserOpt.cmu, ftol=UserOpt.ftol, xtol=UserOpt.xtol,
                                          memory=UserOpt.memory, screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_cmaes":
        OptAlg = PyGMO.algorithm.cmaes(gen=UserOpt.gen, cc=UserOpt.cc, cs=UserOpt.cs, c1=UserOpt.c1,
                                       cmu=UserOpt.cmu, ftol=UserOpt.ftol, xtol=UserOpt.xtol,
                                       memory=UserOpt.memory, screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_scipy_fmin":
        OptAlg = PyGMO.algorithm.scipy_fmin(maxiter=UserOpt.maxiter, xtol=UserOpt.xtol, ftol=UserOpt.ftol, maxfun=UserOpt.maxfun, disp=UserOpt.disp)
    elif Alg == "PyGMO_scipy_l_bfgs_b":
        OptAlg = PyGMO.algorithm.scipy_l_bfgs_b(maxfun=UserOpt.maxfun, m=UserOpt.m, factr=UserOpt.factr, pgtol=UserOpt.pgtol,
                                                epsilon=UserOpt.epsilon, screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_scipy_slsqp":
        OptAlg = PyGMO.algorithm.scipy_slsqp(max_iter=UserOpt.max_iter, acc=UserOpt.acc, epsilon=UserOpt.epsilon, screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_scipy_tnc":
        OptAlg = PyGMO.algorithm.scipy_tnc(maxfun=UserOpt.maxfun, xtol=UserOpt.xtol, ftol=UserOpt.ftol, pgtol=UserOpt.pgtol, epsilon=UserOpt.epsilon,
                                           screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_scipy_cobyla":
        OptAlg = PyGMO.algorithm.scipy_cobyla(max_fun=UserOpt.max_fun, rho_end=UserOpt.rho_end, screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_nlopt_cobyla":
        OptAlg = PyGMO.algorithm.nlopt_cobyla(max_iter=UserOpt.max_iter, ftol=UserOpt.ftol, xtol=UserOpt.xtol)
    elif Alg == "PyGMO_nlopt_bobyqa":
        OptAlg = PyGMO.algorithm.nlopt_bobyqa(max_iter=UserOpt.max_iter, ftol=UserOpt.ftol, xtol=UserOpt.xtol)
    elif Alg == "PyGMO_nlopt_sbplx":
        OptAlg = PyGMO.algorithm.nlopt_sbplx(max_iter=UserOpt.max_iter, ftol=UserOpt.ftol, xtol=UserOpt.xtol)
    elif Alg == "PyGMO_nlopt_mma":
        OptAlg = PyGMO.algorithm.nlopt_mma(max_iter=UserOpt.max_iter, ftol=UserOpt.ftol, xtol=UserOpt.xtol)
    elif Alg == "PyGMO_nlopt_auglag":
        OptAlg = PyGMO.algorithm.nlopt_auglag(aux_algo_id=UserOpt.aux_algo_id, max_iter=UserOpt.max_iter, ftol=UserOpt.ftol, xtol=UserOpt.xtol,
                                              aux_max_iter=UserOpt.aux_max_iter, aux_ftol=UserOpt.aux_ftol, aux_xtol=UserOpt.aux_xtol)
    elif Alg == "PyGMO_nlopt_auglag_eq":
        OptAlg = PyGMO.algorithm.nlopt_auglag_eq(aux_algo_id=UserOpt.aux_algo_id, max_iter=UserOpt.max_iter, ftol=UserOpt.ftol, xtol=UserOpt.xtol,
                                              aux_max_iter=UserOpt.aux_max_iter, aux_ftol=UserOpt.aux_ftol, aux_xtol=UserOpt.aux_xtol)
    elif Alg == "PyGMO_nlopt_slsqp":
        OptAlg = PyGMO.algorithm.nlopt_slsqp(max_iter=UserOpt.max_iter, ftol=UserOpt.ftol, xtol=UserOpt.xtol)
    elif Alg == "PyGMO_gsl_nm2rand":
        OptAlg = PyGMO.algorithm.gsl_nm2rand(max_iter=UserOpt.max_iter)
    elif Alg == "PyGMO_gsl_nm2":
        OptAlg = PyGMO.algorithm.gsl_nm2(max_iter=UserOpt.max_iter)
    elif Alg == "PyGMO_gsl_nm":
        OptAlg = PyGMO.algorithm.gsl_nm(max_iter=UserOpt.max_iter)
    elif Alg == "PyGMO_gsl_pr":
        OptAlg = PyGMO.algorithm.gsl_pr(max_iter=UserOpt.max_iter, step_size=UserOpt.step_size, tol=UserOpt.tol,
                                        grad_step_size=UserOpt.grad_step_size, grad_tol=UserOpt.grad_tol)
    elif Alg == "PyGMO_gsl_fr":
        OptAlg = PyGMO.algorithm.gsl_fr(max_iter=UserOpt.max_iter, step_size=UserOpt.step_size, tol=UserOpt.tol,
                                        grad_step_size=UserOpt.grad_step_size, grad_tol=UserOpt.grad_tol)
    elif Alg == "PyGMO_gsl_bfgs2":
        OptAlg = PyGMO.algorithm.gsl_bfgs2(max_iter=UserOpt.max_iter, step_size=UserOpt.step_size, tol=UserOpt.tol,
                                        grad_step_size=UserOpt.grad_step_size, grad_tol=UserOpt.grad_tol)
    elif Alg == "PyGMO_gsl_bfgs":
        OptAlg = PyGMO.algorithm.gsl_bfgs(max_iter=UserOpt.max_iter, step_size=UserOpt.step_size, tol=UserOpt.tol,
                                        grad_step_size=UserOpt.grad_step_size, grad_tol=UserOpt.grad_tol)
    elif Alg == "PyGMO_snopt":
        OptAlg = PyGMO.algorithm.snopt(major_iter=UserOpt.major_iter, feas_tol=UserOpt.feas_tol, opt_tol=UserOpt.opt_tol,
                                       screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_ipopt":
        OptAlg = PyGMO.algorithm.ipopt(max_iter=UserOpt.max_iter, constr_viol_tol=UserOpt.contr_viol_tol, dual_inf_tol=UserOpt.dual_inf_tol,
                                       compl_inf_tol=UserOpt.compl_inf_tol, nlp_scaling_method=UserOpt.nlp_scaling_method,
                                       obj_scaling_factor=UserOpt.obj_scaling_factor, mu_init=UserOpt.mu_init, screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_cstrs_self_adaptive":
        OptAlg = PyGMO.algorithm.cstrs_self_adaptive(algorithm=UserOpt.algorithm, max_iter=UserOpt.max_iter, f_tol=UserOpt.f_tol, x_tol=UserOpt.x_tol)
    else:
        sys.exit("Error on line "+ str(inspect.currentframe().f_lineno) + " of file "+ __file__ + ": Algorithm misspelled or not supported")
    return OptAlg

