# -*- coding: utf-8 -*-
'''
Title:    OptAlgOptions.py
Units:    -
Author:   E. J. Wehrle, S. Rudolph
Date:     August 6, 2017
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
Description:
-------------------------------------------------------------------------------
'''
from __future__ import absolute_import, division, print_function
import sys
import inspect

class setDefault():
    def __init__(self, Alg):
        self.Alg = Alg
        if Alg[:5] == "PyGMO":
            import pygmo as pg
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
            self.ACC = 1.0e-6      # Convergence Accurancy
            self.ACCQP = 1.0e-6    # QP Solver Convergence Accurancy
            self.STPMIN = 1.0e-6   # Minimum Step Length
            self.MAXFUN = 10       # Maximum Number of Function Calls During Line Search
            self.MAXIT = 100        # Maximum Number of Outer Iterations
            self.RHOB = 0.         # BFGS-Update Matrix Initialization Parameter
            self.MODE = 0          # NLPQL Mode (0 - Normal Execution, 1 to 18 - See Manual)
            self.LQL = True        # QP Solver (True - Quasi-Newton, False - Cholesky)
            self.IFILE = True
        elif Alg == "IPOPT":
            self.tol = 1.0e-6
            self.print_level = 0
            self.print_user_options = "no"
            self.linear_system_scaling = "none"
            self.max_iter = 60
            self.IFILE = False
            # self.IFILE = True
            # self.output_file = OptName+"_Outfile.out"
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
        elif Alg == "FSQP":
            self.mode = 100    # FSQP Mode
            self.iprint = 2    # Output Level (0- None, 1- Final, 2- Major, 3- Major Details)
            self.miter = 500    # Maximum Number of Iterations
            self.bigbnd = 1e10    # Plus Infinity Value
            self.epstol = 1e-8    # Convergence Tolerance
            self.epseqn = 0    # Equality Constraints Tolerance
            self.iout = 6    # Output Unit Number
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
            self.IFILE = False
        elif Alg == "ALGENCAN":
            self.epsfeas = 1e-2
            self.epsopt = 1e-1
            self.efacc = 1e-2          # Feasibility Level for Newton-KKT Acceleration
            self.eoacc = 1e-1           # Optimality Level for Newton-KKT Acceleration
            self.checkder = False       # Check Derivatives Flag
            self.iprint = 2             # Output Level (0 - None, 10 - Final, >10 - Iter Details)
            self.ncomp = 6              # Print Precision
            self.IFILE = True
        elif Alg == "NSGA2":
            self.PopSize = 300
            self.maxGen = 10
            # self.IFILE = True
            # self.pCross_real = 0.6
            # self.pMut_real = 0.2
            # self.eta_c = 10.
            # self.eta_m = 20.
            # self.pCross_bin = 0.
            # self.pMut_bin = 0.
            # self.seed = 0.
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
        elif Alg == "ALHSO":
            self.hms = 5             # Memory Size [1,50]
            self.hmcr = 0.95         # Probability rate of choosing from memory [0.7,0.99]
            self.par = 0.65          # Pitch adjustment rate [0.1,0.99]
            self.dbw = 2000          # Variable Bandwidth Quantization
            self.maxoutiter = 2e3    # Maximum Number of Outer Loop Iterations (Major Iterations)
            self.maxinniter = 2e2    # Maximum Number of Inner Loop Iterations (Minor Iterations)
            self.stopcriteria = 1    # Stopping Criteria Flag
            self.stopiters = 10      # Consecutively Number of Outer Iterations for convergence
            self.etol = 1e-6         # Absolute Tolerance for Equality constraints
            self.itol = 1e-6         # Absolute Tolerance for Inequality constraints
            self.atol = 1e-6         # Absolute Tolerance for Objective Function
            self.rtol = 1e-6         # Relative Tolerance for Objective Function
            self.prtoutiter = 0      # Number of Iterations Before Print Outer Loop Information
            self.prtinniter = 0      # Number of Iterations Before Print Inner Loop Information
            self.xinit = 0           # Initial Position Flag (0 - no position, 1 - position given)
            self.rinit = 1.0         # Initial Penalty Factor
            self.fileout = 1         # Flag to Turn On Output to filename
            self.seed = 0            # Random Number Seed (0 - Auto-Seed based on time clock)
            self.scaling = 1         # Design Variables Scaling (0- no scaling, 1- scaling [-1,1])
            self.IFILE = False
        elif Alg == "MIDACO":
            self.ACC = 1e-3
            self.ISEED = 1
            # self.QSTART = 0
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
        elif Alg == "SDPEN":
            self.alfa_stop = 1.0e-6     # Convergence Accurancy
            self.nf_max = 5000 	        # Maximum Number of Function Evaluations
            self.iprint = 0             # Print Flag (<0-None, 0-Final, 1,2-Iteration)
            self.iout = 6               # Output Unit Number
            self.IFILE = True           # Output File Name
        # elif Alg[:5] == "PyGMO":
        #    import pygmo as pg
        #    self = eval("pg." + Alg[6:]+"()")
        elif Alg == "PyGMO_de":
            self.gen = 100
            self.f = 0.8
            self.cr = 0.9
            self.variant = 2
            self.ftol = 1e-3
            self.xtol = 1e-3
            self.screen_output = False
        elif Alg == "PyGMO_sade":
            self.gen = 100
            self.f = 0.8
            self.cr = 0.9
            self.variant = 2
            self.ftol = 1e-3
            self.xtol = 1e-3
            self.screen_output = False
        elif Alg == "PyGMO_jde":
            self.gen = 100
            self.variant = 2
            self.variant_adptv = 1
            self.ftol = 1e-3
            self.xtol = 1e-3
            self.memory = False
            self.screen_output = False
        elif Alg == "PyGMO_mde_pbx":
            self.gen = 100
            self.qperc = 0.15
            self.nexp = 1.5
            self.ftol = 1e-06
            self.xtol = 1e-06
            self.screen_output = False
        elif Alg == "PyGMO_de1220":
            self.gen = 100
            self.variant_adptv = 1
            self.allowed_variants = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            self.memory = False
            self.ftol = 1e-06
            self.xtol = 1e-06
            self.screen_output = False
        elif Alg == "PyGMO_pso":
            self.gen = 1
            self.omega = 0.7298
            self.eta1 = 2.05
            self.eta2 = 2.05
            self.vcoeff = 0.5
            self.variant = 5
            self.neighb_type = 2
            self.neighb_param = 4
        elif Alg == "PyGMO_pso_gen":
            self.gen = 10
            self.omega = 0.7298
            self.eta1 = 2.05
            self.eta2 = 2.05
            self.vcoeff = 0.5
            self.variant = 5
            self.neighb_type = 2
            self.neighb_param = 4
        elif Alg == "PyGMO_sea":
            self.gen = 100
            self.limit = 20
        elif Alg == "PyGMO_sga":
            self.gen = 30
            self.cr = 0.95
            self.m = 0.02
            self.elitism = 1
            #self.mutation = pg._algorithm._sga_mutation_type.GAUSSIAN
            self.width = 0.1
            #self.selection = pg._algorithm._sga_selection_type.ROULETTE
            #self.crossover = pg._algorithm._sga_crossover_type.EXPONENTIAL
        elif Alg == "PyGMO_vega":
            self.gen = 1
            self.cr = 0.95
            self.m = 0.02
            self.elitism = 1
            self.mutation = pg._algorithm._vega_mutation_type.GAUSSIAN
            self.width = 0.1
            self.crossover = pg._algorithm._vega_crossover_type.EXPONENTIAL
        elif Alg == "PyGMO_sga_gray":
            self.gen = 10
            self.cr = 0.95
            self.m = 0.02
            self.elitism = 1
            self.mutation = pg._algorithm._gray_mutation_type.UNIFORM
            self.selection = pg._algorithm._gray_selection_type.ROULETTE
            self.crossover = pg._algorithm._gray_crossover_type.SINGLE_POINT
        elif Alg == "PyGMO_nsga_II":
            self.gen = 100
            self.cr = 0.95
            self.eta_c = 10
            self.m = 0.01
            self.eta_m = 10
        elif Alg == "PyGMO_sms_emoa":
            self.hv_algorithm = None
            self.gen = 100
            self.sel_m = 2
            self.cr = 0.95
            self.eta_c = 10
            self.m = 0.01
            self.eta_m = 10
        elif Alg == "PyGMO_pade":
            self.gen = 10
            self.decomposition = 'tchebycheff'
            self.weights = 'grid'
            self.solver = None
            self.threads = 8
            self.T = 8
            self.z = []
        elif Alg == "PyGMO_nspso":
            self.gen = 100
            self.minW = 0.4
            self.maxW = 1.0
            self.C1 = 2.0
            self.C2 = 2.0
            self.CHI = 1.0
            self.v_coeff = 0.5
            self.leader_selection_range = 5
            # self.diversity_mechanism='crowding distance'
        elif Alg == "PyGMO_spea2":
            self.gen = 100
            self.cr = 0.95
            self.eta_c = 10
            self.m = 0.01
            self.eta_m = 50
            self.archive_size = 0
        elif Alg == "PyGMO_simulated_annealing":
            self.gen = 10
            self.iter = 1000
            self.Ts = 10
            self.Tf = 0.1
            self.steps = 1
            self.bin_size = 20
            self.range = 1
        elif Alg == "PyGMO_xnes":
            self.gen = 100
            self.iter = 10000
            self.limit = 20
        elif Alg == "PyGMO_bee_colony":
            self.gen = 100
            self.limit = 20
        elif Alg == "PyGMO_ms":
            self.algorithm = None
            self.iter = 1
        elif Alg == "PyGMO_mbh":
            self.algorithm = None
            self.stop = 5
            self.perturb = 0.05
            self.screen_output = False
        elif Alg == "PyGMO_cstrs_co_evolution":
            self.original_algo = None
            self.original_algo_penalties = None
            self.pop_penalties_size = 30
            self.gen = 20
            # self.method = pg._algorithm._co_evo_method_type.SIMPLE
            self.pen_lower_bound = 0.0
            self.pen_upper_bound = 100000.0
            self.f_tol = 1e-15
            self.x_tol = 1e-15
        elif Alg == "PyGMO_cstrs_immune_system":
            self.algorithm = None
            self.algorithm_immune = None
            self.gen = 1
            self.select_method = pg._algorithm._immune_select_method_type.BEST_ANTIBODY
            self.inject_method = pg._algorithm._immune_inject_method_type.CHAMPION
            self.distance_method = pg._algorithm._immune_distance_method_type.EUCLIDEAN
            self.phi = 0.5
            self.gamma = 0.5
            self.sigma = 0.3333333333333333
            self.f_tol = 1e-15
            self.x_tol = 1e-15
        elif Alg == "PyGMO_cstrs_core":
            self.algorithm = None
            self.repair_algorithm = None
            self.gen = 1
            self.repair_frequency = 10
            self.repair_ratio = 1.0
            self.f_tol = 1e-15
            self.x_tol = 1e-15
        elif Alg == "PyGMO_cs":
            self.max_eval = 1
            self.stop_range = 0.01
            self.start_range = 0.1
            self.reduction_coeff = 0.5
        elif Alg == "PyGMO_ihs":
            self.iter = 100
            self.hmcr = 0.85
            self.par_min = 0.35
            self.par_max = 0.99
            self.bw_min = 1e-05
            self.bw_max = 1
        elif Alg == "PyGMO_monte_carlo":
            self.iter = 10000
        elif Alg == "PyGMO_py_example":
            self.iter = 10
        elif Alg == "PyGMO_py_cmaes":
            self.gen = 500
            self.cc = -1
            self.cs = -1
            self.c1 = -1
            self.cmu = -1
            # self.sigma0 = 0.5
            self.ftol = 1e-06
            self.xtol = 1e-06
            self.memory = False
            self.screen_output = False
        elif Alg == "PyGMO_cmaes":
            self.gen = 500
            self.cc = -1
            self.cs = -1
            self.c1 = -1
            self.cmu = -1
            # self.sigma0 = 0.5
            self.ftol = 1e-06
            self.xtol = 1e-06
            self.memory = False
            self.screen_output = False
        elif Alg == "PyGMO_scipy_fmin":
            self.maxiter = 1
            self.xtol = 0.0001
            self.ftol = 0.0001
            self.maxfun = None
            self.disp = False
        elif Alg == "PyGMO_scipy_l_bfgs_b":
            self.maxfun = 1
            self.m = 10
            self.factr = 10000000.0
            self.pgtol = 1e-05
            self.epsilon = 1e-08
            self.screen_output = False
        elif Alg == "PyGMO_scipy_slsqp":
            self.max_iter = 100
            self.acc = 1e-08
            self.epsilon = 1.4901161193847656e-08
            self.screen_output = False
        elif Alg == "PyGMO_scipy_tnc":
            self.maxfun = 15000
            self.xtol = -1
            self.ftol = -1
            self.pgtol = 1e-05
            self.epsilon = 1e-08
            self.screen_output = False
        elif Alg == "PyGMO_scipy_cobyla":
            self.max_fun = 1
            self.rho_end = 1e-05
            self.screen_output = False
        elif Alg == "PyGMO_nlopt_cobyla":
            self.max_iter = 100
            self.ftol = 1e-06
            self.xtol = 1e-06
        elif Alg == "PyGMO_nlopt_bobyqa":
            self.max_iter = 100
            self.ftol = 1e-06
            self.xtol = 1e-06
        elif Alg == "PyGMO_nlopt_sbplx":
            self.max_iter = 100
            self.ftol = 1e-06
            self.xtol = 1e-06
        elif Alg == "nlopt_mma":
            self.max_iter = 100
            self.ftol = 1e-06
            self.xtol = 1e-06
        elif Alg == "PyGMO_nlopt_auglag":
            self.aux_algo_id = 1
            self.max_iter = 100
            self.ftol = 1e-06
            self.xtol = 1e-06
            self.aux_max_iter = 100
            self.aux_ftol = 1e-06
            self.aux_xtol = 1e-06
        elif Alg == "PyGMO_nlopt_auglag_eq":
            self.aux_algo_id = 1
            self.max_iter = 100
            self.ftol = 1e-06
            self.xtol = 1e-06
            self.aux_max_iter = 100
            self.aux_ftol = 1e-06
            self.aux_xtol = 1e-06
        elif Alg == "PyGMO_nlopt_slsqp":
            self.max_iter = 100
            self.ftol = 1e-06
            self.xtol = 1e-06
        elif Alg == "PyGMO_gsl_nm2rand":
            self.max_iter = 100
            self.ftol = 1e-06
            self.xtol = 1e-06
        elif Alg == "PyGMO_gsl_nm2":
            self.max_iter = 100
            self.ftol = 1e-06
            self.xtol = 1e-06
        elif Alg == "PyGMO_gsl_nm":
            self.max_iter = 100
            self.ftol = 1e-06
            self.xtol = 1e-06
        elif Alg == "PyGMO_gsl_pr":
            self.max_iter = 100
            self.step_size = 1e-08
            self.tol = 1e-08
            self.grad_step_size = 0.01
            self.grad_tol = 0.0001
        elif Alg == "PyGMO_gsl_fr":
            self.max_iter = 100
            self.step_size = 1e-08
            self.tol = 1e-08
            self.grad_step_size = 0.01
            self.grad_tol = 0.0001
        elif Alg == "PyGMO_gsl_bfgs2":
            self.max_iter = 100
            self.step_size = 1e-08
            self.tol = 1e-08
            self.grad_step_size = 0.01
            self.grad_tol = 0.0001
        elif Alg == "PyGMO_gsl_bfgs":
            self.max_iter = 100
            self.step_size = 1e-08
            self.tol = 1e-08
            self.grad_step_size = 0.01
            self.grad_tol = 0.0001
        elif Alg == "PyGMO_snopt":
            self.major_iter = 100
            self.feas_tol = 1e-08
            self.opt_tol = 1e-06
            self.screen_output = False
        elif Alg == "PyGMO_ipopt":
            self.max_iter = 100
            self.constr_viol_tol = 1e-08
            self.dual_inf_tol = 1e-08
            self.compl_inf_tol = 1e-08
            self.nlp_scaling_method = True
            self.obj_scaling_factor = 1.0
            self.mu_init = 0.1
            self.screen_output = False
        else:
            sys.exit("Error on line " + str(inspect.currentframe().f_lineno) +
                     " of file " + __file__ +
                     ": Algorithm misspelled or not supported")
        if Alg[:5] == "PyGMO":
            self.nIndiv = 50
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


def setUserOptions(UserOpt, Alg, OptName, OptAlg):
    if Alg[:5] == "PyGMO":
        import pygmo as pg
    elif Alg == "NSGA2":
        pass
    elif UserOpt.IFILE is True:     # needs to be changed as all non-pygmo algorithms land here! only for pyOpt!
        if Alg in ["SDPEN", "FSQP", "ALGENCAN"]:
            OptAlg.setOption("ifile", OptName+"_Outfile.out")
        else:
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
        OptAlg.setOption("ACC", UserOpt.ACC)        # Convergence Accurancy
        OptAlg.setOption("ACCQP", UserOpt.ACCQP)    # QP Solver Convergence Accurancy
        OptAlg.setOption("STPMIN", UserOpt.STPMIN)  # Minimum Step Length
        OptAlg.setOption("MAXFUN", UserOpt.MAXFUN)  # Maximum Number of Function Calls During Line Search
        OptAlg.setOption("MAXIT", UserOpt.MAXIT)    # Maximum Number of Outer Iterations
        OptAlg.setOption("RHOB", UserOpt.RHOB)      # BFGS-Update Matrix Initialization Parameter
        OptAlg.setOption("MODE", UserOpt.MODE)      # NLPQL Mode (0 - Normal Execution, 1 to 18 - See Manual)
        OptAlg.setOption("LQL", UserOpt.LQL)        # QP Solver (True - Quasi-Newton, False - Cholesky)
    elif Alg == "IPOPT":
        OptAlg.setOption("tol", UserOpt.tol)
        OptAlg.setOption("print_level", UserOpt.print_level)
        OptAlg.setOption("print_user_options", UserOpt.print_user_options)
        OptAlg.setOption('linear_system_scaling',
                         UserOpt.linear_system_scaling)
        OptAlg.setOption("max_iter", UserOpt.max_iter)
    elif Alg == "SLSQP":
        OptAlg.setOption("ACC", UserOpt.ACC)
        OptAlg.setOption("MAXIT", UserOpt.MAXIT)
    elif Alg == "FSQP":
        OptAlg.setOption("mode", UserOpt.mode)
        OptAlg.setOption("iprint", UserOpt.iprint)
        OptAlg.setOption("miter", UserOpt.miter)
        OptAlg.setOption("bigbnd", UserOpt.bigbnd)
        OptAlg.setOption("epstol", UserOpt.epstol)
        OptAlg.setOption("epseqn", UserOpt.epseqn)
        OptAlg.setOption("iout", UserOpt.iout)
        OptAlg.setOption("ifile", UserOpt.IFILE)
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
        OptAlg.setOption("efacc", UserOpt.efacc)
        OptAlg.setOption("eoacc", UserOpt.eoacc)
        OptAlg.setOption("checkder", UserOpt.checkder)
        OptAlg.setOption("iprint", UserOpt.iprint)
        OptAlg.setOption("ncomp", UserOpt.ncomp)
    elif Alg == "NSGA2":
        OptAlg.setOption("PopSize", UserOpt.PopSize)
        OptAlg.setOption("maxGen", UserOpt.maxGen)
        # OptAlg.setOption("pCross_real",0.6)
        # OptAlg.setOption("pMut_real",0.2)
        # OptAlg.setOption("eta_c", 10.)
        # OptAlg.setOption("eta_m",20.)
        # OptAlg.setOption("pCross_bin",0.)
        # OptAlg.setOption("pMut_bin",0.)
        # OptAlg.setOption("seed",0.)
    elif Alg == "SDPEN":
        OptAlg.setOption("alfa_stop", UserOpt.alfa_stop)     # Convergence Accurancy
        OptAlg.setOption("nf_max", UserOpt.nf_max)        # Maximum Number of Function Evaluations
        OptAlg.setOption("iprint", UserOpt.iprint)             # Print Flag (<0-None, 0-Final, 1,2-Iteration)
        OptAlg.setOption("iout", UserOpt.iout)               # Output Unit Number
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
    elif Alg == "ALHSO":
        OptAlg.setOption("hms", UserOpt.hms)
        OptAlg.setOption("hmcr", UserOpt.hmcr)
        OptAlg.setOption("par", UserOpt.par)
        OptAlg.setOption("dbw", UserOpt.dbw)
        #OptAlg.setOption("maxoutiter", UserOpt.maxoutiter)
        #OptAlg.setOption("maxinniter", UserOpt.maxinniter)
        OptAlg.setOption("stopcriteria", UserOpt.stopcriteria)
        OptAlg.setOption("stopiters", UserOpt.stopiters)
        OptAlg.setOption("etol", UserOpt.etol)
        OptAlg.setOption("itol", UserOpt.itol)
        OptAlg.setOption("atol", UserOpt.atol)
        OptAlg.setOption("rtol", UserOpt.rtol)
        OptAlg.setOption("prtoutiter", UserOpt.prtoutiter)
        OptAlg.setOption("prtinniter", UserOpt.prtinniter)
        OptAlg.setOption("xinit", UserOpt.xinit)
        OptAlg.setOption("rinit", UserOpt.rinit)
        OptAlg.setOption("fileout", UserOpt.fileout)
        #OptAlg.setOption("filename", UserOpt.filename)
        #OptAlg.setOption("seed", UserOpt.seed)
        OptAlg.setOption("scaling", UserOpt.scaling)
        # OptAlg.setOption("IFILE", UserOpt.)
    elif Alg == "MIDACO":
        OptAlg.setOption("ACC", UserOpt.ACC)
        OptAlg.setOption("ISEED", UserOpt.ISEED)
        # OptAlg.setOption("QSTART",0)
        OptAlg.setOption("FSTOP", UserOpt.FSTOP)
        OptAlg.setOption("AUTOSTOP", UserOpt.AUTOSTOP)
        OptAlg.setOption("ORACLE", float(UserOpt.ORACLE))
        OptAlg.setOption("FOCUS", UserOpt.FOCUS)
        OptAlg.setOption("ANTS", UserOpt.ANTS)
        OptAlg.setOption("KERNEL", UserOpt.KERNEL)
        OptAlg.setOption("CHARACTER", UserOpt.CHARACTER)
        OptAlg.setOption("MAXEVAL",  int(UserOpt.MAXEVAL))
        OptAlg.setOption("MAXTIME", int(UserOpt.MAXTIME))
    elif Alg == "PyGMO_de":
        OptAlg = pg.de(gen=UserOpt.gen, f=UserOpt.f,
                       cr=UserOpt.cr, variant=UserOpt.variant,
                        tol=UserOpt.ftol, xtol=UserOpt.xtol,
                                    screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_jde":
        OptAlg = pg.jde(gen=UserOpt.gen, variant=UserOpt.variant,
                                     variant_adptv=UserOpt.variant_adptv,
                                     ftol=UserOpt.ftol, xtol=UserOpt.xtol,
                                     memory=UserOpt.memory,
                                     screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_mde_pbx":
        OptAlg = pg.mde_pbx(gen=UserOpt.gen, qperc=UserOpt.qperc,
                                         nexp=UserOpt.nexp, ftol=UserOpt.ftol,
                                         xtol=UserOpt.xtol,
                                         screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_de1220":
        OptAlg = pg.de1220(gen=UserOpt.gen,
                                         variant_adptv=UserOpt.variant_adptv,
                                         allowed_variants=UserOpt.allowed_variants,
                                         ftol=UserOpt.ftol, xtol=UserOpt.xtol,
                                         screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_pso":
        OptAlg = pg.pso(gen=UserOpt.gen, omega=UserOpt.omega,
                                     eta1=UserOpt.eta1, eta2=UserOpt.eta2,
                                     vcoeff=UserOpt.vcoeff,
                                     variant=UserOpt.variant,
                                     neighb_type=UserOpt.neighb_type,
                                     neighb_param=UserOpt.neighb_param)
    elif Alg == "PyGMO_pso_gen":
        OptAlg = pg.pso_gen(gen=UserOpt.gen, omega=UserOpt.omega,
                                         eta1=UserOpt.eta1, eta2=UserOpt.eta2,
                                         vcoeff=UserOpt.vcoeff,
                                         variant=UserOpt.variant,
                                         neighb_type=UserOpt.neighb_type,
                                         neighb_param=UserOpt.neighb_param)
    elif Alg == "PyGMO_sea":
        OptAlg = pg.sea(gen=UserOpt.gen, limit=UserOpt.limit)
    elif Alg == "PyGMO_sga":
        OptAlg = pg.sga(gen=UserOpt.gen, cr=UserOpt.cr,
                                     m=UserOpt.m, elitism=UserOpt.elitism,
                                     mutation=UserOpt.mutation,
                                     width=UserOpt.width,
                                     selection=UserOpt.selection,
                                     crossover=UserOpt.crossover)
    elif Alg == "PyGMO_vega":
        OptAlg = pg.vega(gen=UserOpt.gen, cr=UserOpt.cr,
                                      m=UserOpt.m, elitism=UserOpt.elitism,
                                      mutation=UserOpt.mutation,
                                      width=UserOpt.width,
                                      crossover=UserOpt.crossover)
    elif Alg == "PyGMO_sga_gray":
        OptAlg = pg.sga_gray(gen=UserOpt.gen, cr=UserOpt.cr,
                                          m=UserOpt.m, elitism=UserOpt.elitism,
                                          mutation=UserOpt.mutation,
                                          selection=UserOpt.selection,
                                          crossover=UserOpt.crossover)
    elif Alg == "PyGMO_nsga_II":
        OptAlg = pg.nsga_II(gen=UserOpt.gen, cr=UserOpt.cr,
                                         eta_c=UserOpt.eta_c, m=UserOpt.m,
                                         eta_m=UserOpt.eta_m)
    elif Alg == "PyGMO_sms_emoa":
        OptAlg = pg.sms_emoa(hv_algorithm=UserOpt.hv_algorithm,
                                          gen=UserOpt.gen, cr=UserOpt.cr,
                                          eta_c=UserOpt.eta_c,
                                          m=UserOpt.m, eta_m=UserOpt.eta_m)
    elif Alg == "PyGMO_pade":
        OptAlg = pg.pade(gen=UserOpt.gen,
                                      decomposition=UserOpt.decomposition,
                                      weights=UserOpt.weights,
                                      solver=UserOpt.weights,
                                      threads=UserOpt.threads,
                                      T=UserOpt.T, z=UserOpt.z)
    elif Alg == "PyGMO_nspso":
        OptAlg = pg.nspso(gen=UserOpt.gen, minW=UserOpt.minW,
                                       maxW=UserOpt.maxW, C1=UserOpt.C1,
                                       C2=UserOpt.C2, CHI=UserOpt.CHI,
                                       v_coeff=UserOpt.v_coeff,
                                       leader_selection_range=UserOpt.leader_selection_range)
    elif Alg == "PyGMO_spea2":
        OptAlg = pg.spea2(gen=UserOpt.gen, cr=UserOpt.cr,
                                       eta_c=UserOpt.eta_c, m=UserOpt.m,
                                       eta_m=UserOpt.eta_m,
                                       archive_size=UserOpt.archive_size)
    elif Alg == "PyGMO_sa_corana":
        OptAlg = pg.sa_corana(iter=UserOpt.iter, Ts=UserOpt.Ts,
                                           Tf=UserOpt.Tf, steps=UserOpt.steps,
                                           bin_size=UserOpt.bin_size,
                                           range=UserOpt.range)
    elif Alg == "PyGMO_bee_colony":
        OptAlg = pg.bee_colony(gen=UserOpt.gen,
                                            limit=UserOpt.limit)
    elif Alg == "PyGMO_ms":
        OptAlg = pg.ms(algorithm=UserOpt.algorithm,
                                    iter=UserOpt.iter)
    elif Alg == "PyGMO_mbh":
        OptAlg = pg.mbh(algorithm=UserOpt.algorithm,
                                     stop=UserOpt.stop,
                                     perturb=UserOpt.perturb,
                                     screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_cstrs_co_evolution":
        OptAlg = pg.cstrs_co_evolution(original_algo=UserOpt.original_algo,
                                                    original_algo_penalties=UserOpt.original_algo_penalties,
                                                    pop_penalties_size=UserOpt.pop_penalties_size,
                                                    gen=UserOpt.gen,
                                                    pen_lower_bound=UserOpt.pen_lower_bound,
                                                    pen_upper_bound=UserOpt.pen_upper_bound,
                                                    f_tol=UserOpt.f_tol,
                                                    x_tol=UserOpt.x_tol)
    elif Alg == "PyGMO_cstrs_immune_system":
        OptAlg = pg.cstrs_immune_system(algorithm=UserOpt.algorithm,
                                                     algorithm_immune=UserOpt.algorithm_immune,
                                                     gen=UserOpt.gen,
                                                     select_method=UserOpt.select_method,
                                                     inject_method=UserOpt.inject_method,
                                                     distance_method=UserOpt.distance_method,
                                                     phi=UserOpt.phi,
                                                     gamma=UserOpt.gamma,
                                                     sigma=UserOpt.sigma,
                                                     f_tol=UserOpt.f_tol,
                                                     x_tol=UserOpt.x_tol)
    elif Alg == "PyGMO_cstrs_core":
        OptAlg = pg.cstrs_core(algorithm=UserOpt.algorithm,
                                            repair_algorithm=UserOpt.repair_algorithm,
                                            gen=UserOpt.gen,
                                            repair_frequency=UserOpt.repair_frequency,
                                            repair_ratio=UserOpt.repair_ratio,
                                            f_tol=UserOpt.f_tol,
                                            x_tol=UserOpt.x_tol)
    elif Alg == "PyGMO_cs":
        OptAlg = pg.cs(max_eval=UserOpt.max_eval,
                                    stop_range=UserOpt.stop_range,
                                    start_range=UserOpt.start_range,
                                    reduction_coeff=UserOpt.reduction_coeff)
    elif Alg == "PyGMO_ihs":
        OptAlg = pg.ihs(iter=UserOpt.iter, hmcr=UserOpt.hmcr,
                                     par_min=UserOpt.par_min,
                                     par_max=UserOpt.par_max,
                                     bw_min=UserOpt.bw_min,
                                     bw_max=UserOpt.bw_max)
    elif Alg == "PyGMO_monte_carlo":
        OptAlg = pg.monte_carlo(iter=UserOpt.iter)
    elif Alg == "PyGMO_py_example":
        OptAlg = pg.py_example(iter=UserOpt.iter)
    elif Alg == "PyGMO_py_cmaes":
        OptAlg = pg.py_cmaes(gen=UserOpt.gen, cc=UserOpt.cc,
                                          cs=UserOpt.cs, c1=UserOpt.c1,
                                          cmu=UserOpt.cmu, ftol=UserOpt.ftol,
                                          xtol=UserOpt.xtol,
                                          memory=UserOpt.memory,
                                          screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_cmaes":
        OptAlg = pg.cmaes(gen=UserOpt.gen, cc=UserOpt.cc,
                                       cs=UserOpt.cs, c1=UserOpt.c1,
                                       cmu=UserOpt.cmu, ftol=UserOpt.ftol,
                                       xtol=UserOpt.xtol,
                                       memory=UserOpt.memory,
                                       screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_scipy_fmin":
        OptAlg = pg.scipy_fmin(maxiter=UserOpt.maxiter,
                                            xtol=UserOpt.xtol,
                                            ftol=UserOpt.ftol,
                                            maxfun=UserOpt.maxfun,
                                            disp=UserOpt.disp)
    elif Alg == "PyGMO_scipy_l_bfgs_b":
        OptAlg = pg.scipy_l_bfgs_b(maxfun=UserOpt.maxfun,
                                                m=UserOpt.m,
                                                factr=UserOpt.factr,
                                                pgtol=UserOpt.pgtol,
                                                epsilon=UserOpt.epsilon,
                                                screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_scipy_slsqp":
        OptAlg = pg.scipy_slsqp(max_iter=UserOpt.max_iter,
                                             acc=UserOpt.acc,
                                             epsilon=UserOpt.epsilon,
                                             screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_scipy_tnc":
        OptAlg = pg.scipy_tnc(maxfun=UserOpt.maxfun,
                                           xtol=UserOpt.xtol,
                                           ftol=UserOpt.ftol,
                                           pgtol=UserOpt.pgtol,
                                           epsilon=UserOpt.epsilon,
                                           screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_scipy_cobyla":
        OptAlg = pg.scipy_cobyla(max_fun=UserOpt.max_fun,
                                              rho_end=UserOpt.rho_end,
                                              screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_nlopt_cobyla":
        OptAlg = pg.nlopt_cobyla(max_iter=UserOpt.max_iter,
                                              ftol=UserOpt.ftol,
                                              xtol=UserOpt.xtol)
    elif Alg == "PyGMO_nlopt_bobyqa":
        OptAlg = pg.nlopt_bobyqa(max_iter=UserOpt.max_iter,
                                              ftol=UserOpt.ftol,
                                              xtol=UserOpt.xtol)
    elif Alg == "PyGMO_nlopt_sbplx":
        OptAlg = pg.nlopt_sbplx(max_iter=UserOpt.max_iter,
                                             ftol=UserOpt.ftol,
                                             xtol=UserOpt.xtol)
    elif Alg == "PyGMO_nlopt_mma":
        OptAlg = pg.nlopt_mma(max_iter=UserOpt.max_iter,
                                           ftol=UserOpt.ftol,
                                           xtol=UserOpt.xtol)
    elif Alg == "PyGMO_nlopt_auglag":
        OptAlg = pg.nlopt_auglag(aux_algo_id=UserOpt.aux_algo_id,
                                              max_iter=UserOpt.max_iter,
                                              ftol=UserOpt.ftol,
                                              xtol=UserOpt.xtol,
                                              aux_max_iter=UserOpt.aux_max_iter,
                                              aux_ftol=UserOpt.aux_ftol,
                                              aux_xtol=UserOpt.aux_xtol)
    elif Alg == "PyGMO_nlopt_auglag_eq":
        OptAlg = pg.nlopt_auglag_eq(aux_algo_id=UserOpt.aux_algo_id,
                                                 max_iter=UserOpt.max_iter,
                                                 ftol=UserOpt.ftol,
                                                 xtol=UserOpt.xtol,
                                                 aux_max_iter=UserOpt.aux_max_iter,
                                                 aux_ftol=UserOpt.aux_ftol,
                                                 aux_xtol=UserOpt.aux_xtol)
    elif Alg == "PyGMO_nlopt_slsqp":
        OptAlg = pg.nlopt_slsqp(max_iter=UserOpt.max_iter,
                                             ftol=UserOpt.ftol,
                                             xtol=UserOpt.xtol)
    elif Alg == "PyGMO_gsl_nm2rand":
        OptAlg = pg.gsl_nm2rand(max_iter=UserOpt.max_iter)
    elif Alg == "PyGMO_gsl_nm2":
        OptAlg = pg.gsl_nm2(max_iter=UserOpt.max_iter)
    elif Alg == "PyGMO_gsl_nm":
        OptAlg = pg.gsl_nm(max_iter=UserOpt.max_iter)
    elif Alg == "PyGMO_gsl_pr":
        OptAlg = pg.gsl_pr(max_iter=UserOpt.max_iter,
                                        step_size=UserOpt.step_size,
                                        tol=UserOpt.tol,
                                        grad_step_size=UserOpt.grad_step_size,
                                        grad_tol=UserOpt.grad_tol)
    elif Alg == "PyGMO_gsl_fr":
        OptAlg = pg.gsl_fr(max_iter=UserOpt.max_iter,
                                        step_size=UserOpt.step_size,
                                        tol=UserOpt.tol,
                                        grad_step_size=UserOpt.grad_step_size,
                                        grad_tol=UserOpt.grad_tol)
    elif Alg == "PyGMO_gsl_bfgs2":
        OptAlg = pg.gsl_bfgs2(max_iter=UserOpt.max_iter,
                                           step_size=UserOpt.step_size,
                                           tol=UserOpt.tol,
                                           grad_step_size=UserOpt.grad_step_size,
                                           grad_tol=UserOpt.grad_tol)
    elif Alg == "PyGMO_gsl_bfgs":
        OptAlg = pg.gsl_bfgs(max_iter=UserOpt.max_iter,
                                          step_size=UserOpt.step_size,
                                          tol=UserOpt.tol,
                                          grad_step_size=UserOpt.grad_step_size,
                                          grad_tol=UserOpt.grad_tol)
    elif Alg == "PyGMO_snopt":
        OptAlg = pg.snopt(major_iter=UserOpt.major_iter,
                                       feas_tol=UserOpt.feas_tol,
                                       opt_tol=UserOpt.opt_tol,
                                       screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_ipopt":
        OptAlg = pg.ipopt(max_iter=UserOpt.max_iter,
                                       constr_viol_tol=UserOpt.contr_viol_tol,
                                       dual_inf_tol=UserOpt.dual_inf_tol,
                                       compl_inf_tol=UserOpt.compl_inf_tol,
                                       nlp_scaling_method=UserOpt.nlp_scaling_method,
                                       obj_scaling_factor=UserOpt.obj_scaling_factor,
                                       mu_init=UserOpt.mu_init,
                                       screen_output=UserOpt.screen_output)
    elif Alg == "PyGMO_cstrs_self_adaptive":
        OptAlg = pg.cstrs_self_adaptive(algorithm=UserOpt.algorithm,
                                                     max_iter=UserOpt.max_iter,
                                                     f_tol=UserOpt.f_tol,
                                                     x_tol=UserOpt.x_tol)
    else:
        sys.exit("Error on line " + str(inspect.currentframe().f_lineno) +
                 " of file " + __file__ +
                 ": Algorithm misspelled or not supported")
    return OptAlg
