# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 12:30:04 2013

@author: m.richter
"""

import os, time, pprint, copy, glob, sys
import cPickle as pickle
import numpy as np
import scipy.io as spio


#---------------------------------------------------------------------------------------------------
# Parallel
#---------------------------------------------------------------------------------------------------
class info:
    def __init__(self, x, f, g, xdiff, xit, nproc, nodes, OptModel, jobName, optDir, currDir, mainDir, localDir, workDir, user):
        self.x              =   x
        self.f              =   f                   # value of obj Fn e.g. mass
        self.g              =   g                   # equality constraints, dim(g)
        self._xdiff         =   xdiff               # deltax for finite Differences(FD)
        self._xit           =   xit
        self._nproc         =   nproc               # Number of processors
        self._nodes         =   nodes               # job-Name of the interactive session, e.g. 146339
        self._OptModel      =   OptModel            # Optimization Problem
        self.jobName       =   jobName             # Sort of analysis, e.g. ParaFD
        self.optDir         =   optDir              # /home/$User/DesOpt/
        self.currDir        =   currDir             # /home/$User/DesOpt/OptModel
        self.mainDir        =   mainDir             # /home/$User/DesOptRun/OptName/ParallelRun/   or /tmp/$USER.$JOB.ID/OptName
        self.localDir       =   localDir            # /home/$User/DesOptRun/OptName/ParallelRun/nIter_X/
        self.workDir        =   workDir             # /home/$User/DesOptRun/OptName/ParallelRun/nIter_X/0 ... numberOf(DesVar)
        self._user          =   user                # cp batch: $User

    def dump(self, data):
        pkl_file = open("input.pkl", "wb")
        pickle.dump(data, pkl_file)
        pkl_file.close()
        return 0


def schedule(ii, info):
    info._xit = copy.deepcopy(info.x)
    info._xit[ii] = info.x[ii] + info._xdiff
    info.workDir = info.localDir + str(ii+1)
    os.mkdir(info.workDir)
    os.system("cp -r %s %s" %(os.path.join(info.optDir, "ParaPythonFn/*"), info.workDir))
    os.system("cp -r %s %s" %(info.currDir ,info.workDir))


    # Saving the variables to a pickle file im workDir
    os.chdir(info.workDir)        # might be faster to use object pickling instead
    data = {}
    data['OptModel']        = info._OptModel
    data['user']            = info._user
    data['x']               = info.x
    data['f']               = info.f
    data['g']               = info.g
    data['nproc']           = info._nproc
    data['xdiff']           = info._xdiff
    data['currDir']         = info.currDir
    data['optDir']          = info.optDir
    data['mainDir']         = info.mainDir
    data['localDir']        = info.localDir
    data['workDir']         = info.workDir
    data['xit']             = info._xit
    data['nodes']           = info._nodes
    data['jobName']         = info.jobName
    info.dump(data)
    os.chdir('..')
    return 1

def submit(ii, info, *nodes):
    hostname = os.getenv('HOSTNAME')
    try: host = int(hostname.replace('node',''))
    except: host =0
    os.chdir(info.workDir)
    if 0 < host < 27:
        if nodes:
            if nodes[0] == True:
                os.system(". /etc/profile; qsub -v workdir=%s -v OptModel=%s -N %s -pe shm %i ParallelFD.sh" %(info.workDir, info._OptModel, info.jobName, info._nproc))
            elif nodes[0] == False:
                os.system(". /etc/profile; qsub -q smallnodes.q -v workdir=%s -v OptModel=%s -N %s -pe shm %i ParallelFD.sh" %(info.workDir, info._OptModel, info.jobName, info._nproc))
            else:pass
            #endif
        else: pass
        #endif
    elif 26 < host < 41:
        #submission for SLURM, no distinction neccessary since all nodes are the same
#        os.system(". /etc/profile; sbatch --export=PATH,workdir=%s,OptModel=%s -J%s ParallelFD_SLURM.sh" %(info.workDir, info._OptModel, info._jobName))
        os.system(". /etc/profile; sbatch --export=PATH,workdir=%s,OptModel=%s -J%s ParallelFD_SLURM.sh" %(os.getcwd(), info._OptModel, info.jobName))
    elif host == 0:
        os.system("python ParallelFD.py&")
    else:
        print 'Check the environmental variables! There is something wrong here! No job has been submitted!'
    #endif
    os.chdir('..')
    return 1

def Para(x, f, g, deltax, OptName, OptNodes):
    #Situation from DesOpt.py i.e. in problem folder
    """

    :rtype : object
    """
    OptModel                = OptName.split('_')[0]
    #Definition of local variables
    nIter                   = 1                                                             #Number of Iteration of the optimization
    nb                      = 0                                                             #Number of function evaluation per iteration
    optDoc                  = True
    jobRec                  = False
    nodes                   = False                                                         #False allows onle the use of smallnodes
#    jobName                 = 'ParaFD'
    nproc                   = 2
    xdiff                   = deltax
    optDir                  = os.path.normpath(os.path.join(os.getcwd(), ".."))             #returns a string !!!
    currDir                 = os.getcwd()                                                   #alternativ = os.getenv('HOME') + "jobs" + OptModel
    user                    = os.getenv('USER')                                             #returns a string


    #Setting up the directories for the parallelization
    os.chdir(os.getenv('HOME'))
    try:    os.mkdir(os.getenv('HOME') + 'DesOptRun')
    except: pass
    try:    os.mkdir(os.getenv('HOME') + "/DesOptRun/" + OptName)
    except: pass
    mainDir = os.getenv('HOME')+ "/DesOptRun/" + OptName + "/ParallelRun/"
    try:    os.mkdir(mainDir)
    except: pass
    os.chdir(mainDir)
    while ("nIter_" + str(nIter) in os.listdir(mainDir)):
        nIter += 1
    else: pass
    localDir = mainDir + "nIter_" + str(nIter) + "/"
    try:    os.mkdir(localDir)
    except: pass

    workDir     = []
    xit         = []
    jobName     = []
    obj         = info(x, f, g, xdiff, xit, nproc, nodes, OptModel, jobName, optDir, currDir, mainDir, localDir, workDir, user)      # instantiating infos

    print "###################################################"
    print "\tStart submission !!!"
    print "###################################################"
    for iii in xrange(len(x)):
        obj.jobName = 'iter' + str(nIter) + '_' + str(iii+1)
        schedule(iii,obj)
        submit(iii,obj,OptNodes)
        #endfor
    time.sleep(1)

    print "###################################################"
    print "\t Start collection  !!!"
    print "\t    nIter:   %i" %(nIter)
    print "###################################################"

    #Declaration of the lists
    f = [f]
    fnew    =       np.zeros([len(f),len(x)])
    gnew    =       np.zeros([len(g),len(x)])
    for ii in xrange(len(x)):
        workDir = localDir + str(ii+1)
        os.chdir(workDir)
        if optDoc == True:
            print "cwd: \t" + os.getcwd()
        else:   pass
        LoopError = 1
        iii = 0
        while LoopError == 1:           # File not found
            if ('finished.txt' in os.listdir(os.getcwd())):
                with open("output.pkl", "rb") as pkl_rb:
                    data = pickle.load(pkl_rb)
                    #pprint.pprint(data)

                fit             = data['f']
                git             = data['g']
                fnew[0,ii]      = data['fnew']
                gnew[:,ii]      = data['gnew']
                LoopError = 0
                nb +=1
            #endif
            else:
                time.sleep(1)
                os.system('ls -l %s *.txt' %(workDir))
                print "No output .txt-files found !!!"
                iii += 1
        if iii > 10*len(x):
            print "There must be an error with the system response"
            print "Check out ANSYS or another analysis tool!!!"
    #endfor

    dgdx    = np.zeros([len(g),len(x)])
    dfdx    = (fnew - fit)/xdiff
    for ii in xrange(np.shape(dgdx)[1]):
        dgdx[:,ii]    = ((gnew[:,ii] - git)/xdiff)          # Calculation column-wise!
    #endof

    grad = {}
    grad['f']       =   f
    grad['g']       =   g
    grad['fnew']    =   fnew
    grad['gnew']    =   gnew
    grad['dfdx']    =   dfdx
    grad['dgdx']    =   dgdx

    os.chdir(localDir)
    #  Save in Python format (.pkl)
    output = open(OptModel + "_grad.pkl" , 'wb')
    pickle.dump(grad,output)
    output.close()
    # Save in MATLAB format (.mat)
    spio.savemat(OptModel + '_grad.mat' , grad,oned_as='row')

    print "###################################################"
    print "\t Collection succesful %i ;)" %(nIter)
    print "###################################################"
    if optDoc == True:
        print "f"       + "\t" + str(f)
        print "g"       + "\t" + str(g)
        print "dfdx"    + "\t" + str(dfdx)
        print "dgdx"    + "\t" + str(dgdx)
    else:   pass
    # Removing all protocols from the submitted jobs -> put on True if still needed !!!
    if jobRec == False:
        hostname = os.getenv('HOSTNAME')
        try: host = int(hostname.replace('node',''))
        except: host=99
        if 0 < host < 27:
            os.chdir(os.getenv('HOME'))
            os.system("rm -r iter*.*")
        else: pass  #Slurm puts the submission file in the workDir, deleting after collection ?????
    else: pass

    os.chdir(currDir)
    return dfdx, dgdx, nb
#enddef



if __name__ == "__main__":
    #Test: calling the function !!!
    x = []
    xL = []
    xU = []
    g = []
    xdiff = 1e-2
    OptNodes = False
    OptModel = 'Truss45BarShapeSize'               #Truss12BarShapeTopoANSYS, Truss45BarShapeSizeANSYS, Truss10BarShapeSizeANSYS
    os.chdir(OptModel)

    if OptModel == 'DevExample':
        a = 3
    elif OptModel == 'Truss38Bar':
        a = 38
        for i in range(a):
            x.append(2500)
            xL.append(0)
            xU.append(250)
        sys.path.append(os.getcwd())
        import SysEq
        f, g = SysEq.SysEq(x)
    elif OptModel == 'Truss12BarShapeTopoANSYS':
        x = np.loadtxt('x0.txt')
        OptName = OptModel + "_MMA_201404150855"
        with open("init12.pkl", "rb") as pkl_rb:
            data = pickle.load(pkl_rb)
            f = data['f']
            g = data['g']
    elif OptModel == 'Truss10BarShapeSizeANSYS':
        x = np.loadtxt('x0.txt')
        OptName = OptModel + "_MMA_201404150855"
        with open("initial12.pkl", "rb") as pkl_rb:
            data = pickle.load(pkl_rb)
            f = data['f']
            g = data['g']
    elif OptModel == 'Truss45BarShapeSize':
        x = np.loadtxt('x0.txt')
        OptName = OptModel + "_MMA_201404151443"
        with open("init45.pkl", "rb") as pkl_rb:
            data = pickle.load(pkl_rb)
            f = data['f']
            g = data['g']

    dfdx, dgdx, nb = Para(x, f, g, xdiff, OptName, OptNodes)
