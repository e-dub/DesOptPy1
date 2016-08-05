# -*- coding: utf-8 -*-
'''
-------------------------------------------------------------------------------
Title:          OptHis2HTML.py
Units:          Unitless
Date:           July 9, 2016
Authors:        F. Wachter, E.J. Wehrle
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
Description
-------------------------------------------------------------------------------
Open the optimization history (the .cue and .bin file) and processes them into
HTML format during runtime so one can check the status and progress of the
optimization.

-------------------------------------------------------------------------------
To do and ideas
-------------------------------------------------------------------------------

Only read and write last evaluation to save time?
csv write only new line!!!!!!!!! How?????
last lines repeated way too much!!!!!!
-------------------------------------------------------------------------------
'''

import csv
import glob
import os
import shutil
from time import localtime, strftime, time
import numpy as np
from pyOpt import History
from Normalize import normalize, denormalize
from OptReadHis import OptReadHis

def OptHis2HTML(OptName, Alg, AlgOptions, DesOptDir, x0, xL, xU, gc, DesVarNorm, inform,
                starttime, StatusDirectory=""):
    StartTime = str(starttime)[0:10] + "000"
    EndTime = ""
    RefRate = '2000'
    if inform != "Running":
        EndTime = str(time())[0:10] + "000"
        RefRate = '1000000'
    if Alg in ["CONMIN", "MMA", "GCMMA", "NLPQLP", "SLSQP", "FSQP", "KSOPT"]:
        EvalGenIter = "Iteration"
    else:
        EvalGenIter = 'Evaluation'
       # Label and Legends may be Iteration, Generation or Evaluations depending on Algorithm
    if StatusDirectory == "":  # Change the target directory for the status report files if the user wants to
        StatusDirectory = DesOptDir
    # Variables for the data extraction
    pos_of_best_ind = []  # position of the best individual if a GA or ES is used as algorithm
    fIter = []  # objective function array
    xIter = []  # design vector array
    gIter = []  # constraint vector array
    # template_directory= DesOpt_Base + "/.DesOptPy/_OptStatusReport/"  # directory with the html files etc.
    template_directory = os.path.dirname(os.path.realpath(__file__)) + \
                                         "/StatusReportFiles/"  # directory with the html files etc.
    fIter, xIter, gIter, gGradIter, fGradIter, inform =  OptReadHis(OptName,
                                                                    Alg,
                                                                    AlgOptions,
                                                                    x0, xL, xU,
                                                                    DesVarNorm,
                                                                    Iter="All")
    nIter = np.shape(xIter)[0]-1
    xLabel = [EvalGenIter]
    if  xIter.size != 0:
        if DesVarNorm == False:
            xIterDenorm = xIter
        else:
            xIterDenorm = denormalize(xIter, x0, xL, xU, DesVarNorm)
        with open('f_gMax.csv', 'ab') as csvfile:
            datawriter = csv.writer(csvfile, dialect='excel')
            if np.size(gIter) == 0:
                datawriter.writerow([nIter, str(float(fIter)), []])
            else:
                datawriter.writerow([nIter, str(float(fIter[nIter])),
                                     float(np.max(gIter[nIter]))])
        datasets = str(xIter.tolist()).strip('[]')
        with open('xNorm.csv', 'ab') as csvfile:
            datawriter = csv.writer(csvfile, dialect='excel',
                                    quotechar=' ')
            datawriter.writerow([nIter, datasets])
        datasets_denorm = str(xIterDenorm.tolist()).strip('[]')
        with open('x.csv', 'ab') as csvfile:
            datawriter = csv.writer(csvfile, dialect='excel',
                                    quotechar=' ')
            datawriter.writerow([nIter, datasets_denorm])
        if gIter.size != 0:
            datasetsg = str(gIter.tolist()).strip('[]')
            with open('g.csv', 'ab') as csvfile:
                datawriter = csv.writer(csvfile, dialect='excel', quotechar=' ')
                datawriter.writerow([nIter, datasetsg])
    else:
        with open('f_gMax.csv', 'wb') as csvfile:
            datawriter = csv.writer(csvfile, dialect='excel')
            datawriter.writerow([EvalGenIter, 'Objective function', 'Max constraint function'])
        with open('xNorm.csv', 'wb') as csvfile:
            datawriter = csv.writer(csvfile, delimiter=',', escapechar=' ',
                                    quoting=csv.QUOTE_NONE)
            labels = xLabel
            for i in range(1, xL.size + 1):
                labels = labels + ['x' + str(i)]
            datawriter.writerow(labels)
        with open('x.csv', 'wb') as csvfile:
            datawriter = csv.writer(csvfile, delimiter=',', escapechar=' ',
                                    quoting=csv.QUOTE_NONE)
            labels = xLabel
            for i in range(1, xL.size + 1):
                labels = labels + ['x' + str(i)]
            datawriter.writerow(labels)
        with open('g.csv', 'wb') as csvfile:
            datawriter = csv.writer(csvfile, delimiter=',', escapechar=' ',
                                    quoting=csv.QUOTE_NONE)
            labels = xLabel
            if np.size(gc) != 0:
                for i in range(1, gc.size + 1):
                    labels = labels + ['g' + str(i)]
            datawriter.writerow(labels)
    time_now = strftime("%Y-%b-%d %H:%M:%S", localtime())  # update the time for the information table
    number_des_vars = "0"
    number_constraints = "0"

# -----------------------------------------------------------------------------
# Everything is computed, now the html master template is opened and the
# placeholders are replaced with the right values
# -----------------------------------------------------------------------------
    html = open(template_directory + '/initial.html', 'r')  # open template
    hstr = html.read()
    html.close()
    # replace the placeholder values with the true values
    if gIter.size != 0 or gIter.size > 100:
        hstrnew = hstr.replace('xxxxName', OptName)
        hstrnew = hstrnew.replace('xxxxTime', time_now)
        #hstrnew = hstrnew.replace('xxxxtableObjFct', ObjFct_table)
        #hstrnew = hstrnew.replace('xxxxtableDesVar', DesVar_table)
        hstrnew = hstrnew.replace('xxxxnumber_des_var', number_des_vars * 2)
        #hstrnew = hstrnew.replace('xxxxtableConstr', Constraint_table)
        hstrnew = hstrnew.replace('xxxxnumber_constraints', number_constraints)
        hstrnew = hstrnew.replace('xxxxAlg', Alg)
        #hstrnew = hstrnew.replace('xxxxStatus', str(inform))
        hstrnew = hstrnew.replace('xxxxRefRate', RefRate)
        hstrnew = hstrnew.replace('xxxxStartTime', StartTime)
        hstrnew = hstrnew.replace('xxxxEndTime', EndTime)
        hstrnew = hstrnew.replace('xxxxIteration', EvalGenIter)
    else:
        hstrnew = hstr.replace('xxxxName', OptName)
        hstrnew = hstrnew.replace('xxxxTime', time_now)
        #hstrnew = hstrnew.replace('xxxxtableObjFct', ObjFct_table)
        #hstrnew = hstrnew.replace('xxxxtableDesVar', DesVar_table)
        hstrnew = hstrnew.replace('xxxxAlg', Alg)
        #hstrnew = hstrnew.replace('xxxxStatus', inform)
        hstrnew = hstrnew.replace('xxxxRefRate', RefRate)
        hstrnew = hstrnew.replace('xxxxStartTime', StartTime)
        hstrnew = hstrnew.replace('xxxxEndTime', EndTime)
        hstrnew = hstrnew.replace('xxxxIteration', EvalGenIter)
        # remove the hmtl parts which are only needed for constrained problems
        try:
            for i in range(0, 10):
                hstrnew = hstrnew[0:hstrnew.find("<!--Start of constraint html part-->")] + hstrnew[hstrnew.find("<!--End of constraint html part-->") + 34:-1]
        except:
            print ""
    # generate a new html file which is filled with the actual content
    html = open('initial1.html', 'w')
    html.write(hstrnew)
    html.close()
    # copy everything needed to the result directory
    # no more coping of everything!
    # TODO give absolute path above!!
    if not os.path.exists(StatusDirectory + os.sep + "Results" + os.sep +
                          OptName):
        os.makedirs(StatusDirectory + os.sep + "Results" + os.sep + OptName)
    shutil.copy("initial1.html",
                StatusDirectory + os.sep + "Results" + os.sep + OptName +
                os.sep + OptName + "_Status.html")
    shutil.copy("f_gMax.csv",
                StatusDirectory + os.sep + "Results" + os.sep + OptName +
                os.sep + "f_gMax.csv")
    shutil.copy("x.csv",
                StatusDirectory + os.sep + "Results" + os.sep + OptName +
                os.sep + "x.csv")
    shutil.copy("xNorm.csv",
                StatusDirectory + os.sep + "Results" + os.sep + OptName +
                os.sep + "xNorm.csv")
    shutil.copy("g.csv",
                StatusDirectory + os.sep + "Results" + os.sep + OptName +
                os.sep + "g.csv")
    for file in glob.glob(template_directory + "*.png"):
        shutil.copy(file, StatusDirectory + os.sep + "Results" + os.sep +
        OptName + os.sep)
    for file in glob.glob(template_directory + "*.js"):
        shutil.copy(file, StatusDirectory + os.sep + "Results" + os.sep +
        OptName + os.sep)
    for file in glob.glob(template_directory + "*.css"):
        shutil.copy(file, StatusDirectory + os.sep + "Results" + os.sep +
        OptName + os.sep)
    for file in glob.glob(template_directory + "*.ico"):
        shutil.copy(file, StatusDirectory + os.sep + "Results" + os.sep +
        OptName + os.sep)
    for file in glob.glob(template_directory + "view_results.py"):
        shutil.copy(file, StatusDirectory + os.sep + "Results" + os.sep +
        OptName + os.sep)
    return 0
