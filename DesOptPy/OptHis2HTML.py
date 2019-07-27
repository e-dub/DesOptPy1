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
see DesOpt.py
'''
from __future__ import absolute_import, division, print_function
import csv
import glob
import os
import shutil
from time import localtime, strftime, time
import numpy as np
from pyOpt import History
from DesOptPy.Normalize import normalize, denormalize
from DesOptPy.OptReadHis import OptReadHis


def OptHis2HTML(OptName, Alg, AlgOptions, DesOptDir, x0, xL, xU, gc, DesVarNorm, inform,
                starttime, StatusDirectory=""):
# -----------------------------------------------------------------------------
# General calculations for the uppermost information table are computed like
# time running, algorithm name, optimization problem name etc.
# -----------------------------------------------------------------------------
    StartTime = str(starttime)[0:10] + "000"
    EndTime = ""
    RefRate = '2000'
    if inform != "Running":
        EndTime = str(time())[0:10] + "000"
        RefRate = '1000000'
    Iteration = 'Iteration'   # Label and Legends may be Iteration, Generation or Evaluations depending on Algorithm
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
                                                                    gc,
                                                                    DesVarNorm)
    if xIter.size > 0:
        nIter = np.shape(xIter)[1]-1
    else:
        nIter = -1
# -----------------------------------------------------------------------------
# The design variables are normalized or denormalized so both can be displayed
# in the graphs and tables
# -----------------------------------------------------------------------------
    if  xIter.size != 0:
        if DesVarNorm == False:
            xIterDenorm = np.zeros((nIter + 1, len(xIter[0])))
            for y in range(0, nIter + 1):
                xIterDenorm[y] = xIter[y]
#            for y in range(0, nIter + 1):
#                [xIter[y, :], xLnorm, xUnorm] = normalize(xIterDenorm[y, :],
#                                                          x0, xL, xU, "xLxU")
        else:
            xIter = xIter[0:np.size(xL), :]
            xIterDenorm = np.zeros(np.shape(xIter))
            for ii in range(np.shape(xIterDenorm)[1]):
                xIterDenorm[:, ii] = denormalize(xIter[:, ii], x0, xL, xU, DesVarNorm)
            #xIterDenorm = np.zeros((nIter + 1, len(x0)))
#            for y in range(0, nIter + 1):
#                print(denormalize(xIter[y, 0:len(x0)], x0, xL, xU, DesVarNorm)
#                xIterDenorm[y, :] = denormalize(xIter[y, 0:len(x0)],
#                                                       x0, xL, xU, DesVarNorm)
    time_now = strftime("%Y-%b-%d %H:%M:%S", localtime())  # update the time for the information table
    number_des_vars = "0"
    number_constraints = "0"

# -----------------------------------------------------------------------------
# The .csv files are created and the first row is filled with the correct
# labels. Those .csv files
# are loaded by the javascript library. Afterwards the files are closed.
# -----------------------------------------------------------------------------
    with open('objFct_maxCon.csv', 'w') as csvfile:
        datawriter = csv.writer(csvfile, dialect='excel')
        datawriter.writerow(['Iteration', 'Objective function', 'Constraint'])
    csvfile.close()
    with open('desVarsNorm.csv', 'w') as csvfile:
        datawriter = csv.writer(csvfile, delimiter=',', escapechar=' ',
                                quoting=csv.QUOTE_NONE)
        labels = ['Iteration']
        if xIter.size != 0:
            for i in range(1, xIter.shape[1] + 1):
                labels = labels + ['x' + str(i)]
        datawriter.writerow(labels)
    csvfile.close()
    with open('desVars.csv', 'w') as csvfile:
        datawriter = csv.writer(csvfile, delimiter=',', escapechar=' ',
                                quoting=csv.QUOTE_NONE)
        labels = ['Iteration']
        if xIter.size != 0:
            for i in range(1, xIter.shape[1] + 1):
                labels = labels + ['x' + str(i)]
        datawriter.writerow(labels)
    csvfile.close()
    with open('constraints.csv', 'w') as csvfile:
        datawriter = csv.writer(csvfile, delimiter=',', escapechar=' ',
                                quoting=csv.QUOTE_NONE)
        labels = ['Iteration']
        if gIter.size != 0:
            for i in range(1, gIter.shape[1] + 1):
                labels = labels + ['g' + str(i)]
        datawriter.writerow(labels)
    csvfile.close()

# -----------------------------------------------------------------------------
# Now the real data like obj fct value and constraint values are writen into
# the .csv files
# -----------------------------------------------------------------------------
    # Extremely slow for large number of evaluations!!!! Needs to be redone!
    # Objective function and maximum constraint values
    for x in range(0, nIter + 1):
        with open('objFct_maxCon.csv', 'a') as csvfile:
            datawriter = csv.writer(csvfile, dialect='excel')
            if np.size(gIter[x]) == 0:
                datawriter.writerow([x, str(float(fIter[x])),  []])
            else:
                datawriter.writerow([x, str(float(fIter[x])),
                                     float(np.max(gIter[x]))])
        csvfile.close()
    # Normalized design variables
    if xIter.size != 0:
        for x in range(0, nIter + 1):
            datasets = str(xIter[x][:].tolist()).strip('[]')
            with open('desVarsNorm.csv', 'a') as csvfile:
                datawriter = csv.writer(csvfile, dialect='excel',
                                        quotechar=' ')
                datawriter.writerow([x, datasets])
            csvfile.close()
    # non normalized design variables
    if xIter.size != 0:
        for x in range(0, nIter + 1):
            datasets_denorm = str(xIterDenorm[x][:].tolist()).strip('[]')
            with open('desVars.csv', 'a') as csvfile:
                datawriter = csv.writer(csvfile, dialect='excel',
                                        quotechar=' ')
                datawriter.writerow([x, datasets_denorm])
            csvfile.close()
    # constraint variables
    if gIter.size != 0:
        for x in range(0, nIter + 1):
            datasetsg = str(gIter[x][:].tolist()).strip('[]')
            with open('constraints.csv', 'a') as csvfile:
                datawriter = csv.writer(csvfile, dialect='excel', quotechar=' ')
                datawriter.writerow([x, datasetsg])
            csvfile.close()

# -----------------------------------------------------------------------------
# The data for the graphs is generated, now follows the table generation
# routine
# -----------------------------------------------------------------------------
    # Objective function table generation
    ObjFct_table = "<td></td>"
    if xIter.size != 0:
        if gIter.size != 0:
            for x in range(0, nIter + 1):
                ObjFct_table += "<tr>\n<td>" + str(x) + "</td>\n<td>" + \
                                str(round(fIter[x][0], 4)) + "</td>\n<td>" + \
                                str(round(np.max(gIter[x]), 4)) + \
                                "</td>\n</tr>"
        else:
            for x in range(0, nIter + 1):
                ObjFct_table += "<tr>\n<td>" + str(x) + "</td>\n<td>" + \
                                str(round(fIter[x][0], 4)) + \
                                "</td>\n<td> no constraints </td>\n</tr>"
    # Design Variable table generation
    DesVar_table = "<td></td>"
    if xIter.size != 0:
        number_des_vars = str(len(xIter[0]))
        for x in range(0, len(xIter[0])):
            DesVar_table += "<td>" + "x&#770;<sub>" + str(x + 1) + \
                            "</sub></td>" + "<td>" + "x<sub>" + str(x + 1) +  \
                            " </sub></td>"
        for y in range(0, nIter + 1):
            DesVar_table += "<tr>\n<td>" + str(y) + "</td>"
            for x in range(0, len(xIter[0])):
                DesVar_table += "<td>" + str(round(xIter[y][x], 4)) + \
                                "</td><td>" + \
                                str(round(xIterDenorm[y][x], 4)) + \
                                "</td>"
            DesVar_table += "</tr>"
    # Constraint table generation
    Constraint_table = "<td></td>"
    if gIter.size != 0:
        number_constraints = str(len(gIter[0]))
        for x in range(0, len(gIter[0])):
            Constraint_table += "<td>" + "g<sub>" + str(x + 1) + "</sub></td>"
        for y in range(0, nIter + 1):
            Constraint_table += "<tr>\n<td>" + str(y) + "</td>"
            for x in range(0, len(gIter[0])):
                if (round(gIter[y][x], 4) > 0):
                    Constraint_table += "<td class=\"negativ\">" + \
                                        str(round(gIter[y][x], 4)) + "</td>"
                else:
                    Constraint_table += "<td class=\"positiv\">" + \
                                        str(round(gIter[y][x], 4)) + "</td>"
            Constraint_table += "</tr>"

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
        hstrnew = hstrnew.replace('xxxxtableObjFct', ObjFct_table)
        hstrnew = hstrnew.replace('xxxxtableDesVar', DesVar_table)
        hstrnew = hstrnew.replace('xxxxnumber_des_var', number_des_vars * 2)
        hstrnew = hstrnew.replace('xxxxtableConstr', Constraint_table)
        hstrnew = hstrnew.replace('xxxxnumber_constraints', number_constraints)
        hstrnew = hstrnew.replace('xxxxAlg', Alg)
        hstrnew = hstrnew.replace('xxxxStatus', str(inform))
        hstrnew = hstrnew.replace('xxxxRefRate', RefRate)
        hstrnew = hstrnew.replace('xxxxStartTime', StartTime)
        hstrnew = hstrnew.replace('xxxxEndTime', EndTime)
        hstrnew = hstrnew.replace('xxxxIteration', Iteration)
    else:
        hstrnew = hstr.replace('xxxxName', OptName)
        hstrnew = hstrnew.replace('xxxxTime', time_now)
        hstrnew = hstrnew.replace('xxxxtableObjFct', ObjFct_table)
        hstrnew = hstrnew.replace('xxxxtableDesVar', DesVar_table)
        hstrnew = hstrnew.replace('xxxxAlg', Alg)
        hstrnew = hstrnew.replace('xxxxStatus', inform)
        hstrnew = hstrnew.replace('xxxxRefRate', RefRate)
        hstrnew = hstrnew.replace('xxxxStartTime', StartTime)
        hstrnew = hstrnew.replace('xxxxEndTime', EndTime)
        hstrnew = hstrnew.replace('xxxxIteration', Iteration)
        # remove the hmtl parts which are only needed for constrained problems
        try:
            for i in range(0, 10):
                hstrnew = hstrnew[0:hstrnew.find("<!--Start of constraint html part-->")] + hstrnew[hstrnew.find("<!--End of constraint html part-->") + 34:-1]
        except:
            print("")
    # generate a new html file which is filled with the actual content
    html = open('initial1.html', 'w')
    html.write(hstrnew)
    html.close()
    # copy everything needed to the result directory
    if not os.path.exists(StatusDirectory + os.sep + "Results" + os.sep +
                          OptName):
        os.makedirs(StatusDirectory + os.sep + "Results" + os.sep + OptName)
    shutil.copy("initial1.html",
                StatusDirectory + os.sep + "Results" + os.sep + OptName +
                os.sep + OptName + "_Status.html")
    shutil.copy("objFct_maxCon.csv",
                StatusDirectory + os.sep + "Results" + os.sep + OptName +
                os.sep + "objFct_maxCon.csv")
    shutil.copy("desVars.csv",
                StatusDirectory + os.sep + "Results" + os.sep + OptName +
                os.sep + "desVars.csv")
    shutil.copy("desVarsNorm.csv",
                StatusDirectory + os.sep + "Results" + os.sep + OptName +
                os.sep + "desVarsNorm.csv")
    shutil.copy("constraints.csv",
                StatusDirectory + os.sep + "Results" + os.sep + OptName +
                os.sep + "constraints.csv")
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
