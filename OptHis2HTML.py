# -*- coding: utf-8 -*-
'''
----------------------------------------------------------------------------------------------------
Title:          OptHis2HTML.py
Units:          Unitless
Date:           November 15, 2013
Author:         F. Wachter
Contributors:   E.J. Wehrle
----------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------
Description
----------------------------------------------------------------------------------------------------
Open the optimization history (the .cue and .bin file) and processes them into HTML format
during runtime so one can check the status and progress of the optimization.

----------------------------------------------------------------------------------------------------
To do and ideas
----------------------------------------------------------------------------------------------------
see DesOpt.py
'''
import numpy as np
from time import localtime, strftime, time
import calendar
from pyOpt import History
from Normalize import normalize, denormalize
import shutil
import os
import sys
import csv
import glob
import webbrowser


def OptHis2HTML(OptName, Alg, DesOptDir, xL, xU, DesVarNorm, inform, starttime, StatusDirectory=""):

    with open('objFct_maxCon.csv', 'wb') as csvfile:
            datawriter = csv.writer(csvfile, dialect='excel')
    csvfile.close()

    with open('desVarsNorm.csv', 'wb') as csvfile:
            datawriter = csv.writer(csvfile, dialect='excel')
    csvfile.close()

    with open('desVars.csv', 'wb') as csvfile:
            datawriter = csv.writer(csvfile, dialect='excel')
    csvfile.close()

    with open('constraints.csv', 'wb') as csvfile:
            datawriter = csv.writer(csvfile, dialect='excel')
    csvfile.close()



    StartTime = str(starttime)[0:10] + "000"
    EndTime = ""
    if inform != "Running":
        EndTime = str(time())[0:10] + "000"



    if StatusDirectory == "":           #Change the target directory for the status report files if the user wants to
        StatusDirectory = DesOptDir

    pos_of_best_ind = []
    fIter = []
    xIter = []
    gIter = []



    # template_directory= DesOpt_Base + "/.DesOptPy/_OptStatusReport/"  # directory with the html files etc.
    template_directory = os.path.dirname(
        os.path.realpath(__file__)) + "/StatusReportFiles/"  # directory with the html files etc.


    OptHist = History(OptName, "r")  # Instanz einer History erstellen

    fAll = OptHist.read([0, -1], ["obj"])[0]["obj"]
    xAll = OptHist.read([0, -1], ["x"])[0]["x"]
    gAll = OptHist.read([0, -1], ["con"])[0]["con"]

    if Alg.name == "NLPQLP":
        gAll = [x * -1 for x in gAll]

    fGradIter = OptHist.read([0, -1], ["grad_obj"])[0]["grad_obj"]

    if Alg.name == "COBYLA":
        fIter = fAll
        xIter = xAll
        gIter = gAll
    elif Alg.name == "NSGA-II":
        PopSize = Alg.options['PopSize'][1]

        for i in range(0,fAll.__len__() / PopSize):  # Iteration trough the Populations

            best_fitness = 9999999
            max_violation_of_all_g = np.empty(PopSize)
            max_violation_of_all_g.fill(99999999)

            for u in range(0,PopSize):                      # Iteration trough the Individuals of the actual population
                if np.max(gAll[i*PopSize+u]) < max_violation_of_all_g[u]:
                    max_violation_of_all_g[u] = np.max(gAll[i*PopSize+u])

            pos_smallest_violation = np.argmin(max_violation_of_all_g)

            if max_violation_of_all_g[pos_smallest_violation] > 0:   # only not feasible designs, so choose the less violated one as best
                fIter.append(fAll[i*PopSize + pos_smallest_violation])
                xIter.append(xAll[i*PopSize + pos_smallest_violation])
                gIter.append(gAll[i*PopSize + pos_smallest_violation])
            else:                                                   # find the best feasible one
                for u in range(0,PopSize):                      # Iteration trough the Individuals of the actual population
                    if np.max(fAll[i*PopSize+u]) < best_fitness:
                        if np.max(gAll[i*PopSize+u]) <= 0:
                            best_fitness = fAll[i*PopSize+u]
                            pos_of_best_ind = i*PopSize +u

                fIter.append(fAll[pos_of_best_ind])
                xIter.append(xAll[pos_of_best_ind])
                gIter.append(gAll[pos_of_best_ind])

        #print fAll.__len__() / PopSize
    else:
        fIter = [[]] * len(fGradIter)
        xIter = [[]] * len(fGradIter)
        gIter = [[]] * len(fGradIter)

        for ii in range(len(fIter)):

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

    if Alg.name != "NSGA-II":
        if len(fGradIter) == 0:  # first calculation
            fIter = fAll
            xIter = xAll
            gIter = gAll

    OptHist.close()

    fIter = np.asarray(fIter)
    xIter = np.asarray(xIter)
    gIter = np.asarray(gIter)
    niter = len(fIter) - 1

    if xIter.size != 0:
        if DesVarNorm == False:
            xIter_denormalized = np.zeros((niter + 1, len(xIter[0])))
            for y in range(0, niter + 1):
                xIter_denormalized[y] = xIter[y]
            for y in range(0, niter + 1):
                [xIter[y, :],xLnorm, xUnorm] = normalize(xIter_denormalized[y, :], xL, xU, "xLxU")
        else:
            xIter_denormalized = np.zeros((niter + 1, len(xIter[0])))
            for y in range(0, niter + 1):
                xIter_denormalized[y, :] = denormalize(xIter[y, :], xL, xU, DesVarNorm)

    time_now = strftime("%Y-%b-%d %H:%M:%S", localtime())  # Aktualisierungszeit auslesen
    ymax = -200000
    ymin = 20000
    ymax_denorm = -200000
    ymin_denorm = 20000
    arr_gmin = [[]] * len(fIter)

    gmax = -200000
    gmin = 20000
    number_des_vars = "0"
    number_constraints = "0"
    objFctmax = -20000
    objFctmin = 20000

    value = ""
    value2 = ""
    # value = Werte der Zielfkt
    # value2 = maximale werte aller Nebenbedingungen


    for x in range(0, niter + 1):
        value = value + '[' + str(x) + ',' + str(float(fIter[x])) + '],'  # Daten für Zielfkt-diagramm aufbereiten
        if gIter.size != 0:
            value2 = value2 + '[' + str(x) + ',' + str(
                float(np.max(gIter[x]))) + '],'  # Daten für Nebenb-diagramm aufbereiten

        with open('objFct_maxCon.csv', 'ab') as csvfile:
            datawriter = csv.writer(csvfile, dialect='excel')
            datawriter.writerow([x, str(float(fIter[x])), float(np.max(gIter[x]))])
        csvfile.close()


    for x in range(0, niter + 1):  # Maximale y-Achsen Werte bestimmen
        if (np.max(fIter[x]) > objFctmax):
            objFctmax = np.max(fIter[x])
        if (np.min(fIter[x]) < objFctmin):
            objFctmin = np.min(fIter[x])

        if (np.max(xIter[x]) > ymax):
            ymax = np.max(xIter[x])
        if (np.min(xIter[x]) < ymin):
            ymin = np.min(xIter[x])

        if (np.max(xIter_denormalized[x]) > ymax_denorm):  # fuer denormalized
            ymax_denorm = np.max(xIter_denormalized[x])
        if (np.min(xIter_denormalized[x]) < ymin_denorm):
            ymin_denorm = np.min(xIter_denormalized[x])

        if gIter.size != 0:
            if (np.max(gIter[x]) > gmax):
                gmax = np.max(gIter[x])
        if gIter.size != 0:
            if (np.min(gIter[x]) < gmin):
                gmin = np.min(gIter[x])

    datasets = ""
    datasets_denorm = ""
    datasetsg = ""

    if xIter.size != 0:
        for x in range(0, niter +1):  # Datasets von Designvariables erstellen
            datasets = str(xIter[x][:].tolist()).strip('[]')

            with open('desVarsNorm.csv', 'ab') as csvfile:
                datawriter = csv.writer(csvfile, dialect='excel', quotechar=' ')
                datawriter.writerow([x, datasets] )
            csvfile.close()
            datasets = ""

    if xIter.size != 0:
        for x in range(0, niter +1):  # Datasets von denormalisierten Designvariables erstellen
            datasets_denorm = str(xIter_denormalized[x][:].tolist()).strip('[]')
            with open('desVars.csv', 'ab') as csvfile:
                datawriter = csv.writer(csvfile, dialect='excel', quotechar=' ')
                datawriter.writerow([x, datasets_denorm] )
            csvfile.close()
            datasets = ""


    if gIter.size != 0:
        for x in range(0, niter +1):  # Datasets von Con-fkt erstellen
            datasetsg = str(gIter[x][:].tolist()).strip('[]')

            with open('constraints.csv', 'ab') as csvfile:
                datawriter = csv.writer(csvfile, dialect='excel', quotechar=' ')
                datawriter.writerow([x, datasetsg] )
            csvfile.close()

        for u in range(0, len(gIter)):
            arr_gmin[u] = np.max(gIter[u])

    allDesVar = ""

    if xIter.size != 0:
        for y in range(0, len(xIter[0])):
            allDesVar = allDesVar + ',data' + str(y)

    allConVar = ""

    if gIter.size != 0:
        for y in range(0, len(gIter[0])):
            allConVar = allConVar + ',data' + str(y)


    # Tables erstellen

    ##ObjFct and constraint table

    ObjFct_table = ""
    if xIter.size != 0:
        if gIter.size != 0:
            for x in range(0, niter + 1):
                ObjFct_table += "<tr>\n<td>" + str(x) + "</td>\n<td>" + str(round(fIter[x], 4)) + "</td>\n<td>" + str(
                    round(np.max(gIter[x]), 4)) + "</td>\n</tr>"
        else:
            for x in range(0, niter + 1):
                ObjFct_table += "<tr>\n<td>" + str(x) + "</td>\n<td>" + str(
                    round(fIter[x], 4)) + "</td>\n<td> no constraints </td>\n</tr>"

    ##Design Variable table generation

    DesVar_table = "<td></td>"
    if xIter.size != 0:
        number_des_vars = str(len(xIter[0]))

        for x in range(0, len(xIter[0])):  #header erzeugen
            DesVar_table += "<td>" + "x&#770;<sub>" + str(x + 1) + "</sub></td>" + "<td>" + "x<sub>" + str(x + 1) + " </sub></td>"

        for y in range(0, niter + 1):  # daten befuellen
            DesVar_table += "<tr>\n<td>" + str(y) + "</td>"
            for x in range(0, len(xIter[0])):
                DesVar_table += "<td>" + str(round(xIter[y][x], 4)) + "</td><td>" + str(
                    round(xIter_denormalized[y][x], 4)) + "</td>"
            DesVar_table += "</tr>"

    ##Constraint  table generation

    Constraint_table = "<td></td>"
    if gIter.size != 0:
        number_constraints = str(len(gIter[0]))
        for x in range(0, len(gIter[0])):  #header erzeugen
            Constraint_table += "<td>" + "g<sub>" + str(x + 1) + "</sub></td>"
        for y in range(0, niter + 1):  # daten befuellen
            Constraint_table += "<tr>\n<td>" + str(y) + "</td>"
            for x in range(0, len(gIter[0])):
                if (round(gIter[y][x], 4) > 0):
                    Constraint_table += "<td class=\"negativ\">" + str(round(gIter[y][x], 4)) + "</td>"
                else:
                    Constraint_table += "<td class=\"positiv\">" + str(round(gIter[y][x], 4)) + "</td>"
            Constraint_table += "</tr>"

    html = open(template_directory + '/initial.html', 'r')  # HTML Template öffnen
    hstr = html.read()
    html.close()

    if len(fIter) > 50:
        xxxxNumLabels = str(30)
    else:
        xxxxNumLabels = str(niter)

    # Neue HTML Datei erstellen
    if gIter.size != 0 or gIter.size > 100:
        hstrnew = hstr.replace('xxxxName', OptName)
        hstrnew = hstrnew.replace('xxxxTime', time_now)
        hstrnew = hstrnew.replace('xxxxtableObjFct', ObjFct_table)
        hstrnew = hstrnew.replace('xxxxtableDesVar', DesVar_table)
        hstrnew = hstrnew.replace('xxxxnumber_des_var', number_des_vars*2)
        hstrnew = hstrnew.replace('xxxxtableConstr', Constraint_table)
        hstrnew = hstrnew.replace('xxxxnumber_constraints', number_constraints)
        hstrnew = hstrnew.replace('xxxxAlg', Alg.name)
        hstrnew = hstrnew.replace('xxxxStatus', str(inform))
        hstrnew = hstrnew.replace('xxxxStartTime', StartTime)
        hstrnew = hstrnew.replace('xxxxEndTime', EndTime)
    else:
        hstrnew = hstr.replace('xxxxName', OptName)
        hstrnew = hstrnew.replace('xxxxTime', time_now)
        hstrnew = hstrnew.replace('xxxxtableObjFct', ObjFct_table)
        hstrnew = hstrnew.replace('xxxxtableDesVar', DesVar_table)
        hstrnew = hstrnew.replace('xxxxAlg', Alg.name)
        hstrnew = hstrnew.replace('xxxxStatus', inform)
        hstrnew = hstrnew.replace('xxxxStartTime', StartTime)
        hstrnew = hstrnew.replace('xxxxEndTime', EndTime)

        try:  # remove the hmtl parts which are only needed for constrained problems
            for i in range(0, 10):
                hstrnew = hstrnew[0:hstrnew.find("<!--Start of constraint html part-->")] + hstrnew[hstrnew.find(
                    "<!--End of constraint html part-->") + 34:-1]
        except:
            print ""

    html = open('initial1.html', 'w')
    html.write(hstrnew)
    html.close()

    if not os.path.exists(StatusDirectory + os.sep + "Results" + os.sep + OptName):
        os.makedirs(StatusDirectory + os.sep + "Results" + os.sep + OptName)

    shutil.copy("initial1.html",
                StatusDirectory + os.sep + "Results" + os.sep + OptName + os.sep + OptName + "_Status.html")

    shutil.copy("objFct_maxCon.csv",
                StatusDirectory + os.sep + "Results" + os.sep + OptName + os.sep + "objFct_maxCon.csv")

    shutil.copy("desVars.csv",
                StatusDirectory + os.sep + "Results" + os.sep + OptName + os.sep + "desVars.csv")

    shutil.copy("desVarsNorm.csv",
                StatusDirectory + os.sep + "Results" + os.sep + OptName + os.sep + "desVarsNorm.csv")

    shutil.copy("constraints.csv",
                StatusDirectory + os.sep + "Results" + os.sep + OptName + os.sep + "constraints.csv")

    if not os.path.exists(StatusDirectory + os.sep + "Results" + os.sep + "dygraph-combined.js"):
        for file in glob.glob(template_directory + "*.png"):
            shutil.copy(file, StatusDirectory + os.sep + "Results" + os.sep)
        for file in glob.glob(template_directory + "*.js"):
            shutil.copy(file, StatusDirectory + os.sep + "Results" + os.sep)
        for file in glob.glob(template_directory + "*.css"):
            shutil.copy(file, StatusDirectory + os.sep + "Results" + os.sep)
        for file in glob.glob(template_directory + "*.ico"):
            shutil.copy(file, StatusDirectory + os.sep + "Results" + os.sep)

    #if len(fAll)==0:
    #    webbrowser.get('firefox').open_new_tab(StatusDirectory + os.sep + "Results" + os.sep + OptName + os.sep + OptName + "_Status.html")

    return 0



def picture(number):
    str_picture = '\'<img src="./Pictures/DesignVarXXXX.png"  width="600" />\''
    str_picture = str_picture.replace('XXXX', str(number))
    return str_picture
    ###neuer Kommentar