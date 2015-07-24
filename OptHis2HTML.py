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
from time import localtime, strftime
from pyOpt import History
from Normalize import denormalize
import shutil
import os
import sys
import glob


def OptHis2HTML(OptName, Alg, DesOptDir, xL, xU, DesVarNorm, StatusDirectory=""):


    if StatusDirectory == "":           #Change the target directory for the status report files if the user wants to
        StatusDirectory = DesOptDir

    pos_of_best_ind = []
    fIter = []
    xIter = []
    gIter = []

    directory_startscript = sys.argv[0]
    (DesOpt_Base, tail) = os.path.split(directory_startscript)

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

        for i in range(0,fAll.__len__() / PopSize):  #TODO: Calculate the best individual in another (correct) way!
            pos_of_best_ind.append(np.argmin(fAll[i * PopSize:i * PopSize + PopSize]) + PopSize * i)
            fIter.append(fAll[pos_of_best_ind[i]])
            xIter.append(xAll[pos_of_best_ind[i]])
            gIter.append(gAll[pos_of_best_ind[i]])

        print fAll.__len__() / PopSize
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
        for x in range(0, len(xIter[0])):  # Datasets von Designvariables erstellen
            datasets += 'var ' + 'data' + str(x) + '=['
            for y in range(0, niter + 1):
                datasets += '[' + str(y) + ',' + str(xIter[y][x]) + '],'
            datasets += '];\n\t\t\t'

    if xIter.size != 0:
        for x in range(0, len(xIter[0])):  # Datasets von denormalisierten Designvariables erstellen

            datasets_denorm += 'var ' + 'data' + str(x) + '=['
            for y in range(0, niter + 1):
                datasets_denorm += '[' + str(y) + ',' + str(xIter_denormalized[y][x]) + '],'
            datasets_denorm += '];\n\t\t\t'

    if gIter.size != 0:
        for x in range(0, len(gIter[0])):  # Datasets von Con-fkt erstellen
            datasetsg += 'var ' + 'data' + str(x) + '=['
            for y in range(0, niter + 1):
                datasetsg += '[' + str(y) + ',' + str(gIter[y][x]) + '],'
            datasetsg += '];\n\t\t\t'
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
            DesVar_table += "<td>" + "x_" + str(x + 1) + "</td>" + "<td>" + "x_" + str(x + 1) + " denormalized</td>"

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
            Constraint_table += "<td>" + "g_" + str(x + 1) + "</td>"
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

    if len(fAll) > 50:
        xxxxNumLabels = str(30)
    else:
        xxxxNumLabels = str(niter)

    # Neue HTML Datei erstellen
    if gIter.size != 0 or gIter.size > 100:
        hstrnew = hstr.replace('xxxxName', OptName)
        hstrnew = hstrnew.replace('xxxxTime', time_now)
        hstrnew = hstrnew.replace('xxxxValue1', value)
        hstrnew = hstrnew.replace('xxxxValue2', value2)
        hstrnew = hstrnew.replace('xxxxallDesVar_denorm', allDesVar)
        hstrnew = hstrnew.replace('xxxxdatasetf_denorm', datasets_denorm)
        hstrnew = hstrnew.replace('xxxxXmax', str(ymax_denorm))
        hstrnew = hstrnew.replace('xxxxXmin', str(ymin_denorm))
        hstrnew = hstrnew.replace('xxxxnIter', str(niter))
        hstrnew = hstrnew.replace('xxxxymax', str(ymax))
        hstrnew = hstrnew.replace('xxxxymin', str(ymin))
        hstrnew = hstrnew.replace('xxxxObjFctmin', str(objFctmin))
        hstrnew = hstrnew.replace('xxxxObjFctmax', str(objFctmax))
        hstrnew = hstrnew.replace('xxxxdatasetf', datasets)
        hstrnew = hstrnew.replace('xxxxallDesVar', allDesVar)
        hstrnew = hstrnew.replace('xxxxgmax', str(gmax))
        hstrnew = hstrnew.replace('xxxxgmin', str(gmin))
        hstrnew = hstrnew.replace('xxxxdatasetg', datasetsg)
        hstrnew = hstrnew.replace('xxxxallConVar', allConVar)
        hstrnew = hstrnew.replace('xxxxallConVar', allConVar)
        hstrnew = hstrnew.replace('xxxxtableObjFct', ObjFct_table)
        hstrnew = hstrnew.replace('xxxxtableDesVar', DesVar_table)
        hstrnew = hstrnew.replace('xxxxnumber_des_var', number_des_vars*2)
        hstrnew = hstrnew.replace('xxxxtableConstr', Constraint_table)
        hstrnew = hstrnew.replace('xxxxnumber_constraints', number_constraints)
        hstrnew = hstrnew.replace('xxxxNumLabels', xxxxNumLabels)
    else:
        hstrnew = hstr.replace('xxxxName', OptName)
        hstrnew = hstrnew.replace('xxxxTime', time_now)
        hstrnew = hstrnew.replace('xxxxValue1', value)
        hstrnew = hstrnew.replace('xxxxValue2', value2)
        hstrnew = hstrnew.replace('xxxxallDesVar_denorm', allDesVar)
        hstrnew = hstrnew.replace('xxxxdatasetf_denorm', datasets_denorm)
        hstrnew = hstrnew.replace('xxxxXmax', str(ymax_denorm))
        hstrnew = hstrnew.replace('xxxxXmin', str(ymin_denorm))
        hstrnew = hstrnew.replace('xxxxnIter', str(niter))
        hstrnew = hstrnew.replace('xxxxymax', str(ymax))
        hstrnew = hstrnew.replace('xxxxymin', str(ymin))
        hstrnew = hstrnew.replace('xxxxObjFctmin', str(objFctmin))
        hstrnew = hstrnew.replace('xxxxObjFctmax', str(objFctmax))
        hstrnew = hstrnew.replace('xxxxdatasetf', datasets)
        hstrnew = hstrnew.replace('xxxxallDesVar', allDesVar)
        hstrnew = hstrnew.replace('xxxxgmax', str(gmax))
        hstrnew = hstrnew.replace('xxxxgmin', str(gmin))
        hstrnew = hstrnew.replace('xxxxdatasetg', datasetsg)
        hstrnew = hstrnew.replace('xxxxtableObjFct', ObjFct_table)
        hstrnew = hstrnew.replace('xxxxtableDesVar', DesVar_table)
        hstrnew = hstrnew.replace('xxxxNumLabels', xxxxNumLabels)

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

    if not os.path.exists(StatusDirectory + os.sep + "Results" + os.sep + "RGraph.scatter.js"):
        for file in glob.glob(template_directory + "*.png"):
            shutil.copy(file, StatusDirectory + os.sep + "Results" + os.sep)
        for file in glob.glob(template_directory + "*.js"):
            shutil.copy(file, StatusDirectory + os.sep + "Results" + os.sep)
        for file in glob.glob(template_directory + "*.css"):
            shutil.copy(file, StatusDirectory + os.sep + "Results" + os.sep)
    return 0


def picture(number):
    str_picture = '\'<img src="./Pictures/DesignVarXXXX.png"  width="600" />\''
    str_picture = str_picture.replace('XXXX', str(number))
    return str_picture
    ###neuer Kommentar