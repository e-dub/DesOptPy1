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
during runtime so one can check the status and prgress of the optimization.

----------------------------------------------------------------------------------------------------
To do and ideas
----------------------------------------------------------------------------------------------------
see DesOpt.py
'''
import numpy as np
from time import localtime, strftime
from pyOpt import History

import shutil
import os
import sys
import glob



def OptHis2HTML(OptName, Alg, DesOptDir):

    pos_of_best_ind = []
    fIter = []
    xIter = []
    gIter = []

    directory_startscript = sys.argv[0]
    (DesOpt_Base, tail) = os.path.split(directory_startscript)

    # template_directory= DesOpt_Base + "/.DesOptPy/_OptStatusReport/"  # directory with the html files etc.
    template_directory = os.path.dirname(
        os.path.realpath(__file__)) + "/StatusReportFiles/"  # directory with the html files etc.

    html_index = open(template_directory + "/index.html", 'r')  # website root Template öffnen
    html_index_string = html_index.read()
    html_index_split = html_index_string.split('<!append new project after here!>')
    if (html_index_split[0].find(OptName) == -1):
        html_index_string = html_index_split[
                                0] + "<br><a href=\"" + OptName + "/" + OptName + "_Status.html\">" + OptName + "</a>\r\n" + '<!append new project after here!>\r\n' + \
                            html_index_split[1]
    html_index.close()
    html_index = open(DesOptDir + "\\Results\\index.html", 'w')
    html_index.write(html_index_string)
    html_index.close()

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

        for i in range(0, fAll.__len__() / PopSize):
            pos_of_best_ind.append(np.argmin(fAll[i * PopSize:i * PopSize + PopSize]) + PopSize * i)
            fIter.append(fAll[pos_of_best_ind[i]])
            xIter.append(xAll[pos_of_best_ind[i]])
            gIter.append(gAll[pos_of_best_ind[i]])
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

    time_now = strftime("%Y-%b-%d %H:%M:%S", localtime())  # Aktualisierungszeit auslesen
    ymax = 0
    ymin = 0
    arr_gmin = [[]] * len(fIter)
    gmin1 = 0
    gmax = 0
    gmin = 0

    value = ""
    value2 = ""
    # value = Werte der Zielfkt
    # value2 = maximale werte aller Nebenbedingungen


    for x in range(0, niter + 1):
        value = value + '[' + str(x) + ',' + str(float(fIter[x])) + '],'  #Daten für Zielfkt-diagramm aufbereiten
        if gIter.size != 0:
            value2 = value2 + '[' + str(x) + ',' + str(
                float(np.max(gIter[x]))) + '],'  # Daten für Nebenb-diagramm aufbereiten

    for x in range(0, niter + 1):  # Maximale y-Achsen Werte bestimmen
        if (np.max(xIter[x]) > ymax):
            ymax = np.max(xIter[x])
        if (np.min(xIter[x]) < ymin):
            ymin = np.min(xIter[x])
        if gIter.size != 0:
            if (np.max(gIter[x]) > gmax):
                gmax = np.max(gIter[x])
        if gIter.size != 0:
            if (np.min(gIter[x]) < gmin):
                gmin = np.min(gIter[x])

    datasets = ""
    datasetsg = ""

    if xIter.size != 0:
        for x in range(0, len(xIter[0])):  # Datasets von Obj-fkt erstellen
            datasets += 'var ' + 'data' + str(x) + '=['
            for y in range(0, niter + 1):
                datasets += '[' + str(y) + ',' + str(xIter[y][x]) + '],'
            datasets += '];\n\t\t\t'

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


    html = open(template_directory + '/initial.html', 'r')  # HTML Template öffnen
    hstr = html.read()
    html.close()

    # Neue HTML Datei erstellen
    if gIter.size != 0:
        hstrnew = hstr.replace('xxxxName',OptName)
        hstrnew = hstrnew.replace('xxxxTime',time_now)
        hstrnew = hstrnew.replace('xxxxValue1',value)
        hstrnew = hstrnew.replace('xxxxValue2',value2)
        hstrnew = hstrnew.replace('xxxxnIter',str(niter))
        hstrnew = hstrnew.replace('xxxxymax',str(ymax))
        hstrnew = hstrnew.replace('xxxxymin',str(ymin))
        hstrnew = hstrnew.replace('xxxxdatasetf',datasets)
        hstrnew = hstrnew.replace('xxxxallDesVar',allDesVar)
        hstrnew = hstrnew.replace('xxxxgmax',str(gmax))
        hstrnew = hstrnew.replace('xxxxgmin',str(gmin))
        hstrnew = hstrnew.replace('xxxxdatasetg',datasetsg)
        hstrnew = hstrnew.replace('xxxxallConVar',allConVar)
        hstrnew = hstrnew.replace('xxxxallConVar',allConVar)
    else:
        hstrnew = hstr.replace('xxxxName',OptName)
        hstrnew = hstrnew.replace('xxxxTime',time_now)
        hstrnew = hstrnew.replace('xxxxValue1',value)
        hstrnew = hstrnew.replace('xxxxValue2',value2)
        hstrnew = hstrnew.replace('xxxxnIter',str(niter))
        hstrnew = hstrnew.replace('xxxxymax',str(ymax))
        hstrnew = hstrnew.replace('xxxxymin',str(ymin))
        hstrnew = hstrnew.replace('xxxxdatasetf',datasets)
        hstrnew = hstrnew.replace('xxxxallDesVar',allDesVar)
        hstrnew = hstrnew.replace('xxxxgmax',str(gmax))
        hstrnew = hstrnew.replace('xxxxgmin',str(gmin))
        hstrnew = hstrnew.replace('xxxxdatasetg',datasetsg)
        hstrnew = hstrnew.replace('xxxxallConVar',allConVar)
        hstrnew = hstrnew.replace('xxxxallConVar',allConVar)

        hstrnew = hstrnew[0:hstrnew.find("<!--Start of constraint part-->")] + hstrnew[hstrnew.find("<!--End of constraint part-->"):-1]
        hstrnew = hstrnew[0:hstrnew.find("<!--Start of constraint html part-->")] + hstrnew[hstrnew.find("<!--End of constraint html part-->"):-1]


    html = open('initial1.html', 'w')
    html.write(hstrnew)
    html.close()

    if not os.path.exists(DesOptDir + os.sep + "Results" + os.sep + OptName):
        os.makedirs(DesOptDir + os.sep + "Results" + os.sep + OptName)

    shutil.copy("initial1.html",
                DesOptDir + os.sep + "Results" + os.sep + OptName + os.sep + OptName + "_Status.html")

    if not os.path.exists(DesOptDir + os.sep + "Results" + os.sep + "RGraph.scatter.js"):
        for file in glob.glob(template_directory + "*.png"):
            shutil.copy(file,DesOptDir + os.sep + "Results" + os.sep)
        for file in glob.glob(template_directory + "*.js"):
            shutil.copy(file,DesOptDir + os.sep + "Results" + os.sep)
    return 0


def picture(number):
    str_picture = '\'<img src="./Pictures/DesignVarXXXX.png"  width="600" />\''
    str_picture = str_picture.replace('XXXX', str(number))
    return str_picture