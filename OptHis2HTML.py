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
import glob
import shutil
import os
import sys
import getpass
import platform

def OptHis2HTML(OptName, Alg, DesOptDir):

    Alg = Alg.name
    operatingSystem = platform.uname()[0]
    if operatingSystem == "Linux":
        DirSplit = "/"
        homeDir = "/home/"
    else:
        DirSplit = "\\"
        homeDir = "c:\\Users\\"
    user = getpass.getuser()                  # Benutzer anfragen

    directory_startscript = sys.argv[0]
    (DesOpt_Base, tail) = os.path.split(directory_startscript)
    (DesOpt_Base, tail) = os.path.split(DesOpt_Base)    #DesOpt_Base is the base directory now
    #template_directory= DesOpt_Base + "/.DesOptPy/_OptStatusReport/"  # directory with the html files etc.
    template_directory= os.path.dirname(os.path.realpath(__file__)) + "/StatusReportFiles/"  # directory with the html files etc.

    html_index = open(template_directory + "/index.html", 'r')                                       # website root Template öffnen
    html_index_string = html_index.read()
    html_index_split = html_index_string.split('<!append new project after here!>')
    if(html_index_split[0].find(OptName) == -1):
        html_index_string = html_index_split[0] + "<br><a href=\"" + OptName + "/" + OptName + "_Status.html\">" + OptName + "</a>\r\n" + '<!append new project after here!>\r\n' + html_index_split[1]
    html_index.close()
    html_index = open(DesOptDir + "\\Results\\index.html", 'w')
    html_index.write(html_index_string)
    html_index.close()



    value = ""
    value2 = ""
    # value = Werte der Zielfkt
    # value2 = maximale werte aller Nebenbedingungen

    OptHist = History(OptName, "r")                    # Instanz einer History erstellen

    fAll = OptHist.read([0, -1], ["obj"])[0]["obj"]
    xAll = OptHist.read([0, -1], ["x"])[0]["x"]
    gAll = OptHist.read([0, -1], ["con"])[0]["con"]
    if Alg == "NLPQLP":
        gAll = [x * -1 for x in gAll]
    gGradIter = OptHist.read([0, -1], ["grad_con"])[0]["grad_con"]
    fGradIter = OptHist.read([0, -1], ["grad_obj"])[0]["grad_obj"]

    fIter = [[]] * len(fGradIter)
    xIter = [[]] * len(fGradIter)
    gIter = [[]] * len(fGradIter)
    #print("length of fGradIter %d"  % len(fGradIter))
    #print("length of fAll %d"  % len(fAll))
    # video    if np.size(fAll)==1 and len(glob.glob('DesignIt*.png'))==0:
    # video       shutil.copy2('Design.png', 'DesignIt'+"{0:04d}".format(0)+'.png')
    for ii in range(len(fGradIter)):
        # video        if ii==len(fGradIter)-1:
        # video            shutil.copy2('Design.png', 'DesignIt'+"{0:04d}".format(ii+1)+'.png')
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

    OptHist.close()


    fIter = np.asarray(fIter)
    xIter = np.asarray(xIter)
    gIter = np.asarray(gIter)
    niter = len(fIter)

    for x in range(0, niter):
        value = value + '[' + str(x) + ',' + str(float(fIter[x])) + '],'            # Daten für Zielfkt-diagramm aufbereiten
        if gIter.size != 0:
            value2 = value2 + '[' + str(x) + ',' + str(float(np.max(gIter[x]))) + '],'  # Daten für Nebenb-diagramm aufbereiten
    html = open(template_directory + '/initial.html', 'r')                                       # HTML Template öffnen
    hstr = html.read()
    part1 = hstr[:(hstr.find('<br><br>') + 4)]                                        # html slicen
    hstr = hstr[hstr.find('<br><br>')+4:]
    part2 = hstr[:hstr.find('</font></h1></center>')]
    hstr = hstr[hstr.find('</font></h1></center>'):]
    part3 = hstr[hstr.find('</font></h1></center>'):hstr.find('var data1 = [') + 13]
    hstr = hstr[hstr.find('var data1 = ['):]
    part4 = hstr[hstr.find('var data1 = [') + 13:hstr.find('var data2=[') + 11]
    hstr = hstr[hstr.find('var data2=['):]
    part5 = hstr[hstr.find('var data2=[') + 11:hstr.find('var niter=') + 10]
    hstr = hstr[hstr.find('var niter='):]
    part6 = hstr[hstr.find('var niter=') + 10:hstr.find('var ymax=') + 9]
    hstr = hstr[hstr.find('var ymax='):]
    part7 = hstr[hstr.find('var ymax=') + 9:hstr.find('var ymin=') + 9]
    hstr = hstr[hstr.find('var ymin='):]
    part8 = hstr[hstr.find('var ymin=') + 9:hstr.find('var ymin=') + 11]
    hstr = hstr[hstr.find('var ymin='):]
    part9 = hstr[hstr.find('var ymin=') + 11:hstr.find('Scatter(\'cvs2\'') + 14]
    hstr = hstr[hstr.find('Scatter(\'cvs2\''):]
    part10 = hstr[hstr.find('Scatter(\'cvs2\'') + 14:hstr.find('var gmax=') + 9]
    hstr = hstr[hstr.find('var gmax='):]
    part11 = hstr[hstr.find('var gmax=') + 9:hstr.find('var gmax=') + 11]
    hstr = hstr[hstr.find('var gmax='):]
    part12 = hstr[hstr.find('var gmax=') + 11:hstr.find('RGraph.Scatter(\'cvs3\'') + 21]
    hstr = hstr[hstr.find('RGraph.Scatter(\'cvs3\''):]
    part13 = hstr[hstr.find('RGraph.Scatter(\'cvs3\'') + 21:hstr.find('.Set(\'ymin\',') + 12]
    hstr = hstr[hstr.find('.Set(\'ymin\','):]
    part14 = hstr[hstr.find('.Set(\'ymin\',') + 12:hstr.find('RGraph.Scatter(\'cvs4\'') + 21]
    hstr = hstr[hstr.find('RGraph.Scatter(\'cvs4\''):]
    part15 = hstr[hstr.find('RGraph.Scatter(\'cvs4\'') + 21:hstr.find('.Draw()') + 7]
    hstr = hstr[hstr.find('.Draw()') + 7:]
    part16 = hstr[hstr.find('.Draw()') + 7:hstr.find('<p>Convergence')]
    hstr = hstr[hstr.find('<p>Convergence'):]
    part17 = hstr[hstr.find('<p>Convergence')+14:]
    html.close()
    time_now = strftime("%Y-%b-%d %H:%M:%S", localtime())          # Aktualisierungszeit auslesen
    ymax = 0
    ymin = 0
    arr_gmin = [[]] * len(fGradIter)
    gmin1 = 0
    gmax = 0
    gmin = 0
    for x in range(0, niter):                                    # Maximale y-Achsen Werte bestimmen
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
        for x in range(0, len(xIter[0])):                                    # Datasets von Obj-fkt erstellen
            datasets += 'var ' + 'data' + str(x) + '=['
            for y in range(0, niter):
                datasets += '[' + str(y) + ',' + str(xIter[y][x]) + '],'
            datasets += '];\n\t\t\t'
    if gIter.size != 0:
        for x in range(0, len(gIter[0])):                                     # Datasets von Con-fkt erstellen
            datasetsg += 'var ' + 'data'+str(x)+'=['
            for y in range(0, niter):
                datasetsg += '[' +str(y) + ','+str(gIter[y][x])+'],'
            datasetsg += '];\n\t\t\t'
        for u in range(0, niter):
            arr_gmin[u] = np.max(gIter[u])
        gmin1 = np.min(arr_gmin)
    allDesVar = ""
    if xIter.size != 0:
        for y in range(0, len(xIter[0])):
            allDesVar = allDesVar+',data'+str(y)
    allConVar = ""
    if gIter.size != 0:
        for y in range(0, len(gIter[0])):
            allConVar = allConVar+',data'+str(y)
    # Neue HTML Datei erstellen
    if gIter.size != 0:
        hstrnew = part1+OptName+part2+time_now+part3+value+part4+value2+part5+str(niter)+part6+str(ymax)+part7+str(ymin)+part8+datasets
        hstrnew += part9+allDesVar+part10+str(gmax)+part11+datasetsg+part12+allConVar+part13+str(gmin)+part14+allConVar+part15+part16+part17
    else:
        hstrnew = part1+OptName+part2+time_now+part3+value+part4+value2+part5+str(niter)+part6+str(ymax)+part7+str(ymin)+part8+datasets
        hstrnew += part9+allDesVar+part10+str(gmax)+part11+datasetsg+part16+"</center></body></html>"
    hstrnew = hstrnew.replace("gmin1", str(gmin1))
    hstrnew = hstrnew.replace("#template_directory#", template_directory)
    html = open('initial1.html', 'w')
    html.write(hstrnew)
    html.close()


    if not os.path.exists(DesOptDir + os.sep + "Results"+ os.sep + OptName):
        os.makedirs(DesOptDir + os.sep + "Results"+ os.sep + OptName)

    shutil.copy("initial1.html",
               DesOptDir + os.sep + "Results"+ os.sep + OptName + os.sep + OptName + "_Status.html")
    if not os.path.exists(DesOptDir + os.sep + "Results" + os.sep + "RGraph.scatter.js"):
        for file in glob.glob(template_directory + "*.PNG"):
            shutil.copy(file,
                        DesOptDir + os.sep + "Results" + os.sep)
        for file in glob.glob(template_directory + "*.js"):
            shutil.copy(file,
                        DesOptDir + os.sep + "Results" + os.sep)

#        shutil.copytree(template_directory,
#                        D+ os.sep,
#                        ignore=shutil.ignore_patterns('*.html'))
    # shutil.copy("initial1.html","M:/Git/history-to-html/_OptResultReports/"+OptName+"/"+OptName+".html")
    # print "done creating html"
    return 0
