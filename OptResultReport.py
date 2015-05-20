# -*- coding: utf-8 -*-
'''
----------------------------------------------------------------------------------------------------
Title:          OptResultReport.py
Units:          Unitless
Date:           December 15, 2014
Authors:        S. Rudolph, E.J. Wehrle, F. Wachter
----------------------------------------------------------------------------------------------------

----------------------------------------------------------------------------------------------------
Description
----------------------------------------------------------------------------------------------------
Make result report for the optimization run


----------------------------------------------------------------------------------------------------
To do and ideas
----------------------------------------------------------------------------------------------------
'''
import matplotlib.pyplot as plt
import matplotlib.pylab as pyl
from matplotlib.font_manager import FontProperties
import matplotlib.ticker as ticker
from matplotlib import cm
from matplotlib import rc
import matplotlib.colors as colors
import numpy as np
import sys
import getopt
import math
import os
import pickle
import fnmatch
import shutil
import subprocess
import time
from subprocess import Popen


def OptResultReport(optname,diagrams=1,tables=0,lyx=0):
    #model_dir = ""
    #model_dir = optname.split("_", 1)[0]
    DirRunFiles = "../DesOpt/Results/"+optname+"/RunFiles/"
    file_OptSolData = open(DirRunFiles+"/"+optname+"_OptSol.pkl")
    OptSolData = pickle.load(file_OptSolData)
    x0 = OptSolData['x0']
    xOpt = OptSolData['xOpt']
    xOptNorm = OptSolData['xOptNorm']
    xIter = OptSolData['xIter']
    xIterNorm = OptSolData['xIterNorm']
    fOpt = OptSolData['fOpt']
    fIter = OptSolData['fIter']
    fIterNorm = OptSolData['fIterNorm']
    gOpt = OptSolData['gOpt']
    gIter = OptSolData['gIter']
    gMaxIter = OptSolData['gMaxIter']
    fGradIter = OptSolData['fGradIter']
    gGradIter = OptSolData['gGradIter']
    fGradOpt = OptSolData['fGradOpt']
    gGradOpt = OptSolData['gGradOpt']
    OptName = OptSolData['OptName']
    OptModel = OptSolData['OptModel']
    OptTime = OptSolData['OptTime']
    #loctime = OptSolData['loctime']
    today = OptSolData['today']
    computerName = OptSolData['computerName']
    operatingSystem = OptSolData['operatingSystem']
    architecture = OptSolData['architecture']
    nProcessors = OptSolData['nProcessors']
    userName = OptSolData['userName']
    Alg = OptSolData['Alg']
    DesVarNorm = OptSolData['DesVarNorm']
    KKTmax = OptSolData['KKTmax']
    lambda_c = OptSolData['lambda_c']
    nEval = OptSolData['nEval']
    nIter = OptSolData['nIter']
    SPg = OptSolData['SPg']
    gc = OptSolData['gc']
    OptAlg = OptSolData['OptAlg']
    x0norm = OptSolData['x0norm']
    xL = OptSolData['xL']
    xU = OptSolData['xU']
    ng = OptSolData['ng']
    nx = OptSolData['nx']
    Opt1Order = OptSolData['Opt1Order']
    hhmmss0 = OptSolData['hhmmss0']
    hhmmss1 = OptSolData['hhmmss1']
    ResultsFolder = "../DesOpt/Results/"+OptName+"/ResultReport/"
    if operatingSystem != 'Windows':
        InkscapeCall = "inkscape"
        LyxCall = "lyx"
    elif operatingSystem == 'Windows':
        InkscapeCall = "C:\\inkscape\\inkscape.exe"
        LyxCall = "c:\\lyx\\lyx.exe"
    # ---------------------------------------------------------------------------------------------------
    #         Write and save plots
    # ---------------------------------------------------------------------------------------------------
    if diagrams == 1:
        print '# --------------  DIAGRAM GENERATION PROGRESS:  -------------- #\n'
        progressbar(0,82)
        fontP = FontProperties()
        fontP.set_size(9.5)
        fontPP = FontProperties()
        fontPP.set_size(8)
        numColg = int(math.ceil(ng/20.))
        numColx = int(math.ceil(nx/20.))
        if np.size(gIter[0]) > 0:
            #---------------------------------------------------------------------------------------------------
            # Normalized objective function and maximum constraint iteration plot
            #---------------------------------------------------------------------------------------------------
            plt.rc('text', usetex=True)
            #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
            fgMaxIterNormFig=plt.figure(figsize=(5, 4), dpi=200)
            ax1 = fgMaxIterNormFig.add_subplot(111)
            ax1.plot(fIterNorm,label="$\hat{f}$")
            plt.xlabel("Iteration")
            plt.ylabel("Normalized objective function $\hat{f}$")
            ax2 = ax1.twinx()
            ax2.plot(gMaxIter,"g",label="$g_{max}$")
            plt.ylabel("Maximum constraint $g_{max}$")
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles=handles1+handles2
            labels=labels1+labels2
            ax1.legend(handles, labels, loc='best',prop = fontP)
            plt.xlim(xmin=0, xmax=len(fIterNorm)-1)
            plt.tight_layout()                                                  #Feature to autofit the graphics
            plt.savefig(ResultsFolder+OptName+'_fgMaxIterNorm.png')
            plt.savefig(ResultsFolder+OptName+'_fgMaxIterNorm.svg')
            plt.savefig(ResultsFolder+OptName+'_fgMaxIterNorm_wo.pdf')
            plt.close()

            plt.rc('text', usetex=False)
            fgMaxIterNormFig=plt.figure(figsize=(5, 4), dpi=200)
            ax1 = fgMaxIterNormFig.add_subplot(111)
            ax1.plot(fIterNorm,label=r"\$\fn$")
            plt.xlabel(r"Iteration")
            plt.ylabel(r"Normalized objective function \$\fn$")#,fontsize=1)
            ax2 = ax1.twinx()
            ax2.plot(gMaxIter,"g",label=r"\$\gmax$")
            plt.ylabel(r"Maximum constraint \$\gmax$")#,fontsize=1)
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles=handles1+handles2
            labels=labels1+labels2
            ax1.legend(handles, labels, loc='best',prop = fontP)
            plt.xlim(xmin=0, xmax=len(fIterNorm)-1)
            plt.tight_layout()
            plt.savefig(ResultsFolder+OptName+'_fgMaxIterNorm.pdf')
            plt.close()

            progressbar(6,82)

            #---------------------------------------------------------------------------------------------------
            # Objective function and maximum constraint iteration plot
            #---------------------------------------------------------------------------------------------------,
            plt.rc('text', usetex=True)
            fgMaxIterFig=plt.figure(figsize=(5, 4), dpi=200)
            ax1 = fgMaxIterFig.add_subplot(111)
            ax1.plot(fIter,label="$f$")
            plt.xlabel("Iteration")
            ax2 = ax1.twinx()
            ax2.plot(gMaxIter,'g',label="$g_{max}$")
            ax1.set_ylabel("Objective function $f$")
            ax2.set_ylabel("Maximum constraint $g_{max}$")
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles=handles1+handles2
            labels=labels1+labels2
            ax1.legend(handles, labels,  loc='best',prop = fontP,handleheight=2)  #changed location to best
            plt.xlim(xmin=0, xmax=len(fIter)-1)
            plt.tight_layout()                                                  #Feature to autofit the graphics
            plt.savefig(ResultsFolder+OptName+'_fgMaxIter.svg')
            plt.savefig(ResultsFolder+OptName+'_fgMaxIter.png')
            plt.savefig(ResultsFolder+OptName+'_fgMaxIter_wo.pdf')
            plt.close()

            plt.rc('text', usetex=False)
            fgMaxIterFig=plt.figure(figsize=(5, 4), dpi=200)
            ax1 = fgMaxIterFig.add_subplot(111)
            ax1.plot(fIter,label=r"\$\f$")
            plt.xlabel(r"Iteration")
            ax2 = ax1.twinx()
            ax2.plot(gMaxIter,'g',label=r"\$\gmax$")
            ax1.set_ylabel(r"Objective function \$\f$")
            ax2.set_ylabel(r"Maximum constraint \$\gmax$")
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            handles=handles1+handles2
            labels=labels1+labels2
            ax1.legend(handles, labels,  loc='best',prop = fontP,handleheight=2)  #changed location to best
            plt.xlim(xmin=0, xmax=len(fIter)-1)
            plt.tight_layout()                                                  #Feature to autofit the graphics
            plt.savefig(ResultsFolder+OptName+'_fgMaxIter.pdf')
            plt.close()

            progressbar(12,82)

            #---------------------------------------------------------------------------------------------------
            # Constraint iteration plot
            #---------------------------------------------------------------------------------------------------
            plt.rc('text', usetex=True)
            def gIterPlot(ax):
                jet = plt.get_cmap('jet')
                cNorm  = colors.Normalize(vmin=0, vmax=len(gIter[0]))
                scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
                for n in range(len(gIter[0])):
                    ax.plot(gIter[:,n], color=scalarMap.to_rgba(n), label="$g_{"+str(n+1)+"}$")
            gIterFig=plt.figure(figsize=(5, 4), dpi=200)
            ax2 = gIterFig.add_subplot(111)
            gIterPlot(ax2)
            plt.xlabel("Iteration")
            plt.ylabel("$g$")
            plt.xlim(xmin=0, xmax=len(gIter)-1)
            ax2.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.,ncol=numColg,prop=fontPP)
            plt.subplots_adjust(left=0.07, right=(0.96-(numColg*0.075)), top=0.93, bottom=0.12)
            plt.savefig(ResultsFolder+OptName+'_gIter.svg')
            plt.savefig(ResultsFolder+OptName+'_gIter.png')
            plt.savefig(ResultsFolder+OptName+'_gIter_wo.pdf')
            plt.close()

            plt.rc('text', usetex=False)
            def gIterPlot2(ax):
                jet = plt.get_cmap('jet')
                cNorm  = colors.Normalize(vmin=0, vmax=len(gIter[0]))
                scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
                for n in range(len(gIter[0])):
                    ax.plot(gIter[:,n], color=scalarMap.to_rgba(n), label=r"\$\g_{"+str(n+1)+"}$")
            gIterFig=plt.figure(figsize=(5, 4), dpi=200)
            ax2 = gIterFig.add_subplot(111)
            gIterPlot2(ax2)
            plt.xlabel(r"Iteration")
            plt.ylabel(r"\$\g$")
            plt.xlim(xmin=0, xmax=len(gIter)-1)
            ax2.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.,ncol=numColg,prop=fontPP)
            plt.subplots_adjust(left=0.07, right=(0.96-(numColg*0.075)), top=0.93, bottom=0.12)
            plt.savefig(ResultsFolder+OptName+'_gIter.pdf')
            plt.close()

            progressbar(18,82)

            #---------------------------------------------------------------------------------------------------
            # Constraint at optimum bar plot:
            #---------------------------------------------------------------------------------------------------
            plt.rc('text', usetex=True)
            #TODO turn off when no constraints!
            gBarfig = plt.figure(figsize=(6, 3), dpi=200)
            width = 0.5
            xspacing=0.25
            ax = gBarfig.add_subplot(111)
            ind = np.arange(ng)
            rects1 = ax.bar(ind+xspacing,gIter[0], width, color='#FAA43A')
            rects2 = ax.bar(ind+xspacing+width/2.,gIter[-1],width, color='#5DA5DA')
            ax.legend( (rects1[0], rects2[0]), ("$g_0$", "$g^*$"), prop = fontP )
            ax.set_xticks(ind+(width+width/2.)/2.+xspacing)
            plt.xlim(xmin=0, xmax=ng+xspacing)
            plt.ylim(ymin=np.min((np.min(gIter[0]),np.min(gOpt))), ymax=np.max((np.max(gIter[0]),np.max(gOpt))))
            #ax.set_xticklabels(ind+1, rotation=90)
            #ax.set_xticklabels(ind+1)
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            plt.xlabel("Constraints")
            plt.ylabel("$g$")
            plt.tight_layout()
            plt.savefig(ResultsFolder+OptName+'_gBar.svg')
            plt.savefig(ResultsFolder+OptName+'_gBar.png')
            plt.savefig(ResultsFolder+OptName+'_gBar_wo.pdf')
            plt.close()

            plt.rc('text', usetex=False)
            gBarfig = plt.figure(figsize=(6, 3), dpi=200)
            width = 0.5
            xspacing=0.25
            ax = gBarfig.add_subplot(111)
            ind = np.arange(ng)
            rects1 = ax.bar(ind+xspacing,gIter[0], width, color='#FAA43A')
            rects2 = ax.bar(ind+xspacing+width/2.,gIter[-1],width, color='#5DA5DA')
            ax.legend( (rects1[0], rects2[0]), (r"\$\gin$", r"\$\go$"), prop = fontP )
            ax.set_xticks(ind+(width+width/2.)/2.+xspacing)
            plt.xlim(xmin=0, xmax=ng+xspacing)
            plt.ylim(ymin=np.min((np.min(gIter[0]),np.min(gOpt))), ymax=np.max((np.max(gIter[0]),np.max(gOpt))))
            #ax.set_xticklabels(ind+1, rotation=90)
            #ax.set_xticklabels(ind+1)
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            plt.xlabel(r"Constraints")
            plt.ylabel(r"\$\g$")
            plt.tight_layout()
            plt.savefig(ResultsFolder+OptName+'_gBar.pdf')
            plt.close()

            progressbar(24,82)
        else:
            #---------------------------------------------------------------------------------------------------
            # Normalized objective function and maximum constraint iteration plot
            #---------------------------------------------------------------------------------------------------
            plt.rc('text', usetex=True)
            #rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
            fgMaxIterNormFig=plt.figure(figsize=(5, 4), dpi=200)
            ax1 = fgMaxIterNormFig.add_subplot(111)
            ax1.plot(fIterNorm,label="$\hat{f}$")
            plt.xlabel("Iteration")
            plt.ylabel("Normalized objective function $\hat{f}$")
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles, labels, loc='best',prop = fontP)
            plt.xlim(xmin=0, xmax=len(fIterNorm)-1)
            plt.tight_layout()                                                  #Feature to autofit the graphics
            plt.savefig(ResultsFolder+OptName+'_fgMaxIterNorm.png')
            plt.savefig(ResultsFolder+OptName+'_fgMaxIterNorm.svg')
            plt.savefig(ResultsFolder+OptName+'_fgMaxIterNorm_wo.pdf')
            plt.close()

            plt.rc('text', usetex=False)
            fgMaxIterNormFig=plt.figure(figsize=(5, 4), dpi=200)
            ax1 = fgMaxIterNormFig.add_subplot(111)
            ax1.plot(fIterNorm,label=r"\$\fn$")
            plt.xlabel(r"Iteration")
            plt.ylabel(r"Normalized objective function \$\fn$")#,fontsize=1)
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles, labels, loc='best',prop = fontP)
            plt.xlim(xmin=0, xmax=len(fIterNorm)-1)
            plt.tight_layout()
            plt.savefig(ResultsFolder+OptName+'_fgMaxIterNorm.pdf')
            plt.close()

            plt.rc('text', usetex=True)
            fgMaxIterFig=plt.figure(figsize=(5, 4), dpi=200)
            ax1 = fgMaxIterFig.add_subplot(111)
            ax1.plot(fIter,label="$f$")
            plt.xlabel("Iteration")
            ax1.set_ylabel("Objective function $f$")
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles, labels,  loc='best',prop = fontP,handleheight=2)  #changed location to best
            plt.xlim(xmin=0, xmax=len(fIter)-1)
            plt.tight_layout()                                                  #Feature to autofit the graphics
            plt.savefig(ResultsFolder+OptName+'_fgMaxIter.svg')
            plt.savefig(ResultsFolder+OptName+'_fgMaxIter.png')
            plt.savefig(ResultsFolder+OptName+'_fgMaxIter_wo.pdf')
            plt.close()

            plt.rc('text', usetex=False)
            fgMaxIterFig=plt.figure(figsize=(5, 4), dpi=200)
            ax1 = fgMaxIterFig.add_subplot(111)
            ax1.plot(fIter,label=r"\$\f$")
            plt.xlabel(r"Iteration")
            ax1.set_ylabel(r"Objective function \$\f$")
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles, labels,  loc='best',prop = fontP,handleheight=2)  #changed location to best
            plt.xlim(xmin=0, xmax=len(fIter)-1)
            plt.tight_layout()                                                  #Feature to autofit the graphics
            plt.savefig(ResultsFolder+OptName+'_fgMaxIter.pdf')
            plt.close()
        #---------------------------------------------------------------------------------------------------
        # Objective function at optimum bar plot
        #---------------------------------------------------------------------------------------------------
        plt.rc('text', usetex=True)
        fBarfig = plt.figure(figsize=(5, 3), dpi=200)
        width = 0.5
        xspacing=0.25
        ax = fBarfig.add_subplot(111)
        ind = 1
        rects1 = ax.bar(xspacing,fIter[0], width, color='#FAA43A')
        rects2 = ax.bar(xspacing+width/2.,fOpt,width, color='#5DA5DA')
        ax.legend( (rects1[0], rects2[0]), ("$f_0$", "$f^*$"),prop = fontP )
        ax.set_ylabel("$f$")
        #ax.xaxis.set_major_formatter(ticker.NullFormatter())
        #ax.set_xticks(ind+(width+width/2.)/2.+xspacing)
        #plt.xlim(xmin=0, xmax=nx+xspacing )
        #plt.ylim(ymin=0, ymax=np.max((np.max(fOpt),np.max(fOpt))))
        #ax.set_xticklabels(ind+1)
        plt.tick_params(axis='x', which='both', bottom='off', labelbottom='off')
        plt.tight_layout()
        plt.savefig(ResultsFolder+OptName+'_fBar.svg')
        plt.savefig(ResultsFolder+OptName+'_fBar.png')
        plt.savefig(ResultsFolder+OptName+'_fBar_wo.pdf')
        plt.close()

        plt.rc('text', usetex=False)
        fBarfig = plt.figure(figsize=(5, 3), dpi=200)
        width = 0.5
        xspacing=0.25
        ax = fBarfig.add_subplot(111)
        ind = 1
        rects1 = ax.bar(xspacing,fIter[0], width, color='#FAA43A')
        rects2 = ax.bar(xspacing+width/2.,fOpt,width, color='#5DA5DA')
        ax.legend( (rects1[0], rects2[0]), (r"\$\fin$", r"\$\fo$"),prop = fontP )
        ax.set_ylabel(r"\$\f$")
        plt.tick_params(axis='x', which='both', bottom='off', labelbottom='off')
        plt.tight_layout()
        plt.savefig(ResultsFolder+OptName+'_fBar.pdf')
        plt.close()

        progressbar(26,82)

        #---------------------------------------------------------------------------------------------------
        # Gradiants of objective function iteration plot
        #---------------------------------------------------------------------------------------------------
        if len(fGradIter) is not 0:
            plt.rc('text', usetex=True)
            def fGradIterPlot(ax):
                jet = plt.get_cmap('jet')
                cNorm  = colors.Normalize(vmin=0, vmax=len(fGradIter[0]))
                scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
                for n in range(len(fGradIter[0])):
                    ax.plot(fGradIter[:,n], color=scalarMap.to_rgba(n), label='$d_{x_{'+str(n+1)+r'}}f$') #something wrong here

            fGradIterFig=plt.figure(figsize=(5, 4), dpi=200)
            ax2 = fGradIterFig.add_subplot(111)
            fGradIterPlot(ax2)
            plt.xlabel("Iteration")
            plt.ylabel("$d_{x_j}f$")
            plt.xlim(xmin=0, xmax=len(xIter)-1)
            ax2.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.,ncol=numColx,prop=fontPP)
            plt.subplots_adjust(left=0.07, right=(0.96-(numColx*0.078)), top=0.93, bottom=0.12)
            plt.savefig(ResultsFolder+OptName+'_fGradIter.svg')
            plt.savefig(ResultsFolder+OptName+'_fGradIter.png')
            plt.savefig(ResultsFolder+OptName+'_fGradIter_wo.pdf')
            plt.close()

            plt.rc('text', usetex=False)
            def fGradIterPlot2(ax):
                jet = plt.get_cmap('jet')
                cNorm  = colors.Normalize(vmin=0, vmax=len(fGradIter[0]))
                scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
                for n in range(len(fGradIter[0])):
                    ax.plot(fGradIter[:,n], color=scalarMap.to_rgba(n), label=r'\$\d_{x_{'+str(n+1)+r'}}\f$') #something wrong here
            fGradIterFig=plt.figure(figsize=(5, 4), dpi=200)
            ax2 = fGradIterFig.add_subplot(111)
            fGradIterPlot2(ax2)
            plt.xlabel(r"Iteration")
            plt.ylabel(r"\$\d_{x} \f$")
            plt.xlim(xmin=0, xmax=len(xIter)-1)
            ax2.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.,ncol=numColx,prop=fontPP)
            plt.subplots_adjust(left=0.07, right=(0.96-(numColx*0.078)), top=0.93, bottom=0.12)
            plt.savefig(ResultsFolder+OptName+'_fGradIter.pdf')
            plt.close()

        progressbar(35,82)

        #---------------------------------------------------------------------------------------------------
        # Design variables iteration plot
        #---------------------------------------------------------------------------------------------------
        plt.rc('text', usetex=True)
        def xIterPlot(ax):
            jet = plt.get_cmap('jet')
            cNorm  = colors.Normalize(vmin=0, vmax=len(xIter[0]))
            scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
            for n in range(len(xIter[0])):
                ax.plot(xIter[:,n], color=scalarMap.to_rgba(n), label="$x_{"+str(n+1)+"}$")
        xIterFig=plt.figure(figsize=(5, 4), dpi=200)
        ax2 = xIterFig.add_subplot(111)
        xIterPlot(ax2)
        plt.xlabel("Iteration")
        plt.ylabel("$x$")
        plt.xlim(xmin=0, xmax=len(xIter)-1)
        ax2.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.,ncol=numColx,prop=fontPP)
        plt.subplots_adjust(left=0.07, right=(0.96-(numColx*0.078)), top=0.93, bottom=0.12)
        plt.savefig(ResultsFolder+OptName+'_xIter.svg')
        plt.savefig(ResultsFolder+OptName+'_xIter.png')
        plt.savefig(ResultsFolder+OptName+'_xIter_wo.pdf')
        plt.close()

        plt.rc('text', usetex=False)
        def xIterPlot2(ax):
            jet = plt.get_cmap('jet')
            cNorm  = colors.Normalize(vmin=0, vmax=len(xIter[0]))
            scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
            for n in range(len(xIter[0])):
                ax.plot(xIter[:,n], color=scalarMap.to_rgba(n), label=r"\$x_{"+str(n+1)+"}$")
        xIterFig=plt.figure(figsize=(5, 4), dpi=200)
        ax2 = xIterFig.add_subplot(111)
        xIterPlot2(ax2)
        plt.xlabel(r"Iteration")
        plt.ylabel(r"\$\x$")
        plt.xlim(xmin=0, xmax=len(xIter)-1)
        ax2.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.,ncol=numColx,prop=fontPP)
        plt.subplots_adjust(left=0.07, right=(0.96-(numColx*0.078)), top=0.93, bottom=0.12)
        plt.savefig(ResultsFolder+OptName+'_xIter.pdf')
        plt.close()

        progressbar(41,82)

        #---------------------------------------------------------------------------------------------------
        # Normalized design variables iteration plot
        #---------------------------------------------------------------------------------------------------
        ######commment out if not normalized!
        if DesVarNorm in ["None", None, False]:
            pass
        else:
            plt.rc('text', usetex=True)
            def xIterNormPlot(ax):
                jet = plt.get_cmap('jet')
                cNorm  = colors.Normalize(vmin=0, vmax=len(xIter[0]))
                scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
                for n in range(len(xIterNorm[0])):
                    ax.plot(xIterNorm[:,n], color=scalarMap.to_rgba(n), label="$\hat{x}_{"+str(n+1)+"}$")
            xIterNormFig=plt.figure(figsize=(5, 4), dpi=200)
            ax2 = xIterNormFig.add_subplot(111)
            xIterNormPlot(ax2)
            plt.xlim(xmin=0, xmax=len(xIter)-1)
            ax2.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.,ncol=numColx,prop=fontPP)
            plt.subplots_adjust(left=0.07, right=(0.96-(numColx*0.078)), top=0.93, bottom=0.12)
            plt.xlabel("Iteration")
            plt.ylabel("$\hat{x}_j$")
            plt.savefig(ResultsFolder+OptName+'_xIterNorm.svg')
            plt.savefig(ResultsFolder+OptName+'_xIterNorm.png')
            plt.savefig(ResultsFolder+OptName+'_xIterNorm_wo.pdf')
            plt.close()

            plt.rc('text', usetex=False)
            def xIterNormPlot2(ax):
                jet = plt.get_cmap('jet')
                cNorm  = colors.Normalize(vmin=0, vmax=len(xIter[0]))
                scalarMap = cm.ScalarMappable(norm=cNorm, cmap=jet)
                for n in range(len(xIterNorm[0])):
                    ax.plot(xIterNorm[:,n], color=scalarMap.to_rgba(n), label=r"\$\xn_{"+str(n+1)+"}$")
            xIterNormFig=plt.figure(figsize=(5, 4), dpi=200)
            ax2 = xIterNormFig.add_subplot(111)
            xIterNormPlot2(ax2)
            plt.xlim(xmin=0, xmax=len(xIter)-1)
            ax2.legend(loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.,ncol=numColx,prop=fontPP)
            plt.subplots_adjust(left=0.07, right=(0.96-(numColx*0.078)), top=0.93, bottom=0.12)
            plt.xlabel(r"Iteration")
            plt.ylabel(r"\$\xn$")
            plt.savefig(ResultsFolder+OptName+'_xIterNorm.pdf')
            plt.close()

        progressbar(49,82)

        #---------------------------------------------------------------------------------------------------
        # Design variables bar plot
        #---------------------------------------------------------------------------------------------------
        plt.rc('text', usetex=True)
        xBarfig = plt.figure(figsize=(5, 4), dpi=200)
        width = 0.5
        xspacing=0.25
        ax = xBarfig.add_subplot(111)
        ind = np.arange(nx)
        rects1 = ax.bar(ind+xspacing,x0, width, color='#FAA43A')
        rects2 = ax.bar(ind+xspacing+width/2.,xOpt,width, color='#5DA5DA')
        ax.legend( (rects1[0], rects2[0]), ("$x_0$", "$x^*$"),prop = fontP)
        ax.set_xticks(ind+(width+width/2.)/2.+xspacing)
        #ax.locator_params(tight=False, nbins=4)
        plt.xlabel("Design variables")
        plt.ylabel("$x_j$")
        plt.xlim(xmin=0, xmax=nx+xspacing )
        plt.ylim(ymin=0, ymax=np.max((np.max(xOpt),np.max(xOpt))))
        #ax.set_xticklabels(ind+1)
        #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        plt.tight_layout()
        plt.savefig(ResultsFolder+OptName+'_xBar.svg')
        plt.savefig(ResultsFolder+OptName+'_xBar.png')
        plt.savefig(ResultsFolder+OptName+'_xBar_wo.pdf')
        plt.close()

        plt.rc('text', usetex=False)
        xBarfig = plt.figure(figsize=(5, 4), dpi=200)
        width = 0.5
        xspacing=0.25
#        ax2 = xBarfig2.add_subplot(111)
#        ind = np.arange(nx)
#        rects1 = ax2.bar(ind+xspacing,x0, width, color='#FAA43A')
#        rects2 = ax2.bar(ind+xspacing+width/2.,xOpt,width, color='#5DA5DA')
#        ax2.legend( (rects1[0], rects2[0]), (r"\$\xin$", r"\$\xo$"),prop = fontP)
#        ax2.set_xticks(ind+(width+width/2.)/2.+xspacing)
        #ax.locator_params(tight=False, nbins=4)
        ax = xBarfig.add_subplot(111)
        ind = np.arange(nx)
        rects1 = ax.bar(ind+xspacing,x0, width, color='#FAA43A')
        rects2 = ax.bar(ind+xspacing+width/2., xOpt, width, color='#5DA5DA')
        #ax.legend( (rects1[0], rects2[0]), ("$x_0$", "$x^*$"),prop = fontP)
        ax.legend( (rects1[0], rects2[0]), (r"\$\xin$", r"\$\xo$"), prop = fontP)
        ax.set_xticks(ind+(width+width/2.)/2.+xspacing)

        plt.xlabel(r"Design variables")
        plt.ylabel(r"\$\x$")
        plt.xlim(xmin=0, xmax=nx+xspacing )
        #plt.ylim(ymin=0, ymax=np.max((np.max(x0),np.max(xOpt))))
        plt.ylim(ymin=0, ymax=np.max((np.max(xOpt),np.max(xOpt))))
        #ax.set_xticklabels(ind+1)
        #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        ax.xaxis.set_major_locator(ticker.AutoLocator())
        plt.tight_layout()
        plt.savefig(ResultsFolder+OptName+'_xBar.pdf')
        plt.savefig(ResultsFolder+OptName+'_xBarTry.pdf')
        plt.close()

        progressbar(52,82)

        #---------------------------------------------------------------------------------------------------
        # Normalized design variables bar plot
        #---------------------------------------------------------------------------------------------------
        if DesVarNorm in ["None", None, False]:
            pass
        else:
            plt.rc('text', usetex=True)
            xBarNormfig = plt.figure(figsize=(5, 4), dpi=200)
            width = 0.5
            xspacing=0.25
            ax = xBarNormfig.add_subplot(111)
            ind = np.arange(nx)
            rects1 = ax.bar(ind+xspacing,x0norm, width, color='#FAA43A')
            rects2 = ax.bar(ind+xspacing+width/2.,xOptNorm,width, color='#5DA5DA')
            ax.legend( (rects1[0], rects2[0]), ("$\hat{x}_0$", "$\hat{x}^*$"), prop=fontP)
            #ax.set_xticks(ind+(width+width/2.)/2.+xspacing)
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            plt.xlim(xmin=0, xmax=nx+xspacing )
            plt.ylim(ymin=0, ymax=np.max((np.max(x0norm),np.max(xOptNorm))))
            plt.xlabel("Design variables")
            plt.ylabel("$\hat{x}_j$")
            plt.tight_layout()
            plt.savefig(ResultsFolder+OptName+'_xBarNorm.svg')
            plt.savefig(ResultsFolder+OptName+'_xBarNorm.png')
            plt.savefig(ResultsFolder+OptName+'_xBarNorm_wo.pdf')
            plt.close()

            plt.rc('text', usetex=False)
            xBarNormfig = plt.figure(figsize=(5, 4), dpi=200)
            width = 0.5
            xspacing=0.25
            ax = xBarNormfig.add_subplot(111)
            ind = np.arange(nx)
            rects1 = ax.bar(ind+xspacing,x0norm, width, color='#FAA43A')
            rects2 = ax.bar(ind+xspacing+width/2.,xOptNorm,width, color='#5DA5DA')
            ax.legend( (rects1[0], rects2[0]), (r"\$\xnin$", r"\$\xno$"), prop=fontP)
            ax.xaxis.set_major_locator(ticker.AutoLocator())
            plt.xlim(xmin=0, xmax=nx+xspacing )
            plt.ylim(ymin=0, ymax=np.max((np.max(x0norm),np.max(xOptNorm))))
            plt.xlabel(r"Design variables")
            plt.ylabel(r"\$\xn$")
            plt.tight_layout()
            plt.savefig(ResultsFolder+OptName+'_xBarNorm.pdf')
            plt.close()

        progressbar(56,82)

        #---------------------------------------------------------------------------------------------------
        # Gradients plot
        #---------------------------------------------------------------------------------------------------
        plt.rc('text', usetex=True)
        try:
            plt.pcolor(gGradOpt.T)
            plt.colorbar()
        except: pass
        plt.xlabel("Design variables")
        plt.ylabel("Constraints")
        plt.xlim(xmin=0, xmax=nx )
        plt.ylim(ymin=0, ymax=ng)
        plt.grid(True, which='both')
        plt.tight_layout()
        plt.savefig(ResultsFolder+OptName+'_gGradOpt.svg')
        plt.savefig(ResultsFolder+OptName+'_gGradOpt.png')
        plt.savefig(ResultsFolder+OptName+'_gGradOpt_wo.pdf')
        plt.close()

        plt.rc('text', usetex=False)
        try:
            plt.pcolor(gGradOpt.T)
            plt.colorbar()
        except: pass
        plt.xlabel(r"Design variables")
        plt.ylabel(r"Constraints")
        plt.xlim(xmin=0, xmax=nx )
        plt.ylim(ymin=0, ymax=ng)
        plt.grid(True, which='both')
        plt.tight_layout()
        plt.savefig(ResultsFolder+OptName+'_gGradOpt.pdf')
        plt.close()

        progressbar(63,82)

        #---------------------------------------------------------------------------------------------------
        # Convert the PDF's with Inkscape to PDF-Latex files
        #---------------------------------------------------------------------------------------------------
        PlotFiles = ["gBar", "fBar", "fgMaxIterNorm", "fgMaxIter", "gIter",
                      "fGradIter", "xIter", "xIterNorm", "gGradOpt", "xBar", "xBarNorm"]
        mProc = [[]]*len(PlotFiles)
        for ii in range(len(PlotFiles)):
            mProc[ii] = Popen(InkscapeCall + " -D -z --file="+
                              ResultsFolder+OptName+"_"+ PlotFiles[ii] + ".pdf"+
                              " --export-pdf="+
                              ResultsFolder+OptName+"_"+ PlotFiles[ii] + ".pdf"+
                              " --export-latex", shell=True).wait()
        progressbar(82,82)
        print '# --------------- DIAGRAM GENERATION FINISHED! --------------- #'

    if tables==1:
        #---------------------------------------------------------------------------------------------------
        # Options Table
        #---------------------------------------------------------------------------------------------------
        tRR = open(""+ResultsFolder+OptName+"_OptAlgOptionsTable.tex","w")
        optCrit = []
        optCrit.append('\\begin{table}[H] \n')
        optCrit.append('\\caption{Algorithm options} \n')
        optCrit.append('\\noindent \n')
        optCrit.append('\\begin{centering} \n')
        optCrit.append('\\begin{tabular}{cc} \n')
        optCrit.append('\\toprule \n')
        optCrit.append('Property & Value \\tabularnewline \n')
        #optCrit.append('\\midrule \n')
        #optCrit.append('\\midrule \n')os.system(
        optCrit.append('\\midrule \n')
        for uu in OptAlg.options:
            if str(uu) != "defaults" and str(uu) != "IFILE":
                temp = [str(uu),'&',str(OptAlg.options[uu][1]),'\\tabularnewline \n']
                optCrit=optCrit + temp
                #optCrit.append('\\bottomrule \n')
        optCrit.append('\\bottomrule \n')
        optCrit.append('\\end{tabular} \n')
        optCrit.append('\\par\\end{centering} \n')
        optCrit.append('\\end{table}')
        tRR.writelines(optCrit)
        tRR.close()
        os.system("tex2lyx "+ResultsFolder+OptName+"_OptAlgOptionsTable.tex")

        # Normalized data
        if DesVarNorm in ["None", None, False]:
            pass
        else:
            #---------------------------------------------------------------------------------------------------
            # Normalized design variables table
            #---------------------------------------------------------------------------------------------------
            tRR = open(""+ResultsFolder+OptName+"_DesignVarTableNorm.tex","w")
            dvT = []
            dvT.append('\\begin{table}[H] \n')
            dvT.append('\\caption{Details of normalized design variables $\hat{x}$} \n')
            dvT.append('\\noindent \n')
            dvT.append('\\begin{centering} \n')
            dvT.append('\\begin{tabular}{cccccc} \n')
            dvT.append('\\toprule \n')
            dvT.append(r'\begin{minipage}{2.7cm} \centering Normalized \\ design variable \end{minipage} & Symbol & \begin{minipage}{2cm} \centering Start \\ value $\xn^{0}$ \end{minipage} & \begin{minipage}{2cm} \centering Lower \\ bound $\xn^{L}$ \end{minipage} &')
            dvT.append(r'\begin{minipage}{2cm} \centering Upper \\ bound $\xn^{U}$ \end{minipage} & \begin{minipage}{2cm} \centering Optimal \\ value $\xn^{*}$ \end{minipage} \tabularnewline')
            dvT.append('\n \\midrule \n')
            if nx==1:
                #dvT.append('\\midrule \n')
                temp=[str(1),r'&$\xn_{1}$&',str(round(x0norm,4)),'&',str(0.),'&',str(1.),'&',str(round(xOptNorm,4)),'\\tabularnewline \n']
                dvT=dvT+temp
            else:
                for ii in range(nx):
                    #dvT.append('\\midrule \n')
                    temp=[str(ii+1),r'&$\xn_{',str(ii+1),'}$&',str(round(x0norm[ii],4)),'&',str(0.),'&',str(1.),'&',str(round(xOptNorm[ii],4)),'\\tabularnewline \n']
                    dvT=dvT+temp
            dvT.append('\\bottomrule \n')
            dvT.append('\\end{tabular} \n')
            dvT.append('\\par\\end{centering} \n')
            dvT.append('\\end{table}')
            tRR.writelines(dvT)
            tRR.close()
            os.system("tex2lyx "+ResultsFolder+OptName+"_DesignVarTableNorm.tex")

        #---------------------------------------------------------------------------------------------------
        # Design variables table
        #---------------------------------------------------------------------------------------------------
        tRR = open(""+ResultsFolder+OptName+"_DesignVarTable.tex","w")
        dvT = []
        dvT.append('\\begin{table}[H] \n')
        dvT.append('\\caption{Details of design variables $x$} \n')
        dvT.append('\\noindent \n')
        dvT.append('\\begin{centering} \n')
        dvT.append('\\begin{tabular}{cccccc} \n')
        dvT.append('\\toprule \n')
        dvT.append(r'\begin{minipage}{2cm} \centering Design \\ variable \end{minipage} & Symbol & \begin{minipage}{2cm} \centering Start \\ value $\x^{0}$ \end{minipage} & \begin{minipage}{2cm} \centering Lower \\ bound $\x^{L}$ \end{minipage} &')
        dvT.append(r'\begin{minipage}{2cm} \centering Upper \\ bound $\x^{U}$ \end{minipage} & \begin{minipage}{2cm} \centering Optimal \\ value $\x^{*}$ \end{minipage} \tabularnewline')
        dvT.append('\n \\midrule  \n')
        if nx==1:
            #dvT.append('\\midrule \n')
            temp=[str(1),r'&$\x_{1}$&',str(round(x0,4)),'&',str(round(xL,4)),'&',str(round(xU,4)),'&',str(round(xOpt,4)),'\\tabularnewline \n']
            dvT=dvT+temp
        else:
            for ii in range(nx):
                #dvT.append('\\midrule \n')
                temp=[str(ii+1),r'&$\x_{',str(ii+1),'}$&',str(round(x0[ii],4)),'&',str(round(xL[ii],4)),'&',str(round(xU[ii],4)),'&',str(round(xOpt[ii],4)),'\\tabularnewline \n']
                dvT=dvT+temp
        dvT.append('\\bottomrule \n')
        dvT.append('\\end{tabular} \n')
        dvT.append('\\par\\end{centering} \n')
        dvT.append('\\end{table}')
        tRR.writelines(dvT)
        tRR.close()
        # Convert tex file to lyx file
        os.system("tex2lyx "+ResultsFolder+OptName+"_DesignVarTable.tex")

        # ---------------------------------------------------------------------------------------------------
        # System responses table
        # ---------------------------------------------------------------------------------------------------
        tRR = open(""+ResultsFolder+OptName+"_SystemResponseTable.tex", "w")
        srT = []
        srT.append('\\begin{table}[H] \n')
        srT.append('\\caption{System responses} \n')
        srT.append('\\noindent \n')
        srT.append('\\begin{centering} \n')
        srT.append('\\begin{tabular}{cccc} \n')
        srT.append('\\toprule \n')
        srT.append('Response & Symbol & Start value  & Optimal value \\tabularnewline \n')
        srT.append('\\midrule \n')
        # srT.append('\\midrule \n')
        # temp = [r'Objective function  & $\f$ & ',str(round(fIter[0][0],4)),'&',str(round(fOpt[0],4)),'\\tabularnewline']
        temp = [r'Objective function  & $\f$ & ', str(round(fIter[0][0], 4)), '&', str(round(fOpt[0], 4)), '\\tabularnewline \n']
        srT = srT + temp
        if np.size(gc) > 0:
            for ii in range(ng):
                # srT.append('\\midrule \n')
                temp = ['Inequality constraint ', str(ii+1), r'&$\g_{', str(ii+1), '}$&', str(round(gIter[0][ii], 4)), '&', str(round(gOpt[ii],4)), '\\tabularnewline \n']
                srT = srT+temp
        srT.append('\\bottomrule \n')
        srT.append('\\end{tabular} \n')
        srT.append('\\par\\end{centering} \n')
        srT.append('\\end{table}')
        tRR.writelines(srT)
        tRR.close()
        # Convert tex file to lyx file
        os.system("tex2lyx "+ResultsFolder+OptName+"_SystemResponseTable.tex")

        # ---------------------------------------------------------------------------------------------------
        # First order and lagrange table
        # ---------------------------------------------------------------------------------------------------
        if np.size(gc) > 0:
            tRR = open(""+ResultsFolder+OptName+"_FirstOrderLagrangeTable.tex", "w")
            optCrit = []
            optCrit.append('\\begin{table}[H] \n')
            optCrit.append('\\caption{First-order optimality as well as non-zero Lagrangian multipliers} \n')
            optCrit.append('\\noindent \n')
            optCrit.append('\\begin{centering} \n')
            optCrit.append('\\begin{tabular}{ccc} \n')
            optCrit.append('\\toprule \n')
            optCrit.append('Property & Symbol & Value \\tabularnewline \n')
            optCrit.append('\\midrule \n')
            # optCrit.append('\\midrule \n')
            try:
                temp = ['First-order optimality & $\\left\\Vert \\mathcal{\\nabla L}\\right\\Vert $ & ', str(round(Opt1Order, 4)), '\\tabularnewline \n']
                optCrit = optCrit + temp
                for uu in range(len(lambda_c)):
                    #optCrit.append('\\midrule \n')
                    temp = [r'Lagrangian multiplier of $\g_{',str(uu+1),r'}$ & $\lambda_{\g_{',str(uu+1),'}}$ & ',str(round(lambda_c[uu],4)),'\\tabularnewline \n']
                    optCrit = optCrit + temp
                optCrit.append('\\bottomrule \n')
                optCrit.append('\\end{tabular} \n')
                optCrit.append('\\par\\end{centering} \n')
                optCrit.append('\\end{table}')
                tRR.writelines(optCrit)
                tRR.close()
                os.system("tex2lyx "+ResultsFolder+OptName+"_FirstOrderLagrangeTable.tex")
            except:
                pass
    # ---------------------------------------------------------------------------------------------------
    # Shadow prices table
    # ---------------------------------------------------------------------------------------------------
            tRR = open(""+ResultsFolder+OptName+"_ShadowPricesTable.tex", "w")
            optCrit = []
            optCrit.append('\\begin{table}[H] \n')
            optCrit.append('\\caption{Shadow prices} \n')
            optCrit.append('\\noindent \n')
            optCrit.append('\\begin{centering} \n')
            optCrit.append('\\begin{tabular}{ccc} \n')
            optCrit.append('\\toprule \n')
            optCrit.append('Property & Symbol & Value \\tabularnewline \n')
            optCrit.append('\\midrule \n')
            for uu in range(len(SPg)):
                # optCrit.append('\\midrule \n')
                temp = ['Shadow price of $g_{', str(uu+1), '}$ & $S_{g_{', str(uu+1), '}}$ & ', str(round(SPg[uu], 4)), '\\tabularnewline \n']
                optCrit = optCrit + temp
            optCrit.append('\\bottomrule \n')
            optCrit.append('\\end{tabular} \n')
            optCrit.append('\\par\\end{centering} \n')
            optCrit.append('\\end{table}')
            tRR.writelines(optCrit)
            tRR.close()
            # Convert tex file to lyx file
            os.system("tex2lyx "+ResultsFolder+OptName+"_ShadowPricesTable.tex")

        # ---------------------------------------------------------------------------------------------------
        # Write LyX solution document and print as pdf
        # ---------------------------------------------------------------------------------------------------
        templatePath = os.path.dirname(os.path.realpath(__file__)) + "/ResultReportFiles/"
        shutil.copy(templatePath + "/TUM.eps", ResultsFolder+"TUM.eps")
        shutil.copy(templatePath + "/FG_CM_blau_oZ_CMYK.eps", ResultsFolder + "FG_CM_blau_oZ_CMYK.eps")
        shutil.copy(templatePath + "/FGCM_Background.pdf", ResultsFolder + "FGCM_Background.pdf")
        FileName = ["_ResultPresentationPy.lyx", "_ResultReportPy.lyx"]
        for ii in range(2):
            fRR = open(templatePath + FileName[ii], "r+")
            contentRR = fRR.readlines()
            # Replace Modelname and Date
            contentRR = [w.replace('XXXmodelname', OptModel) for w in contentRR]
            contentRR = [w.replace('XXXdate', today) for w in contentRR]
            # Replace computername, operating system, computer-architecture, number of processors, user
            contentRR = [w.replace('XXXcomputername', computerName) for w in contentRR]
            contentRR = [w.replace('XXXoperatingsystem', operatingSystem) for w in contentRR]
            contentRR = [w.replace('XXXarchitecture', architecture) for w in contentRR]
            contentRR = [w.replace('XXXnprocessors', nProcessors) for w in contentRR]
            contentRR = [w.replace('XXXuser', userName) for w in contentRR]
            # Replace modelname, number of design variables, number of constraints, algorithm name and options
            contentRR = [w.replace('XXXmodelname', OptModel) for w in contentRR]
            contentRR = [w.replace('XXXnx', str(np.size(x0))) for w in contentRR]
            if np.size(gc) > 0:
                contentRR = [w.replace('XXXnc', str(np.size(gc))) for w in contentRR]
            else:
                contentRR = [w.replace('XXXnc', str(0)) for w in contentRR]
            contentRR = [w.replace('XXXalgorithmname', Alg) for w in contentRR]
            contentRR = [w.replace('XXXdesvarnorm', str(DesVarNorm)) for w in contentRR]
            contentRR = [w.replace('XXXoptions', "Options") for w in contentRR]
            # Replace number of iterations, number of evaluations, starting time, ending time, elapsed time
            contentRR = [w.replace('XXXnit', str(nIter)) for w in contentRR]
            contentRR = [w.replace('XXXneval', str(nEval)) for w in contentRR]
            contentRR = [w.replace('XXXt0', hhmmss0) for w in contentRR]
            contentRR = [w.replace('XXXt1', hhmmss1) for w in contentRR]
            contentRR = [w.replace('XXXtopt', str(OptTime)) for w in contentRR]
            # Replace AAA for pictures and tables
            contentRR = [w.replace('AAA', OptName) for w in contentRR]
            fRR.close()
            fRR = open(ResultsFolder+OptName+FileName[ii], "w")
            fRR.writelines(contentRR)
            fRR.close()
    if lyx == 1:
        for ii in range(2):
            os.system(LyxCall + " --export pdf2 " + ResultsFolder + OptName + FileName[ii])


def progressbar(actTime, totTime):
    toolbar_width = 60
    percentage = float(actTime)/float(totTime)
    numOfChars = int(percentage*toolbar_width)
    Char = '='
    sys.stdout.write('\r[')
    sys.stdout.flush()
    for i in range(numOfChars):
        if i == toolbar_width/2-1:
            sys.stdout.write('%03i' % int(percentage*100)+'%')
        else:
            sys.stdout.write(Char)
            sys.stdout.flush()
    for i in range(numOfChars, toolbar_width):
        if i == toolbar_width/2-1:
            sys.stdout.write('%03i' % int(percentage * 100) + '%')
        else:
            sys.stdout.write(' ')
            sys.stdout.flush()
    sys.stdout.write(']')
    sys.stdout.flush()


def elapsed(starttime):
    time_now = time.time()
    print('time Elapsed:\t' + str(time_now-starttime))
