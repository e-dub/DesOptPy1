# -*- coding: utf-8 -*-
'''
-------------------------------------------------------------------------------
Title:          OptVideo.py
Units:          Unitless
Date:           October 16, 2013
Author:         E.J. Wehrle
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
Description
-------------------------------------------------------------------------------
Optimization suite for Python...

TODO several videos for system responses
TODO check frames per second
TODO check codec
TODO is mencorder on cluster, master, workstations?
TODO integrate in file OptPostProc?
TODO save pngs in result folder? Yes!
-------------------------------------------------------------------------------
'''
from __future__ import absolute_import, division, print_function
import os


def OptVideo(OptName):
    fps = 4
    # cd in _OptResultReports/OptName
    os.system("mencoder 'mf://DesVar***.png' -mf type=png:fps=" + str(fps) +
              " -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o DesVar.mpg")
    os.system("mencoder 'mf://SysRes***.png' -mf type=png:fps=" + str(fps) +
              " -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o SysRes.mpg")
