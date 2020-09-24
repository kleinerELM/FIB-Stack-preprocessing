#########################################################
# Automated Alignment and data preparation for FIB/SEM 
# image stacks
#
# © 2019 Florian Kleiner
#   Bauhaus-Universität Weimar
#   Finger-Institut für Baustoffkunde
#
# programmed using python 3.7, gnuplot 5.2,
# Fiji/ImageJ 1.52k
# don't forget to install PIL (pip install Pillow)
#
#########################################################


import csv
import os, sys, getopt
import subprocess
import math
import tkinter as tk
import mmap
import shutil
import xml.etree.ElementTree as ET
import statistics
from PIL import Image
from tkinter import filedialog
from subprocess import check_output

print("#########################################################")
print("# Automated Alignment and data preparation for FIB/SEM  #")
print("# image stacks                                          #")
print("#                                                       #")
print("# © 2019 Florian Kleiner                                #")
print("#   Bauhaus-Universität Weimar                          #")
print("#   Finger-Institut für Baustoffkunde                   #")
print("#                                                       #")
print("#########################################################")
print()

#### directory definitions
#outputDir_Pores = "/pores/"
#suffix_Pores = "_pores_sqnm.csv"
#suffix_Pores = "_pores_sqpx.csv"
home_dir = os.path.dirname(os.path.realpath(__file__))

#### global var definitions
root = tk.Tk()
root.withdraw()

voxelSizeX = 0
voxelSizeY = 0
voxelSizeZ = 0
thicknesses = []
measuredThickness = 0
resX = 0
resY = 0
resZ = 0

runImageJ_Script = True #False
useMeasuredThickness = False
removeCurtaining = 1
createLogVideos = "n"
showDebuggingOutput = False
outputType = 0 # standard output type (y-axis value) is area-%
thresholdLimit = 140
infoBarHeight = 0
metricScale = 0
pixelScale  = 0
startFrame = 0
endFrame = 0

def processArguments():
    argv = sys.argv[1:]
    usage = sys.argv[0] + " [-h] [-i] [-s <start frame>] [-e <end frame>] [-l <i / a>] [-m] [-c] [-o <outputType>] [-t <thresholdLimit>] [-d]"
    try:
        opts, args = getopt.getopt(argv,"hims:e:cl:t:d",["noImageJ="])
        for opt, arg in opts:
            if opt == '-h':
                print( 'usage: ' + usage )
                print( '-h,                  : show this help' )
                print( '-i, --noImageJ       : skip ImageJ processing' )
                print( '-s                   : set starting frame (only for stack)' )
                print( '-e                   : set end frame (only for stack)' )
                print( '-m                   : use measured mean stack thickness instead of defined thickness' )
                print( '-c                   : define curtaining removal processes (1-4) or disable curtaining removal (0)' )
                print( '-l                   : create log videos (n=none, i=ion only, a=all)' )
                print( '-t                   : set threshold limit (0-255)' )
                print( '-d                   : show debug output' )
                print( '' )
                sys.exit()
            elif opt in ("-i", "-noImageJ"):
                print( 'deactivating ImageJ processing!' )
                global runImageJ_Script
                runImageJ_Script = False
            elif opt in ("-m"):
                print( 'using measured mean stack thickness!' )
                global useMeasuredThickness
                useMeasuredThickness = True
            elif opt in ("-s"):
                if ( int( arg ) > 0 ):
                    global startFrame
                    startFrame = int( arg )
                    print( 'start frame is set to: ' + str( startFrame ) )
            elif opt in ("-e"):
                if ( int( arg ) > 0 ):
                    global endFrame
                    endFrame = int( arg )
                    print( 'end frame is set to:   ' + str( endFrame ) )
            elif opt in ("-c"):
                print( 'curtaining removal deactivatd!' )
                global removeCurtaining
                removeCurtaining = int(arg)
                if( removeCurtaining == 0 ):
                    print( 'curtaining removal disabled' )
                else:
                    print( 'curtaining removal will be started using ' + str( removeCurtaining ) + ' process(es)' )
            elif opt in ("-l"):
                if ( arg == "i" or arg == "a" ):
                    global createLogVideos
                    createLogVideos = arg
                    if( arg == "i" ):
                        print( 'creating ion alignment video!' )
                    else:
                        print( 'creating all log videos!' )
            elif opt in ("-t"):
                if ( int( arg ) < 256 and int( arg ) > -1 ):
                    global thresholdLimit
                    thresholdLimit = int( arg )
                    print( 'set threshold limit to ' + str( thresholdLimit ) )
            elif opt in ("-d"):
                print( 'show debugging output' )
                global showDebuggingOutput
                showDebuggingOutput = True
    except getopt.GetoptError:
        print( usage )
    print( '' )

def analyseImages( directory ):
    sizeZ = voxelSizeZ
    if ( useMeasuredThickness ):
        sizeZ = round( measuredThickness/resZ*1000000000, 7 )
    command = "ImageJ-win64.exe -macro \"" + home_dir +"\FFT_Stack.ijm\" \"" + directory + "/|" + str(thresholdLimit) + "|" + str(voxelSizeX) + "|" + str(voxelSizeY) + "|" + str(sizeZ) + "|" + str(removeCurtaining) + "|" + str(startFrame) + "|" + str(endFrame) + "\""
    print( "starting ImageJ Macro..." )
    if ( showDebuggingOutput ) : print( command )
    try:
        subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print( "Error" )#"returned error (code {}): {}".format(e.returncode, e.output))
        pass

def convertToMP4( directory, filename ):
    sizeZ = voxelSizeZ
    if ( useMeasuredThickness ):
        sizeZ = round( measuredThickness/resZ*1000000000, 7 )
    command = "ffmpeg.exe -i '" + directory + filename + ".avi' -c:v libx264 -crf 19 -preset slow -vf \"scale='min(1920,iw)':'min(1080,ih)':force_original_aspect_ratio=decrease\" '" + directory + filename + ".mp4'"
    print( "starting ffmpeg conversion..." )
    if ( showDebuggingOutput ) : print( command )
    try:
        subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print( "Error" )#"returned error (code {}): {}".format(e.returncode, e.output))
        pass
def cmdExists(cmd):
    return shutil.which(cmd) is not None

def imageJInPATH():
    if ( not cmdExists( "ImageJ-win64.exe" ) ):
        if os.name == 'nt':
            print( "make sure you have Fiji/ImageJ installed and added the program path to the PATH variable" )
            command = "rundll32 sysdm.cpl,EditEnvironmentVariables"
            try:
                subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                print( "Error" )#"returned error (code {}): {}".format(e.returncode, e.output))
                pass
            print( "make sure you have Fiji/ImageJ installed and added the program path to the PATH variable" )
        else:
            print( "make sure Fiji/ImageJ is accessible from command line" )
        return False
    elif ( showDebuggingOutput ) : print( "Fiji/ImageJ found!" )
    return True

def getPixelSizeFromMetaData( directory, filename ):
    global voxelSizeX
    global voxelSizeY
    global voxelSizeZ
    if ( voxelSizeZ == 0 ):
        voxelSizeZ = int( input("Please enter the stack slice thickness [nm]: ") )
    with open(directory + '/' + filename, 'rb', 0) as file, \
        mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as s:
        if s.find(b'PixelWidth') != -1:
            file.seek(s.find(b'PixelWidth'))
            tempLine = str( file.readline() ).split("=",1)[1]
            voxelSizeX = float( tempLine.split("\\",1)[0] )*1000000000
            file.seek(s.find(b'PixelHeight'))
            tempLine = str( file.readline() ).split("=",1)[1]
            voxelSizeY = float( tempLine.split("\\",1)[0] )*1000000000
            print( " image scale X: " + str( voxelSizeX ) + " nm / px" )
            print( " image scale Y: " + str( voxelSizeY ) + " nm / px" )
            print( " image scale Z: " + str( voxelSizeZ ) + " nm / slice" )
    return True

def getInfoBarHeightFromMetaData( directory, filename ):
    contentHeight = 0
    global resX
    global resY
    global infoBarHeight
    with open(directory + '/' + filename, 'rb', 0) as file, \
        mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as s:
        if s.find(b'ResolutionX') != -1:
            file.seek(s.find(b'ResolutionX'))
            tempLine = str( file.readline() ).split("=",1)[1]
            resX = float( tempLine.split("\\",1)[0] )
        if s.find(b'ResolutionY') != -1:
            file.seek(s.find(b'ResolutionY'))
            tempLine = str( file.readline() ).split("=",1)[1]
            resY = float( tempLine.split("\\",1)[0] )
    if ( resY > 0 ):
        im = Image.open( directory + '/' + filename )
        width, height = im.size
        infoBarHeight = int( height - resY )
        if ( infoBarHeight < 1 ):
            if ( showDebuggingOutput ) : print( " no info bar detected" )
        else:
            print( " detected info bar height: " + str( infoBarHeight ) + " px" )
    else:
        print( " info bar height not detected" )
    return infoBarHeight

def scaleInMetaData( directory ):
    global infoBarHeight
    result = False
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if ( filename.endswith(".tif") or filename.endswith(".TIF")):
            if getPixelSizeFromMetaData( directory, filename ):
                getInfoBarHeightFromMetaData( directory, filename )
                result = True
                break
    return result

def getZResolution( directory ):
    global resZ
    global voxelSizeZ
    global measuredThickness
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if ( filename.endswith(".tif") or filename.endswith(".TIF")):
            resZ = resZ + 1
    print( " dataset thickness: " + str( resZ ) + " slices = " + str( resZ*voxelSizeZ/1000 ) + " µm" )
    print( " measured thickness: " + str( round( measuredThickness*1000000, 2 ) ) + " µm" )
    print( " mean measured image scale Z: " + str( round( measuredThickness/resZ*1000000000, 2) ) + "±" + str( round( statistics.stdev( thicknesses )*1000000000, 2 ) ) + " nm / slice" )

def createSizeDefinitionFile():
    global projectDir
    global voxelSizeX
    global voxelSizeY
    global voxelSizeZ
    global resX
    global resY
    global resZ
    global measuredThickness
    f = open( projectDir + "\size_definition.txt","w+")
    f.write( "name, X, Y, defined Z, measured Z, unit \n" )
    f.write( "voxel size, " + str(voxelSizeX) + ", " + str(voxelSizeY) + ", " + str(voxelSizeZ) + ", " + str(round(measuredThickness/resZ*1000000000, 4)) + ", nm\n" )
    f.write( "resolution, " + str(round(resX)) + ", " + str(round(resY)) + ", " + str(round(resZ)) + ", , px/slices\n" )
    f.write( "data set size, " + str(resX*voxelSizeX/1000) + ", " + str(resY*voxelSizeY/1000) + ", " + str(resZ*voxelSizeZ/1000) + ", " + str(round(measuredThickness*1000000, 7) ) + ", µm\n" )
    f.close()

def readProjectData( directory ):
    global voxelSizeZ
    global measuredThickness
    global projectDir
    parentDir = os.path.abspath(os.path.join(directory, os.pardir))
    projectDir = os.path.abspath(os.path.join(parentDir, os.pardir))
    projectDataXML = projectDir + "\ProjectData.dat"
    #print( projectDataXML )
    if ( not os.path.isfile( projectDataXML ) ):
        projectDataXML = filedialog.askopenfilename(title='Please select ProjectData.dat')
        projectDir = os.path.abspath(os.path.join(projectDataXML, os.pardir))
    if ( createLogVideos != "n" ): processLogImages( projectDir )
    tree = ET.parse( projectDataXML )
    root = tree.getroot()
    for child in root:
        if ( str( child.tag ) == "Name" ):
            print( " project title: " + str( child.text ) )
        if ( str( child.tag ) == "TimeCreated" ):
            print( " data set created on: " + str( child.text ) )
        if ( str( child.tag ) == "ExecutionResult" ):
            print( " project execution: " + str( child.text ) )
        if ( str( child.tag ) == "Results" ):
            for slice in child:
                if ( str( slice.tag ) == "Slice" ):
                    voxelSizeZ = float( slice.attrib['TargetThickness'] )*1000000000
                    if ( not str( slice.attrib['MeasuredThickness'] ) == "NaN" ):
                        measuredThickness = measuredThickness + float( slice.attrib['MeasuredThickness'] )
                        thicknesses.append(float( slice.attrib['MeasuredThickness'] ))
                    else:
                        thicknesses.append(0)
                #if ( str( child.tag ) == "Image" ):
                    
                    #print( str( slice.attrib['MeasuredThickness'] ) )
        #print( str( child.tag ) + ": " + str( child.attrib.text ) )

def logImagesToAvi( directory, workingDirectory, filename ):
    aviFPS = 50
    command = "ImageJ-win64.exe -macro \"" + home_dir +"\makeAvi.ijm\" \"" + workingDirectory + "|" + str(aviFPS) + "\""
    print( "starting ImageJ Macro..." )
    if ( showDebuggingOutput ) : print( command )
    try:
        subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print( "Error" )#"returned error (code {}): {}".format(e.returncode, e.output))
        pass
    command = "ffmpeg.exe -i '" + directory +  "\\" + filename + ".avi' -c:v libx265 -crf 19 -preset slow " + filename + ".mp4"

def processLogImages( directory ):
    global createLogVideos
    logDirectory = directory + "\LogImages\\"
    eAlignBeam = "Electron Alignment BeamShift"
    eAlignStage = "Electron Alignment StageMove"
    ionAlign = "Ion Alignment"
    eAlignBeamDir = logDirectory + eAlignBeam + "\\"
    eAlignStageDir = logDirectory + eAlignStage + "\\"
    ionAlignDir = logDirectory + ionAlign + "\\"
    print( " processing LogImages ..." )
    if ( os.path.isdir( logDirectory ) ):
        for file in os.listdir(logDirectory):
            filename = os.fsdecode(file)
            if ( filename.endswith(".tif") or filename.endswith(".TIF")):
                if ( createLogVideos == "a" ):
                    if ( "Electron - Alignment (BeamShift)" in filename ) :
                        if ( not os.path.isdir( eAlignBeamDir ) ) : os.mkdir( eAlignBeamDir )
                        os.rename( logDirectory + filename, eAlignBeamDir + filename )
                    if ( "Electron - Alignment (StageMove)" in filename ) :
                        if ( not os.path.isdir( eAlignStageDir ) ) : os.mkdir( eAlignStageDir )
                        os.rename( logDirectory + filename, eAlignStageDir + filename )
                if ( "Ion - Alignment" in filename ) :
                    if ( not os.path.isdir( ionAlignDir ) ) : os.mkdir( ionAlignDir )
                    os.rename( logDirectory + filename, ionAlignDir + filename )
    if ( createLogVideos == "a" ):
        if ( os.path.isdir( eAlignBeamDir ) ) :
            if ( runImageJ_Script ) : logImagesToAvi( directory, eAlignBeamDir, eAlignBeam )
            convertToMP4( directory + "\\", eAlignBeam )
        if ( os.path.isdir( eAlignStageDir ) ) :
            if ( runImageJ_Script ) : logImagesToAvi( directory, eAlignStageDir, eAlignStage )
            convertToMP4( directory + "\\", eAlignStage )
    if ( os.path.isdir( ionAlignDir ) ) :
        if ( runImageJ_Script ) : logImagesToAvi( directory, ionAlignDir, ionAlign )
        convertToMP4( directory + "\\", ionAlign )

processArguments()
if ( showDebuggingOutput ) : print( "I am living in '" + home_dir + "'" )

workingDirectory = filedialog.askdirectory(title='Please select the image / working directory')

if ( workingDirectory != "" ) :
    print( "Selected working directory: " + workingDirectory )
    readProjectData( workingDirectory )

    #main process
    if scaleInMetaData( workingDirectory ) :
        # use metaData in files to determine scale
        getZResolution( workingDirectory )
        createSizeDefinitionFile()
        if ( runImageJ_Script and imageJInPATH() ):
            analyseImages( workingDirectory )
    else:
        print( "No matching metadata found!" )
else:
    print("No directory selected")


print("-------")
print("DONE!")