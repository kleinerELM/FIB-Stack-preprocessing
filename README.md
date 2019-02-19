# FIB-Stack-preprocessing
Preprocess a FIB/SEM-Stack using Fiji from "Auto Slice &amp; View" by Thermofischer Scientific

## Requirements
[Fiji](https://fiji.sc/) (tested with ImageJ 1.52i)

[Python](https://www.python.org/) (tested with Python 3.7.2)

[ffmpeg](https://www.ffmpeg.org/) (tested with Python 3.7.2)


The tool is developed for the windows world. Therefore, it won't work correctly using linux.

Add the binary folders of Fiji and Python to the Windows PATH variable!

## Usage

run the script using the following parameters:
```
.\start_preprocess.py [-h] [-i] [-m] [-c] [-o <outputType>] [-t <thresholdLimit>] [-d]
-h,                  : show this help
-i, --noImageJ       : skip ImageJ processing
-m                   : use measured mean stack thickness instead of defined thickness
-c                   : disable curtaining removal
-l                   : create log videos
-t                   : set threshold limit (0-255)
-d                   : show debug output
```
Select the folder containing the image-stack within the project folder of "Auto Slice &amp; View" by Thermofischer Scientific.

Select the Area of interest when ImageJ asks you to.

## Results
