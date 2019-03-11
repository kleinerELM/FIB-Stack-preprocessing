// Macro for ImageJ 1.52d for Windows
// written by Florian Kleiner 2019
// run from commandline as follows
// ImageJ-win64.exe -macro "C:\path\to\FFT_Stack.ijm" "D:\path\to\data\|threshold|scaleX|scaleY|scaleZ|removeCurtaining|startFrame|endFrame"

macro "REMPorenanalyse" {
	// check if an external argument is given or define the options
	arg = getArgument();
	removeCurtaining = true;
	startFrame = 0;
	endFrame = 0;
	frameNumber = endFrame - startFrame;
	if ( arg == "" ) {
		dir = getDirectory("Choose the working directory");
		scaleX = getNumber("Voxel size in X direction [nm]", 1);
		scaleY = getNumber("Voxel size in Y direction [nm]", 1);
		scaleZ = getNumber("Voxel size in Z direction [nm]", 1);
	} else {
		arg_split = split(getArgument(),"|");
		dir				= arg_split[0];
		scaleX			= arg_split[2];
		scaleY			= arg_split[3];
		scaleZ			= arg_split[4];
		if ( parseInt(arg_split[5]) == 0 ) {
			removeCurtaining = false;
		}
		startFrame		= arg_split[6];
		endFrame		= arg_split[7];
	}
	print("Starting process using the following arguments...");
	print("Directory: " + dir);
	print("Parent directory: " + File.getParent(dir));
	
	//directory handling
	invTmpDir = "inverse";
	outputDirInverse = dir + "/" + invTmpDir + "/";
	File.makeDirectory(outputDirInverse);
	outputDirStacks = File.getParent(dir) + "/";
	directoryStack = split(dir, "/");
	outputFileName = directoryStack[directoryStack.length-1];
	print( "file: " + outputFileName );
	list = getFileList(dir);
	
	// running main loop
	setBatchMode(true);
	for (i=0; i<list.length; i++) {
		path = dir+list[i];
		// get all files
		showProgress(i, list.length);
		// select only images
		if (!endsWith(path,"/") && ( endsWith(path,".tif") || endsWith(path,".jpg") || endsWith(path,".JPG") ) ) {
			open(path);
			imageId = getImageID();
			// get image id to be able to close the main image
			if (nImages>=1) {
				//////////////////////
				// name definitions & image constants
				//////////////////////
				filename = getTitle();
				print( filename );
				if ( !File.exists(outputDirInverse + filename ) ) {
					baseName		= substring(filename, 0, lengthOf(filename)-4);
					fftName			= baseName + "-fft.tif";
					cleanName		= baseName + "-fft.tif";
					width			= getWidth();
					height			= getHeight();

					//////////////////////
					// processing FFT
					//////////////////////
					run("FFT");
					distanceFromCenter = 17;
					FFTsize			= getWidth(); // height = width!
					ovalHeight = FFTsize * 0.015; // 15
					ovalWidth = FFTsize / 2 - distanceFromCenter;

					//////////////////////
					// removing curtaining
					//////////////////////
					print( "  remove horizontal structures in FFT..." );
					makeOval(0, ((FFTsize-ovalHeight)/2), ovalWidth, ovalHeight);
					setBackgroundColor(0, 0, 0);
					run("Clear", "slice");
					makeOval((FFTsize/2 + distanceFromCenter), ((FFTsize-ovalHeight)/2), ovalWidth, ovalHeight);
					run("Clear", "slice");
					print( "  inverse FFT..." );
					run("Inverse FFT");
					selectWindow("FFT of " + filename);
					//saveAs("Tiff", outputDirInverse + "FFT_" + filename );
					close();
					selectWindow("Inverse FFT of " + filename);
					saveAs("Tiff", outputDirInverse + filename );
					close();
				} else {
					print( "  Inverse FFT already exists! Aborting..." );
				}

				//////////////////////
				// close the main file
				//////////////////////
				print( "  closing file ..." );
				selectImage(imageId);
				close();
				print( "" );
			}
		}
	}
	print( "Creating Image Stack ..." );
	// calculating the selected stack range
	// Todo: plausibility check if out of range!
	imageSeqOptions = "";
	if ( startFrame > 0 ) imageSeqOptions = " starting=" + startFrame;
	if ( endFrame > 0 ) {
		imageSeqOptions = " number=" + frameNumber + imageSeqOptions;
	}
	run("Image Sequence...", "open=[" + outputDirInverse + filename + "]" + imageSeqOptions + " sort use");
	invImageStackId = getImageID();
	run("Linear Stack Alignment with SIFT", "initial_gaussian_blur=1.60 steps_per_scale_octave=3 minimum_image_size=64 maximum_image_size=1024 feature_descriptor_size=4 feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 maximal_alignment_error=25 inlier_ratio=0.05 expected_transformation=Translation interpolate"); // -> Plugins -> Registration
	saveAs("Tiff", outputDirStacks + "/" + outputFileName + "_Cleaned_Stack_Aligned.Tif" );
	imageId = getImageID();
	selectImage(invImageStackId);
	close();
	setBatchMode(false);
	//select rectangle tool
	setTool(0);
	selectImage(imageId);
	waitForUser("Please crop the ROI", "Place a Rectangle to Crop and close this message afterwards.");
	//make sure we have got a rectangular selection
	if (selectionType() != 0) exit("Sorry, no rectangle selected! Aborting Macro!");
	run("Crop");
	saveAs("Tiff", outputDirStacks + "/" + outputFileName + "_Cropped_Stack.Tif" );
	run("Mean 3D...", "x=2 y=2 z=2");
	saveAs("Tiff", outputDirStacks + "/" + outputFileName + "_Cropped_Stack_Mean3D.Tif" );

	// exit script
	print("Done!");
	if ( arg != "" ) {
		run("Quit");
	}
}
