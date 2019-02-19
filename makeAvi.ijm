// Macro for ImageJ 1.52d for Windows
// written by Florian Kleiner 2019
// run from commandline as follows
// ImageJ-win64.exe -macro "C:\path\to\makeAvi.ijm" "D:\path\to\data\|aviFPS"

macro "REMPorenanalyse" {
	// check if an external argument is given or define the options
	arg = getArgument();
	removeCurtaining = true
	if ( arg == "" ) {
		dir = getDirectory("Choose the working directory");
		aviFPS = 50;
	} else {
		arg_split = split(getArgument(),"|");
		dir				= arg_split[0];
		aviFPS			= arg_split[1];
	}
	print("Starting process using the following arguments...");
	print("Directory: " + dir);
	mainDir = File.getParent(dir);
	print("Avi FPS: " + aviFPS);
	directoryStack = split(dir, "\\");
	outputFileName = directoryStack[directoryStack.length-1];
	print( "target file: " + outputFileName + ".avi" );
    
    
    
	list = getFileList(dir);
	
	// running main loop
	setBatchMode(true);
	for (i=0; i<list.length; i++) {
		path = dir+list[i];
		// select only images
		if (!endsWith(path,"/") && ( endsWith(path,".tif") ) ) {
			open(path);
			imageId = getImageID();
			// get image id to be able to close the main image
			if (nImages>=1) {
				i = list.length;
				width			= getWidth();
				height			= getHeight();
                close();
            }
        }
    }
	wBorder = 0.02*width;
    hBorder = 0.02*height;
    print("Image size X: " + width + "px , Y: " + height + "px, border X: " + wBorder + "pxm border Y: " + hBorder + "px");
    print("border X: " + wBorder + "pxm border Y: " + hBorder + "px");
    print("open image stack");
	run("Image Sequence...", "open=[" + dir + "] sort");
    
    print("Start stack alignment");
	run("Linear Stack Alignment with SIFT", "initial_gaussian_blur=1.60 steps_per_scale_octave=3 minimum_image_size=64 maximum_image_size=1024 feature_descriptor_size=4 feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 maximal_alignment_error=25 inlier_ratio=0.05 expected_transformation=Translation interpolate"); // -> Plugins -> Registration
	makeRectangle(wBorder, hBorder, width-wBorder, height-hBorder);
	run("Crop");
    width	= getWidth();
    height	= getHeight();
    if ( width > 1920 || height > 1080 ) {
        factor = 1920/width;
        print("resize factor: " + factor);
        run("Scale...", "x=" + factor + " y=" + factor + " depth=587 interpolation=Bilinear average process title=Scaled");
    }

    print("enhancing contrast");
	run("Enhance Contrast...", "saturated=0.3 normalize process_all");
    print("save as AVI");
	run("AVI... ", "compression=JPEG frame=" + aviFPS + " save=[" + File.getParent( mainDir ) + "/" + outputFileName + ".avi]");
	
	// exit script
	print("Done!");
	if ( arg != "" ) {
		run("Quit");
	}
}