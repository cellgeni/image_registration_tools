# Overview
This code is using *pystackreg* to register sequentially H&E images of tissue sections (but potentially any other RGB images of sections). It can work as with the list of single plane files or with one multiplane image file (tiff or ndpi).

# Preparation
## Environment
Copy folder to your local machine. Create conda environment by running line:
`conda env create -f environment.yml`
Activate environment:
`conda activate serial_registration`

## Configuration file
Configuration file should contain mandatory fields, as in example.
 - Path: path to either multiplane image file or to csv table with image paths. In case of single multiplane image make sure that image has 4 dimensions (XYZC) in any order, and metadata is available. In case of csv file make sure it contains the column with the name "image_path" where paths to all images are stored. Please make sure that order of image paths corresponds to sections order
 - Downscale: integer number, which describes the level of downscaling. If you want to use full-resolution image use value 1 for downscale. Be aware that due to limitations of warping function code can not process images with sizes more than 30000 pixels in one side
 - Output_fodler: output folder path
 - Sample_name: string that will be used to create output image and json file
 - Background_Intensity: that value will be used to replace black edges (purely esthetical reason), usually use values close to backround intensity (we assume white-gray-black backgrounds). For H&E images typical values will be in range: 220-240
 - Regularization: at the moment is not used
 - Regularization_strength: at the moment is not used
