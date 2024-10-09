# Overview
This code does image alignment accordingly to [MOSCOT](https://github.com/theislab/moscot) alignment. It uses transformation matrices for scaling and rotation from MOSCOT and complete them with shift components.

# Preparation
## Environment
Copy folder to your local machine. Create conda environment by running line:
`conda env create -f environment.yml`
Activate environment:
`conda activate MOSCOT_registration`

## Configuration file
Configuration file should contain mandatory fields, as in example (see **conf.yaml**):
 - *path_adata*: path to *h5ad* anndata file, which is output from MOSCOT
 - *downscale*: integer number, which describes the level of downscaling. If you want to use full-resolution image use value 1 for downscale. 
 - *out_file_path*: path for final image (tiff)
 - *final_image_size*: the image size (after downscaling) for the final image in one side (it assume quadratic image). Be aware that due to limitations of warping function code can not process images with sizes more than 30000 pixels in one side
 - *path_table_img_path*: the path to csv file which contains paths to all raw images (tiff or ndpi). Please make sure that column names are: "image_path" - full path to each image; "sample_name" - name of the section. Make sure that "sample_name" from csv is matching section names from anndata!

# Running
In conda environment simply run:
`python align_sections_from_MOSCOT.py path/to/conf.yaml`

# Output
As output code produces one image file with sizes: *N_planes* x *final_image_path* x *final_image_path*

# Postregistration
## Overview
MOSCOT alignment can be improved by using [Serial Registration of H&E images](https://github.com/cellgeni/image_registration_tools/tree/main/serial_registration_HE). For this run Serial Registration on output image stack from MOSCOT alignment. 

## Running Serial Alignment
In configuration file for Serial Registration specify path to output image from MOSCOT alignment. Downscale can be 1 if you don't want to downscale image even more, but keep current level of downscaling. After you runned Serial Alignment check output image stack, and if you find it aligned better than image stack from MOSCOT alignment, go to the next step of reassigning spot positions in anndata file.

## Configuration file for postregistration spot reassignment
If postregistration is successful you can update visium spot positions accordingly to postregistered images. For this firstly create configuration file (see conf_assignment.py):
 - *path_adata*: path to *h5ad* anndata file, which is output from MOSCOT (the same as in conf fiel for MOSCOT image alignment)
 - *path_tmat_json*: path to *json* file which contains all computed transformation matrices frm Serial Alignment step
 - *downscale*: this is total downscale applied to the images including (i) MOSCOT alignment and (ii) Serial Alignment. For example, if downscale in MOSCOT alignment was 2 and downscale in Serial Alignment was 5, the total downscale should be 2*5=10
 - *out_file_path*: path for final save anndata *h5ad*
 - *cx*, *cy*: center points of the image saved at Serial Alignment step. For example if image size was [1000, 1000]: *cx* = 500, *cy* = 500

## Running
In conda environment simply run:
`python reassign_spots.py path/to/conf_reassign.yaml`

## Output
As output program will save anndata h5ad file which will be the same anndata file as from MOSCOT output with additional new spot positions in `adata.obsm['spatial_affine_postreg']`
