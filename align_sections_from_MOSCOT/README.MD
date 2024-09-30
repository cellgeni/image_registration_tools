# Overview
This code does image alignment accordingly to [MOSCOT](https://github.com/theislab/moscot) alignment. It uses transformation matrices for scaling and rotation from MOSCOT and complete them with shift components.

# Preparation
## Environment
Copy folder to your local machine. Create conda environment by running line:
`conda env create -f environment.yml`
Activate environment:
`conda activate MOSCOT_registration`

## Configuration file
Configuration file should contain mandatory fields, as in example.
 - **path_adata**: path to *h5ad* anndata file, which is output from MOSCOT
 - **downscale**: integer number, which describes the level of downscaling. If you want to use full-resolution image use value 1 for downscale. 
 - **out_file_path**: path for final image (tiff)
 - **final_image_size**: the image size (after downscaling) for the final image in one side (it assume quadratic image). Be aware that due to limitations of warping function code can not process images with sizes more than 30000 pixels in one side
 - **path_table_img_path**: the path to csv file which contains paths to all raw images (tiff or ndpi). Please make sure that column names are: "image_path" - full path to each image; "sample_name" - name of the section. Make sure that "sample_name" from csv is matching section names from anndata!

# Running
In conda environment simply run:
`python align_sections_from_MOSCOT.py path/to/conf.yaml`

# Output
As output code produces one image file with sizes: *N_planes* x *final_image_path* x *final_image_path*
