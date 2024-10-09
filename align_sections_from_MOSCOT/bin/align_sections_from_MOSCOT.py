import pandas as pd
import tifffile
import anndata
import numpy as np
import cv2 as cv
import fire
from tifffile import memmap
import yaml
import gc

def find_center_spot_pos(adata, section_name):
    idx = adata.obs.index.str.contains(section_name)
    spot_pos_section = adata.obsm['spatial'][idx]
    #note: here x positions are in first column and y positions are in second!
    #xc = (np.max(spot_pos_section[:,0])-np.min(spot_pos_section[:,0]))/2+np.min(spot_pos_section[:,0])
    #yc = (np.max(spot_pos_section[:,1])-np.min(spot_pos_section[:,1]))/2+np.min(spot_pos_section[:,1])
    xc = np.mean(spot_pos_section[:,0]); yc = np.mean(spot_pos_section[:,1])
    return xc, yc


def ReadConfFile(FilePath):
    with open(FilePath, 'r') as file:
        data = yaml.safe_load(file)
    path_adata = data['path_adata']
    downscale = data['downscale']
    out_file_path = data['out_file_path']
    final_image_size = data['final_image_size']
    path_table_img_paths = data['path_table_img_paths']
    return path_adata, downscale, out_file_path, final_image_size, path_table_img_paths

def take_img_and_tr_mat(table_img_paths, tr_mat_all_dict, section_name):
    tr_mat = tr_mat_all_dict['alignment_metadata'][section_name]
    img_path = table_img_paths[table_img_paths['sample_name'].str.contains(section_name)]['image_path'].item()
    img = tifffile.imread(img_path)
    return img, tr_mat

def register_one_image(img, final_image_size, tr_mat, xc, yc, downscale):
    img_d = img[::downscale, ::downscale]
    huge_image_one = np.zeros((final_image_size, final_image_size, 3), dtype = 'uint8')
    #print(xc); print(yc)
    #idea is to use centre of the spot positions and place it exactly in the center of the big image
    ystart = (final_image_size/2-yc/downscale); xstart = (final_image_size/2-xc/downscale)
    yend = ystart+img_d.shape[0]; xend = xstart+img_d.shape[1];
    #print(ystart); print(xstart)
    #check for boundaries
    if ystart>=0 and xstart>=0 and yend<huge_image_one.shape[0] and xend<huge_image_one.shape[1]:
        huge_image_one[int(ystart):int(yend),int(xstart):int(xend)] = img_d
    else:
        x_d_start=0; y_d_start=0; y_d_end = img_d.shape[0]; x_d_end = img_d.shape[1]
        #print(ystart)
        #print(xstart)
        #print(yend)
        #print(xend)
        if ystart<0:
            y_d_start = -ystart+1; ystart=0;
        if xstart<0:
            x_d_start = -xstart+1; xstart=0;
        if yend>huge_image_one.shape[0]:
            y_d_end = img_d.shape[0]-(yend-huge_image_one.shape[0])+1; yend = huge_image_one.shape[0]
        if xend>huge_image_one.shape[1]:
            x_d_end = img_d.shape[1]-(xend-huge_image_one.shape[1])+1; xend = huge_image_one.shape[1]
        huge_image_one[int(ystart):int(yend),int(xstart):int(xend)] = img_d[int(y_d_start):int(y_d_end),int(x_d_start):int(x_d_end)]
    
    tr_mat_full = np.zeros((3,3)); tr_mat_full[:2,:2] = tr_mat
    #here idea that I add shift values, such that rotation will be applied around center of the image
    Cx = huge_image_one.shape[0]//2; Cy = huge_image_one.shape[1]//2
    a = tr_mat_full[0,0]; b = tr_mat_full[0,1]
    #this formula is taken from https://docs.opencv.org/4.x/da/d6e/tutorial_py_geometric_transformations.html "Rotation" paragraph
    tr_mat_full[0,2] = ((1-a)*Cx - b*Cy)
    tr_mat_full[1,2] = (b*Cx + (1-a)*Cy)
    img_reg = cv.warpAffine(huge_image_one, tr_mat_full[:2], (huge_image_one.shape[0], huge_image_one.shape[1]))
    img_reg[img_reg==0] = 225 #background color to fill the black frame around the image
    return img_reg



def main(ConfFilePath):
    path_adata, downscale, out_file_path, final_image_size, path_table_img_paths = ReadConfFile(ConfFilePath)
    
    print('opening adata')
    adata = anndata.read_h5ad(path_adata)
    tr_mat_all =  adata.uns['spatial_affine']
    list_of_section_names = list(tr_mat_all['alignment_metadata'].keys())
    
    table_img_paths = pd.read_csv(path_table_img_paths)
    
    final_image = memmap(out_file_path, shape = (len(list_of_section_names),final_image_size, final_image_size, 3), dtype = 'uint8', photometric = 'rgb', metadata={'axes':'ZCYX'}, bigtiff = True, imagej = True)
    
    i = 0
    for section_name in list_of_section_names:
        print(section_name)
        xc, yc = find_center_spot_pos(adata, section_name)
        img, tr_mat = take_img_and_tr_mat(table_img_paths, tr_mat_all, section_name)
        if img.shape[0] ==3:
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 0, 1)
        print(img.shape)
        img_reg = register_one_image(img, final_image_size, tr_mat, xc, yc, downscale)
        final_image[i] = img_reg
        final_image.flush()
        del img_reg, img
        gc.collect()
        i+=1
    
if __name__ == "__main__":
    fire.Fire(main) 
