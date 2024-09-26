import numpy as np
import fire
import tifffile
from tifffile import TiffFile, imwrite, memmap
import gc
import pandas as pd
from pystackreg import StackReg
from skimage.transform import AffineTransform, warp
import json
import os
import yaml
import sys


def read_conf_file(ConfFilePath):
    with open(ConfFilePath, 'r') as file:
        data = yaml.safe_load(file)
    Path = data['Path']
    Downscale = data['Downscale']
    OutFolder = data['Output_folder']
    Name = data['Sample_name']
    BackgroundInt = data['Background_Intensity']
    Regularization = data['Regularization']
    RegularizationStrength = data['Regularization_strength']
    return Path, Downscale, OutFolder, Name, Regularization, RegularizationStrength, BackgroundInt

def search_the_largest_image_size(table_paths, col_name, N_sections):
    sx_list = []; sy_list = []
    for i in range(N_sections):
        path_img = table_paths[col_name][i]
        with TiffFile(path_img) as tif1:
            S = tif1.series[0].shape
            if S[0]<=3:
                sx_list.append(S[2]); sy_list.append(S[1])
            elif S[2]<=3:
                sx_list.append(S[1]); sy_list.append(S[0])          
    return np.max(np.array(sx_list)), np.max(np.array(sy_list))

def mode_csv_prepare(csvPath, Downscale, OutFolder, Name, dtype = 'uint8'):
    # in this mode we assume the csv table with (at least) one column "image_path". If no column with such name we take the first column
    table_paths = pd.read_csv(csvPath)
    if 'image_path' not in table_paths.columns:
        col_name = table_paths.columns[0]
    else:
        col_name = 'image_path'
    N_sections = table_paths.shape[0] #take all lines from csv
    sx, sy = search_the_largest_image_size(table_paths, col_name, N_sections)# define size of the final image by choosing the largest sizes in each dimension
    sx = int(np.ceil(sx/Downscale)); sy = int(np.ceil(sy/Downscale))
    ImageName = Name + '.tif'; OutImgPath = os.path.join(OutFolder, ImageName)
    memmap_image_d = memmap(OutImgPath, shape = (N_sections, sy, sx, 3), dtype = dtype, photometric = 'rgb', metadata={'axes':'ZCYX'}, bigtiff = True, imagej = True)
    TrMatDict = {}; TrMatName = Name + '.json'; TrMatPath = os.path.join(OutFolder, TrMatName)
    return N_sections, col_name, sy, sx, memmap_image_d, TrMatDict, TrMatPath, table_paths
    
    
    
def mode_img_prepare(imgPath, Downscale, OutFolder, Name, dtype = 'uint8'):
    #open first image to get metadata
    with TiffFile(imgPath) as tif:
        axs = tif.series[0].axes
        assert len(axs)==4, 'Image file should have 4 dimensions'
        z_ax = np.max([axs.find('Z'), axs.find('I')]) #one of the 2 will be present, other will give "-1"
        c_ax = np.max([axs.find('C'), axs.find('S')]) #one of the 2 will be present, other will give "-1"
        assert z_ax>=0, 'Can not find z plane axis from image file metadata'
        assert c_ax>=0, 'Can not find channel axis from image file metadata'
        axs_n = tif.series[0].shape
        N_sections = axs_n[z_ax]
        axs_n_crop = np.delete(axs_n, [z_ax, c_ax])
        sy = axs_n_crop[0]
        sx = axs_n_crop[1]
        
    sx = int(np.ceil(sx/Downscale)); sy = int(np.ceil(sy/Downscale))
    ImageName = Name + '.tif'; OutImgPath = os.path.join(OutFolder, ImageName)
    memmap_image_d = memmap(OutImgPath, shape = (N_sections, sy, sx, 3), dtype = 'uint8', photometric = 'rgb', metadata={'axes':'ZCYX'}, bigtiff = True, imagej = True)
    TrMatDict = {}; TrMatName = Name + '.json'; TrMatPath = os.path.join(OutFolder, TrMatName)
    return N_sections, z_ax, sy, sx, memmap_image_d, TrMatDict, TrMatPath

def get_img_from_table(table, col_name, num):
    path_img = table[col_name][num]
    with TiffFile(path_img) as tif1:
        img = tif1.asarray()
    return img

def get_img_plane(filepath, z_ax, num):
    with TiffFile(filepath) as tif:
        img = np.take(tif.series[0], num, z_ax).asarray()
    return img

def check_axis(img1, img2):
    assert len(img1.shape)==3, str('image ' + str(table_paths[column_name][i]) + ' should contain 3 dimensions (YXC or CYX)')
    assert len(img2.shape)==3, str('image ' + str(table_paths[column_name][i+1]) + ' should contain 3 dimensions (YXC or CYX)')
        
        #check if color channels are on the first place in the image shape - swap axes. We assume than we have not more than 3 channels - RGB
    if img1.shape[0]<=3:
        img1 = np.swapaxes(img1, 0, 2)
        img1 = np.swapaxes(img1, 0, 1)
    if img2.shape[0]<=3:
        img2 = np.swapaxes(img2, 0, 2)
        img2 = np.swapaxes(img2, 0, 1)
    return img1, img2

def resizing(img1, img2, sy, sx, background_int):
    img1_resized = np.ones((sy,sx,3), dtype = img1.dtype)*background_int
    img2_resized = np.ones((sy,sx,3), dtype = img2.dtype)*background_int
    img1_resized[:img1.shape[0],:img1.shape[1],:] = img1.copy()
    img2_resized[:img2.shape[0],:img2.shape[1],:] = img2.copy()
    del img1, img2
    gc.collect()
    return img1_resized, img2_resized

def register_3ch_img(img1_r, img2_r, Downscale, sy, sx, i, background_int):
    if i==0:
        img1 = img1_r[::Downscale, ::Downscale, :]
    else:
        img1 = img1_r
    img2 = img2_r[::Downscale, ::Downscale, :]
    print('resizing and grayscaling')
    img1, img2 = resizing(img1, img2, sy, sx, background_int)
    
    img1_grey = img1[:,:,1]; img2_grey = img2[:,:,1]
    print('finding transformation matrix')
    sr = StackReg(StackReg.AFFINE)
    tmat= sr.register(img1_grey, img2_grey)
    #tmat[0,2]*=downscale; tmat[1,2]*=downscale #those values are x and y shifts
    del img1_grey, img2_grey
    img2_reg = np.zeros_like(img2)
    gc.collect()
    print('registering')
    for i in range(img2.shape[2]):
        #print(i)
        AT = AffineTransform(tmat)
        img2_reg[:,:,i] = warp(img2[:,:,i], AT, output_shape=img2_reg[:,:,i].shape, preserve_range=True).astype(
            img2.dtype
        )
    
    img2_reg[img2_reg==0] = background_int
    img1[img1==0] = background_int
    del img2
    return img1, img2_reg, tmat




def sequential_registration(N_sections, col_name, z_ax, sy, sx, memmap_image_d, TrMatDict, TrMatPath, TablePaths, ImgPath, BackgroundInt, Downscale, mode):

    print('Sequential registration of planes')
    for i in range(N_sections-1):
        print(str(i) + '/' + str(N_sections-1))
        print('Reading images')
        if i==0:
            if mode=='csv':
                img1 = get_img_from_table(TablePaths, col_name, i)
                img2 = get_img_from_table(TablePaths, col_name, i+1)
            if mode=='img':
                img1 = get_img_plane(ImgPath, z_ax, i)
                img2 = get_img_plane(ImgPath, z_ax, i+1)
        else:
            img1 = img2_reg.copy()
            del img2_reg
            if mode=='csv':
                img2 = get_img_from_table(TablePaths, col_name, i+1)
            if mode=='img':
                img2 = get_img_plane(ImgPath, z_ax, i+1)
        
        print('Checking axes')
        img1, img2 = check_axis(img1, img2)
        
        img1, img2_reg, tmat = register_3ch_img(img1, img2, Downscale, sy, sx, i, BackgroundInt)
        TrMatDict[str(i)] = tmat.tolist()
        
        #saving plane into the multiplane image file
        if i==0:
            memmap_image_d[0] = img1
            memmap_image_d[1] = img2_reg
            memmap_image_d.flush()
        else:
            memmap_image_d[i+1] = img2_reg
            memmap_image_d.flush()
        
        with open(TrMatPath, "w") as outfile: 
            json.dump(TrMatDict, outfile)

            
def main(ConfFilePath):
    
    Path, Downscale, OutFolder, Name, Regularization, RegularizationStrength, BackgroundInt = read_conf_file(ConfFilePath)
    #there are two options - if the path leads to the csv with table with paths of single plane images or it point to the single mutliplane image
    if Path[-3:] == 'csv':
        N_sections, col_name, sy, sx, memmap_image_d, TrMatDict, TrMatPath, TablePaths = mode_csv_prepare(Path,  Downscale, OutFolder, Name)
        mode = 'csv'; z_ax = -1
    else:
        N_sections, z_ax, sy, sx, memmap_image_d, TrMatDict, TrMatPath = mode_img_prepare(Path,  Downscale, OutFolder, Name)
        mode = 'img'; col_name = 'None'; TablePaths = 'None'
    
    
    sequential_registration(N_sections, col_name, z_ax, sy, sx, memmap_image_d, TrMatDict, TrMatPath, TablePaths, Path, BackgroundInt, Downscale, mode)


if __name__ == "__main__":
    fire.Fire(main) 
