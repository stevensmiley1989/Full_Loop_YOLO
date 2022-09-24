import os
import shutil
import time
import cv2
import numpy as np
import shutil
from tqdm import tqdm
import collections

def create_video_from_imgs(path_JPEGImages='None',fps='DEFAULT',delete_previous_DESIRED=True,EXTOUT='.mp4',EXT_UNIQUE='*.jpg',SPLIT_UNIQUE='frame'):
    
    '''
    This script uses FFMPEG to convert a directory of images into a video
    cmd_i="ffmpeg -framerate {} -pattern_type  glob -i '{}' -c:v libx264 -pix_fmt yuv420p {}".format(fps,UNIQUE_MOVIE_NAME+EXT_UNIQUE,movie_i_name)

    -------------------------------------------------------------------------------------------------------------------------------------------------
    path_JPEGImages = path to your list of JPEGImages generated from create_imgs_from_video.py

    fps="DEFAULT" or "1/2" or "30" etc., if "DEFAULT" then it will try to parse the name of the parent directory for the fps specified.  This works well with create_imgs_from_video.py

    delete_previous_DESIRED= removes previous output.mp4 video if it exists when set to True, if set to False then it backs it up with the timestamp.

    EXTOUT= the output extension, i.e. .mp4 for an mp4 file to output.

    EXT_UNIQUE= the extension of the images in the directory to use FFMPEG on for stitching together a video with.  Use a * before the extension to let FFMPEG find the pattern.

    SPLIT_UNIQUE= the unique identifier of the frames in the path_JPEGImages directory.  When using create_imgs_from_video.py, this results in "frame" as a unique identifier.


    
    '''
    EXT="."+EXT_UNIQUE.split('.')[-1]
    if os.path.exists(path_JPEGImages):
        print('\npath_JPEGImages = {} \n'.format(path_JPEGImages))
    else:
        return print(f'This directory does not exist: {path_JPEGImages}\n')
    if str(type(fps)).find('str')==-1:
        return print('This is not a string for the fps! {}'.format(str(type(fps))))   
    elif fps=='DEFAULT':
        try:
            float(eval(os.path.basename(os.path.dirname(path_JPEGImages)).split('_')[-1].replace('d','/')))
            fps=os.path.basename(os.path.dirname(path_JPEGImages)).split('_')[-1].replace('d','/')
            print(f'using path_JPEGImages specified fps of {fps}')
            print('\nfps={}\n'.format(fps))
        except:
            print('defaulting to 30 fps.\n')
            fps='30'
            print('\nfps={}\n'.format(fps))
    else:
        print('\nfps={}\n'.format(fps))


    JPEGImages=os.listdir(path_JPEGImages)
    JPEGImages=[w for w in JPEGImages if w.find(EXT)!=-1]
    if len(JPEGImages)==0:
        return print('No images to create .mp4 from.')
    time_i=str(time.time())
    time_i=time_i.split('.')[0]
    
    UNIQUE_IMG_DIC={}
    for img_i in JPEGImages:
        UNIQUE_IMG_DIC[os.path.basename(img_i.split(SPLIT_UNIQUE)[0])]=img_i
    if len(UNIQUE_IMG_DIC)>1:
        return print(f"FOUND too many image prefixes in this directory {path_JPEGImages}\n This is what I found {UNIQUE_IMG_DIC.keys()}\n")
    UNIQUE_MOVIE_NAME=list(UNIQUE_IMG_DIC.keys())
    UNIQUE_MOVIE_NAME=UNIQUE_MOVIE_NAME[0]

    return_dir=os.getcwd()

    basepath=path_JPEGImages
    os.chdir(basepath)
    print(basepath)

    movie_i_name=UNIQUE_MOVIE_NAME+'_FPS_{}'.format(fps.replace("/","d"))+EXTOUT
    movie_i_name=os.path.join(basepath,movie_i_name)
    if os.path.exists(movie_i_name) and delete_previous_DESIRED:
        print(f'Removing previous {movie_i_name}\n')
        os.remove(movie_i_name)
    elif os.path.exists(movie_i_name):
        print(f'Backing-up previous {movie_i_name}\n')
        shutil.move(movie_i_name,movie_i_name.replace(EXTOUT,f'_BACKUP_{time_i}_'+EXTOUT))





    cmd_i="ffmpeg -framerate {} -pattern_type  glob -i '{}' -c:v libx264 -pix_fmt yuv420p {}".format(fps,UNIQUE_MOVIE_NAME+EXT_UNIQUE,movie_i_name)
    print(cmd_i)
    os.system(cmd_i)
    print(cmd_i)

    os.chdir(return_dir)

if __name__=='__main__':
    create_video_from_imgs('/media/steven/Elements/youtube_videos/hemtt_tank2/FPS_ACTUAL_30_FILTERED/JPEGImages')

