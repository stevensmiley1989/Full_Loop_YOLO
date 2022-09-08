import os
import shutil
import time
import cv2
import numpy as np
import shutil
from tqdm import tqdm
import collections
from resources.create_img_list import create_img_list
def create_imgs_from_video(path_movie=None,fps='1/2',delete_previous_ACTUAL=True,delete_previous_DESIRED=True):
    '''path_change should be the MOV or mp4 path '''
    print('\npath_movie = {} \n'.format(path_movie))
    print('\nfps={}\n'.format(fps))
    time_i=str(time.time())
    time_i=time_i.split('.')[0]
    if str(type(fps)).find('str')==-1:
        print('This is not a string for the fps! {}'.format(str(type(fps))))
    elif path_movie.lower().find('.mp4')!=-1 or path_movie.lower().find('.mov')!=-1:
        return_dir=os.getcwd()

        basepath=os.path.dirname(path_movie)
        os.chdir(basepath)
        print(basepath)
        video_files=os.listdir(basepath)
        video_files=[w for w in video_files if w.lower().find('.mp4')!=-1 or w.lower().find('.mov')!=-1]
        if len(video_files)>0 and os.path.basename(basepath)!=os.path.basename(path_movie.split('.')[0]):
            print('MULTIPLE VIDEO FILES, moving to separate directories')
            for video_file_i in tqdm(video_files):
                video_file_i_path=os.path.join(basepath,video_file_i)
                video_file_i_dir_path_new=video_file_i_path.split('.')[0]
                try:
                    os.makedirs(video_file_i_dir_path_new)
                except:
                    pass
                shutil.move(video_file_i_path,video_file_i_dir_path_new)
            path_movie=os.path.join(os.path.join(basepath,os.path.basename(path_movie).split('.')[0]),os.path.basename(path_movie))
            basepath=os.path.dirname(path_movie)
            os.chdir(basepath)
            print(basepath)
        video=cv2.VideoCapture(path_movie)
        actual_video_fps=str(int(np.ceil(video.get(cv2.CAP_PROP_FPS))))
        actual_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        print("Actual video's FPS is = {}".format(actual_video_fps))
        print('Actual frame count = {}'.format(actual_frames))
        movie_i_name=os.path.basename(path_movie).split('.')[0]
        basepath_desired=os.path.join(basepath,'FPS_DESIRED_{}'.format(fps.replace("/","d")))
        basepath=os.path.join(basepath,'FPS_ACTUAL_{}'.format(actual_video_fps,fps.replace("/","d")))
        if os.path.exists(basepath)==False:
            os.makedirs(basepath)
        if os.path.exists(basepath_desired)==False:
            os.makedirs(basepath_desired)
        folders_in_basepath=os.listdir(basepath)
        folders_in_basepath=[w for w in folders_in_basepath if os.path.isdir(os.path.join(basepath,w))]
        folders_in_basepath_desired=os.listdir(basepath_desired)
        folders_in_basepath_desired=[w for w in folders_in_basepath_desired if os.path.isdir(os.path.join(basepath_desired,w))]
        JPEGImages_path=os.path.join(basepath,'JPEGImages')
        Annotations_path=os.path.join(basepath,'Annotations')
        if 'Annotations' not in folders_in_basepath:
            os.makedirs(Annotations_path)
        else:
            #os.system('mv {} {}'.format(Annotations_path,Annotations_path+'_backup_{}'.format(time_i)))
            len_annos=len(os.listdir(Annotations_path))
            if len_annos>0:
                delete_previous_ACTUAL=False
                Annotations_path=Annotations_path+'_'+time_i
            else:
                os.system(f'rm -rf {Annotations_path}')
            os.makedirs(Annotations_path)
        if 'JPEGImages' not in folders_in_basepath:
            os.makedirs(JPEGImages_path)  
        else:
            #os.system('mv {} {}'.format(JPEGImages_path,JPEGImages_path+'_backup_{}'.format(time_i)))
            if delete_previous_ACTUAL:
                os.system(f'rm -rf {JPEGImages_path}')
            else:
                JPEGImages_path=JPEGImages_path+'_'+time_i
            os.makedirs(JPEGImages_path)
        JPEGImages_path_desired=os.path.join(basepath_desired,'JPEGImages')
        Annotations_path_desired=os.path.join(basepath_desired,'Annotations')
        if 'Annotations' not in folders_in_basepath_desired:
            os.makedirs(Annotations_path_desired)
        else:
            #os.system('mv {} {}'.format(Annotations_path,Annotations_path+'_backup_{}'.format(time_i)))
            len_annos_desired=len(os.listdir(Annotations_path_desired))
            if len_annos_desired>0:
                delete_previous_DESIRED=False
                Annotations_path_desired=Annotations_path_desired+'_'+time_i
            else:
                os.system(f'rm -rf {Annotations_path_desired}')
            os.makedirs(Annotations_path_desired)
        if 'JPEGImages' not in folders_in_basepath_desired:
            os.makedirs(JPEGImages_path_desired)  
        else:
            #os.system('mv {} {}'.format(JPEGImages_path,JPEGImages_path+'_backup_{}'.format(time_i)))
            if delete_previous_DESIRED:
                os.system(f'rm -rf {JPEGImages_path_desired}')
            else:
                JPEGImages_path_desired=JPEGImages_path_desired+'_'+time_i
            os.makedirs(JPEGImages_path_desired)



        #os.system('ffmpeg -i {} -qscale:v 2 -vf fps={} {}/{}_fps{}_%08d.jpg'.format(path_movie,fps,JPEGImages_path,movie_i_name,fps.replace('/','d').replace('.','p')))
        os.system('ffmpeg -i {} -qscale:v 2 -vf fps={} {}/{}_fps{}_frame%08d.jpg'.format(path_movie,actual_video_fps,JPEGImages_path,movie_i_name,actual_video_fps.replace('.','p')))
        actual_frames_found=os.listdir(JPEGImages_path)
        actual_frames_found=[os.path.join(JPEGImages_path,w) for w in actual_frames_found if w.find('.jpg')!=-1]
        actual_frames_dic={}
        for frame in tqdm(actual_frames_found):
            frame_count_i=int(frame.split('frame')[1].split('.jpg')[0])
            actual_frames_dic[frame_count_i]=frame
        od = collections.OrderedDict(sorted(actual_frames_dic.items()))
        counter=0.0
        desired_frames=[]
        last_frame=max(od.keys())
        frames_every=float(actual_video_fps)/eval(fps)
        for frame, frame_path in tqdm(od.items()):
            counter+=1
            if counter>frames_every:
                counter=0
                desired_frames.append(frame_path)
                shutil.copy(frame_path,JPEGImages_path_desired) #copy instead of move
            elif frame==1 or frame==last_frame:
                desired_frames.append(frame_path)
                shutil.copy(frame_path,JPEGImages_path_desired) #copy instead of move         
                
        os.chdir(return_dir)
        print('creating img list')
        create_img_list(JPEGImages_path)
        print('finished creating img list')
        print('creating img list for desired')
        create_img_list(JPEGImages_path_desired)
        print('finished creating img list for desired')


    else:
        print('This is not a valid movie file.  Needs to be .mp4 or .MOV.  \n Provided: {}'.format(path_movie))

if __name__=='__main__':
    create_imgs_from_video('/media/steven/Elements/Drone_Videos/20220526_ADKDJI_P4V2_0027/DJI_0027.MOV')

