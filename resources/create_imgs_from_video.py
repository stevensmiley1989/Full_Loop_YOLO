import os
import shutil
import time
from resources.create_img_list import create_img_list
def create_imgs_from_video(path_movie=None,fps='1/2'):
    '''path_change should be the MOV or mp4 path '''
    time_i=str(time.time())
    time_i=time_i.split('.')[0]
    if str(type(fps)).find('str')==-1:
        print('This is not a string for the fps! {}'.format(str(type(fps))))
    elif path_movie.lower().find('.mp4')!=-1 or path_movie.lower().find('.mov')!=-1:
        return_dir=os.getcwd()
        
        basepath=os.path.dirname(path_movie)
        os.chdir(basepath)
        print(basepath)
        movie_i_name=path_movie.split('/')[-1].split('.')[0]
        folders_in_basepath=os.listdir(basepath)
        folders_in_basepath=[w for w in folders_in_basepath if os.path.isdir(os.path.join(basepath,w))]
        JPEGImages_path=os.path.join(basepath,'JPEGImages')
        Annotations_path=os.path.join(basepath,'Annotations')
        if 'Annotations' not in folders_in_basepath:
            os.makedirs(Annotations_path)
        else:
            #os.system('mv {} {}'.format(Annotations_path,Annotations_path+'_backup_{}'.format(time_i)))
            Annotations_path=Annotations_path+'_'+time_i
            os.makedirs(Annotations_path)
        if 'JPEGImages' not in folders_in_basepath:
            os.makedirs(JPEGImages_path)  
        else:
            #os.system('mv {} {}'.format(JPEGImages_path,JPEGImages_path+'_backup_{}'.format(time_i)))
            JPEGImages_path=JPEGImages_path+'_'+time_i
            os.makedirs(JPEGImages_path)


        os.system('ffmpeg -i {} -qscale:v 2 -vf fps={} {}/{}_fps{}_%08d.jpg'.format(path_movie,fps,JPEGImages_path,movie_i_name,fps.replace('/','d').replace('.','p')))
        os.chdir(return_dir)
        print('creating img list')
        create_img_list(JPEGImages_path)
        print('finished creating img list')
    else:
        print('This is not a valid movie file.  Needs to be .mp4 or .MOV.  \n Provided: {}'.format(path_movie))

if __name__=='__main__':
    create_imgs_from_video('/media/steven/Elements/Drone_Videos/20220526_ADKDJI_P4V2_0027/DJI_0027.MOV')

