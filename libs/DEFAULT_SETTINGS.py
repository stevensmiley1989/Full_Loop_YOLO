import os
WIDTH_NUM=640
HEIGHT_NUM=640
TRAIN_SPLIT=70 #70/30 train/val split only need to say 70
increment=10000 #for writing out pandas pickle files RAM efficient
num_div=0
num_classes=12
path_prefix_elements=r''
path_prefix_mount_mac=r''
path_prefix_volumes_one=r''
if os.path.exists(path_prefix_elements):
    path_prefix=path_prefix_elements
elif os.path.exists(path_prefix_mount_mac):
    path_prefix=path_prefix_mount_mac
else:
    path_prefix=os.getcwd()
print('path_prefix',path_prefix)
PREFIX='tiny_yolo'+'-'+path_prefix.split('/')[-2].strip().replace(' ','_')
darknet_path=r"/home/steven/darknet"
base_path_OG=r"{}/Yolo_Models".format(path_prefix)
if os.path.exists(base_path_OG)==False:
    os.makedirs(base_path_OG)
MODEL_PATHS=os.path.join(base_path_OG,'MODEL_PATHS')
path_JPEGImages=r'{}/dataset/sample_rc_car/JPEGImages'.format(path_prefix)
path_Annotations=r'{}/dataset/sample_rc_car/Annotations'.format(path_prefix) #default Annotations directory
path_Yolo=path_JPEGImages.replace('JPEGImages','YOLO_OBJS')
if os.path.exists(path_Yolo)==False:
    os.makedirs(path_Yolo)
mp4_video_path=r'{}/dataset/sample_rc_car/DJI_0015.mp4'.format(path_prefix) 
root_background_img=r'misc/gradient_green.jpg' #path to background image for app
DEFAULT_ENCODING = 'utf-8'
XML_EXT = '.xml'
JPG_EXT = '.jpg'
COLOR='red'
root_bg='#000000'#'black'
root_fg='#b7f731'#'lime'
canvas_columnspan=50
canvas_rowspan=50
