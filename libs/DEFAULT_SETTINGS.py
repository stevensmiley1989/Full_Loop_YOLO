import os
WIDTH_NUM=640
HEIGHT_NUM=640
TRAIN_SPLIT=70 #70/30 train/val split only need to say 70
increment=10000000 #for writing out pandas pickle files RAM efficient
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
PREFIX='tiny_yolo'+'-'+os.path.basename(os.path.dirname(path_prefix)).strip().replace(' ','_')
darknet_path=r"/home/steven/darknet"
base_path_OG=r"{}".format(os.path.join(path_prefix,"Yolo_Models"))
if os.path.exists(base_path_OG)==False:
    os.makedirs(base_path_OG)
MODEL_PATHS=os.path.join(base_path_OG,'MODEL_PATHS')
path_JPEGImages=r'{}'.format(os.path.join(path_prefix,os.path.join("dataset",os.path.join("sample_rc_car","JPEGImages"))))
path_Annotations=r'{}'.format(os.path.join(path_prefix,os.path.join("dataset",os.path.join("sample_rc_car","Annotations"))))
path_Yolo=path_JPEGImages.replace('JPEGImages','YOLO_OBJS')
if os.path.exists(path_Yolo)==False:
    os.makedirs(path_Yolo)
mp4_video_path=r'{}'.format(os.path.join(path_prefix,os.path.join("dataset",os.path.join("sample_rc_car","DJI_0015.mp4"))))
root_background_img=r'{}'.format(os.path.join('misc','gradient_green.jpg')) #path to background image for app
DEFAULT_ENCODING = 'utf-8'
XML_EXT = '.xml'
JPG_EXT = '.jpg'
COLOR='red'
root_bg='#000000'#'black'
root_fg='#b7f731'#'lime'
canvas_columnspan=50
canvas_rowspan=50
