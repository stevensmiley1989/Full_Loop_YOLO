global UNIQUE_DEVICE
import datetime
UNIQUE_DEVICE='jetson'
import time
from threading import Thread
from multiprocessing import Process, Queue, Pipe
import os
import traceback
import time
import cv2

time_found=time.time()
import threading as th
from threading import Thread
target_of_interest='car'
target_found=False
path_prefix_elements=r'/media/steven/Elements/'
path_prefix_mount_mac=r'/media/pi/Elements/'
path_prefix_volumes_one=os.getcwd()
if os.path.exists(path_prefix_elements):
    path_prefix=path_prefix_elements
elif os.path.exists(path_prefix_mount_mac):
    path_prefix=path_prefix_mount_mac
else:
    path_prefix=path_prefix_volumes_one
print('path_prefix',path_prefix)
time_start=str(time_found).split('.')[0]
path_save=os.path.join(path_prefix,'CustomDronePhantom')
if os.path.exists(path_save)==False:
    os.makedirs(path_save)
path_save=os.path.join(path_save,time_start)
if os.path.exists(path_save)==False:
    os.makedirs(path_save)
# import the necessary packages
import pandas as pd
import numpy as np
import argparse
import time
import os
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
from tqdm import tqdm
import codecs
try:
    from PyQt5.QtGui import QImage
except ImportError:
    from PyQt4.QtGui import QImage
DEFAULT_ENCODING = 'utf-8'
XML_EXT = '.xml'
ENCODE_METHOD = DEFAULT_ENCODING


def move_files(list_to_move,save_path_jpg_todo):
    for file in list_to_move:
        try:
            os.system('mv {} {}'.format(file,save_path_jpg_todo))
        except:
            pass
def run_cmd(cmd_i):
    os.system(cmd_i)
def ustr(x):
    '''py2/py3 unicode helper'''

    if sys.version_info < (3, 0, 0):
        from PyQt4.QtCore import QString
        if type(x) == str:
            return x.decode(DEFAULT_ENCODING)
        if type(x) == QString:
            #https://blog.csdn.net/friendan/article/details/51088476
            #https://blog.csdn.net/xxm524/article/details/74937308
            return unicode(x.toUtf8(), DEFAULT_ENCODING, 'ignore')
        return x
    else:
        return x

class PascalVocWriter:
    def __init__(self, foldername, filename, imgSize,databaseSrc='Unknown', localImgPath=None):
        self.foldername = foldername
        self.filename = filename
        self.databaseSrc = databaseSrc
        self.imgSize = imgSize
        self.boxlist = []
        self.localImgPath = localImgPath
        self.verified = False

    def prettify(self, elem):
        """
            Return a pretty-printed XML string for the Element.
        """
        rough_string = ElementTree.tostring(elem, 'utf8')
        root = etree.fromstring(rough_string)
        return etree.tostring(root, pretty_print=True, encoding=ENCODE_METHOD).replace("  ".encode(), "\t".encode())
        # minidom does not support UTF-8
        '''reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="\t", encoding=ENCODE_METHOD)'''

    def genXML(self):
        """
            Return XML root
        """
        # Check conditions
        if self.filename is None or \
                self.foldername is None or \
                self.imgSize is None:
            return None

        top = Element('annotation')
        if self.verified:
            top.set('verified', 'yes')

        folder = SubElement(top, 'folder')
        folder.text = self.foldername

        filename = SubElement(top, 'filename')
        filename.text = self.filename

        if self.localImgPath is not None:
            localImgPath = SubElement(top, 'path')
            localImgPath.text = self.localImgPath

        source = SubElement(top, 'source')
        database = SubElement(source, 'database')
        database.text = self.databaseSrc

        size_part = SubElement(top, 'size')
        width = SubElement(size_part, 'width')
        height = SubElement(size_part, 'height')
        depth = SubElement(size_part, 'depth')
        width.text = str(self.imgSize[1])
        height.text = str(self.imgSize[0])
        if len(self.imgSize) == 3:
            depth.text = str(self.imgSize[2])
        else:
            depth.text = '1'

        segmented = SubElement(top, 'segmented')
        segmented.text = '0'
        return top

    def addBndBox(self, xmin, ymin, xmax, ymax, name, difficult,score):
        bndbox = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax}
        bndbox['name'] = name
        bndbox['difficult'] = difficult
        bndbox['score']=score
        self.boxlist.append(bndbox)

    def appendObjects(self, top):
        for each_object in self.boxlist:
            object_item = SubElement(top, 'object')
            name = SubElement(object_item, 'name')
            name.text = ustr(each_object['name'])
            pose = SubElement(object_item, 'pose')
            pose.text = "Unspecified"
            truncated = SubElement(object_item, 'truncated')
            if int(float(each_object['ymax'])) == int(float(self.imgSize[0])) or (int(float(each_object['ymin']))== 1):
                truncated.text = "1" # max == height or min
            elif (int(float(each_object['xmax']))==int(float(self.imgSize[1]))) or (int(float(each_object['xmin']))== 1):
                truncated.text = "1" # max == width or min
            else:
                truncated.text = "0"
            difficult = SubElement(object_item, 'difficult')
            difficult.text = str( bool(each_object['difficult']) & 1 )
            score = SubElement(object_item, 'score')
            score.text = str(float(each_object['score']))
            bndbox = SubElement(object_item, 'bndbox')
            xmin = SubElement(bndbox, 'xmin')
            xmin.text = str(each_object['xmin'])
            ymin = SubElement(bndbox, 'ymin')
            ymin.text = str(each_object['ymin'])
            xmax = SubElement(bndbox, 'xmax')
            xmax.text = str(each_object['xmax'])
            ymax = SubElement(bndbox, 'ymax')
            ymax.text = str(each_object['ymax'])

    def save(self, targetFile=None):
        root = self.genXML()
        self.appendObjects(root)
        out_file = None
        if targetFile is None:
            out_file = codecs.open(
                self.filename + XML_EXT, 'w', encoding=ENCODE_METHOD)
        else:
            out_file = codecs.open(targetFile, 'w', encoding=ENCODE_METHOD)

        prettifyResult = self.prettify(root)
        out_file.write(prettifyResult.decode('utf8'))
        out_file.close()
def savePascalVocFormat(filename, bndboxes, labels, imagePath, imageData,scores,
                            lineColor=None, fillColor=None, databaseSrc=None):
        imgFolderPath = os.path.dirname(imagePath)
        imgFolderName = os.path.split(imgFolderPath)[-1]
        imgFileName = os.path.basename(imagePath)
        #imgFileNameWithoutExt = os.path.splitext(imgFileName)[0]
        # Read from file path because self.imageData might be empty if saving to
        # Pascal format
        if isinstance(imageData, QImage):
            image = imageData
        else:
            image = QImage()
            image.load(imagePath)
        imageShape = [image.height(), image.width(),
                      1 if image.isGrayscale() else 3]
        writer = PascalVocWriter(imgFolderName, imgFileName,
                                 imageShape, localImgPath=imagePath)
        writer.verified = True

        for bndbox,label,score in zip(bndboxes,labels,scores):
            difficult = 'NA'
            #bndbox = LabelFile.convertPoints2BndBox(points)
            writer.addBndBox(bndbox[0], bndbox[1], bndbox[2], bndbox[3], label, difficult, bndbox[4])

        writer.save(targetFile=filename)
        return


import subprocess as sp
class YOUTUBE_RTMP:
    def __init__(self,YOUTUBE_STREAM_KEY):
        self.YOUTUBE_STREAM_KEY=YOUTUBE_STREAM_KEY
        self.initiate=True
    def Preprocess(self,HEIGHT_i,WIDTH_i,VBR_i):
        self.HEIGHT_i=HEIGHT_i
        self.WIDTH_i=WIDTH_i
        self.VBR_i=VBR_i     
        self.initiate=False
        self.startFFmpeg_Process()
    def startFFmpeg_Process(self):
        self.cmd=["ffmpeg","-y","-f","lavfi","-i","anullsrc","-f","rawvideo","-vcodec","rawvideo", "-s","{}x{}".format(self.HEIGHT_i,self.WIDTH_i),
        "-pix_fmt","bgr24","-i","-","-acodec","aac","-ar","44100","-b:a","712000","-vcodec","libx264","-preset","medium","-b:v","{}".format(self.VBR_i),"-bufsize","0","-pix_fmt",
        "yuv420p","-f","flv","-crf","18","rtmp://a.rtmp.youtube.com/live2/{}".format(self.YOUTUBE_STREAM_KEY)]
        self.process = sp.Popen(self.cmd, stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.STDOUT)
    def write(self,frame,VBR_i):
        # initiate FFmpeg process on first run
        if self.initiate:
            # start pre-processing and initiate process
            self.Preprocess(frame.shape[1],frame.shape[0],VBR_i)
            # Check status of the process
            assert self.process is not None
        # write the frame
        try:
            self.process.stdin.write(frame.tostring())
        except (OSError, IOError):
            # log something is wrong!
            print(
                "BrokenPipeError caught"
            )
            raise ValueError  # for testing purpose only
    def close(self):
        if self.process.stdin:
            self.process.stdin.close()  # close `stdin` output
        self.process.wait()  # wait if still process is still processing some information
        self.process = None
  

def YOUTUBE_STREAM_RESOLUTION(res='720p'):
    #returns the height,width, and video bit rate
    if res=='720p':
        return 720,1280,'4000k'
    elif res=='1080p':
        return 1080,1920,'6000k'
    elif res=='480p':
        return 480,854,'2000k'
    elif res=='360p':
        return 640,360,'1000k'
    else:
        print('DID NOT RECOGNIZE res=={}\n so using res==720p'.format(res))
        return 720,1280,'4000k'
def send_imgs(sender,im0):
    try:
        sender.send_image("YOLO OUPUT", im0)
    except:
        pass
def run_cmd(cmd_i):
    os.system(cmd_i)
weightsPath1="{}Images/Drone_Images/Yolo/backup_models/custom-yolov4-tiny-detector_VisDrone_best.weights".format(path_prefix)
labelsPath1="{}Images/Drone_Images/Yolo/obj.names".format(path_prefix)
configPath1="{}Jetson_Stuff/YOLOv4/darknet/cfg/custom-yolov4-tiny-detector_VisDrone_test.cfg".format(path_prefix)
imW=str(640)
imH=str(640)
video=str(0)
save='No'
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", default='None',
    help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applying non-maxima suppression")
ap.add_argument("--weightsPath",type=str,default=weightsPath1,help='path to yolo weights')
ap.add_argument("--labelsPath",type=str,default=labelsPath1,help='path to obj.names')
ap.add_argument("--configPath",type=str,default=configPath1,help='path to yolo cfg for test')
ap.add_argument("--imW",type=str,default=imW,help='Width of input image')
ap.add_argument("--imH",type=str,default=imH,help='Height of input image')
ap.add_argument("--video",type=str,default=video,help='0 for webcam etc')
ap.add_argument("--save",type=str,default=save,help='save annotations')
ap.add_argument("--YOUTUBE_RTMP",type=str,default="xxxx-xxxx-xxxx-xxxx-xxxx",help="The YOUTUBE STREAM RTMP Key")
ap.add_argument("--YOUTUBE_STREAM_RES",type=str,default='720p',help="Youtube Stream Height")
ap.add_argument("--RTSP_PATH",type=str,default="xxxx-xxxx-xxxx-xxxx-xxxx",help="The RTSP Path")
ap.add_argument("--RTSP_SERVER_PATH",type=str,default="/media/steven/Elements/Full_Loop_YOLO/resources/rtsp_server.py",help="The path to rtsp_server.py")
ap.add_argument("--fps",default=30,help="fps of incoming images for rtsp_server",type=int)
ap.add_argument("--width",default=None,help="width of incoming images for rtsp_server",type=int)
ap.add_argument("--height",default=None,help="height of incoming images for rtsp_server",type=int)
ap.add_argument("--port",default=8554,help="port for rtsp_server",type=int)
ap.add_argument("--stream_key",default="/video_stream",help="rtsp image stream uri for rtsp_server")
ap.add_argument("--send_image_to_cell",action='store_true',help='Should send text messages with chips?')
ap.add_argument("--send_image_to_cell_path",default="/media/steven/Elements/Full_Loop_YOLO/resources/send_image_to_cell.py",help="Send text message images of chips to cell")
ap.add_argument("--destinations",type=str,default='XXXYYYZZZZ@mms.att.net',help='phone numbers to send text message updates to')
ap.add_argument("--basepath_chips",type=str,default="/media/steven/Elements/chips",help="path for chips stored")
ap.add_argument("--sleep_time_chips",type=float,default=30,help="Seconds to sleep between sending chips")
ap.add_argument("--using_JETSON_NANO",action='store_true', help='If using Jetson NANO, issues with RET, so want to have this flag to handle it')
ap.add_argument("--use_socket_receive_imgs",action='store_true',help='if you want to receive images from a set path given by labelimg and send predictions to labelimg')
ap.add_argument("--noview",action='store_true',help='if you do not want to see output')

global args
global send_image_to_cell,send_image_to_cell_path,destinations,basepath_chips,sleep_time_chips
args = vars(ap.parse_args())
print(args)
image_i=args['image']
weightsPath=args['weightsPath']
labelsPath=args['labelsPath']
configPath=args['configPath']
video=args['video']
imW=args['imW']
imH=args['imH']
save=args['save']
send_image_to_cell=args['send_image_to_cell']
send_image_to_cell_path=args['send_image_to_cell_path']
destinations=args['destinations']
basepath_chips=args['basepath_chips']
sleep_time_chips=args['sleep_time_chips']
YOUTUBE_STREAM_KEY = args['YOUTUBE_RTMP']
YOUTUBE_STREAM_RES = args['YOUTUBE_STREAM_RES']
imW, imH = int(imW), int(imH)
video=video.strip()
if video=='0':
    video=0
elif video=='1':
    video=1
print('video is {}'.format(video))
if save=='No':
    save=False
else:
    save=True

if YOUTUBE_STREAM_KEY!='xxxx-xxxx-xxxx-xxxx-xxxx':
    RTMP=True
    writer=YOUTUBE_RTMP(YOUTUBE_STREAM_KEY)
else:
    RTMP=False
global running,RTSP
RTSP_PATH = args['RTSP_PATH']
RTSP_SERVER_PATH=args['RTSP_SERVER_PATH']
RTSP=False
running=False
if RTSP_PATH != 'xxxx-xxxx-xxxx-xxxx-xxxx' and os.path.exists(RTSP_SERVER_PATH):
    print(RTSP_SERVER_PATH)
   
    if args['width']:
        WIDTH=args['width']
    else:
        WIDTH=imW
    if args['height']:
        HEIGHT=args['height']
    else:
        HEIGHT=imH
    import sys
    from threading import Thread
    sys.path.append(os.path.dirname(RTSP_SERVER_PATH))
    #import rtsp_server as rs
    import imagezmq
    sender = imagezmq.ImageSender()
    RTSP=True
    
    
def create_output_paths(unique_device=UNIQUE_DEVICE,video_device=str(video)):
    global output_day_path,output_day_hour_sec_path,OUTPUT_FILE
    video_device=video_device.replace(':','-').replace('/','').replace('.','p')
    video_device=unique_device+"_"+video_device
    main_path=r"/home/steven/Elements/camera_detections/"
    second_path=r"/media/pi/Elements/camera_detections/"
    third_path=os.path.join(os.getcwd(),'camera_detections')
    if os.path.exists(main_path):
        #print('using main path',main_path)
        main_path=main_path
    elif os.path.exists(second_path):
        main_path=second_path
    else:
        main_path=third_path
        try:
            os.mkdir(main_path)
        except:
            pass
    
    output_day_path=os.path.join(main_path,str(datetime.datetime.now()).split(' ')[0])
    #if os.path.exists(output_day_path)==False:
    #    os.makedirs(output_day_path)
    output_day_path=os.path.join(output_day_path,video_device)
    output_day_hour_sec_path=os.path.join(output_day_path,str(datetime.datetime.now()).split(' ')[1].replace(':','-').split('.')[0])  
    output_day_hour_min_path=output_day_hour_sec_path[0:output_day_hour_sec_path.rfind('-')]+'_'+unique_device+'.avi'
          
    try:
        os.makedirs(output_day_path)
    except:
        #print("This output path already exists")
        pass
    OUTPUT_FILE=output_day_hour_min_path
    #print("OUTPUT_FILE",OUTPUT_FILE)
    return OUTPUT_FILE
OUTPUT_FILE=create_output_paths()

LABELS = open(labelsPath).read().strip().split("\n")
# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")
# derive the paths to the YOLO weights and model configuration
#weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
#configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])


# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
ln = net.getLayerNames()
from pprint import pprint
#pprint(ln)
yolo_layers=[]
for i in net.getUnconnectedOutLayers():
    #print(i)
    if str(i).find('[')==-1:
        if ln[i-1].find('yolo')!=-1:
            yolo_layers.append(ln[i-1])
    else:
        if ln[i[0]-1].find('yolo')!=-1:
            yolo_layers.append(ln[i[0]-1])
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# pprint(ln)
#pprint(yolo_layers)
if str(video).find('rtsp')!=-1:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]="rtsp_transport;udp" #edit sjs 8/4/2021


def load_imges_from_folder(folder):
    found_image=False
    while found_image==False:
        try:
            images = []
            filenames=os.listdir(folder)
            filenames=[w for w in filenames if w.find('.jpg')!=-1 or w.find('.png')!=-1]
            for filename in filenames:
                print('filename',filename)
                #img = cv2.imread(os.path.join(folder,filename))
                #if img is not None:
                if os.path.exists(os.path.join(folder,filename)):
                    image=cv2.imread(os.path.join(folder,filename))
                    return image
                else:
                    pass
        except:
            pass
    return images



class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,video,resolution=(300,300),framerate=40,output_file_path=OUTPUT_FILE): #edit 8/4/2021 sjs, includes 640x640 default with video parameter defined for rtsp arg.
        # Initialize the PiCamera and the camera image stream
        #self.stream = cv2.VideoCapture(video) #edit 8/4/2021 sjs, includes the cv2.CAP_FFMPEG for the os.env set earlier for better feed.
        print('video is set to {}'.format(video))
        if str(video)!='0' and str(video)!='1' and str(video).find('rtsp')==-1:
            print('using path 1')
            #self.stream = cv2.VideoCapture(video,cv2.CAP_FFMPEG) #ed
            self.stream=cv2.VideoCapture(video,cv2.CAP_DSHOW)
        elif str(video).find('rtsp')!=-1:
            print('using path 2')
            print('using rtsp cam {}'.format(str(video)))
            self.stream=cv2.VideoCapture(video)
        else:
            print('using path 3')
            self.stream=cv2.VideoCapture(video)
        #self.stream.set(cv2.CAP_PROP_BUFFERSIZE,3)
        if args['using_JETSON_NANO']==False and str(video).find('rtsp')==-1:
            ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) #if JETSON NANO, Comment out line
        if str(video).find('rtsp')==-1:
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE,3)
        if args['using_JETSON_NANO']==False and str(video).find('rtsp')==-1:
            ret = self.stream.set(3,resolution[0]) #if JETSON NANO, Comment out line
            ret = self.stream.set(4,resolution[1]) #if JETSON NANO, Comment out line

        self.resolution=resolution
        self.output_file_path=output_file_path
        self.output_file_path_temp=output_file_path.replace('.avi','_temp.avi')
        self.fourcc=cv2.VideoWriter_fourcc(*'XVID')
        self.videoOut=cv2.VideoWriter(self.output_file_path,self.fourcc,20.0,self.resolution)
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()


	# Variable to control when the camera is stopped
        self.stopped = False
    def output(self,frame_i):
        self.videoOut.write(frame_i)
    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True
ln=yolo_layers
text=":0"
keep_going=True
def key_capture_thread():
    global keep_going
    while True:
        my_answer=input("Enter 'q' to quit:\n")
        if my_answer.find('q')!=-1:
            keep_going=False
#videostream = VideoStream(video,resolution=(640,480),framerate=30).start()
aspect=640./480.
if imW==imH:
    imH=int(imW/aspect)
if args['image']=='None':
  videostream = VideoStream(video,resolution=(imW,imH),framerate=30).start()
def dataset_loader(dataset_Queue,dataset_path_Queue,msg_i_Queue):
    if args['use_socket_receive_imgs']:
            import socket
            import threading
            from multiprocessing import Queue
            xy=Queue() #bboxes
            ready=Queue()
            PORT_RX=8765
            HOST_RX=socket.gethostname()
            connected=False
            while connected==False:
                print('using Socket for PORT=={} and HOST=={}'.format(PORT_RX,HOST_RX))
                try:
                    yolov4_model=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    yolov4_model.connect((HOST_RX, PORT_RX)) #edit sjs
                    connected=True
                    msg_i_list="&" #get it started
                except:
                    print('Not accepting socket')
            LOOP=True
            dataset_path_i=yolov4_model.recv(1024)
            dataset_path_i=dataset_path_i.decode()
            image=dataset_path_i
    else:
        LOOP=False
        dataset_path_i=dataset_path_Queue.get()
    DO_THE_LOOP=True
    while DO_THE_LOOP:

        print("PUT IMAGE")
        image=load_imges_from_folder(image)
        
        dataset_Queue.put(image)
        msg_i_list=msg_i_Queue.get()
        #print(msg_i_list)
        if LOOP:
            yolov4_model.send(msg_i_list.encode())  
            dataset_path_i=yolov4_model.recv(1024)
            dataset_path_i=dataset_path_i.decode()
            image=dataset_path_i

        if LOOP==False:
            print('FINISHED LOOP')
            DO_THE_LOOP=False 


                                
dataset_Queue=Queue()
dataset_path_Queue=Queue()
msg_i_Queue=Queue()
if args['image']!='None':
    #dataset=load_images_from_folder(os.path.dirname(args['image']))
    dataset_loader_process=Process(target=dataset_loader,args=(dataset_Queue,dataset_path_Queue,msg_i_Queue,)).start()
    #dataset_path_Queue.put(os.path.dirname(args['image']))



window_name='YOLO camera'

def do_stuff(dataset_Queue,dataset_path_Queue,msg_i_Queue):
    global time_found,target_found,myrtmp_addr,running,args,RTSP
    global send_image_to_cell,send_image_to_cell_path,destinations,basepath_chips,sleep_time_chips
    global total_fps, total_fps_count


    total_fps=0
    total_fps_count=0
    th.Thread(target=key_capture_thread,args=(),name='key_capture_thread',daemon=True).start()
    time_last=time.time()
    #Sending images to cell phone via text/email
    #send_image_to_cell=opt.send_image_to_cell
    #send_image_to_cell_path=opt.send_image_to_cell_path
    #destinations=opt.destinations
    #basepath_chips=opt.basepath_chips
    send_allowed=True
    start_time=time.time()
    date_i=str(datetime.datetime.now()).replace(' ','_').replace('.','p').replace(':',"c").replace('-','_')
    if os.path.exists(basepath_chips)==False and send_image_to_cell:
        if os.path.exists(os.path.dirname(basepath_chips)):
            os.makedirs(basepath_chips)
            basepath_chips=os.path.join(basepath_chips,date_i)
            if os.path.exists(basepath_chips)==False:
                os.makedirs(basepath_chips)
        else:
            send_image_to_cell=False
            print('You have a bad path to save chips.  Not sending images to cell')
    elif send_image_to_cell:
        basepath_chips=os.path.join(basepath_chips,date_i)
        if os.path.exists(basepath_chips)==False:
            os.makedirs(basepath_chips)
    while keep_going:
        time_start=time.time()
        # load our input image and grab its spatial dimensions
        #image_path="/run/user/1000/gvfs/afc:host=7566645a7797611a80d631cdedfd746643ecb130,port=3/com.holystone.Ophelia-GO/Photo"
        save_path="{}/Predictions".format(path_save)
        save_path_jpg="{}/JPEGImages".format(path_save)
        save_path_anno="{}/Annotations".format(path_save)
        save_path_jpg_todo="{}/JPEGImages_todo".format(path_save)
        if os.path.exists(save_path)==False:
            try:
                os.makedirs(save_path)
            except:
                pass
        if os.path.exists(save_path_jpg)==False:
            try:
                os.makedirs(save_path_jpg)
            except:
                pass
        if os.path.exists(save_path_jpg_todo)==False:
            try:
                os.makedirs(save_path_jpg_todo)
            except:
                pass
        if os.path.exists(save_path_anno)==False:
            try:
                os.makedirs(save_path_anno)
            except:
                pass
        try:
               if args['image']=='None':            
                image = videostream.read()
               else:
                #image=cv2.imread(args['image'])
                image=dataset_Queue.get()
                
               (H, W) = image.shape[:2]
               msg_i_list="STAART&"

               #Check send chips
               time_now=time_start
               if time_now-start_time>sleep_time_chips:
                send_allowed=True
                start_time=time_now
               else:
                send_allowed=False
                img_list={}
                label_list={}


               blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (imH, imW),
               swapRB=True, crop=False)
               net.setInput(blob)
               start = time.time()
               layerOutputs = net.forward(ln)
               end = time.time()
               image_i=str(start).split('.')[0]+'.jpg'
               # show timing information on YOLO
               #print("[INFO] YOLO took {:.6f} seconds".format(end - start))
               
               # initialize our lists of detected bounding boxes, confidences, and
               # class IDs, respectively
               boxes = []
               confidences = []
               classIDs = []
               # loop over each of the layer outputs
               for output in layerOutputs:
                   # loop over each of the detections
                   for detection in output:
                       # extract the class ID and confidence (i.e., probability) of
                       # the current object detection
                       scores = detection[5:]
                       classID = np.argmax(scores)
                       confidence = scores[classID]
                       # filter out weak predictions by ensuring the detected
                       # probability is greater than the minimum probability
                       if confidence > args["confidence"]:
                           # scale the bounding box coordinates back relative to the
                           # size of the image, keeping in mind that YOLO actually
                           # returns the center (x, y)-coordinates of the bounding
                           # box followed by the boxes' width and height
                           box = detection[0:4] * np.array([W, H, W, H])
                           (centerX, centerY, width, height) = box.astype("int")
                           # use the center (x, y)-coordinates to derive the top and
                           # and left corner of the bounding box
                           x = int(centerX - (width / 2))
                           y = int(centerY - (height / 2))
                           # update our list of bounding box coordinates, confidences,
                           # and class IDs
                           boxes.append([x, y, int(width), int(height)])
                           confidences.append(float(confidence))
                           classIDs.append(classID)
               # apply non-maxima suppression to suppress weak, overlapping bounding
               # boxes
               idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
               args["threshold"])
               # ensure at least one detection exists
               bndboxes=[]
               labels_found=[]
               jpeg_i=os.path.join(save_path_jpg,image_i)
               anno_i=os.path.join(save_path_anno,image_i.replace('.jpg','.xml'))
               if len(idxs) > 0:
                   # loop over the indexes we are keeping
                   for i in idxs.flatten():
                       detection_time_i=str(time_start).replace('.','point')
                       detection_path_i=os.path.join(basepath_chips,detection_time_i)
                       detection_path_i_text=os.path.join(detection_path_i,'message_content.txt')
                       datetime_i=str(datetime.datetime.now())
                       detection_path_i_full=os.path.join(detection_path_i,'FULL')
                       im0_og=image.copy()
                       # extract the bounding box coordinates
                       (x, y) = (boxes[i][0], boxes[i][1])
                       (w, h) = (boxes[i][2], boxes[i][3])
                       # draw a bounding box rectangle and label on the image
                       color = [int(c) for c in COLORS[classIDs[i]]]
                       cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                       text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                       if text.find(target_of_interest)!=-1:
                           target_found=True
                       cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, color, 2)
                       print(text)
                       xmin=max(1,x)
                       xmax=max(1,x+w)
                       ymin=max(1,y)
                       ymax=max(1,y+h)
                       score_i=confidences[i]
                       bndboxes.append([xmin,ymin,xmax,ymax,score_i])
                       labels_found.append(LABELS[classIDs[i]])
                       
                       if (send_image_to_cell and os.path.exists(send_image_to_cell_path) and send_allowed) or args['image']!='None':
                        if os.path.exists(detection_path_i)==False:
                            os.makedirs(detection_path_i)
                        label = LABELS[classIDs[i]]+" Confidence="+str(score_i)
                        label_og=label
                        
                        label=label.replace(' ','_').replace('.','p').replace(':',"c").replace('-','_')
                        chip_i=label+".jpg"
                        chip_i=os.path.join(detection_path_i,chip_i)
                        label_list[chip_i]=label_og
                        MARGIN=0
                        print(im0_og.shape)
                        xmin=max(xmin-MARGIN,0)
                        xmax=min(xmax+MARGIN,im0_og.shape[1])
                        ymin=max(ymin-MARGIN,0)
                        ymax=min(ymax+MARGIN,im0_og.shape[0])
                        print('xmin,xmax,ymin,ymax')
                        print(xmin,xmax,ymin,ymax)
                        prefix='yolov4tiny'
                        msg_i=f'{prefix}_{LABELS[classIDs[i]]};{xmin};{ymin};{xmax};{ymax};{np.round(score_i,2)};{im0_og.shape[1]};{im0_og.shape[0]}' #edit sjs
                        msg_i_list=msg_i_list+msg_i+"&"
                        if send_image_to_cell and os.path.exists(send_image_to_cell_path) and send_allowed:
                            if len(list(im0_og.shape))==3:
                                img_list[chip_i]=im0_og[ymin:ymax,xmin:xmax,:]
                                cv2.imwrite(chip_i,im0_og[ymin:ymax,xmin:xmax,:])
                            elif len(list(im0_og.shape))==2:
                                img_list[chip_i]=im0_og[ymin:ymax,xmin:xmax]
                                cv2.imwrite(chip_i,im0_og[ymin:ymax,xmin:xmax])
                   if send_image_to_cell and os.path.exists(send_image_to_cell_path) and send_allowed and len(img_list)>0:
                    if os.path.exists(detection_path_i_full)==False:
                        os.makedirs(detection_path_i_full)
                    main_message="Targets FOUND"
                    with open(detection_path_i_text,'w') as f:
                        f.writelines('Time found == {};\n'.format(datetime_i))
                    cmd_i='python3 {} --destinations={} --main_message="{}"  --img_path="{}" '.format(send_image_to_cell_path,destinations,main_message,detection_path_i)
                    print(cmd_i)
                    Thread(target=run_cmd,args=(cmd_i,)).start()
                    cv2.imwrite(os.path.join(detection_path_i_full,'Full_Detected.jpg'),image)
                    cv2.imwrite(os.path.join(detection_path_i_full,'Full_OG.jpg'),im0_og)
                    send_allowed=False

               else:
                   pass
               if args['image']!='None':
                if msg_i_Queue.empty():
                    print('len(msg_i_list)',len(msg_i_list))
                    msg_i_Queue.put(msg_i_list+'EENNDD')
               if save:
                   # Save Annotation
                   Thread(target=savePascalVocFormat,args=(anno_i,bndboxes,labels_found,jpeg_i,image,scores,)).start()
                   #savePascalVocFormat(anno_i, bndboxes, labels_found, jpeg_i, image,scores)
                   # show the output image
                   moved_file=os.path.join(save_path,image_i)
                   cv2.imwrite(moved_file,image)
               if target_found==True and float(text.split(':')[1])>0.6:
                   text=":0"
                   target_found=False
                   print('FOUND YOUR TARGET')
                   if time.time()-time_found>0.1:
                        time_found=time.time()
               if not(args['noview']):
                   cv2.imshow('YOLO DNN',image)
               if RTMP:
                YH_i,YW_i,VBR_i=YOUTUBE_STREAM_RESOLUTION(res=YOUTUBE_STREAM_RES)
                image_og=cv2.resize(image,(YW_i,YH_i))
                writer.write(image_og,VBR_i)
               if running and RTSP:
                running=True
               elif RTSP:
                cmd_i='python3 {} --fps={} --width={} --height={} --port={} --stream_key={}'.format(RTSP_SERVER_PATH, args['fps'],image.shape[1],image.shape[0],args['port'],args['stream_key'])
                RunMe=Thread(target=run_cmd,args=(cmd_i,)).start()
                running=True
               if RTSP:
                    Thread(target=send_imgs,args=(sender,image,)).start()
               if cv2.waitKey(1) == ord('q'):
                   break
               #cv2.waitKey(0)
        except:
            print('exception found')
            print(traceback.print_exc())
            pass
        #print('LOOP TIME = {} seconds\n'.format(np.round(time.time()-time_start,2)))
        total_time=time.time() - time_start
        fps_i=(1.0/total_time)
        total_fps+=fps_i
        total_fps_count+=1
        avg_fps=total_fps/total_fps_count
        print("[INFO] FPS {:.6f}; AVG_FPS {:.6F}".format(fps_i,avg_fps))
        if fps_i>500:
            print('Slowing down to debug error\n')
            time.sleep(1)
do_stuff(dataset_Queue,dataset_path_Queue,msg_i_Queue)
