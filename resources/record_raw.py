import cv2
import os
import argparse
import datetime
from threading import Thread
def RESOLUTION(res='720p'):
    '''edit sjs, added YOUTUBE_STREAM_RESOLUTION function'''
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
imH,imW,_=RESOLUTION('1080p')
parser = argparse.ArgumentParser()
parser.add_argument('--video', type=str, default='0', help='video')  # file/folder, 0 for webcam
#parser.add_argument('--video', type=str, default='rtsp://10.5.1.200:8554/unicast', help='video')  # file/folder, 0 for webcam
parser.add_argument("--imW",type=int,default=imW,help='Width of input image')
parser.add_argument("--imH",type=int,default=imH,help='Height of input image')
parser.add_argument("--fps",type=int,default=30,help='fps')
parser.add_argument("--using_JETSON_NANO",action='store_true', help='If using Jetson NANO, issues with RET, so want to have this flag to handle it')
parser.add_argument("--UNIQUE_DEVICE",type=str,default='Jetson',help='device type for name')
parser.add_argument("--UNIQUE_PREFIX",type=str,default='',help='Additional unique prefix (i.e. soccer) for video file made')
parser.add_argument("--second_path",type=str,default=r"/media/steven/Elements/Videos/",help='backup location of recorded videos')
parser.add_argument("--main_path",type=str,default=r"/media/steven/OneTouch4tb/Videos/",help='main location')

args = vars(parser.parse_args())

#fix video
video=args['video'].strip()
if video=='0':
    video=0
elif video=='1':
    video=1
args['video']=video

def run_cmd(cmd_i):
    os.system(cmd_i)
def create_output_paths(args):
    unique_device=args['UNIQUE_DEVICE']
    video_device=str(args['video'])
    video_device=video_device.replace(':','-').replace('/','').replace('.','p')
    video_device=unique_device+"_"+video_device
    main_path=args['main_path']
    second_path=args['second_path'] 
    third_path=os.path.join(os.getcwd(),'Videos')
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
    output_day_hour_min_path=output_day_hour_sec_path[0:output_day_hour_sec_path.rfind('-')]+'_'+args['UNIQUE_PREFIX']+'_'+unique_device+'.mp4'
          
    try:
        os.makedirs(output_day_path)
    except:
        #print("This output path already exists")
        pass
    OUTPUT_FILE=output_day_hour_min_path
    #print("OUTPUT_FILE",OUTPUT_FILE)
    return OUTPUT_FILE
OUTPUT_FILE=create_output_paths(args)
if str(args['video']).find('rtsp')!=-1:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]="rtsp_transport;udp" #edit sjs 8/4/2021
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,video,resolution=(imW,imH),framerate=30,output_file_path=OUTPUT_FILE): #edit 8/4/2021 sjs, includes 640x640 default with video parameter defined for rtsp arg.	
        # Initialize the PiCamera and the camera image stream
        #self.stream = cv2.VideoCapture(video) #edit 8/4/2021 sjs, includes the cv2.CAP_FFMPEG for the os.env set earlier for better feed.
        print('video is set to {}'.format(video))
        if str(video)!='0' and str(video)!='1' and str(video).find('rtsp')==-1 and str(video).find('cam')==-1:
            print('using path 1')
            #self.stream = cv2.VideoCapture(video,cv2.CAP_FFMPEG) #ed
            self.stream=cv2.VideoCapture(video,cv2.CAP_DSHOW)
        elif str(video).find('rtsp')!=-1:
            print('using path 2')
            print('using rtsp cam {}'.format(str(video)))
            self.stream=cv2.VideoCapture(video)
        elif str(video).find('0_cam')!=-1:
            self.stream=cv2.VideoCapture(0)
            output_file_path=output_file_path.replace('.mp4','.mp4')
            imW=int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            imH=int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            resolution=(imW,imH)
            fps=self.stream.get(cv2.CAP_PROP_FPS)
        else:
            print('using path 3')
            self.stream=cv2.VideoCapture(video)
        if str(video).find('rtsp')!=-1:
            imW=int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            imH=int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            resolution=(imW,imH)
            print('resolution=',resolution)
            fps=self.stream.get(cv2.CAP_PROP_FPS)
            if fps<50:
                print('fps=',fps)
            else:
                print('using fps of 30')
                fps=30
        else:
            fps=args['fps']

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
        #self.fourcc=cv2.VideoWriter_fourcc(*'XVID')
        self.fourcc=cv2.VideoWriter_fourcc(*'mp4v')
        #self.fourcc=cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
        if str(video).find('0_cam')!=-1:
            pass
            #self.fourcc=cv2.VideoWriter_fourcc(*'avc2')

        self.videoOut=cv2.VideoWriter(self.output_file_path,self.fourcc,fps,self.resolution)
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
#Capture video from webcam
aspect=640./480.
if imW==imH:
    imH=int(imW/aspect)

videostream = VideoStream(args['video'],resolution=(args['imW'],args['imH']),framerate=args['fps']).start()
while True:
    image = videostream.read()
    cv2.imshow("Press 'q' to Quit recording to: "+OUTPUT_FILE,image)
    videostream.output(image)
    if cv2.waitKey(1) &0XFF == ord('q'):
        break

videostream.stop()
videostream.videoOut.release()
cv2.destroyAllWindows()
os.system('xdg-open {}'.format(os.path.dirname(OUTPUT_FILE)))
