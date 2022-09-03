from venv import create
import imagezmq
import socket
import argparse
import cv2
from threading import Thread
import os
import simplejpeg
from multiprocessing import Process
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,video,resolution=(300,300),framerate=40): #edit 8/4/2021 sjs, includes 640x640 default with video parameter defined for rtsp arg.
        # Initialize the PiCamera and the camera image stream
        #self.stream = cv2.VideoCapture(video) #edit 8/4/2021 sjs, includes the cv2.CAP_FFMPEG for the os.env set earlier for better feed.
        print('video is set to {}'.format(video))
        if str(video).isnumeric():
            print('using path 1')
            video=int(video)
        if str(video)!='0' and str(video)!='1' and str(video).find('rtsp')==-1 and str(video).lower().find('mp4')==-1:
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
        if str(video).find('rtsp')==-1 and str(video).lower().find('mp4')==-1:
            ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) #if JETSON NANO, Comment out line
        if str(video).find('rtsp')==-1:
            self.stream.set(cv2.CAP_PROP_BUFFERSIZE,3)
        if str(video).find('rtsp')==-1:
            ret = self.stream.set(3,resolution[0]) #if JETSON NANO, Comment out line
            ret = self.stream.set(4,resolution[1]) #if JETSON NANO, Comment out line

        self.resolution=resolution

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()


	# Variable to control when the camera is stopped
        self.stopped = False
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
try:
    s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    s.connect(("8.8.8.8",80))
    IP_ADDRESS=s.getsockname()[0]
except:
    IP_ADDRESS="127.0.0.1"
print(f'IP_ADDRESS = {IP_ADDRESS}')
PORT_LIST=['5553','5554','5555']
IP_LIST=[IP_ADDRESS,IP_ADDRESS,IP_ADDRESS]

def create_senders(IP_LIST,PORT_LIST,REQ_REP=False):
    sender_dic={}
    for i,(IP_ADDRESS,PORT) in enumerate(zip(IP_LIST,PORT_LIST)):
        sender_dic[i]=imagezmq.ImageSender(connect_to=f"tcp://{IP_ADDRESS}:{PORT}",REQ_REP=REQ_REP)
    return sender_dic
def run_multi_senders_custom(image,sender_dic):
        host_name= socket.gethostname() # send hostname with each image
        jpeg_quality = 95 
        try:
            jpg_buffer     = simplejpeg.encode_jpeg(image, quality=jpeg_quality, 
                                                                colorspace='BGR')
            for i,sender in enumerate(sender_dic.values()):
                try:
                    response_i = sender.send_jpg(host_name, jpg_buffer)
                except:
                    print('Failed to send to i={}, PORT={}'.format(i,PORT_LIST[i]))
        except:
            print("EXCEPTION")
            pass

def run_multi_senders(args,IP_LIST,PORT_LIST,REQ_REP=False):
    sender_dic=create_senders(args,IP_LIST,PORT_LIST,REQ_REP)
    source=args.source
    imW=args.width
    imH=args.height
    host_name= socket.gethostname() # send hostname with each image
    videostream = VideoStream(source,resolution=(imW,imH),framerate=40).start()
    window_name='Feed from {}'.format(source)
    print(f'source = {source}, imW={imW}')
    while videostream.stopped==False:
        image = videostream.read()
        #print(image)

        jpeg_quality = 95 
        try:
            jpg_buffer     = simplejpeg.encode_jpeg(image, quality=jpeg_quality, 
                                                                colorspace='BGR')
            for i,sender in enumerate(sender_dic.values()):
                try:
                    response_i = sender.send_jpg(host_name, jpg_buffer)
                except:
                    print('Failed to send to i={}, PORT={}'.format(i,PORT_LIST[i]))
        except:
            print("EXCEPTION")
            pass
        cv2.imshow(window_name,image)
        if cv2.waitKey(1) == ord('q'):
            videostream.stop()
            break
def create_receivers(IP_LIST,PORT_LIST,REQ_REP=False):
    receiver_dic={}
    for i,(IP_ADDRESS,PORT) in enumerate(zip(IP_LIST,PORT_LIST)):
        receiver_dic[i]=imagezmq.ImageHub(f'tcp://{IP_ADDRESS}:{PORT}',REQ_REP=REQ_REP)
    return receiver_dic
def run_single_receiver_custom(receiver_dic_i,REQ_REP=False):
        try:

                sent_from, jpg_buffer = receiver_dic_i.recv_jpg()
                image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                        colorspace='BGR')  
                #window_name='Feed from {}, PORT={}'.format(i,PORT_LIST[i])
                #cv2.imshow(window_name,image)  
                if REQ_REP:
                    receiver_dic_i.send_reply(b'OK')
                return image
        except:
            pass 
def run_multi_receivers_custom(receiver_dic,REQ_REP=False):
        try:

            for i,image_hub_i in enumerate(receiver_dic.values()):
                sent_from, jpg_buffer = image_hub_i.recv_jpg()
                image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                        colorspace='BGR')  
                window_name='Feed from {}, PORT={}'.format(i,PORT_LIST[i])
                cv2.imshow(window_name,image)  
                if REQ_REP:
                    image_hub_i.send_reply(b'OK')
        except:
            pass    
def run_multi_receivers(args,IP_LIST,PORT_LIST,REQ_REP=False):
    receiver_dic=create_receivers(IP_LIST,PORT_LIST,REQ_REP)
    while True:       
        try:

            for i,image_hub_i in enumerate(receiver_dic.values()):
                sent_from, jpg_buffer = image_hub_i.recv_jpg()
                image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                        colorspace='BGR')  
                window_name='Feed from {}, PORT={}'.format(i,PORT_LIST[i])
                cv2.imshow(window_name,image)  
                if REQ_REP:
                    image_hub_i.send_reply(b'OK')
        except:
            pass
        if cv2.waitKey(1) == ord('q'):
            break

def generate_PORT_LIST(PORT_LIST_PATH):
    if os.path.exists(PORT_LIST_PATH):
        f=open(PORT_LIST_PATH,'r')
        f_read=f.readlines()
        f.close()
        PORT_LIST=[w.replace('\n','') for w in f_read]
        if len(PORT_LIST)>0:
            all_numeric=True
            PORT_LIST_INT=[]
            for PORT in PORT_LIST:
                if PORT.isnumeric() and len(PORT)==4:
                    all_numeric=True
                    PORT_LIST_INT.append(int(PORT))
                else:
                    all_numeric=False
            if all_numeric==True:
                return PORT_LIST_INT
            else:
                return ['ERROR']
    else:
        print('GENERATING PORT LIST at {}'.format(PORT_LIST_PATH))
        f=open(PORT_LIST_PATH,'w')
        PORT_LIST=[5553,5554,5555]
        for PORT in PORT_LIST:
            print('GENERATED PORT = {}'.format(PORT))
            f.writelines(str(PORT)+'\n')
        f.close()
        return PORT_LIST


class args_stuff:
    def __init__(self):
        self.fps=30
        self.width=640
        self.height=480
        self.PORT_LIST_PATH="PORT_LIST.txt"
        self.source='0'#r'/media/steven/Elements/Drone_Videos_Park/DJI_0028.MP4'
        self.REQ_REP=True
if __name__=="__main__":
    ar=args_stuff()
    parser=argparse.ArgumentParser()
    parser.add_argument("--fps",default=ar.fps,help="fps of incoming images",type=int)
    parser.add_argument("--width",default=ar.width,help="width of incoming images",type=int)
    parser.add_argument("--height",default=ar.height,help="height of incoming images",type=int)
    parser.add_argument("--PORT_LIST_PATH",default=ar.PORT_LIST_PATH,help="port",type=str)
    parser.add_argument("--source",default=ar.source,help='Default is None because other scripts can use imagezmq to publish to this one.  If you wan to access the raw video, then specify 0 or 1 etc for /dev/video0 or /dev/video1 etc')
    parser.add_argument("--REQ_REP",default=ar.REQ_REP,action='store_true',help='Require Response between sending images?')
    args=parser.parse_args()
    PORT_LIST=generate_PORT_LIST(args.PORT_LIST_PATH)
    IP_LIST=[]
    for PORT in PORT_LIST:
        IP_LIST.append(IP_ADDRESS)
    if PORT_LIST[0]!='ERROR':
        Process(target=run_multi_receivers,args=(args,IP_LIST,PORT_LIST,args.REQ_REP,)).start()
        Process(target=run_multi_senders,args=(args,IP_LIST,PORT_LIST,args.REQ_REP,)).start()
    else:
        print('CANNOT RUN BECAUSE BAD PORT LIST PROVIDED at {}\n'.format(args.PORT_LIST_PATH))
    #image_hub=imagezmq.ImageHub(f'tcp://{IP_ADDRESS}:5553',REQ_REP=False)
    #print(args.fps,args.width,args.height,args.port,args.stream_key)
    #run_app=RunMe(args.fps,args.width,args.height,args.port,args.stream_key)
        
