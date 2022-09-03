import gi,cv2,argparse
import imagezmq
import os
import simplejpeg
from threading import Thread
from multiprocessing import Process
#sudo apt-get install gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad libgstreamer1.0-0 gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
#sudo apt-get install libglib2.0-dev libgstrtspserver-1.0-dev gstreamer1.0-rtsp
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject
global image_hub,args


    

def run_cmd(cmd_i):
    os.system(cmd_i)

class VideoStream:
    """Camera object that controls video streaming"""
    def __init__(self,video,resolution=(300,300),framerate=30): #edit 8/4/2021 sjs, includes 640x640 default with video parameter defined for rtsp arg.
        # Initialize the PiCamera and the camera image stream
        #self.stream = cv2.VideoCapture(video) #edit 8/4/2021 sjs, includes the cv2.CAP_FFMPEG for the os.env set earlier for better feed.
        print('video is set to {}'.format(video))
        if str(video).isnumeric():
            print('using path 1')
            video=int(video)
            #self.stream = cv2.VideoCapture(video,cv2.CAP_FFMPEG) #ed
            self.stream=cv2.VideoCapture(video)
        #ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) #if JETSON NANO, Comment out line
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE,3)
        #ret = self.stream.set(3,resolution[0]) #if JETSON NANO, Comment out line
        #ret = self.stream.set(4,resolution[1]) #if JETSON NANO, Comment out line
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

class RTSP_Images(GstRtspServer.RTSPMediaFactory):
    def __init__(self,fps,width,height,**properties):
        super(RTSP_Images,self).__init__(**properties)
        self.ready=True
        self.run=True
        self.fps=fps
        self.width=width
        self.height=height
        self.frame_dur=1 /self.fps*Gst.SECOND
        self.frame_num=0
        self.gstream_string='appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                             'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 ' \
                             '! videoconvert ! video/x-raw,format=I420 ' \
                             '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                             '! rtph264pay config-interval=1 name=pay0 pt=96' \
                             .format(self.width, self.height, self.fps)
        
    def next_data(self,src,length):
        if self.run:
            try:
                if args.source=='None':
                    self.image_name,self.image=image_hub.recv_image()
                else:
                    sent_from, jpg_buffer = image_hub.recv_jpg()
                    self.image                 = simplejpeg.decode_jpeg( jpg_buffer, 
                                                            colorspace='BGR')                    
                self.ready=True
            except:
                self.ready=False
            if self.ready:
                self.image=cv2.resize(self.image,(self.width,self.height),interpolation=cv2.INTER_LINEAR)
                self.image=self.image.tostring()
                self.buffer=Gst.Buffer.new_allocate(None,len(self.image),None)
                self.buffer.fill(0,self.image)
                self.buffer.duration=self.frame_dur
                self.timestamp=self.frame_num*self.frame_dur
                self.buffer.pts=self.buffer.dts=int(self.timestamp)
                self.buffer.offset=self.timestamp
                self.frame_num+=1
                self.ret=src.emit('push-buffer',self.buffer)
                if self.ret != Gst.FlowReturn.OK:
                    print(self.ret)
                if args.source=='None':
                    image_hub.send_reply(b'OK')
                else:
                    #pass
                    image_hub.send_reply(b'OK')
    def do_create_element(self,string_i):
        return Gst.parse_launch(self.gstream_string)
    
    def do_configure(self,rtsp_media):
        self.frame_num=0
        self.appsrc=rtsp_media.get_element().get_child_by_name('source')
        self.appsrc.connect('need-data',self.next_data)

class GstServer_RTSPServer(GstRtspServer.RTSPServer):
    def __init__(self,fps,width,height,port,stream_key,**properties):
        super(GstServer_RTSPServer,self).__init__(**properties)
        self.RTSP_Images=RTSP_Images(fps,width,height)
        self.RTSP_Images.set_shared(True)
        self.port=port
        self.stream_key=stream_key
        self.set_service(str(self.port))
        self.get_mount_points().add_factory(self.stream_key,self.RTSP_Images)
        self.attach(None)


class args_stuff:
    def __init__(self):
        self.fps=30
        self.width=640
        self.height=480
        self.port=8554
        self.stream_key="/video_stream"
        self.source='0'#'None'

class RunMe:
    def __init__(self,fps,width,height,port,stream_key):
        self.fps=fps
        self.width=width
        self.height=height
        self.port=port
        self.stream_key=stream_key
        self.GObject=GObject.threads_init()
        self.Gst=Gst
        self.Gst.init(None)
        self.server=GstServer_RTSPServer(self.fps,self.width,self.height,self.port,self.stream_key)
        self.app=GObject.MainLoop()
        self.app.run()
def run_source_feed(args,IP_ADDRESS):
    #sender = imagezmq.ImageSender(connect_to=f"tcp://*:5553",REQ_REP=False)
    sender = imagezmq.ImageSender(connect_to=f"tcp://{IP_ADDRESS}:5553")
    source=args.source
    imW=args.width
    imH=args.height
    host_name= socket.gethostname() # send hostname with each image
    videostream = VideoStream(source,resolution=(imW,imH),framerate=100).start()
    window_name='RTSP Feed from {}'.format(source)
    print(f'source = {source}, imW={imW}')
    while videostream.stopped==False:
        image = videostream.read()
        #print(image)
        #cv2.imshow(window_name,image)
        jpeg_quality = 85 
        try:
            jpg_buffer     = simplejpeg.encode_jpeg(image, quality=jpeg_quality, 
                                                                colorspace='BGR')
            classification_i = sender.send_jpg(host_name, jpg_buffer)
        except:
            pass
        if cv2.waitKey(1) == ord('q'):
            break


if __name__=="__main__":
    ar=args_stuff()
    parser=argparse.ArgumentParser()
    parser.add_argument("--fps",default=ar.fps,help="fps of incoming images",type=int)
    parser.add_argument("--width",default=ar.width,help="width of incoming images",type=int)
    parser.add_argument("--height",default=ar.height,help="height of incoming images",type=int)
    parser.add_argument("--port",default=ar.port,help="port",type=int)
    parser.add_argument("--stream_key",default=ar.stream_key,help="rtsp image stream uri",type=str)
    parser.add_argument("--source",default=ar.source,help='Default is None because other scripts can use imagezmq to publish to this one.  If you wan to access the raw video, then specify 0 or 1 etc for /dev/video0 or /dev/video1 etc')
    args=parser.parse_args()
    if args.source=='None':
        image_hub=imagezmq.ImageHub()
        run_app=RunMe(args.fps,args.width,args.height,args.port,args.stream_key)
    elif args.source.isnumeric():
        import socket
        try:
            s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            s.connect(("8.8.8.8",80))
            IP_ADDRESS=s.getsockname()[0]
        except:
            IP_ADDRESS="127.0.0.1"
        print(f'IP_ADDRESS = {IP_ADDRESS}')

        Process(target=run_source_feed,args=(args,IP_ADDRESS,)).start()
        #image_hub=imagezmq.ImageHub(f'tcp://{IP_ADDRESS}:5553',REQ_REP=False)
        image_hub=imagezmq.ImageHub(f'tcp://{IP_ADDRESS}:5553')
        print(args.fps,args.width,args.height,args.port,args.stream_key)
        run_app=RunMe(args.fps,args.width,args.height,args.port,args.stream_key)
        
