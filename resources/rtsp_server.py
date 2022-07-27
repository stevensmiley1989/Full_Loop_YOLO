import gi,cv2,argparse
import imagezmq
#sudo apt-get install gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad libgstreamer1.0-0 gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio
#sudo apt-get install libglib2.0-dev libgstrtspserver-1.0-dev gstreamer1.0-rtsp
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject
image_hub=imagezmq.ImageHub()

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
                self.image_name,self.image=image_hub.recv_image()
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

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--fps",required=True,help="fps of incoming images",type=int)
    parser.add_argument("--width",required=True,help="width of incoming images",type=int)
    parser.add_argument("--height",required=True,help="height of incoming images",type=int)
    parser.add_argument("--port",default=8554,help="port",type=int)
    parser.add_argument("--stream_key",default="/video_stream",help="rtsp image stream uri")
    args=parser.parse_args()
    run_app=RunMe(args.fps,args.width,args.height,args.port,args.stream_key)