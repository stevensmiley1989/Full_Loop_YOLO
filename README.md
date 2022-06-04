# Full_Loop_YOLO
Full_Loop_YOLO.py is a wrapper for creating custom darknet YoloV4 tiny &amp; regular sized models on your custom datasets. 


'''
Full_Loop_Yolo
========
Created by Steven Smiley 3/20/2022

Full_Loop_YOLO.py is a wrapper for creating custom darknet YoloV4 tiny & regular sized models on their custom datasets. 

Darknet should be installed from (https://github.com/AlexeyAB/darknet).

Change your DEFAULT_SETTINGS path to point to your installed Darknet path for use.  

Ensure you put the yolov4-tiny.conv.29 weights in your Darknet path.  You can get these weights from:
https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29

Ensure you put the yolov4.conv.137 weights in your Darknet path.  You can get these weights from:
https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137


It is written in Python and uses Tkinter for its graphical interface.


Installation
------------------

Ubuntu Linux
~~~~~~~

Python 3 + Tkinter + Darknet YOLO

.. code:: shell
    cd ~/
    git clone https://github.com/AlexeyAB/darknet
    cd darknet
    make #modify MakeFile before to use cuda
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137

    cd ../Full_Loop_YOLO
    sudo pip3 install -r requirements.txt
    nano libs/DEFAULT_SETTINGS.py #edit the path for darknet to your installed path above
    python3 Full_Loop_YOLO.py
~~~~~~~
'''