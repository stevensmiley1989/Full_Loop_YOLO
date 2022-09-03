# Full_Loop_YOLO
![Full_Loop_YOLO.py](https://github.com/stevensmiley1989/Full_Loop_YOLO/blob/main/misc/Full_Loop_YOLO_GUI_Screenshot.png)

Full_Loop_YOLO.py is a wrapper for creating custom darknet YoloV4-tiny &amp; regular sized models on your custom datasets.
Furthermore, you can train and test Yolov7-tiny as of 7/20/2020.  You may also convert your Yolov4-tiny weights to TFLITE with the click of a button.

It is written in Python and uses Tkinter for its graphical interface.

Prerequisites
------------------

Darknet (Yolov4) should be installed from (https://github.com/AlexeyAB/darknet).  A version controlled fork is shown in these instructions below.

Yolov7 should be installed from (https://github.com/WongKinYiu/yolov7).  A version controlled fork is shown in these instructions below.

tensorflow-yolov4-tflite installed from (https://github.com/stevensmiley1989/tensorflow-yolov4-tflite) and switched to the smiley_yolov4tiny_r0_prior_20220902 branch.

Change your DEFAULT_SETTINGS path (located at libs/DEFAULT_SETTINGS.py) to point to your installed Darknet path for use.  

Change your yolov7 path (located at libs/yolov7_path.py) to point to your installed yolov7 path for use.  NOT REQUIRED.

Change your tensorflow-yolov4-tflite path (located at libs/tensorflow_yolov4_tflite_path.py) to point to your installed path for use.  NOT REQUIRED.

Change your labelImg path (located at libs/labelImg_path.py) to point to your installed path for use.  NOT REQUIRED.

Change your MOSAIC_Chip_Sorter path (located at libs/MOSAIC_Chip_Sorter_path.py) to point to your installed MOSAIC_Chip_Sorter path for use. NOT REQUIRED.

Change your IMAGE_AUG_GUI path (located at libs/IMAGE_AUG_GUI_path.py) to point to your installed IMAGE_AUG_GUI path for use. NOT REQUIRED.

Ensure you put the yolov4-tiny.conv.29 weights in your Darknet path.  You can get these weights from:
https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29

Ensure you put the yolov4.conv.137 weights in your Darknet path.  You can get these weights from:
https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137

~~~~~~~

Darknet Yolov4
.. code:: shell
    cd ~/
    #git clone https://github.com/AlexeyAB/darknet
    git clone https://github.com/stevensmiley1989/darknet.git
    cd darknet
    git switch smiley #if using smiley branch, this is a version control method
    make #modify MakeFile before to use cuda
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29
    wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
    
    #also add to your bashrc file at ~/.bashrc, add the following lines with your cuda paths
    export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

    
~~~~~~~


~~~~~~~

Yolov7
.. code:: shell
    cd ~/
    #git clone https://github.com/WongKinYiu/yolov7.git
    git clone https://github.com/stevensmiley1989/yolov7.git
    cd yolov7
    git switch smiley_r0_prior_20220902 #if using smiley branch, this is a version control method
    pip3 install -r requirements.txt #you might need to adjust things manually here for versions of PyTorch    
~~~~~~~

~~~~~~~

MOSAIC_Chip_Sorter
.. code:: shell
    cd ~/
    git clone https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter.git
    cd MOSAIC_Chip_Sorter
    git switch r0_prior_20220902
    pip3 install -r requirements.txt #you might need to adjust things manually, see repo of MOSAIC_Chip_Sorter for more info 
~~~~~~~

~~~~~~~

tensorflow-yolov4-tflite
.. code:: shell
    cd ~/
    git clone https://github.com/stevensmiley1989/tensorflow-yolov4-tflite.git
    cd tensorflow-yolov4-tflite
    git switch smiley_yolov4tiny_r0_prior_20220902 #
    pip3 install -r requirements_smiley_yolov4_tiny_converter.txt #you might need to adjust things manually    
~~~~~~~

~~~~~~~
labelImg
.. code:: shell
    cd ~/
    git clone https://github.com/stevensmiley1989/labelImg.git
    cd labelImg
    git switch smiley_r0_prior_20220902
    pip3 install -r requirements/requirements-linux-python3.txt #you might need to adjust things manually    
~~~~~~~

~~~~~~~
IMAGE_AUG_GUI
.. code:: shell
    cd ~/
    git clone https://github.com/stevensmiley1989/IMAGE_AUG_GUI.git
    cd IMAGE_AUG_GUI
    git switch r0_prior_20220902
    pip3 install -r requirements.txt #you might need to adjust things manually
~~~~~~~

Installation
------------------
~~~~~~~

Python 3 + Tkinter
.. code:: shell
    cd ~/
    python3 -m venv venv_Full_Loop_YOLO
    source venv_Full_Loop_YOLO/bin/activate
    
    cd ~/Full_Loop_YOLO
    git swtich r0_prior_20220902
    pip3 install -r requirements.txt
    nano libs/DEFAULT_SETTINGS.py #edit the path for darknet to your installed path above
    nano libs/tensorflow_yolov4_tflite_path.py #edit the path for your installed path above
    nano libs/yolov7_path.py #edit the path for yolov7 to your installed path above
    nano libs/labelImg_path.py #edit the path for labelImg to your installed path above
    nano libs/MOSAIC_Chip_Sorter_path.py #edit the path for MOSAIC_Chip_Sorter path above
    nano libs/IMAGE_AUG_GUI_path.py #edit the path for IMAGE_AUG_GUI path above
    python3 Full_Loop_YOLO.py
~~~~~~~

## [YouTube Tutorial (needs updating for new features 8/5/2022) ](https://youtu.be/3cNyFcDw4ks)

Optional
------------------
RTMP to YOUTUBE
Make sure you have a https://www.youtube.com/ account and have started a RTMP Stream.
Use that given key when running Yolo DNN RTMP.


## Contact-Info<a class="anchor" id="4"></a>

Feel free to contact me to discuss any issues, questions, or comments.

* Email: [stevensmiley1989@gmail.com](mailto:stevensmiley1989@gmail.com)
* GitHub: [stevensmiley1989](https://github.com/stevensmiley1989)
* LinkedIn: [stevensmiley1989](https://www.linkedin.com/in/stevensmiley1989)
* Kaggle: [stevensmiley](https://www.kaggle.com/stevensmiley)

### License <a class="anchor" id="5"></a>
MIT License

Copyright (c) 2022 Steven Smiley

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*I am providing code and resources in this repository to you under an open source license.  Because this is my personal repository, the license you receive to my code and resources is from me and not my employer. 
