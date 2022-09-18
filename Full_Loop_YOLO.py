'''
Full_Loop_Yolo
========
Created by Steven Smiley 3/20/2022

Full_Loop_YOLO.py is a wrapper for creating custom darknet YoloV4-tiny &amp; regular sized models on your custom datasets.
Furthermore, you can train and test Yolov7-tiny as of 7/20/2020.  You may also convert your Yolov4-tiny weights to TFLITE with the click of a button.

It is written in Python and uses Tkinter for its graphical interface.

Prerequisites
------------------

Darknet (Yolov4) should be installed from (https://github.com/AlexeyAB/darknet).  A version controlled fork is shown in these instructions below.

Yolov7 should be installed from (https://github.com/WongKinYiu/yolov7).  A version controlled fork is shown in these instructions below.

tensorflow-yolov4-tflite installed from (https://github.com/stevensmiley1989/tensorflow-yolov4-tflite) and switched to the smiley_yolov4tiny branch.

Change your DEFAULT_SETTINGS path (located at libs/DEFAULT_SETTINGS.py) to point to your installed Darknet path for use.  

Change your yolov7 path (located at libs/yolov7_path.py) to point to your installed yolov7 path for use.  NOT REQUIRED.

Change your tensorflow-yolov4-tflite path (located at libs/tensorflow_yolov4_tflite_path.py) to point to your installed path for use.  NOT REQUIRED.

Change your labelImg path (located at libs/labelImg_path.py) to point to your installed path for use.  NOT REQUIRED.

Change your MOSAIC_Chip_Sorter path (located at libs/MOSAIC_Chip_Sorter_path.py) to point to your installed MOSAIC_Chip_Sorter path for use. NOT REQUIRED.

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
    git switch smiley #if using smiley branch, this is a version control method
    pip3 install -r requirements.txt #you might need to adjust things manually here for versions of PyTorch    
~~~~~~~

~~~~~~~

MOSAIC_Chip_Sorter
.. code:: shell
    cd ~/
    git clone https://github.com/stevensmiley1989/MOSAIC_Chip_Sorter.git
    cd MOSAIC_Chip_Sorter
    pip3 install -r requirements.txt #you might need to adjust things manually, see repo of MOSAIC_Chip_Sorter for more info 
~~~~~~~

~~~~~~~

tensorflow-yolov4-tflite
.. code:: shell
    cd ~/
    git clone https://github.com/stevensmiley1989/tensorflow-yolov4-tflite.git
    cd tensorflow-yolov4-tflite
    git switch smiley_yolov4tiny #
    pip3 install -r requirements_smiley_yolov4_tiny_converter.txt #you might need to adjust things manually    
~~~~~~~

~~~~~~~
labelImg
.. code:: shell
    cd ~/
    git clone https://github.com/stevensmiley1989/labelImg.git
    cd labelImg
    git switch smiley
    pip3 install -r requirements/requirements-linux-python3.txt #you might need to adjust things manually    
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
    pip3 install -r requirements.txt
    nano libs/DEFAULT_SETTINGS.py #edit the path for darknet to your installed path above
    nano libs/tensorflow_yolov4_tflite_path.py #edit the path for your installed path above
    nano libs/yolov7_path.py #edit the path for yolov7 to your installed path above
    nano libs/labelImg_path.py #edit the path for labelImg to your installed path above
    nano libs/MOSAIC_Chip_Sorter_path.py #edit the path for MOSAIC_Chip_Sorter path above
    python3 Full_Loop_YOLO.py
~~~~~~~
'''
from pandastable import Table, TableModel
import os
from pprint import pprint
import sys
from sys import platform as _platform
from tkinter.font import names
import pandas as pd
from tqdm import tqdm
import os
import traceback
import matplotlib.pyplot as plt
from functools import partial
from threading import Thread
from multiprocessing import Process
import shutil
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
import numpy as np
import functools
import time
import PIL
from PIL import Image, ImageTk
from PIL import ImageDraw
from PIL import ImageFont
import tkinter as tk
from tkinter import N, ttk
from tkinter import filedialog as fd
from tkinter.messagebox import NO, showinfo
from tkinter.tix import Balloon
DEFAULT_ENCODING = 'utf-8'
XML_EXT = '.xml'
ENCODE_METHOD = DEFAULT_ENCODING
from resources import switch_basepath
from resources import create_img_list
from resources import create_imgs_from_video
import socket
global return_to_main
return_to_main=True
class TestApp(tk.Frame):
    def __init__(self, parent, filepath):
        super().__init__(parent)
        self.table = Table(self, showtoolbar=True, showstatusbar=True)
        self.table.importCSV(filepath)
        #self.table.load(filepath)
        #self.table.resetIndex()
        self.table.show()

#switch_basepath.switch_scripts()
def get_default_settings(SAVED_SETTINGS='SAVED_SETTINGS'):
    global DEFAULT_SETTINGS
    try:
        #from libs import SAVED_SETTINGS as DEFAULT_SETTINGS
        exec('from libs import {} as DEFAULT_SETTINGS'.format(SAVED_SETTINGS),globals())
        if os.path.exists(DEFAULT_SETTINGS.base_path_OG):
            pass
        else:
            from libs import DEFAULT_SETTINGS
    except:
        print(traceback.print_exc())
        print('exception')
        from libs import DEFAULT_SETTINGS 

if _platform=='darwin':
    import tkmacosx
    from tkmacosx import Button as Button
    open_cmd='open'
else:
    from tkinter import Button as Button
    if _platform.lower().find('linux')!=-1:
        open_cmd='xdg-open'
    else:
        open_cmd='start'
#pprint(cfg_vanilla)
global PROCEED 
PROCEED=False
class main_entry:
    global SAVED_SETTINGS_PATH
    def __init__(self,root_tk):
        self.root=root_tk
        self.root.bind('<Escape>',self.close)
        self.root.wm_iconphoto(False,ImageTk.PhotoImage(Image.open("resources/icons/appY.png")))
        self.root.wm_iconphoto(False,ImageTk.PhotoImage(Image.open("resources/icons/appY.png")))
        self.icon_breakup=ImageTk.PhotoImage(Image.open('resources/icons/breakup.png'))
        self.icon_folder=ImageTk.PhotoImage(Image.open("resources/icons/open_folder_search.png"))
        self.icon_single_file=ImageTk.PhotoImage(Image.open('resources/icons/single_file.png'))
        self.icon_load=ImageTk.PhotoImage(Image.open('resources/icons/load.png'))
        self.icon_create=ImageTk.PhotoImage(Image.open('resources/icons/create.png'))
        self.icon_analyze=ImageTk.PhotoImage(Image.open('resources/icons/analyze.png'))
        self.icon_move=ImageTk.PhotoImage(Image.open('resources/icons/move.png'))
        self.icon_labelImg=ImageTk.PhotoImage(Image.open('resources/icons/labelImg.png'))
        self.icon_map=ImageTk.PhotoImage(Image.open('resources/icons/map.png'))
        self.icon_merge=ImageTk.PhotoImage(Image.open('resources/icons/merge.png'))
        self.icon_clear_fix=ImageTk.PhotoImage(Image.open('resources/icons/clear.png'))
        self.icon_clear_checked=ImageTk.PhotoImage(Image.open('resources/icons/clear_checked.png'))
        self.icon_save_settings=ImageTk.PhotoImage(Image.open('resources/icons/save.png'))
        self.icon_yolo_objects=ImageTk.PhotoImage(Image.open('resources/icons/yolo_objects.png'))
        self.icon_divide=ImageTk.PhotoImage(Image.open('resources/icons/divide.png'))
        self.icon_scripts=ImageTk.PhotoImage(Image.open('resources/icons/scripts.png'))
        self.icon_config=ImageTk.PhotoImage(Image.open('resources/icons/config.png'))
        self.icon_open=ImageTk.PhotoImage(Image.open('resources/icons/open_folder.png'))
        self.icon_train=ImageTk.PhotoImage(Image.open('resources/icons/train.png'))
        self.icon_test_mp4=ImageTk.PhotoImage(Image.open('resources/icons/test_mp4.png'))
        self.icon_test_images=ImageTk.PhotoImage(Image.open('resources/icons/test_images.png'))
        self.icon_test=ImageTk.PhotoImage(Image.open('resources/icons/test.png'))
        self.icon_MOSAIC=ImageTk.PhotoImage(Image.open('resources/icons/appM_icon.png'))
        self.icon_IMGAUG=ImageTk.PhotoImage(Image.open('resources/icons/appI_icon.png'))
        self.list_script_path='resources/list_of_scripts/list_of_scripts.txt'
        self.style4=ttk.Style()
        self.style4.configure('Normal.TCheckbutton',
                             background='green',
                             foreground='black')
        self.root_H=int(self.root.winfo_screenheight()*0.95)
        self.root_W=int(self.root.winfo_screenwidth()*0.95)
        self.root.geometry(str(self.root_W)+'x'+str(self.root_H))
        self.root.title("Full-Loop YOLO")
        self.root_bg='black'
        self.root_fg='lime'
        self.canvas_columnspan=50
        self.canvas_rowspan=50
        self.root_background_img=r"misc/gradient_green.jpg"
        
        self.INITIAL_CHECK()
    def INITIAL_CHECK(self):
        self.get_update_background_img()
        self.root.configure(bg=self.root_bg)
        self.dropdown=None
        self.CWD=os.getcwd()
        self.df_settings=pd.DataFrame(columns=['files','Annotations','Number Models','mp4_video_path','path_Annotations','path_JPEGImages','path_Yolo','YOLO_MODEL_PATH'])
        self.SETTINGS_FILE_LIST=[w.split('.py')[0] for w in os.listdir('libs') if w.find('SETTINGS')!=-1 and w[0]!='.'] 
        self.files_keep=[]
        i=0
        for file in self.SETTINGS_FILE_LIST:
            file=file+'.py'
            if file!="DEFAULT_SETTINGS.py":
                #print(file)
                found=False
                f=open(os.path.join('libs',file),'r')
                f_read=f.readlines()
                f.close()
                for line in f_read:
                    if line.find('YOLO_MODEL_PATH')!=-1:
                        self.df_settings.at[i,'YOLO_MODEL_PATH']=line.split('=')[-1].replace("'",'"').split('"')[1].replace('\n','')
                        self.files_keep.append(file.split('.py')[0])
                        self.df_settings.at[i,'files']=file.split('.py')[0]
                        if os.path.exists(os.path.join(line.split('=')[-1].replace("'",'"').split('"')[1],'backup_models')):
                            num_models=len(os.listdir(os.path.join(line.split('=')[-1].replace("'",'"').split('"')[1],'backup_models')))
                            self.df_settings.at[i,'Number Models']=num_models
                        else:
                            self.df_settings.at[i,'Number Models']=0
                        found=True
                    elif line.find('path_Annotations')!=-1:
                        #self.df_settings.at[i,'Annotations']=line.split('=')[-1].replace("'",'"').split('"')[1].split('Annotations')[0].split('/')[-2]
                        self.df_settings.at[i,'Annotations']=line.split('=')[-1].replace("'",'"').split('"')[1].split('Annotations')[0]
                        self.df_settings.at[i,'path_Annotations']=line.split('=')[-1].replace("'",'"').split('"')[1].replace('\n','')
                    elif line.find('path_JPEGImages')!=-1:
                        self.df_settings.at[i,'path_JPEGImages']=line.split('=')[-1].replace("'",'"').split('"')[1].replace('\n','')
                    elif line.find('path_Yolo')!=-1:
                        self.df_settings.at[i,'path_Yolo']=line.split('=')[-1].replace("'",'"').split('"')[1].replace('\n','')
                        

                    elif line.find('mp4_video_path')!=-1:
                        self.df_settings.at[i,'mp4_video_path']=line.split('=')[-1].replace("'",'"').split('"')[1]
                if found==True:
                    i+=1
        self.df_settings=self.df_settings.fillna('0')
        self.files_keep.append('DEFAULT_SETTINGS')
        print(self.df_settings)
        self.checkd_buttons={}
        self.checkd_vars={}
        self.checkd_label=tk.Label(self.root,text='Dataset',bg=self.root_bg,fg=self.root_fg,font=('Arial 14 underline'))
        self.checkd_label.grid(row=1,column=2,sticky='nw')
        f=open('libs/DATASETS_LIST.txt','w')
        print(self.df_settings)
        for i,path_JPEGImages_i in enumerate(list(self.df_settings['path_JPEGImages'])):
            zip_i=self.df_settings['path_Annotations'].iloc[i]+":"+self.df_settings['path_JPEGImages'].iloc[i]+":"+self.df_settings['path_Yolo'].iloc[i]+":"+self.df_settings['YOLO_MODEL_PATH'].iloc[i]
            f.writelines(zip_i+'\n')
        f.close()
        for i,label in enumerate(list(sorted(self.df_settings['Annotations'].unique()))):

            self.checkd_vars[label]=tk.IntVar()
            self.checkd_vars[label].set(0)
            self.checkd_buttons[file]=ttk.Checkbutton(self.root, style='Normal.TCheckbutton',text=label,variable=self.checkd_vars[label], command=self.update_checks,onvalue=1, offvalue=0)
            self.checkd_buttons[file].grid(row=i+1,column=2,sticky='sw')
        self.checkm_buttons={}
        self.checkm_vars={}
        #self.checkm_label=tk.Label(self.root,text='Number of Models',bg=self.root_bg,fg=self.root_fg,font=('Arial 14 underline'))
        #self.checkm_label.grid(row=1,column=3,sticky='nw')
        for i,label in enumerate(sorted(list(self.df_settings['Number Models'].astype(int).unique()))):
            self.checkm_vars[label]=tk.IntVar()
            self.checkm_vars[label].set(1)
            #self.checkm_buttons[file]=ttk.Checkbutton(self.root, style='Normal.TCheckbutton',text=label,variable=self.checkm_vars[label], command=self.update_checks,onvalue=1, offvalue=0)
            #self.checkm_buttons[file].grid(row=i+1,column=3,sticky='sw')           


        self.SETTINGS_FILE_LIST=self.files_keep
        self.df_comb=pd.DataFrame(columns=['times','items'])
        self.df_comb['times']=[os.path.getmtime(os.path.join('libs',w+'.py')) for w in self.SETTINGS_FILE_LIST] #edit sjs 6/11/2022 use to be libs/
        self.df_comb['items']=[w for w in self.SETTINGS_FILE_LIST]
        #self.df_comb=self.df_comb.sort_values(by='times',ascending=True).reset_index().drop('index',axis=1)
        self.df_comb=self.df_comb.sort_values(by='times',ascending=True).reset_index().drop('index',axis=1)
        self.SETTINGS_FILE_LIST=list(self.df_comb['items'])
        self.USER=""
        self.USER_SELECTION=tk.StringVar()
        self.dropdown_menu()
        self.submit_label=Button(self.root,text='Submit',command=self.submit,bg=self.root_fg,fg=self.root_bg,font=('Arial',12))
        self.submit_label.grid(row=1,column=5,sticky='se')
        self.delete_label=Button(self.root,text='Delete',command=self.popupWindow_delete,bg=self.root_bg,fg=self.root_fg,font=('Arial',12))
        self.delete_label.grid(row=2,column=5,sticky='se')
        self.open_libs=Button(self.root,text='Open /libs',command=self.run_cmd_open_libs,bg=self.root_bg,fg=self.root_fg,font=('Arial',12))
        self.open_libs.grid(row=3,column=5,sticky='se')
        # self.submit2_label=Button(self.root,text='Run Script',command=self.submit_script,bg=self.root_fg,fg=self.root_bg,font=('Arial',12))
        # self.submit2_label.grid(row=4,column=1,sticky='se')
        # self.select_file_script_label=Button(self.root,image=self.icon_folder,command=self.select_file_script,bg=self.root_bg,fg=self.root_fg,font=('Arial',8))
        # self.select_file_script_label.grid(row=4,column=2,sticky='sw')


    def update_checks(self):
        checked_models=[]
        for model_num,var in self.checkm_vars.items():
            if var.get()==1:
                checked_models.append(model_num)
        checked_datasets=[]
        for dataset,var in self.checkd_vars.items():
            if var.get()==1:
                checked_datasets.append(dataset)
        df_temp=self.df_settings[(self.df_settings['Number Models'].isin(checked_models))&(self.df_settings['Annotations'].isin(checked_datasets))].copy()
        self.files_keep=list(df_temp['files'])
        self.files_keep.append('DEFAULT_SETTINGS')
        self.dropdown_menu()

            

    def select_file_script(self):
        filetypes=(('sh','*.sh'),('All files','*.*'))
        initialdir_i=os.getcwd()
        self.filename=fd.askopenfilename(title='Open a file',
                                    initialdir=initialdir_i,
                                    filetypes=filetypes)
        if os.path.exists(self.filename):
            print(self.filename)
            f=open(self.list_script_path,'r')
            f_old=f.readlines()
            f.close()
            if len(f_old)>0:
                print('length >0')
                f_old.append(self.filename+'\n')
                f_new=f_old
            else:
                f_old=['None']
                f_old.append(self.filename+'\n')
                f_new=f_old
            print(f_new)
            print(len(f_new))
            print('------')
            f_new_dic={path_i:i for i,path_i in enumerate(f_new) if path_i.find('.sh')!=-1}
            f_new_list=list(f_new_dic.keys())
            f=open(self.list_script_path,'w')
            tmp=[f.writelines(w) for w in f_new_list]
            f.close()
            self.dropdown_menu()
        showinfo(title='Selected File',
                 message=self.filename)
    def dropdown_menu(self):
        if self.dropdown!=None:
            self.dropdown_label.destroy()
            self.dropdown.destroy()
        # self.SETTINGS_FILE_LIST=[w.split('.py')[0] for w in os.listdir('libs') if w.find('SETTINGS')!=-1 and w[0]!='.'] 
        # files_keep=[]
        # for file in self.SETTINGS_FILE_LIST:
        #     file=file+'.py'
        #     if file!="DEFAULT_SETTINGS.py":
        #         f=open(os.path.join('libs',file),'r')
        #         f_read=f.readlines()
        #         f.close()
        #         for line in f_read:
        #             if line.find('YOLO_MODEL_PATH')!=-1:
        #                 files_keep.append(file.split('.py')[0])
        #                 break
        # files_keep.append('DEFAULT_SETTINGS')
        self.SETTINGS_FILE_LIST=self.files_keep
        self.df_comb=pd.DataFrame(columns=['times','items'])
        self.df_comb['times']=[os.path.getmtime(os.path.join('libs',w+'.py')) for w in self.SETTINGS_FILE_LIST] #edit sjs 6/11/2022 use to be libs/
        self.df_comb['items']=[w for w in self.SETTINGS_FILE_LIST]
        self.df_comb=self.df_comb.sort_values(by='items',ascending=True).reset_index().drop('index',axis=1)
        self.SETTINGS_FILE_LIST=list(self.df_comb['items'])
   
        self.USER_SELECTION=tk.StringVar()
        if self.USER in self.SETTINGS_FILE_LIST:
            self.USER_SELECTION.set(self.USER)
        else:
            self.USER_SELECTION.set(self.SETTINGS_FILE_LIST[0])
        self.dropdown=tk.OptionMenu(self.root,self.USER_SELECTION,*self.SETTINGS_FILE_LIST)
        self.dropdown.grid(row=1,column=9,sticky='sw')
        
        self.dropdown_label=Button(self.root,image=self.icon_single_file,command=self.run_cmd_libs,bg=self.root_bg,fg=self.root_fg,font=('Arial',12))
        self.dropdown_label.grid(row=1,column=8,sticky='sw')
        # f=open(self.list_script_path,'r')
        # self.SCRIPTS_LINES=f.readlines()
        # f.close()
        # if len(self.SCRIPTS_LINES)>0:
        #     self.SCRIPTS_FILE_LIST=self.SCRIPTS_LINES
        #     self.SCRIPTS_FILE_LIST=[w.replace('\n','') for w in self.SCRIPTS_FILE_LIST if os.path.exists(w.replace('\n',''))]
        # else:
        #     self.SCRIPTS_FILE_LIST=['None']
        # print([os.path.exists(w) for w in self.SCRIPTS_FILE_LIST])
        # self.USER_SELECTION2=tk.StringVar()
        # if len(self.SCRIPTS_FILE_LIST)==0:
        #     self.SCRIPTS_FILE_LIST=['None']
        # self.USER_SELECTION2.set(self.SCRIPTS_FILE_LIST[0])
        # self.dropdown_2=tk.OptionMenu(self.root,self.USER_SELECTION2,*self.SCRIPTS_FILE_LIST)
        # self.dropdown_2.grid(row=4,column=4,sticky='sw')

        # self.dropdown_2_label=Button(self.root,image=self.icon_single_file,command=self.run_cmd_scripts,bg=self.root_bg,fg=self.root_fg,font=('Arial',12))
        # self.dropdown_2_label.grid(row=4,column=3,sticky='sw')
        # self.dropdown_3_label=Button(self.root,text='Edit Script Paths',command=self.run_cmd_editpaths,bg=self.root_bg,fg=self.root_fg,font=('Arial',8))
        # self.dropdown_3_label.grid(row=5,column=1,sticky='ne')        
    def run_cmd_libs(self):
        cmd_i=open_cmd+" {}.py".format(os.path.join('libs',self.USER_SELECTION.get()))
        os.system(cmd_i)
    def run_cmd_open_libs(self):
        cmd_i=open_cmd+" libs"
        os.system(cmd_i)        
    # def run_cmd_scripts(self):
    #     if os.path.exists(self.USER_SELECTION2.get()):
    #         cmd_i=open_cmd+" {}".format(self.USER_SELECTION2.get())
    #         os.system(cmd_i)
    def run_cmd_editpaths(self):
        cmd_i=open_cmd+" {}".format(self.list_script_path)
        os.system(cmd_i)

    def popupWindow_delete(self):

        self.USER=self.USER_SELECTION.get()
        self.USER=self.USER.strip()
        get_default_settings(self.USER)
        SAVED_SETTINGS_PATH=os.path.join('libs',self.USER)
        if os.path.exists(SAVED_SETTINGS_PATH+".py"):
            f=open(SAVED_SETTINGS_PATH+".py")
            f_read=f.readlines()
            f.close()
            path_of_interest='None'
            for line in f_read:
                if line.find('YOLO_MODEL_PATH')!=-1:                   
                    path_of_interest=line.split('=')[1].replace('\n','').strip().replace('r"','"').replace("r'","'").replace("'","").replace('"',"")

        #self.delete_note1=tk.Label(self.top,text='Are you sure you want to delete all associated files with:',bg=self.root_bg,fg=self.root_fg,font=("Arial", 12))
        #self.delete_note1.grid(row=0,column=0,sticky='w')
        if path_of_interest!='None':
            delete_prompt="{}.py \n and \n {}".format(SAVED_SETTINGS_PATH,path_of_interest)
        else:
            delete_prompt="{}.py".format(SAVED_SETTINGS_PATH)
        self.top=tk.Toplevel(self.root)
        self.top.geometry( "{}x{}".format(int(self.root.winfo_screenwidth()*0.95//2.0),int(self.root.winfo_screenheight()*0.95//6.0)) )
        self.top.configure(background = 'black')
        self.top.title('Are you sure you want to delete all associated files with')
        self.delete_note2=tk.Label(self.top,text='{}'.format(delete_prompt),bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.delete_note2.grid(row=0,column=0,columnspan=5,sticky='ne')
        self.b=Button(self.top,text='Yes',command=self.delete,bg=self.root_fg, fg=self.root_bg)
        self.b.grid(row=4,column=1,sticky='se')
        self.c=Button(self.top,text='No',command=self.cleanup,bg=self.root_bg, fg=self.root_fg)
        self.c.grid(row=4,column=2,sticky='se')

    def delete(self):
        self.USER=self.USER_SELECTION.get()
        self.USER=self.USER.strip()
        get_default_settings(self.USER)
        SAVED_SETTINGS_PATH=os.path.join('libs',self.USER)
        if os.path.exists(SAVED_SETTINGS_PATH+".py"):
            f=open(SAVED_SETTINGS_PATH+".py")
            f_read=f.readlines()
            f.close()
            for line in f_read:
                if line.find('YOLO_MODEL_PATH')!=-1:
                    
                    path_of_interest=line.split('=')[1].replace('\n','').strip().replace('r"','"').replace("r'","'").replace("'","").replace('"',"")
                    print('Found YOLO_MODEL_PATH: \n{} \n'.format(path_of_interest))
                    if os.path.exists(path_of_interest):
                        os.system('rm -rf {}'.format(path_of_interest))
                        print('Deleted all files located at {}'.format(path_of_interest))
        os.remove(SAVED_SETTINGS_PATH+".py")
        print('Deleted SAVED_SETTINGS_PATH: \n',SAVED_SETTINGS_PATH)
        self.cleanup()
        self.INITIAL_CHECK()

    def submit(self):
        global SAVED_SETTINGS_PATH
        global PROCEED 
        PROCEED=True
        self.USER=self.USER_SELECTION.get()
        self.USER=self.USER.strip()
        get_default_settings(self.USER)
        SAVED_SETTINGS_PATH=os.path.join('libs',self.USER)
        self.close()
    # def submit_script(self):
        
    #     if os.path.exists(self.USER_SELECTION2.get()):
    #         switch_basepath.switch_config(self.USER_SELECTION2.get().split('/')[-2])
    #         print('Executing')
    #         os.system('bash {}'.format(self.USER_SELECTION2.get()))
    #     else:
    #         print('This does not exist or is not a script \n {}'.format(self.USER_SELECTION2.get()))
    def get_update_background_img(self):
        self.image=Image.open(self.root_background_img)
        self.image=self.image.resize((self.root_W,self.root_H),Image.ANTIALIAS)
        self.bg=ImageTk.PhotoImage(self.image)
        self.canvas=tk.Canvas(self.root,width=self.root_W,height=self.root_H)
        self.canvas.grid(row=0,column=0,columnspan=self.canvas_columnspan,rowspan=self.canvas_rowspan,sticky='nw')
        self.canvas.create_image(0,0,image=self.bg,anchor='nw')
    def close(self):
        self.root.destroy()
    def cleanup(self):
        self.top.destroy()

class yolo_cfg:
    def __init__(self,root_tk,SAVED_SETTINGS_PATH):
        self.PREFIX=DEFAULT_SETTINGS.PREFIX
        self.HEIGHT_NUM=DEFAULT_SETTINGS.HEIGHT_NUM
        self.WIDTH_NUM=DEFAULT_SETTINGS.WIDTH_NUM
        self.num_div=DEFAULT_SETTINGS.num_div
        self.num_classes=DEFAULT_SETTINGS.num_classes
        self.darknet_path=DEFAULT_SETTINGS.darknet_path
        self.base_path_OG=DEFAULT_SETTINGS.base_path_OG
        self.mp4_video_path=DEFAULT_SETTINGS.mp4_video_path
        self.path_JPEGImages=DEFAULT_SETTINGS.path_JPEGImages
        self.path_Annotations=DEFAULT_SETTINGS.path_Annotations
        self.path_Yolo=DEFAULT_SETTINGS.path_Yolo
        self.TRAIN_SPLIT=DEFAULT_SETTINGS.TRAIN_SPLIT
        self.MODEL_PATHS=DEFAULT_SETTINGS.MODEL_PATHS
        self.increment=DEFAULT_SETTINGS.increment #1000 #increment the df.pkl files for yolo
        self.DEFAULT_ENCODING=DEFAULT_SETTINGS.DEFAULT_ENCODING
        self.XML_EXT=DEFAULT_SETTINGS.XML_EXT
        self.JPG_EXT=DEFAULT_SETTINGS.JPG_EXT
        self.COLOR=DEFAULT_SETTINGS.COLOR
        self.destination_list_file="destination_list.txt"
        self.PHONE_VAR=tk.StringVar()
        self.sec=tk.StringVar()
        self.sec.set('n')
        self.path_JPEGImages_CUSTOM=self.path_JPEGImages
        self.path_Annotations_CUSTOM=self.path_Annotations
        self.open_anno_label_var_CUSTOM=tk.StringVar()
        self.open_anno_label_var_CUSTOM.set(self.path_Annotations_CUSTOM)
        self.open_jpeg_label_var_CUSTOM=tk.StringVar()
        self.open_jpeg_label_var_CUSTOM.set(self.path_JPEGImages_CUSTOM)


        if os.path.exists(self.destination_list_file):
            self.load_destination_list()
        else:
            self.destination_list=["XXXYYYZZZZ@mms.att.net"]
            f=open(self.destination_list_file,'w')
            tmp=[f.writelines(w+'\n') for w in self.destination_list]
            f.close()
        self.phone_dic_trigger={}
        self.phone_dic_trigger_var={}
        self.sleep_time_chips_VAR=tk.StringVar()
        self.sleep_time_chips_VAR.set('30')
        print('SAVED SETTINGS PATH: \n',SAVED_SETTINGS_PATH)
        try:
            self.ITERATION_NUM=DEFAULT_SETTINGS.ITERATION_NUM
        except:
            self.ITERATION_NUM=2000
        print('ITERATION_NUM={}\n'.format(self.ITERATION_NUM))
        try:
            self.epochs_yolov7=DEFAULT_SETTINGS.epochs_yolov7
        except:
            self.epochs_yolov7=40
        print('epochs_yolov7={}\n'.format(self.epochs_yolov7))
        try:
            self.epochs_yolov7_re=DEFAULT_SETTINGS.epochs_yolov7_re
        except:
            self.epochs_yolov7_re=40
        print('epochs_yolov7_re={}\n'.format(self.epochs_yolov7_re))
        try:
            self.epochs_yolov7_e6e=DEFAULT_SETTINGS.epochs_yolov7_e6e
        except:
            self.epochs_yolov7_e6e=40
        print('epochs_yolov7_e6e={}\n'.format(self.epochs_yolov7_e6e))
        self.epochs_yolov7_e6e_VAR=tk.StringVar()
        self.epochs_yolov7_e6e_VAR.set(self.epochs_yolov7_e6e)
        
        self.epochs_yolov7_VAR=tk.StringVar()
        self.epochs_yolov7_VAR.set(self.epochs_yolov7)

        self.epochs_yolov7_re_VAR=tk.StringVar()
        self.epochs_yolov7_re_VAR.set(self.epochs_yolov7_re)
        self.RTSP_CLIENT=True
        self.USE_RTSP_CLIENT_VAR=None
        if os.path.exists('resources/rtsp_server.py'):
            self.RTSP_SERVER_PATH=os.path.abspath('resources/rtsp_server.py')
            self.RTSP_SERVER=True 
            
            try:
                s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
                s.connect(("8.8.8.8",80))
                self.IP_ADDRESS=s.getsockname()[0]
            except:
                self.IP_ADDRESS="127.0.0.1"
            self.PORT=8554
            self.PORT_VAR=None
            self.FPS=10 #RTSP can be kind of slow
            self.FPS_VAR=None
            self.W_RTSP=self.WIDTH_NUM
            self.H_RTSP=self.HEIGHT_NUM
            self.STREAM_KEY="/video_stream"
            self.STREAM_KEY_VAR=None
            self.USE_RTSP_VAR=None
            self.RTSP_FULL_PATH_VAR=None
        else:
            self.RTSP_SERVER=False
        if os.path.exists('YOUTUBE_KEY.txt'):
            f=open('YOUTUBE_KEY.txt')
            f_read=f.readlines()
            f.close()
            self.YOUTUBE_KEY=f_read[0].strip()
        else:
            self.YOUTUBE_KEY="xxxx-xxxx-xxxx-xxxx-xxxx"
        self.YOUTUBE_KEY_VAR=None
        self.DNN_PATH=os.path.join(os.getcwd(),"resources/yolo_dnn_multi_drone_hdmi.py")
        self.THRESH=0.5 #default threshold for Yolo
        self.IOU_THRESH=0.5 #default IOU Threshold for Yolo
        if os.path.basename(SAVED_SETTINGS_PATH).find('_r1_')==-1:
            self.random='0' #default for cfg 
        else:
            self.random='1'
        self.SAVED_SETTINGS_PATH=SAVED_SETTINGS_PATH
        self.root=root_tk
        self.root.bind('<Escape>',self.close)
        self.root.wm_iconphoto(False,ImageTk.PhotoImage(Image.open("resources/icons/appY.png")))
        self.icon_breakup=ImageTk.PhotoImage(Image.open('resources/icons/breakup.png'))
        self.icon_folder=ImageTk.PhotoImage(Image.open("resources/icons/open_folder_search.png"))
        self.icon_single_file=ImageTk.PhotoImage(Image.open('resources/icons/single_file.png'))
        self.icon_load=ImageTk.PhotoImage(Image.open('resources/icons/load.png'))
        self.icon_create=ImageTk.PhotoImage(Image.open('resources/icons/create.png'))
        self.icon_analyze=ImageTk.PhotoImage(Image.open('resources/icons/analyze.png'))
        self.icon_move=ImageTk.PhotoImage(Image.open('resources/icons/move.png'))
        self.icon_labelImg=ImageTk.PhotoImage(Image.open('resources/icons/labelImg.png'))
        self.icon_map=ImageTk.PhotoImage(Image.open('resources/icons/map.png'))
        self.icon_merge=ImageTk.PhotoImage(Image.open('resources/icons/merge.png'))
        self.icon_clear_fix=ImageTk.PhotoImage(Image.open('resources/icons/clear.png'))
        self.icon_clear_checked=ImageTk.PhotoImage(Image.open('resources/icons/clear_checked.png'))
        self.icon_save_settings=ImageTk.PhotoImage(Image.open('resources/icons/save.png'))
        self.icon_yolo_objects=ImageTk.PhotoImage(Image.open('resources/icons/yolo_objects.png'))
        self.icon_divide=ImageTk.PhotoImage(Image.open('resources/icons/divide.png'))
        self.icon_scripts=ImageTk.PhotoImage(Image.open('resources/icons/scripts.png'))
        self.icon_config=ImageTk.PhotoImage(Image.open('resources/icons/config.png'))
        self.icon_open=ImageTk.PhotoImage(Image.open('resources/icons/open_folder.png'))
        self.icon_train=ImageTk.PhotoImage(Image.open('resources/icons/train.png'))
        self.icon_test_mp4=ImageTk.PhotoImage(Image.open('resources/icons/test_mp4.png'))
        self.icon_test_images=ImageTk.PhotoImage(Image.open('resources/icons/test_images.png'))
        self.icon_test=ImageTk.PhotoImage(Image.open('resources/icons/test.png'))
        self.icon_MOSAIC=ImageTk.PhotoImage(Image.open('resources/icons/appM_icon.png'))
        self.icon_IMGAUG=ImageTk.PhotoImage(Image.open('resources/icons/appI_icon.png'))    
        self.icon_CLASSIFY_CHIPS=ImageTk.PhotoImage(Image.open('resources/icons/appC_icon.png'))    

        self.root_H=int(self.root.winfo_screenheight()*0.95)
        self.root_W=int(self.root.winfo_screenwidth()*0.95)
        self.root.geometry(str(self.root_W)+'x'+str(self.root_H))
        self.root.title(os.path.basename(self.SAVED_SETTINGS_PATH))
        #self.root.title("Full-Loop YOLO")
        self.root_bg=DEFAULT_SETTINGS.root_bg#'black'
        self.root_fg=DEFAULT_SETTINGS.root_fg#'lime'
        self.canvas_columnspan=DEFAULT_SETTINGS.canvas_columnspan
        self.canvas_rowspan=DEFAULT_SETTINGS.canvas_rowspan
        self.root_background_img=DEFAULT_SETTINGS.root_background_img #r"misc/gradient_blue.jpg"
        self.get_update_background_img()
        self.root.configure(bg=self.root_bg)
        self.drop_targets=None
        self.CWD=os.getcwd()
        self.img_list_path=None
        self.path_predJPEGImages=None
        self.path_MOVMP4=None
        self.MOVMP4_selected=False
        self.SOCKET_PREFIX='top'
        self.CLASSIFY_CHIPS_SETTINGS_PATH_VAR=tk.StringVar()
        self.CLASSIFY_CHIPS_SETTINGS_PATH_VAR.set('None')
        self.CLASSIFY_CHIPS_LOGIC=tk.StringVar()
        self.CLASSIFY_CHIPS_LOGIC.set('No')
        self.USE_CLASSIFY_CHIPS_VAR=tk.StringVar()
        self.USE_CLASSIFY_CHIPS_VAR.set('None')
        self.MAX_PER_CLASS_VAR=tk.StringVar()
        self.MAX_PER_CLASS=500
        self.MAX_PER_CLASS_VAR.set(self.MAX_PER_CLASS)
        self.TARGET_LIST_VAR=tk.StringVar()
        self.TARGET_LIST='car;truck;van;'
        self.TARGET_LIST_VAR.set(self.TARGET_LIST)
        # self.root.withdraw()
        # self.top=tk.Toplevel(self.root,width=300,height=300)
        # self.canvas_generate=tk.Canvas(self.top,bg='white')
        # self.canvas_generate.pack(expand=tk.YES,fill=tk.BOTH)
        # self.top.destroy()
        # self.root.deiconify()

        self.remaining_buttons_clicked=True


    #Buttons
        self.basepath_now_selected=False
        self.save_cfg_path_train_selected=False
        self.save_cfg_path_test_selected=False
        self.basepath_selected=False
        self.darknet_selected=False
        self.backup_models_selected=False
        self.data_path_selected=False
        self.mp4_selected=False
        self.open_anno_selected=False
        self.open_anno_selected_CUSTOM=False
        self.open_jpeg_selected_CUSTOM=False


        self.open_darknet_label_var=None
        self.dropdowntests=None

        self.save_settings_button=Button(self.root,image=self.icon_save_settings,command=self.save_settings,bg=self.root_bg,fg=self.root_fg)
        self.save_settings_button.grid(row=1,column=4,sticky='se')
        self.save_settings_note=tk.Label(self.root,text='Save Settings',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.save_settings_note.grid(row=2,column=4,sticky='ne')

        self.save_custom_settings_button=Button(self.root,image=self.icon_save_settings,command=self.save_settings_CUSTOM,bg=self.root_bg,fg=self.root_fg)
        self.save_custom_settings_button.grid(row=1,column=3,sticky='se')
        self.save_custom_settings_note=tk.Label(self.root,text='Save Custom Settings',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.save_custom_settings_note.grid(row=2,column=3,sticky='ne')

        self.return_to_main_button=Button(self.root,text='Return to Main Menu',command=self.return_to_main,bg=self.root_fg,fg=self.root_bg)
        self.return_to_main_button.grid(row=0,column=0,sticky='se')
        # self.return_to_main_note=tk.Label(self.root,text='Main Menu',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        # self.return_to_main_note.grid(row=1,column=1,sticky='nw')




        self.open_basepath_label_var=tk.StringVar()
        self.open_basepath_label_var.set(self.base_path_OG)
        self.open_basepath_button=Button(self.root,image=self.icon_folder,command=partial(self.select_folder,self.base_path_OG,'save path',self.open_basepath_label_var),bg=self.root_bg,fg=self.root_fg)
        self.open_basepath_button.grid(row=1,column=5,sticky='se')
        self.open_basepath_note=tk.Label(self.root,text="base_path_OG dir",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_basepath_note.grid(row=2,column=5,sticky='ne')
        cmd_i=open_cmd+" '{}'".format(self.open_basepath_label_var.get())
        self.open_basepath_label=Button(self.root,textvariable=self.open_basepath_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.open_basepath_label.grid(row=1,column=6,columnspan=50,sticky='sw')

        self.PREFIX_VAR=tk.StringVar()
        self.PREFIX_VAR.set(self.PREFIX)
        self.PREFIX_entry=tk.Entry(self.root,textvariable=self.PREFIX_VAR)
        self.PREFIX_entry.grid(row=7,column=0,sticky='se')
        self.PREFIX_label=tk.Label(self.root,text='PREFIX',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.PREFIX_label.grid(row=8,column=0,sticky='ne')

        self.WIDTH_NUM_VAR=tk.StringVar()
        self.WIDTH_NUM_VAR.set(self.WIDTH_NUM)
        self.WIDTH_NUM_entry=tk.Entry(self.root,textvariable=self.WIDTH_NUM_VAR)
        self.WIDTH_NUM_entry.grid(row=9,column=0,sticky='se')
        self.WIDTH_NUM_label=tk.Label(self.root,text='WIDTH',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.WIDTH_NUM_label.grid(row=10,column=0,sticky='ne')

        self.HEIGHT_NUM_VAR=tk.StringVar()
        self.HEIGHT_NUM_VAR.set(self.HEIGHT_NUM)
        self.HEIGHT_NUM_entry=tk.Entry(self.root,textvariable=self.HEIGHT_NUM_VAR)
        self.HEIGHT_NUM_entry.grid(row=11,column=0,sticky='se')
        self.HEIGHT_NUM_label=tk.Label(self.root,text='HEIGHT',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.HEIGHT_NUM_label.grid(row=12,column=0,sticky='ne')

        self.num_div_VAR=tk.StringVar()
        self.num_div_VAR.set(self.num_div)
        self.num_div_entry=tk.Entry(self.root,textvariable=self.num_div_VAR)
        self.num_div_entry.grid(row=13,column=0,sticky='se')
        self.num_div_label=tk.Label(self.root,text='num_div',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.num_div_label.grid(row=14,column=0,sticky='ne')

        self.num_classes_VAR=tk.StringVar()
        self.num_classes_VAR.set(self.num_classes)
        self.num_classes_entry=tk.Entry(self.root,textvariable=self.num_classes_VAR)
        self.num_classes_entry.grid(row=15,column=0,sticky='se')
        self.num_classes_label=tk.Label(self.root,text='num_classes',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.num_classes_label.grid(row=16,column=0,sticky='ne')

        self.random_VAR=tk.StringVar()
        self.random_VAR.set(self.random)
        self.random_options=['0','1']
        self.random_dropdown=tk.OptionMenu(self.root,self.random_VAR,*self.random_options)
        self.random_dropdown.grid(row=17,column=0,sticky='se')       
        self.random_label=tk.Label(self.root,text='random',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.random_label.grid(row=18,column=0,sticky='ne')

        self.ITERATION_NUM_VAR=tk.StringVar()
        self.ITERATION_NUM_VAR.set(self.ITERATION_NUM)
        self.ITERATION_entry=tk.Entry(self.root,textvariable=self.ITERATION_NUM_VAR)
        self.ITERATION_entry.grid(row=19,column=0,sticky='se')
        self.ITERATION_label=tk.Label(self.root,text='max_batches',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.ITERATION_label.grid(row=20,column=0,sticky='ne')

        self.generate_cfg_button=Button(self.root,image=self.icon_config,command=self.generate_cfg,bg=self.root_bg,fg=self.root_fg)
        self.generate_cfg_button.grid(row=3,column=0,sticky='s')
        self.generate_cfg_note=tk.Label(self.root,text='1.b \n Generate Yolo \n Configs (.cfgs)',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.generate_cfg_note.grid(row=4,column=0,sticky='n')

        self.load_cfg_button=Button(self.root,image=self.icon_config,command=self.load_cfg,bg=self.root_bg,fg=self.root_fg)
        self.load_cfg_button.grid(row=1,column=0,sticky='s')
        self.load_cfg_note=tk.Label(self.root,text='1.a \n Load Yolo \n Configs (.cfgs)',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.load_cfg_note.grid(row=2,column=0,sticky='n')

        self.open_anno_label_var=tk.StringVar()
        self.open_anno_label_var.set(self.path_Annotations)

        self.open_jpeg_label_var=tk.StringVar()
        self.open_jpeg_label_var.set(self.path_JPEGImages)

        self.open_predjpeg_label_var=tk.StringVar()
        self.open_predjpeg_label_var.set(self.path_predJPEGImages)

        self.open_yolo_label_var=tk.StringVar()
        if os.path.exists(self.path_Yolo)==False:
            os.makedirs(self.path_Yolo)
        self.open_yolo_label_var.set(self.path_Yolo)

        self.var_yolo_choice=tk.StringVar()
        if self.PREFIX.find('tiny')!=-1:
            self.var_yolo_choice.set('Yolov4-tiny')
        elif self.PREFIX.find('regular')!=-1:
            self.var_yolo_choice.set('Yolov4')
        else:
            print('Did not find the write PREFIX \n FOUND: \t ',self.PREFIX)
        self.style3=ttk.Style()
        self.style3.configure('Normal.TRadiobutton',
                             background='green',
                             foreground='black')
        self.button_yolo_tiny=ttk.Radiobutton(text='Yolov4-tiny',style='Normal.TRadiobutton',variable=self.var_yolo_choice,value='Yolov4-tiny')

        self.button_yolo_tiny.grid(row=5,column=0,stick='se')
        self.button_yolo_regular=ttk.Radiobutton(text='Yolov4',style='Normal.TRadiobutton',variable=self.var_yolo_choice,value='Yolov4')
        self.button_yolo_regular.grid(row=6,column=0,stick='ne')

        self.RECORDRAW_BUTTONS()
        self.GENERATE_CUSTOM_DATASET_BUTTONS()
        self.COMPUTE_METRICS_BUTTONS()

    def COMPUTE_METRICS_BUTTONS(self):
        self.popup_metrics_buttons=Button(self.root,text='Compute Metrics',command=self.popupWindow_mAP,bg=self.root_fg,fg=self.root_bg)
        self.popup_metrics_buttons.grid(row=0,column=7,sticky='sw')

    def TEST_BUTTONS(self):
        self.popup_TEST_button=Button(self.root,text='TEST Script Buttons',command=self.popupWindow_TEST,bg=self.root_fg,fg=self.root_bg)
        self.popup_TEST_button.grid(row=13,column=2,sticky='sw')

    def RECORDRAW_BUTTONS(self):
        self.popup_RECORDRAW_button=Button(self.root,text='RECORD RAW VIDEO Buttons',command=self.popupWindow_RECORD_RAW,bg=self.root_fg,fg=self.root_bg)
        self.popup_RECORDRAW_button.grid(row=0,column=2,sticky='sw')

    def TRAIN_BUTTONS(self):
        self.popup_TRAIN_button=Button(self.root,text='TRAIN Script Buttons',command=self.popupWindow_TRAIN,bg=self.root_fg,fg=self.root_bg)
        self.popup_TRAIN_button.grid(row=12,column=2,sticky='sw')

    def SHOWTABLE_BUTTONS(self):
        self.popup_SHOWTABLE_button=Button(self.root,text='Show df',command=self.popupWindow_showtable,bg=self.root_fg,fg=self.root_bg)
        self.popup_SHOWTABLE_button.grid(row=14,column=2,sticky='sw')

    def return_to_main(self):
        global return_to_main
        return_to_main=True
        self.root.destroy()

    def select_file_validlist(self):
        filetypes=(('txt','*.txt'),('All files','*.*'))
        if os.path.exists(self.path_JPEGS_GT_var.get()):
            initialdir_i=self.path_JPEGS_GT_var.get()
        elif os.path.exists(self.YOLO_MODEL_PATH):
            initialdir_i=self.YOLO_MODEL_PATH
        else:
            initialdir_i=os.getcwd()
        self.filename=fd.askopenfilename(title='Open a file',
                                    initialdir=initialdir_i,
                                    filetypes=filetypes)
        if os.path.exists(self.filename):
            print(self.filename)
            self.valid_list_var.set(self.filename)
        showinfo(title='Selected File',

                 message=self.filename)

    def select_file_objnames(self):
        filetypes=(('names','*.names'),('All files','*.*'))
        if os.path.exists(self.obj_names_path_var.get()):
            initialdir_i=os.path.dirname(self.obj_names_path_var.get())
        elif os.path.exists(self.YOLO_MODEL_PATH):
            initialdir_i=self.YOLO_MODEL_PATH
        else:
            initialdir_i=os.getcwd()
        self.filename=fd.askopenfilename(title='Open a file',
                                    initialdir=initialdir_i,
                                    filetypes=filetypes)
        if os.path.exists(self.filename):
            print(self.filename)
            self.obj_names_path_var.set(self.filename)
        showinfo(title='Selected File',
                 message=self.filename)

    def select_file_optionobjnames(self):
        filetypes=(('names','*.names'),('All files','*.*'))
        if os.path.exists(self.option_obj_names_path_var.get()):
            initialdir_i=os.path.dirname(self.option_obj_names_path_var.get())
        elif os.path.exists(self.YOLO_MODEL_PATH):
            initialdir_i=self.YOLO_MODEL_PATH
        else:
            initialdir_i=os.getcwd()
        self.filename=fd.askopenfilename(title='Open a file',
                                    initialdir=initialdir_i,
                                    filetypes=filetypes)
        if os.path.exists(self.filename):
            print(self.filename)
            self.option_obj_names_path_var.set(self.filename)
        showinfo(title='Selected File',
                 message=self.filename)

    def select_file_mp4(self,file_i):
        filetypes=(('mp4','*.mp4'),('All files','*.*'))
        if os.path.exists(self.mp4_video_path):
            initialdir_i=self.mp4_video_path
        else:
            initialdir_i=os.getcwd()
        self.filename=fd.askopenfilename(title='Open a file',
                                    initialdir=initialdir_i,
                                    filetypes=filetypes)
        if os.path.exists(self.filename):
            print(self.filename)
            self.mp4_video_path=self.filename
            self.open_mp4_label_var.set(self.mp4_video_path)
            self.open_mp4()
        showinfo(title='Selected File',
                 message=self.filename)

    def select_file_weights(self,file_i):
        filetypes=(('weights','*.weights'),('All files','*.*'))
        if os.path.exists(self.best_weights_path):
            initialdir_i=self.best_weights_path
        else:
            initialdir_i=os.getcwd()
        self.filename=fd.askopenfilename(title='Open a file',
                                    initialdir=initialdir_i,
                                    filetypes=filetypes)
        if os.path.exists(self.filename):
            print(self.filename)
            self.best_weights_path=self.filename
            self.open_bestweight_label_var.set(self.best_weights_path)
            self.open_bestweight()
        showinfo(title='Selected File',
                 message=self.filename)

    def select_file_testcfg(self,file_i):
        filetypes=(('test cfg','*.cfg'),('All files','*.*'))
        if os.path.exists(self.testcfg_path):
            initialdir_i=self.testcfg_path
        else:
            initialdir_i=os.getcwd()
        self.filename=fd.askopenfilename(title='Open a file',
                                    initialdir=initialdir_i,
                                    filetypes=filetypes)
        if os.path.exists(self.filename):
            print(self.filename)
            self.testcfg_path=self.filename
            self.open_testcfg_label_var.set(self.testcfg_path)
            self.open_testcfg()
            self.open_testobjdata()
        showinfo(title='Selected File',
                 message=self.filename)

    def select_file_MOVMP4(self,file_i):
        filetypes=(('MOV','*.MOV'),('MP4','*.MP4'),('mp4','*.mp4'),('All files','*.*'))
        if os.path.exists(self.path_MOVMP4):
            initialdir_i=os.path.dirname(self.path_MOVMP4)
        else:
            initialdir_i=os.getcwd()
        self.filename=fd.askopenfilename(title='Open a file',
                                    initialdir=initialdir_i,
                                    filetypes=filetypes)
        if os.path.exists(self.filename):
            print(self.filename)
            self.path_MOVMP4=self.filename
            self.open_MOVMP4_label_var.set(self.path_MOVMP4)
            self.open_MOVMP4()
        showinfo(title='Selected File',
                 message=self.filename)

    def select_file_testobjdata(self,file_i):
        filetypes=(('test obj.data','*.data'),('All files','*.*'))
        if os.path.exists(self.testobjdata_path):
            initialdir_i=self.testobjdata_path
        else:
            initialdir_i=os.getcwd()
        self.filename=fd.askopenfilename(title='Open a file',
                                    initialdir=initialdir_i,
                                    filetypes=filetypes)
        if os.path.exists(self.filename):
            print(self.filename)
            self.testobjdata_path=self.filename
            self.open_testobjdata_label_var.set(self.testobjdata_path)
            self.open_testobjdata()
        showinfo(title='Selected File',
                 message=self.filename)

    def select_folder(self,folder_i,title_i,var_i=None):
            filetypes=(('All files','*.*'))
            if var_i:
                if self.open_anno_label_var_CUSTOM and self.open_jpeg_label_var_CUSTOM:
                    if var_i==self.open_anno_label_var_CUSTOM:
                        if self.open_anno_label_var_CUSTOM.get()==self.path_Annotations:
                            self.YOLO_MODEL_PATH=os.path.join(self.base_path_OG,self.prefix_foldername)
                            initialdir=self.YOLO_MODEL_PATH
                        else:
                            initialdir=os.path.dirname(self.path_Annotations_CUSTOM)
                        folder_i=var_i.get()
                    elif var_i==self.open_jpeg_label_var_CUSTOM:
                        if self.open_jpeg_label_var_CUSTOM.get()==self.path_JPEGImages:
                            self.YOLO_MODEL_PATH=os.path.join(self.base_path_OG,self.prefix_foldername)
                            initialdir=self.YOLO_MODEL_PATH
                        else:
                            initialdir=os.path.dirname(self.path_JPEGImages_CUSTOM)
                        folder_i=var_i.get()
                    else:
                        folder_i=var_i.get() 
                        initialdir=folder_i 
                else:
                    folder_i=var_i.get() 
                    initialdir=folder_i 
                           
            if os.path.exists(folder_i):
                self.foldername=fd.askdirectory(title=title_i,
                                            initialdir=initialdir)
            else:
                self.foldername=fd.askdirectory(title=title_i)
            if self.foldername=='' or len(self.foldername)==0:
                showinfo(title='NOT FOUND! Using previous path',
                        message=self.foldername)
                if var_i==self.open_yolo_label_var:
                    #self.convert_PascalVOC_to_YOLO()
                    pass
            elif self.foldername!='' and len(self.foldername)!=0:
                showinfo(title='Selected Folder',
                    message=self.foldername)
                folder_i=self.foldername
                if var_i==self.open_darknet_label_var:
                    self.darknet_selected=True
                    var_i.set(folder_i)
                    self.open_darknet_label.destroy()
                    del self.open_darknet_label
                    cmd_i=open_cmd+" '{}'".format(self.open_darknet_label_var.get())
                    self.open_darknet_label=Button(self.root,textvariable=self.open_darknet_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
                    self.open_darknet_label.grid(row=11,column=5,columnspan=50,sticky='sw')
                    self.darknet_path=self.foldername
                    print(self.path_darknet)
                if var_i==self.open_basepath_label_var:
                    self.basepath_selected=True
                    var_i.set(folder_i)
                    self.open_basepath_label.destroy()
                    del self.open_basepath_label
                    cmd_i=open_cmd+" '{}'".format(self.open_basepath_label_var.get())
                    self.open_basepath_label=Button(self.root,textvariable=self.open_basepath_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
                    self.open_basepath_label.grid(row=1,column=6,columnspan=50,sticky='sw')
                    self.base_path_OG=self.foldername
                    print(self.base_path_OG)  

                if var_i==self.open_anno_label_var:
                    self.anno_selected=True
                    var_i.set(folder_i)
                    self.open_anno_label.destroy()
                    del self.open_anno_label
                    cmd_i=open_cmd+" '{}'".format(self.open_anno_label_var.get())
                    self.open_anno_label=Button(self.root,textvariable=self.open_anno_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
                    self.open_anno_label.grid(row=11,column=5,columnspan=50,sticky='sw')
                    self.path_Annotations=self.foldername
                    print(self.path_Annotations)

                if var_i==self.open_jpeg_label_var:
                    self.jpeg_selected=True
                    var_i.set(folder_i)
                    self.open_jpeg_label.destroy()
                    del self.open_jpeg_label
                    cmd_i=open_cmd+" '{}'".format(self.open_jpeg_label_var.get())
                    self.open_jpeg_label=Button(self.root,textvariable=self.open_jpeg_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
                    self.open_jpeg_label.grid(row=13,column=5,columnspan=50,sticky='sw')
                    self.path_JPEGImages=self.foldername
                    print(self.path_JPEGImages)  

                if var_i==self.open_predjpeg_label_var:
                    var_i.set(folder_i)
                    self.open_predjpeg_label.destroy()
                    del self.open_predjpeg_label
                    cmd_i=open_cmd+" '{}'".format(self.open_predjpeg_label_var.get())
                    self.open_predjpeg_label=Button(self.root,textvariable=self.open_predjpeg_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
                    self.open_predjpeg_label.grid(row=10+11,column=5,columnspan=50,sticky='sw')
                    self.path_predJPEGImages=self.foldername
                    print(self.path_predJPEGImages)   
                    create_img_list.create_img_list(self.path_predJPEGImages)
                    if os.path.exists(os.path.join(self.path_predJPEGImages,'img_list.txt')):
                        self.img_list_path=os.path.join(self.path_predJPEGImages,'img_list.txt')
                    else:
                        print('no img_list.txt here')
                    self.convert_PascalVOC_to_YOLO_VALID()


                if var_i==self.open_yolo_label_var:
                    self.yolo_selected=True
                    var_i.set(folder_i)
                    self.open_yolo_label.destroy()
                    del self.open_yolo_label
                    cmd_i=open_cmd+" '{}'".format(self.open_yolo_label_var.get())
                    self.open_yolo_label=Button(self.root,textvariable=self.open_yolo_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
                    self.open_yolo_label.grid(row=15,column=5,columnspan=50,sticky='sw')
                    self.path_Yolo=self.foldername
                    print(self.path_Yolo)  
                
                if var_i==self.open_anno_label_var_CUSTOM:
                    self.anno_selected_CUSTOM=True
                    var_i.set(folder_i)
                    self.open_anno_label_CUSTOM.destroy()
                    del self.open_anno_label_CUSTOM
                    cmd_i=open_cmd+" '{}'".format(self.open_anno_label_var_CUSTOM.get())
                    self.open_anno_label_CUSTOM=Button(self.top,textvariable=self.open_anno_label_var_CUSTOM, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
                    self.open_anno_label_CUSTOM.grid(row=3,column=5,columnspan=50,sticky='sw')
                    self.path_Annotations_CUSTOM=self.foldername
                    print(self.path_Annotations_CUSTOM)
                    self.open_anno_CUSTOM()

                if var_i==self.open_jpeg_label_var_CUSTOM:
                    self.jpeg_selected_CUSTOM=True
                    var_i.set(folder_i)
                    self.open_jpeg_label_CUSTOM.destroy()
                    del self.open_jpeg_label_CUSTOM
                    cmd_i=open_cmd+" '{}'".format(self.open_jpeg_label_var_CUSTOM.get())
                    self.open_jpeg_label_CUSTOM=Button(self.top,textvariable=self.open_jpeg_label_var_CUSTOM, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
                    self.open_jpeg_label_CUSTOM.grid(row=4,column=5,columnspan=50,sticky='sw')
                    self.path_JPEGImages_CUSTOM=self.foldername
                    print(self.path_JPEGImages_CUSTOM)  
                    self.open_jpeg_CUSTOM()



    def update_paths(self,generate=True):
        self.PREFIX=str(self.PREFIX_VAR.get().strip())
        yolov4_choice=self.var_yolo_choice.get()
        if yolov4_choice.find('tiny')!=-1:
            self.PREFIX=self.PREFIX.replace('regular','tiny')
        elif yolov4_choice.find('regular')!=-1:
            self.PREFIX=self.PREFIX.replace('tiny','regular')
        self.PREFIX_VAR.set(self.PREFIX)
        try:
            self.WIDTH_NUM=int(self.WIDTH_NUM_VAR.get().strip())
        except:
            print('Sorry but this is an unacceptable value for WIDTH_NUM = {}'.format(self.WIDTH_NUM_VAR.get()))
            self.WIDTH_NUM_VAR.set(self.WIDTH_NUM)
        try:
            self.HEIGHT_NUM=int(self.HEIGHT_NUM_VAR.get().strip())
        except:
            print('Sorry but this is an unacceptable value for HEIGHT_NUM = {}'.format(self.HEIGHT_NUM_VAR.get()))
            self.HEIGHT_NUM_VAR.set(self.HEIGHT_NUM)
        if self.WIDTH_NUM%32!=0:
            print('Sorry but {} is not divisible by 32'.format(self.WIDTH_NUM))
            self.WIDTH_NUM=self.WIDTH_NUM-self.WIDTH_NUM%32 #YOLO must have divisible by 32 for Width and Height 
            print('Reduced value to {}'.format(self.WIDTH_NUM))
            self.WIDTH_NUM_VAR.set(self.WIDTH_NUM)
        if self.HEIGHT_NUM%32!=0:
            print('Sorry but {} is not divisible by 32'.format(self.HEIGHT_NUM))
            self.HEIGHT_NUM=self.HEIGHT_NUM-self.HEIGHT_NUM%32 #YOLO must have divisible by 32 for Width and Height 
            print('Reduced value to {}'.format(self.HEIGHT_NUM))
            self.HEIGHT_NUM_VAR.set(self.HEIGHT_NUM)
        try:
            self.num_div=int(self.num_div_VAR.get().strip())
        except:
            print('Sorry but this is an unacceptable value for num_div = {}'.format(self.num_div_VAR.get()))
            self.num_div_VAR.set(self.num_div)     

        if self.num_div>4:
            print('Sorry but this is an unacceptable value for num_div = {}'.format(self.num_div_VAR.get()))
            self.num_div_VAR.set('4')
            self.num_div=self.num_div_VAR.get()
      
        self.num_classes=int(self.num_classes_VAR.get().strip())
        self.random=self.random_VAR.get()
        self.ITERATION_NUM=self.ITERATION_NUM_VAR.get()
        if generate:
            self.update_height(self.HEIGHT_NUM)
            self.update_width(self.WIDTH_NUM)
            self.divide_filters_by(self.num_div)
            self.update_max_batches(self.ITERATION_NUM)
            self.update_classes()
        mp4_video_path='TBD' #testing video
        try:
            os.makedirs(self.MODEL_PATHS)
        except:
            pass
        if os.path.exists(self.base_path_OG):
            pass
            self.base_path_OG=self.base_path_OG
        else:
            self.base_path_OG=os.path.join(os.getcwd(),'tmp_cfgs')
        try:
            self.base_path_prior=self.base_path
            self.existing_weight_dirs=os.listdir(self.base_path)
            self.existing_weight_dirs=[os.path.join(self.base_path,w) for w in self.existing_weight_dirs]
            self.existing_weight_dirs=[w for w in self.existing_weight_dirs if os.path.isdir(w) and w.find('detections')==-1]
        except:
            self.base_path_prior='None'
            self.existing_weight_dirs=[]
        if self.random=='0':
            self.base_path=os.path.join(self.base_path_OG,'{}_w{}_h{}_d{}_c{}'.format(self.PREFIX,self.WIDTH_NUM,self.HEIGHT_NUM,self.num_div,self.num_classes))
        else:
            self.base_path=os.path.join(self.base_path_OG,'{}_w{}_h{}_d{}_c{}_r{}'.format(self.PREFIX,self.WIDTH_NUM,self.HEIGHT_NUM,self.num_div,self.num_classes,self.random)) 
        try:
            os.makedirs(self.base_path)
        except:
            pass

        if self.base_path_prior!='None' and self.base_path_prior!=self.base_path:
            for dir_i in tqdm(self.existing_weight_dirs):
                new_dirs_in_custom_model_path=os.listdir(self.base_path)
                new_dir_i_base=os.path.basename(dir_i)
                if new_dir_i_base not in new_dirs_in_custom_model_path:
                    print(f'SHUTIL COPYING dir_i={dir_i} to YOLO_MODEL_PATH_CUSTOM={self.base_path}')
                    shutil.copytree(dir_i,os.path.join(self.base_path,new_dir_i_base))
                else:

                    print(f'PASS ON SHUTIL COPYING dir_i={dir_i} to YOLO_MODEL_PATH_CUSTOM={self.base_path}, already exists')

        try:
            os.makedirs(os.path.join(self.base_path_OG,'temp'))
        except:
            pass
        self.data_path=os.path.join(self.base_path,'obj.data')
        self.train_list_path=os.path.join(self.base_path,'train.txt')
        self.valid_list_path=os.path.join(self.base_path,'valid.txt')
        self.prediction_list_path=os.path.join(self.base_path,'predictions.txt')
        self.names_path=os.path.join(self.base_path,'obj.names')
        self.backup_path=os.path.join(self.base_path,'backup_models')
        yolov4_choice=self.var_yolo_choice.get()
        if yolov4_choice.find('tiny')!=-1:
            self.tiny_conv29_path=os.path.join(self.darknet_path,"yolov4-tiny.conv.29")
        elif yolov4_choice.find('regular')!=-1:
            self.tiny_conv29_path=os.path.join(self.darknet_path,"yolov4.conv.137")
        if self.random=='0':
            self.prefix_foldername='{}_w{}_h{}_d{}_c{}'.format(self.PREFIX,self.WIDTH_NUM,self.HEIGHT_NUM,self.num_div,self.num_classes)
            self.save_cfg_path=os.path.join(self.base_path,'{}_w{}_h{}_d{}_c{}.cfg'.format(self.PREFIX,self.WIDTH_NUM,self.HEIGHT_NUM,self.num_div,self.num_classes))
        else:
            self.prefix_foldername='{}_w{}_h{}_d{}_c{}_r{}'.format(self.PREFIX,self.WIDTH_NUM,self.HEIGHT_NUM,self.num_div,self.num_classes,self.random)
            self.save_cfg_path=os.path.join(self.base_path,'{}_w{}_h{}_d{}_c{}_r{}.cfg'.format(self.PREFIX,self.WIDTH_NUM,self.HEIGHT_NUM,self.num_div,self.num_classes,self.random))
        
        self.best_weights_path=os.path.join(self.backup_path,os.path.basename(self.save_cfg_path.replace('.cfg',''))+'_train_best.weights')
        self.testcfg_path=self.save_cfg_path.replace('.cfg','_test.cfg')
        self.testobjdata_path=self.data_path
        if self.random=='0':
            self.model_i_path=os.path.join(self.MODEL_PATHS,'{}_w{}_h{}_d{}_c{}'.format(self.PREFIX,self.WIDTH_NUM,self.HEIGHT_NUM,self.num_div,self.num_classes))
        else:
            self.model_i_path=os.path.join(self.MODEL_PATHS,'{}_w{}_h{}_d{}_c{}_r{}'.format(self.PREFIX,self.WIDTH_NUM,self.HEIGHT_NUM,self.num_div,self.num_classes,self.random))
        switch_basepath.switch_config(self.prefix_foldername)
        if self.path_predJPEGImages==None:
            self.path_predJPEGImages=self.path_JPEGImages
        if self.path_MOVMP4==None:
            self.path_MOVMP4=self.path_JPEGImages
        self.save_settings()   
        self.root.title(os.path.basename(self.SAVED_SETTINGS_PATH))
    def get_header(self,cfg):
        self.net='[net]\n'
        self.Testing="# Testing\n"
        self.Testing_batch="#batch=1\n"
        self.Testing_subdivisions="#subdivisions=1\n"
        self.Training="# Training\n"
        self.batch=[w for w in cfg if w.find('batch=')!=-1 and w.find('#')==-1][0]
        self.subdivisions=[w for w in cfg if w.find('subdivisions=')!=-1 and w.find('#')==-1][0]
        self.width=[w for w in cfg if w.find('width=')!=-1][0]
        self.height=[w for w in cfg if w.find('height=')!=-1][0]
        self.channels=[w for w in cfg if w.find('channels=')!=-1][0]
        self.momentum=[w for w in cfg if w.find('momentum=')!=-1][0]
        self.decay=[w for w in cfg if w.find('decay=')!=-1][0]
        self.angle=[w for w in cfg if w.find('angle=')!=-1][0]
        self.saturation=[w for w in cfg if w.find('saturation=')!=-1][0]
        self.exposure=[w for w in cfg if w.find('exposure=')!=-1][0]
        self.hue=[w for w in cfg if w.find('hue=')!=-1][0]
        self.learning_rate=[w for w in cfg if w.find('learning_rate=')!=-1][0]
        self.burn_in=[w for w in cfg if w.find('burn_in=')!=-1][0]
        self.max_batches=[w for w in cfg if w.find('max_batches=')!=-1][0]
        self.policy=[w for w in cfg if w.find('policy=')!=-1][0]
        self.steps=[w for w in cfg if w.find('steps=')!=-1][0]
        self.scales=[w for w in cfg if w.find('scales=')!=-1][0]

        
    def count_layers(self,cfg):
        self.layers_dic={int(w.split('layer')[1].strip()):i for i,w in enumerate(cfg) if w.find('#layer')!=-1}
        self.layers={}
        for layer,line in self.layers_dic.items():
            start=line
            if layer!=max(list(self.layers_dic.keys())):
                end=self.layers_dic[layer+1]
            else:
                end=len(cfg)
            self.layers[layer]=cfg[start:end]
    def divide_filters_by(self,num):
        for layer,lines in self.layers.items():
            if "".join(lines).find('linear')==-1:
                self.layers[layer]=[w if w.find('filters')==-1 else w.replace(w.split('filters=')[1].strip(),str(int(w.split('filters=')[1].strip())/(2**num)))for w in lines]
    def update_classes(self):
        self.linear_filters=(self.num_classes+5)*3
        self.random=self.random_VAR.get()
        for layer,lines in self.layers.items():
            if "".join(lines).find('classes=')!=-1:
                self.layers[layer]=[w if w.find('classes')==-1 else w.replace(w.split('classes=')[1].strip(),str(self.num_classes))for w in lines]
            if "".join(lines).find('activation=linear')!=-1:
                self.layers[layer]=[w if w.find('filters')==-1 else w.replace(w.split('filters=')[1].strip(),str(int(self.linear_filters))) for w in lines]
    def update_height(self,height):
        self.height=self.height.split('=')[0]+'='+str(height)+'\n'
    def update_width(self,width):
        self.width=self.width.split('=')[0]+'='+str(width)+'\n'
    def update_max_batches(self,max_batches):
        self.max_batches=self.max_batches.split('=')[0]+'='+str(max_batches)+'\n'
        try:
            self.epochs=int(self.max_batches.replace('\n','').strip().split('=')[1])//self.iterations_per_epoch
        except:
            self.epochs=40
    def load_cfg(self):
        self.update_paths(False)
        self.save_cfg_path_train=self.save_cfg_path.replace('.cfg','_train.cfg')
        self.save_cfg_path_test=self.save_cfg_path.replace('.cfg','_test.cfg')
        self.initial_buttons()
    def generate_cfg(self):
        #self.root.withdraw()
        #self.top=tk.Toplevel(self.root,width=300,height=300)
        #self.canvas_generate=tk.Canvas(self.top,bg='white')
        #self.canvas_generate.pack(expand=tk.YES,fill=tk.BOTH)
        #self.top.destroy()
        #self.root.deiconify()
        yolov4_choice=self.var_yolo_choice.get()
        if yolov4_choice.find('tiny')!=-1:
            self.tiny_conv29_path=os.path.join(self.darknet_path,"yolov4-tiny.conv.29")
            self.cfg_vanilla_path=os.path.join('libs','custom-yolov4-tiny-detector.cfg')
        elif yolov4_choice.find('regular')!=-1:
            self.tiny_conv29_path=os.path.join(self.darknet_path,"yolov4.conv.137")
            self.cfg_vanilla_path=os.path.join('libs','yolov4-custom.cfg')
        
        #self.cfg_vanilla_path='libs/yolov4-custom.cfg'
        f=open(self.cfg_vanilla_path,'r')
        self.cfg_vanilla=f.readlines()
        f.close()
        self.count_layers(self.cfg_vanilla)
        self.get_header(self.cfg_vanilla)
        self.update_paths()
        
        #train_path
        self.save_cfg_path_train=self.save_cfg_path.replace('.cfg','_train.cfg')
        self.new_cfg=[
        self.net,
        self.Testing,
        self.Testing_batch,
        self.Testing_subdivisions,
        self.Training,
        self.batch,
        self.subdivisions,
        self.width,
        self.height,
        self.channels,
        self.momentum,
        self.decay,
        self.angle,
        self.saturation,
        self.exposure,
        self.hue,
        self.learning_rate,
        self.burn_in,
        self.max_batches,
        self.policy,
        self.steps,
        self.scales] 
        for i,layer in self.layers.items():
            self.new_cfg+=layer
        f=open(self.save_cfg_path.replace('.cfg','_train.cfg'),'w')
        new_lines=[]
        if self.random=='0':
            for line in self.new_cfg:
                if line.find('random=')!=-1:
                    line='random=0\n'
                else:
                    pass
                new_lines.append(line)
        else:
             for line in self.new_cfg:
                if line.find('random=')!=-1:
                    line='random=1\n'
                else:
                    pass
                new_lines.append(line)           
        self.new_cfg=new_lines
        [f.writelines(w) for w in self.new_cfg]
        f.close()
        print('generated cfg files for train: \n {}'.format(self.save_cfg_path_train))
        #test_path
        self.save_cfg_path_test=self.save_cfg_path.replace('.cfg','_test.cfg')
        self.new_test_cfg=[
        self.net,
        self.Testing,
        self.Testing_batch.replace('#',''),
        self.Testing_subdivisions.replace('#',''),
        self.Training,
        '#'+self.batch,
        '#'+self.subdivisions,
        self.width,
        self.height,
        self.channels,
        self.momentum,
        self.decay,
        self.angle,
        self.saturation,
        self.exposure,
        self.hue,
        self.learning_rate,
        self.burn_in,
        self.max_batches,
        self.policy,
        self.steps,
        self.scales] 
        for i,layer in self.layers.items():
            self.new_test_cfg+=layer
        f=open(self.save_cfg_path.replace('.cfg','_test.cfg'),'w')
        new_lines=[]
        if self.random=='0':
            for line in self.new_cfg:
                if line.find('random=')!=-1:
                    line='random=0\n'
                else:
                    pass
                new_lines.append(line)
        else:
             for line in self.new_cfg:
                if line.find('random=')!=-1:
                    line='random=1\n'
                else:
                    pass
                new_lines.append(line)           
        self.new_cfg=new_lines
        [f.writelines(w) for w in self.new_test_cfg]
        f.close()
        print('generated cfg files for test: \n {}'.format(self.save_cfg_path_test))
        self.initial_buttons()

    def show_numbers(self):
        self.load_destination_list()
        print(self.destination_list)
        self.popupWindow_phones()
    def cleanup_phones(self):
        try:
            sleep_time_i=self.sleep_time_chips_VAR.get()
            float(sleep_time_i) #prove it is able to be floating point
        except:
            print('Invalid sleep time of {}\n should be int/float'.format(sleep_time_i))
            self.sleep_time_chips_VAR.set('30')
        try:
            self.top.destroy()
        except:
            pass
    def popupWindow_phones(self):
        self.cleanup_phones()
        self.load_sender_credentials()
        self.top=tk.Toplevel(self.root)
        self.top.geometry("{}x{}".format(int(self.root.winfo_screenwidth()*0.95//1),int(self.root.winfo_screenheight()*0.95//1)))
        self.top.title('Phone Numbers/Emails to Send')
        self.top.configure(background='black')
        self.b=tk.Button(self.top,text='Close',command=self.cleanup,bg='green',fg='black')
        self.b.grid(row=0,column=0,stick='se')
        self.ADD_PHONE_entry=tk.Entry(self.top,textvariable=self.PHONE_VAR)
        self.ADD_PHONE_entry.grid(row=1,column=2,sticky='sw')
        self.ADD_PHONE_label=tk.Label(self.top,text='Add Phone Number for Alerts',bg=self.root_bg,fg=self.root_fg,font=('Arial',10))
        self.ADD_PHONE_label.grid(row=2,column=2,sticky='nw')
        self.ADD_PHONE_Button=tk.Button(self.top,text='Add',command=self.submit_number,bg=self.root_bg,fg=self.root_fg)
        self.ADD_PHONE_Button.grid(row=3,column=2,sticky='nw')
        self.label_note=tk.Label(self.top,text='{}'.format("Phone Numbers/Emails"),bg='black',fg='green',font=("Arial 20 underline"))
        self.label_note.grid(row=1,column=1,sticky='s')
        self.phone_dic_trigger={}
        self.phone_dic_trigger_var={}
        for i,phone_i in enumerate(self.destination_list):
            self.phone_dic_trigger_var[i]=tk.StringVar()
            self.phone_dic_trigger_var[i].set(phone_i)
            self.phone_dic_trigger[i]=tk.Checkbutton(self.top,text='{}'.format(phone_i),variable=self.phone_dic_trigger_var[i],onvalue=phone_i,offvalue='None',bg='white',fg='blue')
            self.phone_dic_trigger[i].grid(row=i+2,column=1,sticky='n')
        cmd_i=open_cmd+" '{}'".format(self.destination_list_file)
        self.open_phone=tk.Button(self.top,text='Open Phone/Email List',command=partial(self.run_cmd,cmd_i),bg='green',fg='black')
        self.open_phone.grid(row=1,column=3,stick='se')
        self.load_phone=tk.Button(self.top,text='Load Phone/Email List',command=self.show_numbers,bg='green',fg='black')
        self.load_phone.grid(row=1,column=4,stick='se')
        if os.path.exists(self.sender_list_file):
            cmd_i=open_cmd+" '{}'".format(self.sender_list_file)
            self.open_sender=tk.Button(self.top,text='Open Sender Credentials',command=partial(self.run_cmd,cmd_i),bg='green',fg='black')
            self.open_sender.grid(row=1,column=5,stick='se')
            self.load_sender=tk.Button(self.top,text='Load Sender Credentials',command=self.show_numbers,bg='green',fg='black')
            self.load_sender.grid(row=1,column=6,stick='se')
            self.sender_label_note=tk.Label(self.top,text='{}'.format("Sender"),bg='black',fg='green',font=("Arial 20 underline"))
            self.sender_label_note.grid(row=1,column=7,sticky='s')
            self.sender_From_Add=tk.Label(self.top,text=self.From_Add,bg='white',fg='blue',font=('Arial 10 bold'))
            self.sender_From_Add.grid(row=2,column=7,columnspan=3,stick='ne')
        self.sleep_time_entry=tk.Entry(self.top,textvariable=self.sleep_time_chips_VAR)
        self.sleep_time_entry.grid(row=7,column=2,sticky='sw')
        self.sleep_time_label=tk.Label(self.top,text='Time Between Alerts (seconds)',bg=self.root_bg,fg=self.root_fg,font=('Arial',10))
        self.sleep_time_label.grid(row=8,column=2,sticky='sw')
        self.update_sleep_Button=tk.Button(self.top,text='Update Sleep Time',command=self.sleep_time_chips_VAR.set(self.sleep_time_chips_VAR.get()),bg=self.root_bg,fg=self.root_fg)
        self.update_sleep_Button.grid(row=9,column=2,sticky='nw')

    def load_sender_credentials(self):
        self.sender_list_file='resources/EMAIL_INFO.py'
        self.From_Add='None'
        self.UserName='None'
        self.UserPassword='None'
        if os.path.exists(self.sender_list_file):
            f=open(self.sender_list_file,'r')
            f_read=f.readlines()
            f.close()
            for line in f_read:
                if line.find('From_Add')!=-1:
                    self.From_Add=line.split('=')[1].replace('\n','')
                if line.find('UserName')!=-1:
                    self.UserName=line.split('=')[1].replace('\n','')
                if line.find('UserPassword')!=-1:
                    self.UserPassword=line.split('=')[1].replace('\n','')

    def load_destination_list(self):
        f=open(self.destination_list_file,'r')
        f_read=f.readlines()
        f.close()
        self.destination_list=[w.replace('\n','') for w in f_read]   

    def submit_number(self):
        new_phone=self.PHONE_VAR.get()
        if new_phone not in self.destination_list:
            self.destination_list.append(new_phone)
        f=open(self.destination_list_file,'w')
        tmp=[f.writelines(w+'\n') for w in self.destination_list]
        f.close()
        self.popupWindow_phones()

    def initial_buttons(self):
        if self.basepath_now_selected==True:
            self.open_basepath_now_note.destroy()
            del self.open_basepath_now_note
        #     self.open_basepath_now_label.destroy()
        #     del self.open_basepath_now_label

        self.open_basepath_now_label_var=tk.StringVar()
        self.open_basepath_now_label_var.set(self.base_path)
        cmd_i=open_cmd+" '{}'".format(self.open_basepath_now_label_var.get())
        self.open_basepath_now_button=Button(self.root,image=self.icon_open,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
        self.open_basepath_now_button.grid(row=3,column=4,sticky='se')
        #sjs#self.open_basepath_now_note=tk.Label(self.root,text="{}".format(self.open_basepath_now_label_var.get()),bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        #sjs#self.open_basepath_now_note.grid(row=4,column=4,sticky='ne')
        self.open_basepath_now_note=tk.Label(self.root,text="{}".format('Scripts'),bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_basepath_now_note.grid(row=4,column=4,sticky='ne')
        # cmd_i=open_cmd+" '{}'".format(self.open_basepath_now_label_var.get())
        # self.open_basepath_now_label=Button(self.root,textvariable=self.open_basepath_now_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        # self.open_basepath_now_label.grid(row=3,column=5,columnspan=50,sticky='sw')
        self.basepath_now_selected=True

        if self.save_cfg_path_train_selected==True:
            self.open_save_cfg_path_train_note.destroy()
            del self.open_save_cfg_path_train_note
        #     self.open_save_cfg_path_train_label.destroy()
        #     del self.open_save_cfg_path_train_label

        self.open_save_cfg_path_train_label_var=tk.StringVar()
        self.open_save_cfg_path_train_label_var.set(self.save_cfg_path_train)
        cmd_i=open_cmd+" '{}'".format(self.open_save_cfg_path_train_label_var.get())
        self.open_save_cfg_path_train_button=Button(self.root,image=self.icon_single_file,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
        self.open_save_cfg_path_train_button.grid(row=5,column=4,sticky='se')
        #sjs#self.open_save_cfg_path_train_note=tk.Label(self.root,text="{}".format(self.open_save_cfg_path_train_label_var.get().split('/')[-1]),bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        #sjs#self.open_save_cfg_path_train_note.grid(row=6,column=4,sticky='ne')
        self.open_save_cfg_path_train_note=tk.Label(self.root,text="{}".format("train.cfg"),bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_save_cfg_path_train_note.grid(row=6,column=4,sticky='ne')
        cmd_i="netron '{}' -b".format(self.open_save_cfg_path_train_label_var.get())
        self.open_save_cfg_path_train_button_netron=Button(self.root,image=self.icon_map,command=partial(self.run_thread_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
        self.open_save_cfg_path_train_button_netron.grid(row=5,column=5,sticky='sw')
        # cmd_i=open_cmd+" '{}'".format(self.open_save_cfg_path_train_label_var.get())
        # self.open_save_cfg_path_train_label=Button(self.root,textvariable=self.open_save_cfg_path_train_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        # self.open_save_cfg_path_train_label.grid(row=5,column=5,columnspan=50,sticky='sw')
        self.save_cfg_path_train_selected=True

        if self.save_cfg_path_test_selected==True:
            self.open_save_cfg_path_test_note.destroy()
            del self.open_save_cfg_path_test_note
        #     self.open_save_cfg_path_test_label.destroy()
        #     del self.open_save_cfg_path_test_label

        self.open_save_cfg_path_test_label_var=tk.StringVar()
        self.open_save_cfg_path_test_label_var.set(self.save_cfg_path_test)
        cmd_i=open_cmd+" '{}'".format(self.open_save_cfg_path_test_label_var.get())
        self.open_save_cfg_path_test_button=Button(self.root,image=self.icon_single_file,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
        self.open_save_cfg_path_test_button.grid(row=7,column=4,sticky='se')
        #sjs#self.open_save_cfg_path_test_note=tk.Label(self.root,text="{}".format(self.open_save_cfg_path_test_label_var.get().split('/')[-1]),bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        #sjs#self.open_save_cfg_path_test_note.grid(row=8,column=4,sticky='ne')
        self.open_save_cfg_path_test_note=tk.Label(self.root,text="{}".format("test.cfg"),bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_save_cfg_path_test_note.grid(row=8,column=4,sticky='ne')

        cmd_i="netron '{}' -b".format(self.open_save_cfg_path_test_label_var.get())
        self.open_save_cfg_path_test_button_netron=Button(self.root,image=self.icon_map,command=partial(self.run_thread_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
        self.open_save_cfg_path_test_button_netron.grid(row=7,column=5,sticky='sw')
        # cmd_i=open_cmd+" '{}'".format(self.open_save_cfg_path_test_label_var.get())
        # self.open_save_cfg_path_test_label=Button(self.root,textvariable=self.open_save_cfg_path_test_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        # self.open_save_cfg_path_test_label.grid(row=7,column=5,columnspan=50,sticky='sw')
        self.save_cfg_path_test_selected=True

        self.open_anno()

        self.open_jpeg_label_var=tk.StringVar()
        self.open_jpeg_label_var.set(self.path_JPEGImages)
        self.open_jpeg_button=Button(self.root,image=self.icon_folder,command=partial(self.select_folder,self.path_JPEGImages,'Open JPEGImages Folder',self.open_jpeg_label_var),bg=self.root_bg,fg=self.root_fg)
        self.open_jpeg_button.grid(row=13,column=4,sticky='se')
        self.open_jpeg_note=tk.Label(self.root,text="JPEGImages dir",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_jpeg_note.grid(row=14,column=4,sticky='ne')
        cmd_i=open_cmd+" '{}'".format(self.open_jpeg_label_var.get())
        self.open_jpeg_label=Button(self.root,textvariable=self.open_jpeg_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.open_jpeg_label.grid(row=13,column=5,columnspan=50,sticky='sw')

        self.open_predjpeg_label_var=tk.StringVar()
        self.open_predjpeg_label_var.set(self.path_predJPEGImages)
        self.open_predjpeg_button=Button(self.root,image=self.icon_folder,command=partial(self.select_folder,self.path_predJPEGImages,'Open Prediction JPEGImages Folder',self.open_predjpeg_label_var),bg=self.root_bg,fg=self.root_fg)
        self.open_predjpeg_button.grid(row=10+11,column=4,sticky='se')
        self.open_predjpeg_note=tk.Label(self.root,text="Prediction JPEGImages dir",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_predjpeg_note.grid(row=11+11,column=4,sticky='ne')
        cmd_i=open_cmd+" '{}'".format(self.open_predjpeg_label_var.get())
        self.open_predjpeg_label=Button(self.root,textvariable=self.open_predjpeg_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.open_predjpeg_label.grid(row=10+11,column=5,columnspan=50,sticky='sw')

        self.open_MOVMP4()

        self.open_yolo_label_var=tk.StringVar()
        if os.path.exists(self.path_Yolo)==False:
            os.makedirs(self.path_Yolo)
        self.open_yolo_label_var.set(self.path_Yolo)
        self.open_yolo_button=Button(self.root,image=self.icon_folder,command=partial(self.select_folder,self.path_Yolo,'Open Yolo Folder',self.open_yolo_label_var),bg=self.root_bg,fg=self.root_fg)
        self.open_yolo_button.grid(row=15,column=4,sticky='se')
        self.open_yolo_note=tk.Label(self.root,text="Yolo dir",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_yolo_note.grid(row=16,column=4,sticky='ne')
        cmd_i=open_cmd+" '{}'".format(self.open_yolo_label_var.get())
        self.open_yolo_label=Button(self.root,textvariable=self.open_yolo_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.open_yolo_label.grid(row=15,column=5,columnspan=50,sticky='sw')

        self.create_yolo_objs_button=Button(self.root,image=self.icon_yolo_objects,command=self.convert_PascalVOC_to_YOLO,bg=self.root_bg,fg=self.root_fg)
        self.create_yolo_objs_button.grid(row=1,column=1,sticky='se')
        self.create_yolo_objs_button_note=tk.Label(self.root,text='2.a \n Create Yolo \n Objects (.jpg/.txt)',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.create_yolo_objs_button_note.grid(row=2,column=1,sticky='ne')

        

        self.var_overwrite=tk.StringVar()
        # df_pkls=os.listdir(self.path_Yolo)
        # df_pkls=[os.path.join(self.path_Yolo,w) for w in df_pkls if w.find('_df_YOLO.pkl')!=-1]
        # self.found_names={}
        # if len(df_pkls)>0:
        #     for pkl_i in tqdm(df_pkls):
        #         self.df_pkl=pd.read_pickle(pkl_i)
        #         names_possible=list(self.df_pkl['label_i'].unique())
        #         for name in names_possible:
        #             if name not in self.found_names.keys():
        #                 self.found_names[name]=len(self.found_names.keys())+0
        #     f=open(self.names_path,'w')
        #     f.writelines([k+'\n' for k, v in sorted(self.found_names.items(), key=lambda item: item[1])])
        #     f.close()
        if os.path.exists(os.path.join(self.path_Yolo,os.path.basename(self.names_path))) and os.path.exists(self.names_path)==False:
            shutil.copy(os.path.join(self.path_Yolo,os.path.basename(self.names_path)),self.names_path)
        if os.path.exists(self.names_path):
            f=open(self.names_path,'r')
            f_read=f.readlines()
            f.close()
            self.found_names={w.replace('\n',''):j for j,w in enumerate(f_read)}
            self.var_overwrite.set('No')
        else:

            self.var_overwrite.set('Yes')
        self.style2=ttk.Style()
        self.style2.configure('Normal.TRadiobutton',
                             background='green',
                             foreground='black')
        self.button_overwrite_yes=ttk.Radiobutton(text='Create new',style='Normal.TRadiobutton',variable=self.var_overwrite,value='Yes',
                                     command=partial(self.select_yes_no,'Yes'))

        self.button_overwrite_yes.grid(row=1,column=2,stick='nw')
        self.button_overwrite_no=ttk.Radiobutton(text='Keep existing as is',style='Normal.TRadiobutton',variable=self.var_overwrite,value='No',
                                     command=partial(self.select_yes_no,'No'))
        self.button_overwrite_no.grid(row=2,column=2,stick='nw')
        #REMOVING this option to keep it simple
        # self.button_overwrite_add=ttk.Radiobutton(text='Add to existing',style='Normal.TRadiobutton',variable=self.var_overwrite,value='Add',
        #                              command=partial(self.select_yes_no,'Add'))
        # self.button_overwrite_add.grid(row=3,column=2,stick='nw')



    def run_thread_cmd(self,cmd_i):
        Thread(target=self.run_cmd,args=(cmd_i,)).start()

    def open_anno(self):
        if self.open_anno_selected==True:
            self.open_anno_label.destroy()
            self.open_anno_note.destroy()
            del self.open_anno_label
            del self.open_anno_note

        self.open_anno_label_var=tk.StringVar()
        self.open_anno_label_var.set(self.path_Annotations)
        self.open_anno_button=Button(self.root,image=self.icon_folder,command=partial(self.select_folder,self.path_Annotations,'Open Annotations Folder',self.open_anno_label_var),bg=self.root_bg,fg=self.root_fg)
        self.open_anno_button.grid(row=11,column=4,sticky='se')
        self.open_anno_note=tk.Label(self.root,text="Annotations dir",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_anno_note.grid(row=12,column=4,sticky='ne')
        cmd_i=open_cmd+" '{}'".format(self.open_anno_label_var.get())
        self.open_anno_label=Button(self.root,textvariable=self.open_anno_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.open_anno_label.grid(row=11,column=5,columnspan=50,sticky='sw')
        self.open_anno_selected=True

    def open_anno_CUSTOM(self,row_i=3,col_i=6,columnspan=75,sticky='sew',location=None,sticky2='ne',sticky3='se'):
        if self.open_anno_selected_CUSTOM==True:
            self.open_anno_label_CUSTOM.destroy()
            self.open_anno_note_CUSTOM.destroy()
            self.open_anno_button_CUSTOM.destroy()
            del self.open_anno_label_CUSTOM
            del self.open_anno_note_CUSTOM
            del self.open_anno_button_CUSTOM
        try:
            if location==None:
                location=self.top
            else:
                location=location
        except:
            location=location
        print('location=',location)
        self.open_anno_label_var_CUSTOM=tk.StringVar()
        self.open_anno_label_var_CUSTOM.set(self.path_Annotations_CUSTOM)
        
        self.open_anno_button_CUSTOM=Button(location,image=self.icon_folder,command=partial(self.select_folder,os.path.dirname(self.path_Yolo),'Open Custom Annotations Folder',self.open_anno_label_var_CUSTOM),bg=self.root_bg,fg=self.root_fg)

        self.open_anno_note_CUSTOM=tk.Label(location,text="Custom Annotations dir",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))

        if col_i>0:
            self.open_anno_button_CUSTOM.grid(row=row_i,column=col_i,sticky=sticky3)
            self.open_anno_note_CUSTOM.grid(row=row_i+1,column=col_i,sticky=sticky2)

        self.open_anno_label_CUSTOM=Button(location,textvariable=self.open_anno_label_var_CUSTOM, command=self.open_custom_anno_cmd,bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.open_anno_label_CUSTOM.grid(row=row_i,column=col_i+1,columnspan=columnspan,sticky=sticky)
        self.open_anno_selected_CUSTOM=True

    def send_text_buttons(self):
        self.ck2=tk.Checkbutton(self.root,text='Send Text Message/Email Alerts',variable=self.sec,command=self.show_numbers,onvalue='y',offvalue='n',bg=self.root_fg,fg=self.root_bg)
        self.ck2.grid(row=14,column=3,sticky='n')
    def select_yes_no(self,selected):
        if str(selected)=='Yes':
            self.var_overwrite.set('Yes')
        elif str(selected)=='No':
            self.var_overwrite.set('No')
        elif str(selected)=='Add':
            self.var_overwrite.set('Add')

    def create_yolo_files(self):
        self.RTSP()
        self.CLASSIFYCHIPS()
        self.create_obj_data()
        self.create_train_bash()

        self.create_test_bash()
        self.create_test_bash_mp4()
        self.create_test_bash_mp4_record()
        self.create_test_bash_images_with_predictions()
        self.create_test_bash_images_with_predictions_mAP()
        self.create_test_bash_dnn()
        self.create_test_bash_dnn_labelimg()
        self.create_test_bash_dnn_rtsp()
        self.YOUTUBE_RTMP()

        self.create_test_bash_dnn_rtmp()
        self.remaining_buttons()
        self.labelImg_buttons()
        self.MOSAIC_buttons()
        self.IMGAUG_buttons()
        self.CLASSIFY_CHIPS_buttons()
        self.send_text_buttons()
        
        


    def remaining_buttons(self):
        if self.data_path_selected==True:
            self.open_data_path_note.destroy()
            del self.open_data_path_note
        self.open_data_path_label_var=tk.StringVar()
        self.open_data_path_label_var.set(self.data_path)
        cmd_i=open_cmd+" '{}'".format(self.open_data_path_label_var.get())
        self.open_data_path_button=Button(self.root,image=self.icon_single_file,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
        self.open_data_path_button.grid(row=7,column=4+1,sticky='se')
        self.open_data_path_note=tk.Label(self.root,text="{}".format(os.path.basename(self.open_data_path_label_var.get())),bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_data_path_note.grid(row=8,column=4+1,sticky='ne')
        # cmd_i=open_cmd+" '{}'".format(self.open_data_path_label_var.get())
        # self.open_data_path_label=Button(self.root,textvariable=self.open_data_path_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        # self.open_data_path_label.grid(row=17,column=4+50,columnspan=50,sticky='sw')
        self.data_path_selected=True

        self.data_path=os.path.join(self.base_path,'obj.data')
        self.train_list_path=os.path.join(self.base_path,'train.txt')
        self.valid_list_path=os.path.join(self.base_path,'valid.txt')
        self.names_path=os.path.join(self.base_path,'obj.names')

        self.tiny_conv29_path=os.path.join(self.darknet_path,"yolov4-tiny.conv.29")
        if self.random=='0':
            self.save_cfg_path=os.path.join(self.base_path,'{}_w{}_h{}_d{}_c{}.cfg'.format(self.PREFIX,self.WIDTH_NUM,self.HEIGHT_NUM,self.num_div,self.num_classes))
        else:
            self.save_cfg_path=os.path.join(self.base_path,'{}_w{}_h{}_d{}_c{}_r{}.cfg'.format(self.PREFIX,self.WIDTH_NUM,self.HEIGHT_NUM,self.num_div,self.num_classes,self.random))
        self.remaining_buttons_clicked=True
        self.TRAIN_BUTTONS()
        self.TEST_BUTTONS()
        self.SHOWTABLE_BUTTONS()
        #self.test_yolo()
        #self.test_yolo_predict()
        #self.test_yolo_predict_mAP()
        #self.test_yolodnn()
        self.open_mp4()
        #self.test_yolodnn_rtmp()
        self.convert_tflite()
        #self.train_yolo()
        #self.train_yolov7()
        #self.train_yolov7_e6e()
        #self.train_yolov7_re()
        self.calculate_epochs_yolov4()

    def open_mp4(self):
        if self.mp4_selected==True:
            self.open_mp4_label.destroy()
            self.open_mp4_note.destroy()
            del self.open_mp4_label
            del self.open_mp4_note

        self.open_mp4_label_var=tk.StringVar()
        self.open_mp4_label_var.set(self.mp4_video_path)
        self.open_mp4_button=Button(self.root,image=self.icon_folder,command=partial(self.select_file_mp4,self.mp4_video_path),bg=self.root_bg,fg=self.root_fg)
        self.open_mp4_button.grid(row=19,column=4,sticky='se')
        self.open_mp4_note=tk.Label(self.root,text="mp4 file to test with",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_mp4_note.grid(row=20,column=4,sticky='ne')
        cmd_i=open_cmd+" '{}'".format(self.open_mp4_label_var.get())
        self.open_mp4_label=Button(self.root,textvariable=self.open_mp4_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.open_mp4_label.grid(row=19,column=5,columnspan=50,sticky='sw')
        

        if self.mp4_selected==True and self.remaining_buttons_clicked==False:
            self.popupWindow_TEST()
        #self.test_yolo_mp4()
        #self.test_yolov7_mp4()
        self.mp4_selected=True
        self.dropdowntest_menu()
        self.remaining_buttons_clicked=False

    def open_MOVMP4(self):
        if self.MOVMP4_selected==True:
            #self.open_MOVMP4_label.destroy()
            self.open_MOVMP4_note.destroy()
            #del self.open_MOVMP4_label
            del self.open_MOVMP4_note

        self.open_MOVMP4_label_var=tk.StringVar()
        self.open_MOVMP4_label_var.set(self.path_MOVMP4)
        self.open_MOVMP4_button=Button(self.root,image=self.icon_single_file,command=partial(self.select_file_MOVMP4,self.path_MOVMP4),bg=self.root_bg,fg=self.root_fg)
        self.open_MOVMP4_button.grid(row=5,column=8,sticky='sw')
        self.open_MOVMP4_note=tk.Label(self.root,text="MOV/MP4 File to \n Create JPEGImages of",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_MOVMP4_note.grid(row=6,column=7,columnspan=3,sticky='n')
        self.fps_MOVMP4_VAR=tk.StringVar()
        self.fps_MOVMP4_options=['1/10','1/5','1/4','1/2','1','2','3','4','5','6','7','8','9','10','15','20','25','30','40']
        self.fps_MOVMP4_VAR.set('1/2')


        #cmd_i=open_cmd+" '{}'".format(self.open_MOVMP4_label_var.get())
        #self.open_MOVMP4_label=Button(self.root,textvariable=self.open_MOVMP4_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        #self.open_MOVMP4_label.grid(row=5,column=9,columnspan=50,sticky='sw') 
        self.MOVMP4_selected=True 
        self.create_MOVMP4_JPEGImages()
        self.fps_MOVMP4_dropdown=tk.OptionMenu(self.root,self.fps_MOVMP4_VAR,*self.fps_MOVMP4_options)
        self.fps_MOVMP4_dropdown.grid(row=5,column=9,sticky='nw')
        self.fps_MOVMP4_dropdown_label=tk.Label(self.root,text='FPS',bg=self.root_bg,fg=self.root_fg,font=('Arial',8))
        self.fps_MOVMP4_dropdown_label.grid(row=4,column=9,sticky='sw')   

    def create_MOVMP4_JPEGImages(self):
        self.create_MOVMP4_button=Button(self.root,image=self.icon_create,command=self.check_fps,bg=self.root_bg,fg=self.root_fg)
        self.create_MOVMP4_button.grid(row=5,column=7,sticky='se')
        #self.create_MOVMP4_button_note=tk.Label(self.root,text='Create JPEGImages \n from MOV/MP4',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        #self.create_MOVMP4_button_note.grid(row=6,column=6,columnspan=3,sticky='ne')

    def check_fps(self):
        create_imgs_from_video.create_imgs_from_video(self.path_MOVMP4,self.fps_MOVMP4_VAR.get())



    def dropdowntest_menu(self):
        if self.dropdowntests!=None:
            self.dropdowntests_label.destroy()
            self.dropdowntests.destroy()
        self.options=os.listdir(self.MODEL_PATHS)        
        self.model_path_var=tk.StringVar()
        if self.prefix_foldername in self.options:
            self.model_path_var.set(self.prefix_foldername)
            #print('Using current cfg/model')
        else:
            self.model_path_var.set(self.options[0])
            #print('Using custom cfg/model')
        # self.dropdowntests=tk.OptionMenu(self.root,self.model_path_var,*self.options,command=self.read_model_test)
        # self.dropdowntests.grid(row=9,column=7,sticky='nw')
        # cmd_i=open_cmd+" '{}'".format(self.MODEL_PATHS)
        # self.dropdowntests_label=Button(self.root,text='Testing Models',command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg,font=('Arial',12))
        # self.dropdowntests_label.grid(row=8,column=7,sticky='sw')

    def train_yolo(self):
        #cmd_i=" bash '{}'".format(self.save_cfg_path_train.replace('.cfg','.sh'))
        self.train_yolo_objs_button=Button(self.top,image=self.icon_train,command=self.train_yolov4,bg=self.root_bg,fg=self.root_fg)
        self.train_yolo_objs_button.grid(row=3,column=2,sticky='se')
        self.train_yolo_objs_button_note=tk.Label(self.top,text='Train',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.train_yolo_objs_button_note.grid(row=4,column=2,sticky='ne')

        try:
            self.EPOCH_entry.destroy()
            self.EPOCH_label.destroy()
        except:
            self.epochs_VAR=tk.StringVar()
            self.epochs_VAR.set(self.epochs)
            self.epochs_entry=tk.Entry(self.root,textvariable=self.epochs_VAR)
            self.epochs_entry.grid(row=21,column=0,sticky='se')
            self.epochs_label=tk.Label(self.root,text='epochs',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
            self.epochs_label.grid(row=22,column=0,sticky='ne')
        self.epochs_yolov_entry=tk.Entry(self.top,textvariable=self.epochs_VAR)
        self.epochs_yolov_entry.grid(row=5,column=2,sticky='sw')
        self.epochs_yolov_label=tk.Label(self.top,text='epochs',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.epochs_yolov_label.grid(row=6,column=2,sticky='nw')

    def train_yolov4_madness(self):
        if self.epochs_VAR.get()!=self.epochs:
            #self.epochs=int(self.max_batches.replace('\n','').strip().split('=')[1])//self.iterations_per_epoch
            try:
                self.max_batches=int(self.epochs_VAR.get())*int(self.iterations_per_epoch)
                self.ITERATION_NUM=self.max_batches
                self.ITERATION_NUM_VAR.set(self.ITERATION_NUM)
            except:
                print('Could not convert epochs desired to cfg')
            self.generate_cfg()
            self.remaining_buttons()
        self.save_settings()       

    def train_yolov4(self):
        
        self.train_yolov4_madness()
        cmd_i=" bash '{}'".format(self.save_cfg_path_train.replace('.cfg','.sh'))
        if self.train_load_repeat.get()=='None':
            self.run_cmd(cmd_i)
        else:
            self.loop_train_load(cmd_i,'train_yolov4_madness')

    def show_table(self):
        self.app = TestApp(self.top, self.df_filename_csv)
        self.app.pack(fill=tk.BOTH, expand=1)
        
    def RTSP(self):
        spacer=5
          
        if self.RTSP_SERVER and self.PORT_VAR==None:
            self.PORT_VAR=tk.StringVar()
            self.PORT_VAR.set(self.PORT)
            self.PORT_entry=tk.Entry(self.root,textvariable=self.PORT_VAR)
            self.PORT_entry.grid(row=13+spacer,column=2,sticky='sw')
            self.PORT_label=tk.Label(self.root,text='RTSP PORT',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
            self.PORT_label.grid(row=14+spacer,column=2,sticky='nw')  
        if self.RTSP_SERVER and self.FPS_VAR==None:
            self.FPS_VAR=tk.StringVar()
            self.FPS_VAR.set(self.FPS)
            self.FPS_entry=tk.Entry(self.root,textvariable=self.FPS_VAR)
            self.FPS_entry.grid(row=15+spacer,column=2,sticky='sw')
            self.FPS_label=tk.Label(self.root,text='RTSP FPS',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
            self.FPS_label.grid(row=16+spacer,column=2,sticky='nw')  
        if self.RTSP_SERVER and self.STREAM_KEY_VAR==None:
            self.STREAM_KEY_VAR=tk.StringVar()
            self.STREAM_KEY_VAR.set(self.STREAM_KEY)
            self.STREAM_KEY_entry=tk.Entry(self.root,textvariable=self.STREAM_KEY_VAR)
            self.STREAM_KEY_entry.grid(row=17+spacer,column=2,sticky='sw')
            self.STREAM_KEY_label=tk.Label(self.root,text='RTSP STREAM KEY',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
            self.STREAM_KEY_label.grid(row=18+spacer,column=2,sticky='nw')  
        if self.RTSP_SERVER and self.RTSP_FULL_PATH_VAR==None:
            self.RTSP_FULL_PATH_VAR=tk.StringVar()
            self.get_full_path_rtsp()
            self.RTSP_FULL_PATH_label=tk.Label(self.root,textvariable=self.RTSP_FULL_PATH_VAR,bg=self.root_fg,fg=self.root_bg,font=('Arial',10))
            self.RTSP_FULL_PATH_label.grid(row=14+spacer-2,column=2,sticky='nw')  
        if self.RTSP_SERVER and self.USE_RTSP_VAR==None:
            self.USE_RTSP_VAR=tk.StringVar()
            self.USE_RTSP_options=['Yes','No']
            self.USE_RTSP_VAR.set('No')
            self.USE_RTSP_dropdown=tk.OptionMenu(self.root,self.USE_RTSP_VAR,*self.USE_RTSP_options,command=self.get_full_path_rtsp())
            self.USE_RTSP_dropdown.grid(row=13+spacer-2,column=3,sticky='sw')
            self.USE_RTSP_label=tk.Label(self.root,text='RTSP Server?',bg=self.root_bg,fg=self.root_fg,font=('Arial',10))
            self.USE_RTSP_label.grid(row=14+spacer-2,column=3,sticky='nw')   
        if self.RTSP_CLIENT and self.USE_RTSP_CLIENT_VAR==None: 
            self.popupWindow_RTSP()
            self.cleanup()
            self.RTSP_BUTTONS()  

    def CLASSIFYCHIPS(self):
        self.popupWindow_CLASSIFY_CHIPS()
        self.cleanup()
        self.CLASSIFY_CHIPS_BUTTONS()


    def RTSP_BUTTONS(self):
        self.popup_RTSP_button=Button(self.root,text='Client/Socket RTSP Buttons',command=self.popupWindow_RTSP,bg=self.root_fg,fg=self.root_bg)
        self.popup_RTSP_button.grid(row=16+5-2,column=3,sticky='sw')

    def CLASSIFY_CHIPS_BUTTONS(self):
        self.popup_CLASSIFY_CHIPS_button=Button(self.root,text='CLASSIFY CHIPS Buttons',command=self.popupWindow_CLASSIFY_CHIPS,bg=self.root_fg,fg=self.root_bg)
        self.popup_CLASSIFY_CHIPS_button.grid(row=9,column=8,sticky='sw')


        self.style3=ttk.Style()
        self.style3.configure('Normal.TRadiobutton',
                             background='green',
                             foreground='black')
        self.button_classify_yes=ttk.Radiobutton(text='Yes',style='Normal.TRadiobutton',variable=self.CLASSIFY_CHIPS_LOGIC,value='Yes')

        self.button_classify_yes.grid(row=9,column=9,stick='nw')
        self.button_classify_no=ttk.Radiobutton(text='No',style='Normal.TRadiobutton',variable=self.CLASSIFY_CHIPS_LOGIC,value='No')
        self.button_classify_no.grid(row=9,column=10,stick='ne')
        self.label_classify=tk.Label(self.root,text='CLASSIFY CHIPS?',bg=self.root_bg,fg=self.root_fg,font=('Arial',10))
        self.label_classify.grid(row=8,column=9,columnspan=2,stick='sew')

    def GENERATE_CUSTOM_DATASET_BUTTONS(self):
        self.popup_GCD_button=Button(self.root,text='GENERATE CUSTOM DATASET Buttons',command=self.popupWindow_GCD,bg=self.root_fg,fg=self.root_bg)
        self.popup_GCD_button.grid(row=0,column=3,sticky='se')

    def popupWindow_GCD(self):
        try:
            self.top.destroy()
        except:
            pass
        
        self.top=tk.Toplevel(self.root)
        self.top.geometry( "{}x{}".format(int(self.root.winfo_screenwidth()*0.95//1.1),int(self.root.winfo_screenheight()*0.95//1.1)) )
        self.top.title('GENERATE CUSTOM DATASET?')
        self.top.configure(background = 'black')
        
        self.top.columnconfigure(0,weight=1)
        self.top.rowconfigure(0,weight=1)
        self.FMas=tk.Frame(self.top,bg='Black')
        self.FMas.grid(sticky=(tk.N,tk.E,tk.S,tk.W))
        self.FMas.columnconfigure(0,weight=1)



        self.b=Button(self.FMas,text='Close',command=self.cleanup,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.b.grid(row=2,column=0,sticky='sw')

        self.open_GCD_label_CUSTOM=Button(self.FMas,text="Open Custom Dataset LIST OPTIONS",command=self.open_dataset_lists,bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_GCD_label_CUSTOM.grid(row=0,column=0,sticky='nw')

        self.generate_GCD_label_CUSTOM=Button(self.FMas,text="Generate Custom Dataset",command=self.generate_GCD,bg='green',fg=self.root_bg,font=("Arial", 10))
        self.generate_GCD_label_CUSTOM.grid(row=7,column=0,sticky='sw')

        self.MAX_PER_CLASS_VAR=tk.StringVar()
        self.MAX_PER_CLASS_VAR.set(self.MAX_PER_CLASS)
        self.MAX_PER_CLASS_entry=tk.Entry(self.FMas,textvariable=self.MAX_PER_CLASS_VAR)
        self.MAX_PER_CLASS_entry.grid(row=9,column=0,sticky='nw')
        self.MAX_PER_CLASS_label=tk.Label(self.FMas,text='MAX_PER_CLASS',bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.MAX_PER_CLASS_label.grid(row=8,column=0,sticky='sw')

        self.TARGET_LIST_VAR=tk.StringVar()
        self.TARGET_LIST_VAR.set(self.TARGET_LIST)
        self.TARGET_LIST_entry=tk.Entry(self.FMas,textvariable=self.TARGET_LIST_VAR)
        self.TARGET_LIST_entry.grid(row=11,column=0,sticky='nw')
        self.TARGET_LIST_label=tk.Label(self.FMas,text='CLASS LIST',bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.TARGET_LIST_label.grid(row=10,column=0,sticky='sw')

        self.open_anno_CUSTOM(row_i=12,col_i=-1,columnspan=1,sticky='sw',location=self.FMas,sticky2='ne',sticky3='ne')
        self.open_jpeg_CUSTOM(row_i=14,col_i=-1,columnspan=1,sticky='sw',location=self.FMas,sticky2='ne',sticky3='ne')
        
        self.load_dataset_lists_buttons()
        self.frame_canvas=tk.Frame(self.FMas)
        self.frame_canvas.grid(row=17,column=0,sticky='nw')
        self.frame_canvas.grid_rowconfigure(0,weight=1)
        self.frame_canvas.grid_columnconfigure(0,weight=1)
        self.frame_canvas.grid_propagate(False)
        self.canvas=tk.Canvas(self.frame_canvas,bg='black')
        self.canvas.grid(row=0,column=0,sticky='news')
        self.vsbar=tk.Scrollbar(self.frame_canvas,orient="vertical",command=self.canvas.yview)
        self.vsbar.grid(row=0,column=1,sticky='nes')
        self.hsbar=tk.Scrollbar(self.frame_canvas,orient="horizontal",command=self.canvas.xview)
        self.hsbar.grid(row=0,column=0,rowspan=100,sticky='new')
        self.canvas.configure(yscrollcommand=self.vsbar.set)
        self.canvas.configure(xscrollcommand=self.hsbar.set)
        self.frame_table=tk.Frame(self.canvas,bg='black')
        self.canvas.create_window((0,0),window=self.frame_table,anchor='nw')



        self.load_dataset_lists()
    def generate_GCD(self):
        if os.path.exists('resources/find_class_balance.py'):
            from resources import find_class_balance as gcd
            try:
                self.SELECTED_SEARCH_OPTIONS={}
                for k,v in self.DATASET_checkvars.items():
                    if v.get()==1:
                        self.SELECTED_SEARCH_OPTIONS[k]=self.DATASET_OPTIONS[k]
                TARGET_LIST=self.TARGET_LIST_VAR.get()
                if TARGET_LIST.find(';')!=-1:
                    list_targets=TARGET_LIST.split(';')
                else:
                    list_targets=['nothing_to_search']
                list_targets=[w for w in list_targets if w!='']
                print('list_targets',list_targets)
                self.MAX_PER_CLASS=self.MAX_PER_CLASS_VAR.get()
                try:
                    MAX_PER_CLASS=int(self.MAX_PER_CLASS)
                except:
                    print('ISSUE WIHT MAX_PER_CLASS, defaulting to 500')
                    MAX_PER_CLASS=500
                self.path_Annotation_CUSTOM_OG=self.path_Annotations_CUSTOM
                self.path_JPEGImages_CUSTOM_OG=self.path_JPEGImages_CUSTOM
                path_Desired=os.path.join(os.path.dirname(self.path_Annotations_CUSTOM),'Results')
                self.df_results,self.path_Annotations_CUSTOM,self.path_JPEGImages_CUSTOM=gcd.JUST_DO_IT(self.SELECTED_SEARCH_OPTIONS,list_targets,MAX_PER_CLASS,path_Desired)
                self.open_anno_label_var_CUSTOM.set(self.path_Annotations_CUSTOM)
                self.open_jpeg_label_var_CUSTOM.set(self.path_JPEGImages_CUSTOM)
            except:
                try:
                    self.path_Annotation_CUSTOM=self.path_Annotation_CUSTOM_OG
                    self.path_JPEGImages_CUSTOM=self.path_JPEGImages_CUSTOM_OG
                    self.open_anno_label_var_CUSTOM.set(self.path_Annotation_CUSTOM_OG)
                    self.open_jpeg_label_var_CUSTOM.set(self.path_JPEGImages_CUSTOM_OG)
                    print('issue with generating this custom dataset')
                except:
                    print('really an issue')
    def open_dataset_lists(self):
        if not(os.path.exists('libs/DATASETS_LIST.txt')):
            f=open('libs/DATASETS_LIST.txt','w')
            f.close()
        cmd_i=open_cmd+' '+'libs/DATASETS_LIST.txt'
        self.run_cmd(cmd_i)
    def load_dataset_lists_buttons(self):
        self.load_GCD_label_CUSTOM=Button(self.FMas,text="Load Custom Dataset LIST OPTIONS",command=self.load_dataset_lists,bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.load_GCD_label_CUSTOM.grid(row=1,column=0,sticky='nw')   

    def load_dataset_lists(self):
        spacer=0
        dic_i={}
        obj_i={}
        if os.path.exists('libs/DATASETS_LIST.txt'):
            f=open('libs/DATASETS_LIST.txt')
            f_read=f.readlines()
            f.close()
            for line in f_read:
                if line.find(':')!=-1:
                    anno_i=line.split(':')[0].replace('\n','')
                    jpeg_i=line.split(':')[1].replace('\n','')
                    yolo_i=line.split(':')[2].replace('\n','')
                    yolo_model_i=line.split(':')[3].replace('\n','')
                    if os.path.exists(anno_i) and os.path.exists(jpeg_i):
                        if os.path.exists(os.path.join(yolo_i,'obj.names')):
                            obj_i[anno_i]=os.path.join(yolo_i,'obj.names')
                        elif os.path.exists(os.path.join(yolo_model_i,'obj.names')):
                            obj_i[anno_i]=os.path.join(yolo_model_i,'obj.names')
                        else:
                            obj_i[anno_i]='None'

                        dic_i[anno_i]=jpeg_i
        self.DATASET_OPTIONS=dic_i
        self.DATASET_OBJECTS=obj_i
        #print(self.DATASET_OPTIONS)
        try:
            [w.destroy() for w in self.DATASET_checkvars.values()]
        except:
            pass
        try:
            [w.destroy() for w in self.DATASET_checkbuttons.values()]
        except:
            pass
        try:
            self.DATASET_label.destroy()
        except:
            pass
        try:
            [w.destroy() for w in self.DATASET_dropdownvars.values()]
        except:
            pass
        try:
            [w.destroy() for w in self.DATASET_dropdownlists.values()]
        except:
            pass
        try:
            [w.destroy() for w in self.DATASET_dropdownoptions.values()]
        except:
            pass
        self.DATASET_checkvars={}
        self.DATASET_checkbuttons={}
        self.DATASET_label=tk.Label(self.FMas,text='DATASETS USED LIST',bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.DATASET_label.grid(row=8,column=0,sticky='sw')
        self.DATASET_dropdownvars={}
        self.DATASET_dropdownlists={}
        self.DATASET_dropdownoptions={}
        for i,k in enumerate(self.DATASET_OPTIONS.keys()):
            self.DATASET_checkvars[k]=tk.IntVar()
            self.DATASET_checkvars[k].set(0)
            self.DATASET_checkbuttons[k]=tk.Checkbutton(self.frame_table,text='{}'.format(k),variable=self.DATASET_checkvars[k],onvalue=1,offvalue=0,bg=self.root_fg,fg=self.root_bg)
            self.DATASET_checkbuttons[k].grid(row=spacer+i,column=0,sticky='sw')
            if self.DATASET_OBJECTS[k]!='None':
                f=open(self.DATASET_OBJECTS[k],'r')
                f_read=f.readlines()
                f.close()     
                df_i=pd.DataFrame([w.replace('\n','').replace(' ','') for w in f_read])     

            elif os.path.exists(os.path.join(os.path.dirname(k))):
                path_to_read=os.path.join(os.path.dirname(k),'grep_results.txt')
                cmd_i='cd {} && find {} -name obj.names > {}'.format(os.path.dirname(k),os.path.dirname(k),path_to_read)    
                self.run_cmd(cmd_i)
                f=open(path_to_read,'r')
                f_read=f.readlines()
                f.close()     
                if len(f_read)>0:
                    f=open(f_read[0].replace('\n','').replace(' ',''))
                    f_read=f.readlines()
                    f.close()
                    df_i=pd.DataFrame([w.replace('\n','').replace(' ','') for w in f_read])     
                else:  
                    print("DOING THIS THE LONG WAY since no obj.names was found here for:\n\t {}\n".format(k))    
                    path_to_read=os.path.join(os.path.dirname(k),'grep_results.txt')
                    cmd_i='cd {} && grep -r "<name>" Annotations > {}'.format(os.path.dirname(k),path_to_read)
                    self.run_cmd(cmd_i)
                    f=open(path_to_read,'r')
                    f_read=f.readlines()
                    f.close()
                    df_i=pd.DataFrame([w.split("<name>")[1].split("</name")[0] for w in f_read])
            list_i=list(df_i[0].unique())
            list_i=sorted(list_i)
            list_i=";".join([w for w in list_i])
            list_i=[""]+[list_i]
            self.DATASET_dropdownlists[k]=list_i
            self.DATASET_dropdownvars[k]=tk.StringVar()
            self.DATASET_dropdownvars[k].set(list_i[1])
            self.DATASET_dropdownoptions[k]=tk.OptionMenu(self.frame_table,self.DATASET_dropdownvars[k],*self.DATASET_dropdownlists[k])
            self.DATASET_dropdownoptions[k].grid(row=spacer+i,column=1,sticky='sw')

        self.frame_table.update_idletasks()
        width_i=sum([w.winfo_width() for w in self.DATASET_dropdownoptions.values()])
        width_j=sum([w.winfo_width() for w in self.DATASET_checkbuttons.values()])
        height_i=sum([w.winfo_height() for w in self.DATASET_dropdownoptions.values()])
        height_j=sum([w.winfo_height() for w in self.DATASET_checkbuttons.values()])
        total_width=width_i+width_j+self.hsbar.winfo_width()
        total_height=height_i+height_j+self.vsbar.winfo_height()
        print('total_width',total_width)
        print('total_height',total_height)
        self.frame_canvas.config(width=total_width,height=total_height)
        self.canvas.config(scrollregion=self.canvas.bbox('all'))
        self.top.mainloop()




    def popupWindow_CLASSIFY_CHIPS(self):
        spacer=-12
        try:
            self.top.destroy()
        except:
            pass
        self.top=tk.Toplevel(self.root)
        self.top.geometry( "{}x{}".format(int(self.root.winfo_screenwidth()*0.95//1.5),int(self.root.winfo_screenheight()*0.95//1.5)) )
        self.top.title('CLASSIFY CHIPS INFERENCE Options')
        self.top.configure(background = 'black')
        self.b=Button(self.top,text='Close',command=self.cleanup_CLASSIFY_CHIPS,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.b.grid(row=0,column=0,sticky='se')
        self.load_CLASSIFY_CHIPS_INFERENCE_Options(spacer)
        self.USE_CLASSIFY_CHIPS_INFERENCE_label=tk.Label(self.top,text='INFERENCE SCRIPTS FOR SECOND-STAGE CLASSIFIER',bg=self.root_bg,fg=self.root_fg,font=('Arial',10))
        self.USE_CLASSIFY_CHIPS_INFERENCE_label.grid(row=16+spacer-2,column=3,sticky='nw')
        self.OPEN_CLASSIFY_CHIPS_INFERENCE_Button=tk.Button(self.top,text='OPEN CLASSIFY_CHIPS_INFERENCE Options',command=self.open_CLASSIFY_CHIPS_INFERENCE_List,bg=self.root_bg,fg=self.root_fg,font=('Arial',10))
        self.OPEN_CLASSIFY_CHIPS_INFERENCE_Button.grid(row=17+spacer-2,column=3,sticky='sw')
        self.LOAD_CLASSIFY_CHIPS_INFERENCE_Button=tk.Button(self.top,text='LOAD CLASSIFY_CHIPS_INFERENCE Options',command=partial(self.load_CLASSIFY_CHIPS_INFERENCE_Options,spacer),bg=self.root_bg,fg=self.root_fg,font=('Arial',10))
        self.LOAD_CLASSIFY_CHIPS_INFERENCE_Button.grid(row=18+spacer-2,column=3,sticky='sw')



    def cleanup_CLASSIFY_CHIPS(self):
        self.USE_CLASSIFY_CHIPS_VAR.set(self.USE_CLASSIFY_CHIPS_VAR.get())
        if os.path.exists(self.USE_CLASSIFY_CHIPS_VAR.get()):
            SAVED_SETTINGS_PATH=os.path.join(os.path.dirname(self.USE_CLASSIFY_CHIPS_VAR.get()),os.path.basename(os.path.dirname(self.USE_CLASSIFY_CHIPS_VAR.get()))+'_SAVED_SETTINGS.py')
            print(f"SAVED_SETTINGS_PATH = {SAVED_SETTINGS_PATH}")
            if os.path.exists(SAVED_SETTINGS_PATH):
                print("EXISTS")
                self.CLASSIFY_CHIPS_SETTINGS_PATH_VAR.set(os.path.join(os.path.dirname(self.USE_CLASSIFY_CHIPS_VAR.get()),os.path.basename(os.path.dirname(self.USE_CLASSIFY_CHIPS_VAR.get()))+'_SAVED_SETTINGS.py'))
            else:
                print('DOES NOT EXIST')
        self.top.destroy()
    def load_CLASSIFY_CHIPS_INFERENCE_Options_list(self):
        self.CLASSIFY_CHIPS_INFERENCE_LIST='libs/CLASSIFY_CHIPS_LIST.txt'
        if os.path.exists(self.CLASSIFY_CHIPS_INFERENCE_LIST):
            f=open(self.CLASSIFY_CHIPS_INFERENCE_LIST,'r')
            f_read=f.readlines()
            f.close()
            self.CLASSIFY_CHIPS_LIST=[w.replace(' ','').replace('\n','') for w in f_read if w.find('INFERENCE')!=-1]
        else:
            f=open(self.CLASSIFY_CHIPS_INFERENCE_LIST,'w')
            f.writelines('NO OPTIONS AVAILABLE\n')
            f.close()
            f=open(self.CLASSIFY_CHIPS_INFERENCE_LIST,'r')
            f_read=f.readlines()
            f.close()
            self.CLASSIFY_CHIPS_LIST=[w.replace(' ','').replace('\n','') for w in f_read if w.find('INFERENCE')!=-1]
    def open_CLASSIFY_CHIPS_INFERENCE_List(self):
        cmd_i=open_cmd+" '{}'".format(self.CLASSIFY_CHIPS_INFERENCE_LIST)
        self.run_cmd(cmd_i)  

    def load_CLASSIFY_CHIPS_INFERENCE_Options(self,spacer):
        try:
            del self.USE_CLASSIFY_CHIPS_dropdown
        except:
            pass
        self.USE_CLASSIFY_CHIPS_VAR=tk.StringVar()
        self.load_CLASSIFY_CHIPS_INFERENCE_Options_list()
        if len(self.CLASSIFY_CHIPS_LIST)>0:
            pass
        else:
            self.CLASSIFY_CHIPS_LIST=['NO OPTIONS AVAILABLE']
        self.USE_CLASSIFY_CHIPS_VAR.set(self.CLASSIFY_CHIPS_LIST[0])
        self.USE_CLASSIFY_CHIPS_dropdown=tk.OptionMenu(self.top,self.USE_CLASSIFY_CHIPS_VAR,*self.CLASSIFY_CHIPS_LIST,command=self.return_CLASSIFY_CHIPS)
        self.USE_CLASSIFY_CHIPS_dropdown.grid(row=15+spacer-2,column=3,sticky='sw')

    def return_CLASSIFY_CHIPS(self):
        print(self.USE_CLASSIFY_CHIPS_VAR.get())

    def popupWindow_RTSP(self):
        spacer=-12
        try:
            self.top.destroy()
        except:
            pass
        self.top=tk.Toplevel(self.root)
        self.top.geometry( "{}x{}".format(int(self.root.winfo_screenwidth()*0.95//1.5),int(self.root.winfo_screenheight()*0.95//1.5)) )
        self.top.title('RTSP Options')
        self.top.configure(background = 'black')
        self.b=Button(self.top,text='Close',command=self.cleanup_RTSP,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.b.grid(row=0,column=0,sticky='se')
        self.load_RTSP_clients(spacer)
        self.USE_RTSP_CLIENT_label=tk.Label(self.top,text='RTSP Client?',bg=self.root_bg,fg=self.root_fg,font=('Arial',10))
        self.USE_RTSP_CLIENT_label.grid(row=16+spacer-2,column=3,sticky='nw')
        self.OPEN_RTSP_CLIENT_Button=tk.Button(self.top,text='OPEN RTSP CLIENTS',command=self.open_RTSP_CLIENT_List,bg=self.root_bg,fg=self.root_fg,font=('Arial',10))
        self.OPEN_RTSP_CLIENT_Button.grid(row=17+spacer-2,column=3,sticky='sw')
        self.LOAD_RTSP_CLIENT_Button=tk.Button(self.top,text='LOAD RTSP CLIENTS',command=partial(self.load_RTSP_clients,spacer),bg=self.root_bg,fg=self.root_fg,font=('Arial',10))
        self.LOAD_RTSP_CLIENT_Button.grid(row=18+spacer-2,column=3,sticky='sw')

        self.load_SOCKET_RTSP_ports(spacer)
        self.USE_SOCKET_RTSP_PORT_label=tk.Label(self.top,text='SOCKET RTSP PORT',bg=self.root_bg,fg=self.root_fg,font=('Arial',10))
        self.USE_SOCKET_RTSP_PORT_label.grid(row=20+spacer-2,column=3,sticky='nw')
        self.OPEN_SOCKET_RTSP_PORT_Button=tk.Button(self.top,text='OPEN SOCKET PORTS',command=self.open_SOCKET_RTSP_PORT_List,bg=self.root_bg,fg=self.root_fg,font=('Arial',10))
        self.OPEN_SOCKET_RTSP_PORT_Button.grid(row=21+spacer-2,column=3,sticky='sw')
        self.LOAD_SOCKET_RTSP_PORT_Button=tk.Button(self.top,text='LOAD SOCKET PORTS',command=partial(self.load_SOCKET_RTSP_ports,spacer),bg=self.root_bg,fg=self.root_fg,font=('Arial',10))
        self.LOAD_SOCKET_RTSP_PORT_Button.grid(row=22+spacer-2,column=3,sticky='sw')

        self.load_SOCKET_RTSP_hosts(spacer)
        self.USE_SOCKET_RTSP_HOSTS_label=tk.Label(self.top,text='SOCKET RTSP HOSTS',bg=self.root_bg,fg=self.root_fg,font=('Arial',10))
        self.USE_SOCKET_RTSP_HOSTS_label.grid(row=24+spacer-2,column=3,sticky='nw')
        self.OPEN_SOCKET_RTSP_HOSTS_Button=tk.Button(self.top,text='OPEN SOCKET HOSTS',command=self.open_SOCKET_RTSP_HOST_List,bg=self.root_bg,fg=self.root_fg,font=('Arial',10))
        self.OPEN_SOCKET_RTSP_HOSTS_Button.grid(row=25+spacer-2,column=3,sticky='sw')
        self.LOAD_SOCKET_RTSP_HOSTS_Button=tk.Button(self.top,text='LOAD SOCKET HOSTS',command=partial(self.load_SOCKET_RTSP_hosts,spacer),bg=self.root_bg,fg=self.root_fg,font=('Arial',10))
        self.LOAD_SOCKET_RTSP_HOSTS_Button.grid(row=26+spacer-2,column=3,sticky='sw')



        self.SOCKET_PREFIX_VAR=tk.StringVar()
        self.SOCKET_PREFIX_VAR.set(self.SOCKET_PREFIX)
        self.SOCKET_PREFIX_entry=tk.Entry(self.top,textvariable=self.SOCKET_PREFIX_VAR)
        self.SOCKET_PREFIX_entry.grid(row=27+spacer-2,column=3,sticky='nw')
        self.SOCKET_PREFIX_label=tk.Label(self.top,text='SOCKET PREFIX',bg=self.root_bg,fg=self.root_fg,font=('Arial',10))
        self.SOCKET_PREFIX_label.grid(row=28+spacer-2,column=3,sticky='nw')



    def load_SOCKET_RTSP_ports(self,spacer):
        try:
            del self.USE_SOCKET_RTSP_PORT_dropdown
        except:
            pass
        self.USE_SOCKET_RTSP_PORT_VAR=tk.StringVar()
        self.load_SOCKET_RTSP_PORT_list()
        if len(self.SOCKET_RTSP_PORTS)>0:
            success=True
            for port_i in self.SOCKET_RTSP_PORTS:
                try:
                    int(port_i)
                except:
                    success=False
                    print('Bad PORT = {}'.format(port_i))
                    break
        else:
            self.SOCKET_RTSP_PORTS=['8889']
        self.USE_SOCKET_RTSP_PORT_VAR.set(self.SOCKET_RTSP_PORTS[0])
        self.USE_SOCKET_RTSP_PORT_dropdown=tk.OptionMenu(self.top,self.USE_SOCKET_RTSP_PORT_VAR,*self.SOCKET_RTSP_PORTS,command=self.return_SOCKET_RTSP_PORT)
        self.USE_SOCKET_RTSP_PORT_dropdown.grid(row=19+spacer-2,column=3,sticky='sw')

    def load_SOCKET_RTSP_hosts(self,spacer):
        try:
            del self.USE_SOCKET_RTSP_HOST_dropdown
        except:
            pass
        self.USE_SOCKET_RTSP_HOST_VAR=tk.StringVar()
        self.load_SOCKET_RTSP_HOST_list()
        if len(self.SOCKET_RTSP_HOSTS)>0:
            success=True
            for sock_i in self.SOCKET_RTSP_HOSTS:
                try:
                    pass
                except:
                    success=False
                    print('Bad SOCKET = {}'.format(sock_i))
                    break
        else:
            self.SOCKET_RTSP_HOSTS=['10.5.1.201']
        self.USE_SOCKET_RTSP_HOST_VAR.set(self.SOCKET_RTSP_HOSTS[0])
        self.USE_SOCKET_RTSP_HOST_dropdown=tk.OptionMenu(self.top,self.USE_SOCKET_RTSP_HOST_VAR,*self.SOCKET_RTSP_HOSTS,command=self.return_SOCKET_RTSP_HOST)
        self.USE_SOCKET_RTSP_HOST_dropdown.grid(row=23+spacer-2,column=3,sticky='sw')

    def check_int_from_str(self,str_i):
        success=True
        try:
            int(str_i)
        except:
            success=False
        return success

    def load_SOCKET_RTSP_PORT_list(self):
        self.SOCKET_RTSP_PORT_LIST='libs/SOCKET_RTSP_PORT_LIST.txt'
        if os.path.exists(self.SOCKET_RTSP_PORT_LIST):
            f=open(self.SOCKET_RTSP_PORT_LIST,'r')
            f_read=f.readlines()
            f.close()
            self.SOCKET_RTSP_PORTS=[w.replace(' ','').replace('\n','') for w in f_read if self.check_int_from_str(w)]
        else:
            f=open(self.SOCKET_RTSP_PORT_LIST,'w')
            f.writelines('8889\n')
            f.close()
            f=open(self.SOCKET_RTSP_PORT_LIST,'r')
            f_read=f.readlines()
            f.close()
            self.SOCKET_RTSP_PORTS=[w.replace(' ','').replace('\n','') for w in f_read if self.check_int_from_str(w)]

    def load_SOCKET_RTSP_HOST_list(self):
        self.SOCKET_RTSP_HOST_LIST='libs/SOCKET_RTSP_HOST_LIST.txt'
        if os.path.exists(self.SOCKET_RTSP_HOST_LIST):
            f=open(self.SOCKET_RTSP_HOST_LIST,'r')
            f_read=f.readlines()
            f.close()
            self.SOCKET_RTSP_HOSTS=[w.replace(' ','').replace('\n','') for w in f_read]
        else:
            f=open(self.SOCKET_RTSP_HOST_LIST,'w')
            f.writelines('10.5.1.201\n')
            f.close()
            f=open(self.SOCKET_RTSP_HOST_LIST,'r')
            f_read=f.readlines()
            f.close()
            self.SOCKET_RTSP_HOSTS=[w.replace(' ','').replace('\n','') for w in f_read]

    def load_RTSP_clients(self,spacer):
        try:
            del self.USE_RTSP_CLIENT_dropdown
        except:
            pass
        self.USE_RTSP_CLIENT_VAR=tk.StringVar()
        self.load_RTSP_CLIENT_list()
        if len(self.RTSP_CLIENTS)>0:
            pass
        else:
            self.RTSP_CLIENTS=['NO RTSP OPTIONS AVAILABLE']
        self.USE_RTSP_CLIENT_VAR.set(self.RTSP_CLIENTS[0])
        self.USE_RTSP_CLIENT_dropdown=tk.OptionMenu(self.top,self.USE_RTSP_CLIENT_VAR,*self.RTSP_CLIENTS,command=self.return_RTSP_CLIENT)
        self.USE_RTSP_CLIENT_dropdown.grid(row=15+spacer-2,column=3,sticky='sw')


    def open_RTSP_CLIENT_List(self):
        cmd_i=open_cmd+" '{}'".format(self.RTSP_CLIENT_LIST)
        self.run_cmd(cmd_i)

    def open_SOCKET_RTSP_PORT_List(self):
        cmd_i=open_cmd+" '{}'".format(self.SOCKET_RTSP_PORT_LIST)
        self.run_cmd(cmd_i)   
    def open_SOCKET_RTSP_HOST_List(self):
        cmd_i=open_cmd+" '{}'".format(self.SOCKET_RTSP_HOST_LIST)
        self.run_cmd(cmd_i) 

    def return_RTSP_CLIENT(self):
        print(self.USE_RTSP_CLIENT_VAR.get())

    def return_SOCKET_RTSP_PORT(self):
        print(self.USE_SOCKET_RTSP_PORT_VAR.get())

    def return_SOCKET_RTSP_HOST(self):
        print(self.USE_SOCKET_RTSP_HOST_VAR.get())

    def load_RTSP_CLIENT_list(self):
        self.RTSP_CLIENT_LIST='libs/RTSP_CLIENT_LIST.txt'
        if os.path.exists(self.RTSP_CLIENT_LIST):
            f=open(self.RTSP_CLIENT_LIST,'r')
            f_read=f.readlines()
            f.close()
            self.RTSP_CLIENTS=[w.replace(' ','').replace('\n','') for w in f_read if w.find('rtsp')!=-1]
        else:
            f=open(self.RTSP_CLIENT_LIST,'w')
            f.writelines('NO RTSP OPTIONS AVAILABLE\n')
            f.close()
            f=open(self.RTSP_CLIENT_LIST,'r')
            f_read=f.readlines()
            f.close()
            self.RTSP_CLIENTS=[w.replace(' ','').replace('\n','') for w in f_read if w.find('rtsp')!=-1]



    def get_full_path_rtsp(self):
        try:
            s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            s.connect(("8.8.8.8",80))
            self.IP_ADDRESS=s.getsockname()[0]
        except:
            self.IP_ADDRESS="127.0.0.1"
        self.RTSP_FULL_PATH="rtsp://"+self.IP_ADDRESS+":"+str(self.PORT_VAR.get())+self.STREAM_KEY_VAR.get()
        self.RTSP_FULL_PATH_VAR.set(self.RTSP_FULL_PATH)
    def YOUTUBE_RTMP(self):
        if self.YOUTUBE_KEY_VAR==None:
            self.YOUTUBE_KEY_VAR=tk.StringVar()
            self.YOUTUBE_KEY_VAR.set(self.YOUTUBE_KEY)
            self.YOUTUBE_KEY_entry=tk.Entry(self.root,textvariable=self.YOUTUBE_KEY_VAR)
            self.YOUTUBE_KEY_entry.grid(row=11,column=2,sticky='sw')
            self.YOUTUBE_KEY_label=tk.Label(self.root,text='YOUTUBE STREAM KEY',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
            self.YOUTUBE_KEY_label.grid(row=11,column=2,sticky='nw')  
            
            self.SETTINGS_YOUTUBE_LIST=['720p','1080p','480p']
            self.USER_SELECTION_yt=tk.StringVar()
            self.USER_SELECTION_yt.set('720p')
            self.dropdown_yt=tk.OptionMenu(self.root,self.USER_SELECTION_yt,*self.SETTINGS_YOUTUBE_LIST)
            self.dropdown_yt.grid(row=11,column=3,sticky='sw')




    def test_yolo(self):
        self.TMP_create_test_bash()
        cmd_i=" bash '{}'".format(self.save_cfg_path_test.replace('.cfg','.sh'))
        #cmd_i=" bash '{}'".format(self.tmp_test_path)
        self.test_yolo_objs_button=Button(self.top,image=self.icon_test,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
        self.test_yolo_objs_button.grid(row=5,column=2,sticky='se')
        self.test_yolo_objs_button_note=tk.Label(self.top,text='webcam \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.test_yolo_objs_button_note.grid(row=6,column=2,sticky='ne')

    # def test_yolo(self):
    #     self.TMP_create_test_bash()
    #     cmd_i=" bash '{}'".format(self.save_cfg_path_test.replace('.cfg','.sh'))
    #     #cmd_i=" bash '{}'".format(self.tmp_test_path)
    #     self.test_yolo_objs_button=Button(self.root,image=self.icon_test,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
    #     self.test_yolo_objs_button.grid(row=12,column=1,sticky='se')
    #     self.test_yolo_objs_button_note=tk.Label(self.root,text='5.a \n Test Yolov4',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
    #     self.test_yolo_objs_button_note.grid(row=13,column=1,sticky='ne')

    def test_yolodnn(self):
        self.TMP_create_test_dnn_bash()
        cmd_i=" bash '{}'".format(self.save_cfg_path_test.replace('.cfg','dnn.sh'))
        #cmd_i=" bash '{}'".format(self.tmp_test_path)
        self.test_yolo_objsdnn_button=Button(self.top,image=self.icon_test,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
        self.test_yolo_objsdnn_button.grid(row=13,column=2,sticky='se')
        self.test_yolo_objsdnn_button_note=tk.Label(self.top,text='DNN \n webcam \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.test_yolo_objsdnn_button_note.grid(row=14,column=2,sticky='ne')

    def test_yolodnn_labelimg(self):
        if os.path.exists('libs/yolov7_path.py'):
            self.create_test_bash_dnn_labelimg()
            cmd_i=" bash '{}'".format(self.save_cfg_path_test.replace('.cfg','dnn_labelimg.sh'))
            self.test_yolo_labelimg_objs_button=Button(self.top,image=self.icon_labelImg,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
            self.test_yolo_labelimg_objs_button.grid(row=14-7+10,column=2,sticky='se')
            self.test_yolo_labelimg_objs_button_note=tk.Label(self.top,text='inference \n labelImg \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_yolo_labelimg_objs_button_note.grid(row=15-7+10,column=2,sticky='ne')


    def test_yolodnn_rtsp(self):
        self.TMP_create_test_dnn_bash_rtsp()
        cmd_i=" bash '{}'".format(self.save_cfg_path_test.replace('.cfg','dnn_rtsp.sh'))
        #cmd_i=" bash '{}'".format(self.tmp_test_path)
        self.test_yolo_objsdnn_button_rtsp=Button(self.top,image=self.icon_test,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
        self.test_yolo_objsdnn_button_rtsp.grid(row=13+2,column=2,sticky='se')
        self.test_yolo_objsdnn_button_note_rtsp=tk.Label(self.top,text='DNN \n rtsp \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.test_yolo_objsdnn_button_note_rtsp.grid(row=14+2,column=2,sticky='ne')

    def test_yolodnn_rtmp(self):
        self.test_yolo_objsdnn_rtmp_button=Button(self.top,image=self.icon_test,command=self.run_cmd_rtmp,bg=self.root_bg,fg=self.root_fg)
        self.test_yolo_objsdnn_rtmp_button.grid(row=3,column=2,sticky='se')
        self.test_yolo_objsdnn_rtmp_button_note=tk.Label(self.top,text='DNN webcam \n RTMP \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.test_yolo_objsdnn_rtmp_button_note.grid(row=4,column=2,sticky='ne')
    
    def run_cmd_rtmp(self):
        self.create_test_bash_dnn_rtmp()
        self.TMP_create_test_dnn_rtmp_bash()
        cmd_i=" bash '{}'".format(self.save_cfg_path_test.replace('.cfg','dnn_rtmp.sh'))
        self.run_cmd(cmd_i)

    def test_yolo_predict(self):
        self.create_test_bash_images_with_predictions()
        cmd_i=" bash '{}'".format(self.save_cfg_path_test.replace('.cfg','_images_with_predictions.sh'))
        #cmd_i=" bash '{}'".format(self.tmp_test_path)
        self.test_yolo_pred_objs_button=Button(self.top,image=self.icon_test,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
        self.test_yolo_pred_objs_button.grid(row=11,column=2,sticky='se')
        self.test_yolo_pred_objs_button_note=tk.Label(self.top,text='COCO mAP\n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.test_yolo_pred_objs_button_note.grid(row=12,column=2,sticky='ne')

    def test_yolo_predict_mAP(self):
        self.create_test_bash_images_with_predictions_mAP()
        cmd_i=" bash '{}'".format(self.save_cfg_path_test.replace('.cfg','_images_with_predictions_mAP.sh'))
        #cmd_i=" bash '{}'".format(self.tmp_test_path)
        self.test_yolo_res_objs_button=Button(self.top,image=self.icon_test,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
        self.test_yolo_res_objs_button.grid(row=9,column=2,sticky='se')
        self.test_yolo_res_objs_button_note=tk.Label(self.top,text='mAP \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.test_yolo_res_objs_button_note.grid(row=10,column=2,sticky='ne')
        

    def test_yolo_mp4(self):
        if os.path.exists(self.mp4_video_path)==False or (self.mp4_video_path.find('.mp4')==-1 and self.mp4_video_path.find('.MP4')==-1):
            print('This does NOT exist or is not a .mp4 file')
            self.create_test_bash_mp4()
            self.create_test_bash_mp4_record()
        else:
            self.create_test_bash_mp4()
            self.create_test_bash_mp4_record()
            self.TMP_create_test_bash_mp4()
            #cmd_i=" bash '{}'".format(self.save_cfg_path_test.replace('.cfg','.sh'))
            cmd_i=" bash '{}'".format(self.tmp_test_path)
            #os.system('xdg-open {}'.format(self.tmp_test_path))
            #cmd_i=" bash '{}'".format(self.save_cfg_path_test.replace('.cfg','_mp4.sh'))
            #print('cmd_i: \n {}'.format(cmd_i))
            self.test_mp4_yolo_objs_button=Button(self.top,image=self.icon_test_mp4,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
            self.test_mp4_yolo_objs_button.grid(row=7,column=2,sticky='se')
            self.test_mp4_yolo_objs_button_note=tk.Label(self.top,text='mp4 \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_mp4_yolo_objs_button_note.grid(row=8,column=2,sticky='ne')

    def popupWindow_MOSAIC(self):
        try:
            self.top.destroy()
        except:
            pass
        self.top=tk.Toplevel(self.root)
        self.top.geometry( "{}x{}".format(int(self.root.winfo_screenwidth()*0.95//1.1),int(self.root.winfo_screenheight()*0.95//1.1)) )
        self.top.title('LAUNCH MOSAIC CHIP SORTER?')
        self.top.configure(background = 'black')
        self.b=Button(self.top,text='Close',command=self.cleanup,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.b.grid(row=1,column=0,sticky='se')
        self.submit_MOSAIC=Button(self.top,image=self.icon_MOSAIC,command=partial(self.open_MOSAIC,False),bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.submit_MOSAIC.grid(row=0,column=4,sticky='se')
        self.submit_MOSAIC_label=tk.Label(self.top,text="Open Training Dataset",bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.submit_MOSAIC_label.grid(row=1,column=4,sticky='ne')

        self.submit_MOSAIC_CUSTOM=Button(self.top,image=self.icon_MOSAIC,command=partial(self.open_MOSAIC,True),bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.submit_MOSAIC_CUSTOM.grid(row=2,column=4,sticky='se')
        self.submit_MOSAIC_label_CUSTOM=tk.Label(self.top,text="Open Custom Dataset",bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.submit_MOSAIC_label_CUSTOM.grid(row=3,column=4,sticky='ne')

        self.open_anno_CUSTOM()
        self.open_jpeg_CUSTOM()

    def popupWindow_labelImg(self):
        try:
            self.top.destroy()
        except:
            pass
        self.top=tk.Toplevel(self.root)
        self.top.geometry( "{}x{}".format(int(self.root.winfo_screenwidth()*0.95//1.1),int(self.root.winfo_screenheight()*0.95//1.1)) )
        self.top.title('LAUNCH labelImg?')
        self.top.configure(background = 'black')
        self.b=Button(self.top,text='Close',command=self.cleanup,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.b.grid(row=1,column=0,sticky='se')
        self.submit_LABELIMG=Button(self.top,image=self.icon_labelImg,command=partial(self.open_labelImg,False),bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.submit_LABELIMG.grid(row=0,column=4,sticky='se')
        self.submit_LABELIMG_label=tk.Label(self.top,text="Open Training Dataset",bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.submit_LABELIMG_label.grid(row=1,column=4,sticky='ne')

        self.submit_LABELIMG_CUSTOM=Button(self.top,image=self.icon_labelImg,command=partial(self.open_labelImg,True),bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.submit_LABELIMG_CUSTOM.grid(row=2,column=4,sticky='se')
        self.submit_LABELIMG_label_CUSTOM=tk.Label(self.top,text="Open Custom Dataset",bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.submit_LABELIMG_label_CUSTOM.grid(row=3,column=4,sticky='ne')

        self.open_anno_CUSTOM()
        self.open_jpeg_CUSTOM()

    def popupWindow_IMGAUG(self):
        try:
            self.top.destroy()
        except:
            pass
        self.top=tk.Toplevel(self.root)
        self.top.geometry( "{}x{}".format(int(self.root.winfo_screenwidth()*0.95//1.1),int(self.root.winfo_screenheight()*0.95//1.1)) )
        self.top.title('LAUNCH IMGAUG GUI?')
        self.top.configure(background = 'black')
        self.b=Button(self.top,text='Close',command=self.cleanup,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.b.grid(row=1,column=0,sticky='se')
        self.submit_IMGAUG=Button(self.top,image=self.icon_IMGAUG,command=partial(self.open_IMGAUG,False),bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.submit_IMGAUG.grid(row=0,column=4,sticky='se')
        self.submit_IMGAUG_label=tk.Label(self.top,text="Open Training Dataset",bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.submit_IMGAUG_label.grid(row=1,column=4,sticky='ne')

        self.submit_IMGAUG_CUSTOM=Button(self.top,image=self.icon_IMGAUG,command=partial(self.open_IMGAUG,True),bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.submit_IMGAUG_CUSTOM.grid(row=2,column=4,sticky='se')
        self.submit_IMGAUG_label_CUSTOM=tk.Label(self.top,text="Open Custom Dataset",bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.submit_IMGAUG_label_CUSTOM.grid(row=3,column=4,sticky='ne')

        self.open_anno_CUSTOM()
        self.open_jpeg_CUSTOM()
    
    def open_jpeg_CUSTOM(self,row_i=6,col_i=6,columnspan=75,sticky='sew',location=None,sticky2='ne',sticky3='se'):
        if self.open_jpeg_selected_CUSTOM==True:
            self.open_jpeg_label_CUSTOM.destroy()
            self.open_jpeg_note_CUSTOM.destroy()
            self.open_jpeg_button_CUSTOM.destroy()
            del self.open_jpeg_label_CUSTOM
            del self.open_jpeg_note_CUSTOM
            del self.open_jpeg_button_CUSTOM
        try:
            if location==None:
                location=self.top
            else:
                location=location
        except:
            location=location
        print('location=',location)
        
        self.open_jpeg_label_var_CUSTOM=tk.StringVar()
        self.open_jpeg_label_var_CUSTOM.set(self.path_JPEGImages_CUSTOM)
        self.open_jpeg_button_CUSTOM=Button(location,image=self.icon_folder,command=partial(self.select_folder,os.path.dirname(self.path_Yolo),'Open Custom JPEGImages Folder',self.open_jpeg_label_var_CUSTOM),bg=self.root_bg,fg=self.root_fg)

        self.open_jpeg_note_CUSTOM=tk.Label(location,text="Custom JPEGImages dir",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))

        if col_i>0:
            self.open_jpeg_button_CUSTOM.grid(row=row_i,column=col_i,sticky=sticky2)
            self.open_jpeg_note_CUSTOM.grid(row=row_i+1,column=col_i,sticky=sticky3)
        cmd_i=open_cmd+" '{}'".format(self.open_jpeg_label_var_CUSTOM.get())
        self.open_jpeg_label_CUSTOM=Button(location,textvariable=self.open_jpeg_label_var_CUSTOM, command=self.open_custom_jpeg_cmd,bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.open_jpeg_label_CUSTOM.grid(row=row_i,column=col_i+1,columnspan=columnspan,sticky=sticky)
        self.open_jpeg_selected_CUSTOM=True

    def open_custom_jpeg_cmd(self):
        cmd_i=open_cmd+" '{}'".format(self.open_jpeg_label_var_CUSTOM.get())
        self.run_cmd(cmd_i)

    def open_custom_anno_cmd(self):
        cmd_i=open_cmd+" '{}'".format(self.open_anno_label_var_CUSTOM.get())
        self.run_cmd(cmd_i)

    def labelImg_buttons(self):
        if os.path.exists('libs/labelImg_path.py'):

            self.labelImg_button=Button(self.root,image=self.icon_labelImg,command=self.popupWindow_labelImg,bg=self.root_bg,fg=self.root_fg)
            self.labelImg_button.grid(row=9,column=4,sticky='se')
            self.labelImg_button_note=tk.Label(self.root,text='LabelImg',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.labelImg_button_note.grid(row=10,column=4,sticky='ne')    

    def MOSAIC_buttons(self):
        if os.path.exists('libs/MOSAIC_Chip_Sorter_path.py'):

            self.MOSAIC_button=Button(self.root,image=self.icon_MOSAIC,command=self.popupWindow_MOSAIC,bg=self.root_bg,fg=self.root_fg)
            self.MOSAIC_button.grid(row=9,column=5,sticky='se')
            self.MOSAIC_button_note=tk.Label(self.root,text='MOSAIC Chip Sorter',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.MOSAIC_button_note.grid(row=10,column=5,sticky='ne')         

    def IMGAUG_buttons(self):
        if os.path.exists('libs/IMAGE_AUG_GUI_path.py'):

            self.IMGAUG_button=Button(self.root,image=self.icon_IMGAUG,command=self.popupWindow_IMGAUG,bg=self.root_bg,fg=self.root_fg)
            self.IMGAUG_button.grid(row=9,column=6,sticky='se')
            self.IMGAUG_button_note=tk.Label(self.root,text='IMAGE AUG GUI',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.IMGAUG_button_note.grid(row=10,column=6,sticky='ne') 

    def CLASSIFY_CHIPS_buttons(self):
        if os.path.exists('libs/CLASSIFY_CHIPS_path.py'):
            self.CLASSIFY_CHIPS_button=Button(self.root,image=self.icon_CLASSIFY_CHIPS,command=self.open_CLASSIFY_CHIPS,bg=self.root_bg,fg=self.root_fg)
            self.CLASSIFY_CHIPS_button.grid(row=9,column=7,sticky='se')
            self.CLASSIFY_CHIPS_button_note=tk.Label(self.root,text='CLASSIFY CHIPS GUI',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.CLASSIFY_CHIPS_button_note.grid(row=10,column=7,sticky='ne')       

    def open_CLASSIFY_CHIPS(self):
        from libs import CLASSIFY_CHIPS_path
        from multiprocessing import Process
        self.path_CLASSIFY_CHIPS=CLASSIFY_CHIPS_path.path
        self.PYTHON_PATH="python3"
        if os.path.exists(self.path_CLASSIFY_CHIPS):
            self.cmd_i='cd {};{} "{}"'.format(os.path.dirname(self.path_CLASSIFY_CHIPS),self.PYTHON_PATH, self.path_CLASSIFY_CHIPS)
            if os.path.exists(self.CLASSIFY_CHIPS_SETTINGS_PATH_VAR.get()):
                self.cmd_i=self.cmd_i+' --SETTINGS_PATH= "{}"'.format(self.CLASSIFY_CHIPS_SETTINGS_PATH_VAR.get())
            else:
                self.cmd_i=self.cmd_i
            self.CLASSIFY_CHIPS=Process(target=self.run_cmd,args=(self.cmd_i,)).start()
        else:
            self.popup_text='Please provide a valid CLASSIFY_CHIPS.py path. \n  Current path is: {}'.format(self.path_CLASSIFY_CHIPS)     

    def open_MOSAIC(self,custom):
        from libs import MOSAIC_Chip_Sorter_path
        from multiprocessing import Process
        self.path_MOSAIC=MOSAIC_Chip_Sorter_path.path
        self.PYTHON_PATH="python3"
        self.path_JPEGImages_CUSTOM=self.open_jpeg_label_var_CUSTOM.get()
        self.path_Annotations_CUSTOM=self.open_anno_label_var_CUSTOM.get()
        if custom:
            pass
        else:
            self.path_JPEGImages_CUSTOM=self.path_JPEGImages
            self.path_Annotations_CUSTOM=self.path_Annotations

        if os.path.exists(self.path_MOSAIC):
            self.cmd_i='cd {};{} "{}" --path_Annotations "{}" --path_JPEGImages "{}"'.format(os.path.dirname(self.path_MOSAIC),self.PYTHON_PATH ,self.path_MOSAIC,self.path_Annotations_CUSTOM,self.path_JPEGImages_CUSTOM)
            self.MOSAIC=Process(target=self.run_cmd,args=(self.cmd_i,)).start()
        else:
            self.popup_text='Please provide a valid MOSAIC_Chip_Sorter.py path. \n  Current path is: {}'.format(self.path_MOSAIC)

    def open_IMGAUG(self,custom):
        from libs import IMAGE_AUG_GUI_path
        from multiprocessing import Process
        self.path_IMGAUG=IMAGE_AUG_GUI_path.path
        self.PYTHON_PATH="python3"
        self.path_JPEGImages_CUSTOM=self.open_jpeg_label_var_CUSTOM.get()
        self.path_Annotations_CUSTOM=self.open_anno_label_var_CUSTOM.get()
        if custom:
            pass
        else:
            self.path_JPEGImages_CUSTOM=self.path_JPEGImages
            self.path_Annotations_CUSTOM=self.path_Annotations
        if os.path.exists(self.path_IMGAUG):
            self.cmd_i='cd {};{} "{}" --path_Annotations "{}" --path_JPEGImages "{}"'.format(os.path.dirname(self.path_IMGAUG),self.PYTHON_PATH ,self.path_IMGAUG,self.path_Annotations_CUSTOM,self.path_JPEGImages_CUSTOM)
            self.IMGAUG=Process(target=self.run_cmd,args=(self.cmd_i,)).start()
        else:
            self.popup_text='Please provide a valid IMAGE_AUG_GUI.py path. \n  Current path is: {}'.format(self.path_IMGAUG)

    def open_labelImg(self,custom):
        from libs import labelImg_path
        from multiprocessing import Process
        self.path_labelImg=labelImg_path.path
        self.path_labelImg_predefined_classes_file=os.path.join(os.path.dirname(self.names_path),'predefined_classes.txt')
        shutil.copy(self.names_path,self.path_labelImg_predefined_classes_file)
        self.path_JPEGImages_CUSTOM=self.open_jpeg_label_var_CUSTOM.get()
        self.path_Annotations_CUSTOM=self.open_anno_label_var_CUSTOM.get()
        if custom:
            pass
        else:
            self.path_JPEGImages_CUSTOM=self.path_JPEGImages
            self.path_Annotations_CUSTOM=self.path_Annotations
        self.path_labelImg_save_dir=self.path_Annotations_CUSTOM
        self.path_labelImg_image_dir=self.path_JPEGImages_CUSTOM
        self.PYTHON_PATH="python3"
        if os.path.exists(self.path_labelImg):
            self.cmd_i='{} "{}" "{}" "{}" "{}"'.format(self.PYTHON_PATH ,self.path_labelImg,self.path_labelImg_image_dir,self.path_labelImg_predefined_classes_file,self.path_labelImg_save_dir)
            self.labelImg=Process(target=self.run_cmd,args=(self.cmd_i,)).start()
        else:
            self.popup_text='Please provide a valid labelImg.py path. \n  Current path is: {}'.format(self.path_labelImg)

    def train_yolov7(self):
        if os.path.exists('libs/yolov7_path.py'):
            self.train_yolov7_objs_button=Button(self.top,image=self.icon_train,command=self.train_yolov7_cmd,bg=self.root_bg,fg=self.root_fg)
            self.train_yolov7_objs_button.grid(row=10-7,column=10-7,sticky='se')
            self.train_yolov7_objs_button_note=tk.Label(self.top,text='Train',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.train_yolov7_objs_button_note.grid(row=11-7,column=10-7,sticky='ne')



            self.epochs_yolov7_entry=tk.Entry(self.top,textvariable=self.epochs_yolov7_VAR)
            self.epochs_yolov7_entry.grid(row=12-7,column=10-7,sticky='sw')
            self.epochs_yolov7_label=tk.Label(self.top,text='epochs',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
            self.epochs_yolov7_label.grid(row=13-7,column=10-7,sticky='nw')
            #self.test_yolov7_mp4()
            #self.test_yolov7_webcam()
            #self.test_yolov7_webcam_RTMP()
            #self.test_yolov7_mAP()
    def train_yolov7_madness(self):
        self.clear_cache_yolov7()
        self.create_test_bash_mp4_yolov7()
        self.create_train_bash_yolov7()
        self.save_settings()
    def train_yolov7_cmd(self):
        cmd_i=" bash '{}'".format(self.TRAIN_YOLOV7)
        if self.train_load_repeat.get()=='None':
            self.run_cmd(cmd_i)
        else:
            self.loop_train_load(cmd_i,'train_yolov7_madness')


    def train_yolov7_e6e(self):
        if os.path.exists('libs/yolov7_path.py'):
            self.train_yolov7_e6e_objs_button=Button(self.top,image=self.icon_train,command=self.train_yolov7_e6e_cmd,bg=self.root_bg,fg=self.root_fg)
            self.train_yolov7_e6e_objs_button.grid(row=10-7,column=11-7,sticky='se')
            self.train_yolov7_e6e_objs_button_note=tk.Label(self.top,text='Train',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.train_yolov7_e6e_objs_button_note.grid(row=11-7,column=11-7,sticky='ne')


           

            self.epochs_yolov7_e6e_entry=tk.Entry(self.top,textvariable=self.epochs_yolov7_e6e_VAR)
            self.epochs_yolov7_e6e_entry.grid(row=12-7,column=11-7,sticky='sw')
            self.epochs_yolov7_e6e_label=tk.Label(self.top,text='epochs',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
            self.epochs_yolov7_e6e_label.grid(row=13-7,column=11-7,sticky='nw')
            #self.test_yolov7_mp4_e6e()
            #self.test_yolov7_webcam_e6e()
            #self.test_yolov7_webcam_e6e_RTMP()
            #self.test_yolov7_mAP_e6e()
    def clear_cache_yolov7(self):
        self.YOLO_MODEL_PATH=os.path.join(self.base_path_OG,self.prefix_foldername)
        if os.path.exists(os.path.join(self.YOLO_MODEL_PATH,'train.cache')):
            print('Deleting {}'.format(os.path.join(self.YOLO_MODEL_PATH,'train.cache')))
            os.remove(os.path.join(self.YOLO_MODEL_PATH,'train.cache'))
        if os.path.exists(os.path.join(self.YOLO_MODEL_PATH,'valid.cache')):
            print('Deleting {}'.format(os.path.join(self.YOLO_MODEL_PATH,'valid.cache')))
            os.remove(os.path.join(self.YOLO_MODEL_PATH,'valid.cache'))
    def train_yolov7_e6e_madness(self):
        self.clear_cache_yolov7()
        self.create_test_bash_mp4_yolov7_e6e()
        self.create_train_bash_yolov7_e6e()
        self.save_settings()

    def train_yolov7_e6e_cmd(self):
        self.train_yolov7_e6e_madness()
        cmd_i=" bash '{}'".format(self.TRAIN_YOLOV7_e6e)
        if self.train_load_repeat.get()=='None':
            self.run_cmd(cmd_i)
        else:
            self.loop_train_load(cmd_i,'train_yolov7_e6e_madness')

    def train_yolov7_re(self):
        if os.path.exists('libs/yolov7_path.py'):
            self.test_mp4_yolov7_re_objs_button=Button(self.top,image=self.icon_train,command=self.train_yolov7_re_cmd,bg=self.root_bg,fg=self.root_fg)
            self.test_mp4_yolov7_re_objs_button.grid(row=10-7,column=12-7,sticky='se')
            self.test_mp4_yolov7_re_objs_button_note=tk.Label(self.top,text='Train',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_mp4_yolov7_re_objs_button_note.grid(row=11-7,column=12-7,sticky='ne')


            self.epochs_yolov7_re_entry=tk.Entry(self.top,textvariable=self.epochs_yolov7_re_VAR)
            self.epochs_yolov7_re_entry.grid(row=12-7,column=12-7,sticky='sw')
            self.epochs_yolov7_re_label=tk.Label(self.top,text='epochs',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
            self.epochs_yolov7_re_label.grid(row=13-7,column=12-7,sticky='nw')
            #self.test_yolov7_mp4_re()
            #self.test_yolov7_webcam_re()
            #self.test_yolov7_webcam_re_RTMP()
            #self.test_yolov7_mAP_re()

    def train_yolov7_re_madness(self):
        self.clear_cache_yolov7()
        self.create_test_bash_mp4_yolov7_re()
        self.create_train_bash_yolov7_re()
        self.save_settings()

    def train_yolov7_re_cmd(self):
        self.train_yolov7_re_madness()
        cmd_i=" bash '{}'".format(self.TRAIN_YOLOV7_re)
        if self.train_load_repeat.get()=='None':
            self.run_cmd(cmd_i)
        else:
            self.loop_train_load(cmd_i,'train_yolov7_re_madness')

    def test_yolov7_mp4(self):
        if os.path.exists(self.mp4_video_path) and self.mp4_video_path.lower().find('.mp4')!=-1 and os.path.exists('libs/yolov7_path.py'):
            self.create_test_bash_mp4_yolov7()
            cmd_i=" bash '{}'".format(self.TEST_MP4_YOLOV7)
            self.test_mp4_yolov7_objs_button=Button(self.top,image=self.icon_test_mp4,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
            self.test_mp4_yolov7_objs_button.grid(row=14-7,column=10-7,sticky='se')
            self.test_mp4_yolov7_objs_button_note=tk.Label(self.top,text='mp4 \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_mp4_yolov7_objs_button_note.grid(row=15-7,column=10-7,sticky='ne')

    def test_yolov7_labelimg(self):
        if os.path.exists(self.mp4_video_path) and self.mp4_video_path.lower().find('.mp4')!=-1 and os.path.exists('libs/yolov7_path.py'):
            self.create_test_bash_labelimg_yolov7()
            cmd_i=" bash '{}'".format(self.TEST_labelimg_YOLOV7)
            self.test_labelimg_yolov7_objs_button=Button(self.top,image=self.icon_labelImg,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
            self.test_labelimg_yolov7_objs_button.grid(row=14-7+10,column=10-7,sticky='se')
            self.test_labelimg_yolov7_objs_button_note=tk.Label(self.top,text='inference \n labelImg \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_labelimg_yolov7_objs_button_note.grid(row=15-7+10,column=10-7,sticky='ne')

    def test_yolov7_mp4_e6e(self):
        if os.path.exists(self.mp4_video_path) and self.mp4_video_path.lower().find('.mp4')!=-1 and os.path.exists('libs/yolov7_path.py'):
            self.create_test_bash_mp4_yolov7_e6e()
            cmd_i=" bash '{}'".format(self.TEST_MP4_YOLOV7_e6e)
            self.test_mp4_yolov7_e6e_objs_button=Button(self.top,image=self.icon_test_mp4,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
            self.test_mp4_yolov7_e6e_objs_button.grid(row=14-7,column=11-7,sticky='se')
            self.test_mp4_yolov7_e6e_objs_button_note=tk.Label(self.top,text='mp4 \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_mp4_yolov7_e6e_objs_button_note.grid(row=15-7,column=11-7,sticky='ne')

    def test_yolov7_labelimg_e6e(self):
        if os.path.exists(self.mp4_video_path) and self.mp4_video_path.lower().find('.mp4')!=-1 and os.path.exists('libs/yolov7_path.py'):
            self.create_test_bash_labelimg_yolov7_e6e()
            cmd_i=" bash '{}'".format(self.TEST_labelimg_YOLOV7_e6e)
            self.test_labelimg_yolov7_e6e_objs_button=Button(self.top,image=self.icon_labelImg,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
            self.test_labelimg_yolov7_e6e_objs_button.grid(row=14-7+10,column=11-7,sticky='se')
            self.test_labelimg_yolov7_e6e_objs_button_note=tk.Label(self.top,text='inference \n labelImg \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_labelimg_yolov7_e6e_objs_button_note.grid(row=15-7+10,column=11-7,sticky='ne')

    def test_yolov7_mp4_re(self):
        if os.path.exists(self.mp4_video_path) and self.mp4_video_path.lower().find('.mp4')!=-1 and os.path.exists('libs/yolov7_path.py'):
            self.create_test_bash_mp4_yolov7_re()
            cmd_i=" bash '{}'".format(self.TEST_MP4_YOLOV7_re)
            self.test_mp4_yolov7_re_objs_button=Button(self.top,image=self.icon_test_mp4,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
            self.test_mp4_yolov7_re_objs_button.grid(row=14-7,column=12-7,sticky='se')
            self.test_mp4_yolov7_re_objs_button_note=tk.Label(self.top,text='mp4 \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_mp4_yolov7_re_objs_button_note.grid(row=15-7,column=12-7,sticky='ne')

    def test_yolov7_labelimg_re(self):
        if os.path.exists(self.mp4_video_path) and self.mp4_video_path.lower().find('.mp4')!=-1 and os.path.exists('libs/yolov7_path.py'):
            self.create_test_bash_labelimg_yolov7_re()
            cmd_i=" bash '{}'".format(self.TEST_labelimg_YOLOV7_re)
            self.test_labelimg_yolov7_re_objs_button=Button(self.top,image=self.icon_labelImg,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
            self.test_labelimg_yolov7_re_objs_button.grid(row=14-7+10,column=12-7,sticky='se')
            self.test_labelimg_yolov7_re_objs_button_note=tk.Label(self.top,text='inference \n labelImg \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_labelimg_yolov7_re_objs_button_note.grid(row=15-7+10,column=12-7,sticky='ne')

    def test_yolov7_webcam(self):
        if os.path.exists('libs/yolov7_path.py'):
            self.create_test_bash_webcam_yolov7()
            cmd_i=" bash '{}'".format(self.TEST_WEBCAM_YOLOV7)
            self.test_webcam_yolov7_objs_button=Button(self.top,image=self.icon_test,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
            self.test_webcam_yolov7_objs_button.grid(row=12-7,column=10-7,sticky='se')
            self.test_webcam_yolov7_objs_button_note=tk.Label(self.top,text='webcam \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_webcam_yolov7_objs_button_note.grid(row=13-7,column=10-7,sticky='ne')

    def test_yolov7_rtsp(self):
        if os.path.exists('libs/yolov7_path.py'):
            self.create_test_bash_rtsp_yolov7()
            cmd_i=" bash '{}'".format(self.TEST_RTSP_YOLOV7)
            self.test_rtsp_yolov7_objs_button=Button(self.top,image=self.icon_test,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
            self.test_rtsp_yolov7_objs_button.grid(row=12-7+10,column=10-7,sticky='se')
            self.test_rtsp_yolov7_objs_button_note=tk.Label(self.top,text='rtsp \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_rtsp_yolov7_objs_button_note.grid(row=13-7+10,column=10-7,sticky='ne')

    def test_yolov7_webcam_RTMP(self):
        if os.path.exists('libs/yolov7_path.py'):
            self.create_test_bash_webcam_yolov7_RTMP()
            cmd_i=" bash '{}'".format(self.TEST_WEBCAM_YOLOV7_RTMP)
            self.test_webcam_yolov7_RTMP_objs_button=Button(self.top,image=self.icon_test,command=self.run_create_test_bash_webcam_yolov7_RTMP,bg=self.root_bg,fg=self.root_fg)
            self.test_webcam_yolov7_RTMP_objs_button.grid(row=12-7-2,column=10-7,sticky='se')
            self.test_webcam_yolov7_RTMP_objs_button_note=tk.Label(self.top,text='webcam \n RTMP \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_webcam_yolov7_RTMP_objs_button_note.grid(row=13-7-2,column=10-7,sticky='ne')

    def run_create_test_bash_webcam_yolov7_RTMP(self):
        self.create_test_bash_webcam_yolov7_RTMP()
        cmd_i=" bash '{}'".format(self.TEST_WEBCAM_YOLOV7_RTMP)
        self.run_cmd(cmd_i)

    def test_yolov7_webcam_e6e(self):
        if os.path.exists('libs/yolov7_path.py'):
            self.create_test_bash_webcam_yolov7_e6e()
            cmd_i=" bash '{}'".format(self.TEST_WEBCAM_YOLOV7_e6e)
            self.test_webcam_yolov7_e6e_objs_button=Button(self.top,image=self.icon_test,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
            self.test_webcam_yolov7_e6e_objs_button.grid(row=12-7,column=11-7,sticky='se')
            self.test_webcam_yolov7_e6e_objs_button_note=tk.Label(self.top,text='webcam \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_webcam_yolov7_e6e_objs_button_note.grid(row=13-7,column=11-7,sticky='ne')

    def test_yolov7_rtsp_e6e(self):
        if os.path.exists('libs/yolov7_path.py'):
            self.create_test_bash_rtsp_yolov7_e6e()
            cmd_i=" bash '{}'".format(self.TEST_RTSP_YOLOV7_e6e)
            self.test_rtsp_yolov7_e6e_objs_button=Button(self.top,image=self.icon_test,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
            self.test_rtsp_yolov7_e6e_objs_button.grid(row=12-7+10,column=11-7,sticky='se')
            self.test_rtsp_yolov7_e6e_objs_button_note=tk.Label(self.top,text='rtsp \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_rtsp_yolov7_e6e_objs_button_note.grid(row=13-7+10,column=11-7,sticky='ne')

    def test_yolov7_webcam_e6e_RTMP(self):
        if os.path.exists('libs/yolov7_path.py'):
            self.create_test_bash_webcam_yolov7_e6e_RTMP()
            cmd_i=" bash '{}'".format(self.TEST_WEBCAM_YOLOV7_e6e_RTMP)
            self.test_webcam_yolov7_e6e_RTMP_objs_button=Button(self.top,image=self.icon_test,command=self.run_create_test_bash_webcam_yolov7_e6e_RTMP,bg=self.root_bg,fg=self.root_fg)
            self.test_webcam_yolov7_e6e_RTMP_objs_button.grid(row=12-7-2,column=11-7,sticky='se')
            self.test_webcam_yolov7_e6e_RTMP_objs_button_note=tk.Label(self.top,text='webcam \n RTMP \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_webcam_yolov7_e6e_RTMP_objs_button_note.grid(row=13-7-2,column=11-7,sticky='ne')

    def run_create_test_bash_webcam_yolov7_e6e_RTMP(self):
        self.create_test_bash_webcam_yolov7_e6e_RTMP()
        cmd_i=" bash '{}'".format(self.TEST_WEBCAM_YOLOV7_e6e_RTMP)
        self.run_cmd(cmd_i)    

    def test_yolov7_webcam_re(self):
        if os.path.exists('libs/yolov7_path.py'):
            self.create_test_bash_webcam_yolov7_re()
            cmd_i=" bash '{}'".format(self.TEST_WEBCAM_YOLOV7_re)
            self.test_webcam_yolov7_re_objs_button=Button(self.top,image=self.icon_test,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
            self.test_webcam_yolov7_re_objs_button.grid(row=12-7,column=12-7,sticky='se')
            self.test_webcam_yolov7_re_objs_button_note=tk.Label(self.top,text='webcam \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_webcam_yolov7_re_objs_button_note.grid(row=13-7,column=12-7,sticky='ne')

    def test_yolov7_rtsp_re(self):
        if os.path.exists('libs/yolov7_path.py'):
            self.create_test_bash_rtsp_yolov7_re()
            cmd_i=" bash '{}'".format(self.TEST_RTSP_YOLOV7_re)
            self.test_rtsp_yolov7_re_objs_button=Button(self.top,image=self.icon_test,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
            self.test_rtsp_yolov7_re_objs_button.grid(row=12-7+10,column=12-7,sticky='se')
            self.test_rtsp_yolov7_re_objs_button_note=tk.Label(self.top,text='rtsp \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_rtsp_yolov7_re_objs_button_note.grid(row=13-7+10,column=12-7,sticky='ne')

 

    def test_yolov7_webcam_re_RTMP(self):
        if os.path.exists('libs/yolov7_path.py'):
            self.create_test_bash_webcam_yolov7_re_RTMP()
            cmd_i=" bash '{}'".format(self.TEST_WEBCAM_YOLOV7_re_RTMP)
            self.test_webcam_yolov7_re_RTMP_objs_button=Button(self.top,image=self.icon_test,command=self.run_create_test_bash_webcam_yolov7_re_RTMP,bg=self.root_bg,fg=self.root_fg)
            self.test_webcam_yolov7_re_RTMP_objs_button.grid(row=12-7-2,column=12-7,sticky='se')
            self.test_webcam_yolov7_re_RTMP_objs_button_note=tk.Label(self.top,text='webcam \n RTMP \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_webcam_yolov7_re_RTMP_objs_button_note.grid(row=13-7-2,column=12-7,sticky='ne')

    def run_create_test_bash_webcam_yolov7_re_RTMP(self):
        self.create_test_bash_webcam_yolov7_re_RTMP()
        cmd_i=" bash '{}'".format(self.TEST_WEBCAM_YOLOV7_re_RTMP)
        self.run_cmd(cmd_i)  

    def test_yolov7_mAP(self):
        if os.path.exists('libs/yolov7_path.py'):
            self.test_predict_MAP_yolov7_objs_button=Button(self.top,image=self.icon_test,command=self.run_create_predict_bash_mAP_yolov7,bg=self.root_bg,fg=self.root_fg)
            self.test_predict_MAP_yolov7_objs_button.grid(row=16-7,column=10-7,sticky='se')
            self.test_predict_MAP_yolov7_objs_button_note=tk.Label(self.top,text='mAP \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_predict_MAP_yolov7_objs_button_note.grid(row=17-7,column=10-7,sticky='ne')

    def run_create_predict_bash_mAP_yolov7(self):
        self.clear_cache_yolov7()
        self.create_predict_bash_mAP_yolov7()
        cmd_i=" bash '{}'".format(self.TEST_PREDICT_YOLOV7)
        self.run_cmd(cmd_i)

    def test_yolov7_mAP_e6e(self):
        if os.path.exists('libs/yolov7_path.py'):
            self.test_predict_MAP_yolov7_e6e_objs_button=Button(self.top,image=self.icon_test,command=self.run_create_predict_bash_mAP_yolov7_e6e,bg=self.root_bg,fg=self.root_fg)
            self.test_predict_MAP_yolov7_e6e_objs_button.grid(row=16-7,column=11-7,sticky='se')
            self.test_predict_MAP_yolov7_e6e_objs_button_note=tk.Label(self.top,text='mAP \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_predict_MAP_yolov7_e6e_objs_button_note.grid(row=17-7,column=11-7,sticky='ne')

    def run_create_predict_bash_mAP_yolov7_e6e(self):
        self.clear_cache_yolov7()
        self.create_predict_bash_mAP_yolov7_e6e()
        cmd_i=" bash '{}'".format(self.TEST_PREDICT_YOLOV7_e6e)
        self.run_cmd(cmd_i)

    def test_yolov7_mAP_re(self):
        if os.path.exists('libs/yolov7_path.py'):
            self.test_predict_MAP_yolov7_re_objs_button=Button(self.top,image=self.icon_test,command=self.run_create_predict_bash_mAP_yolov7_re,bg=self.root_bg,fg=self.root_fg)
            self.test_predict_MAP_yolov7_re_objs_button.grid(row=16-7,column=12-7,sticky='se')
            self.test_predict_MAP_yolov7_re_objs_button_note=tk.Label(self.top,text='mAP \n',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.test_predict_MAP_yolov7_re_objs_button_note.grid(row=17-7,column=12-7,sticky='ne')

    def run_create_predict_bash_mAP_yolov7_re(self):
        self.clear_cache_yolov7()
        self.create_predict_bash_mAP_yolov7_re()
        cmd_i=" bash '{}'".format(self.TEST_PREDICT_YOLOV7_re)
        self.run_cmd(cmd_i)

    def convert_tflite(self):
        print(os.path.exists(os.path.join(self.CWD,'libs/tensorflow_yolov4_tflite_path.py')))
        print((self.WIDTH_NUM==self.HEIGHT_NUM))
        
        if self.best_weights_path:
            print(os.path.exists(self.best_weights_path))
            if os.path.exists(os.path.join(self.CWD,'libs/tensorflow_yolov4_tflite_path.py')) and (self.WIDTH_NUM==self.HEIGHT_NUM) and os.path.exists(self.best_weights_path):
                self.create_tflite_bash()
                self.convert_tflite_button=Button(self.root,image=self.icon_test,command=self.run_create_tflite_bash,bg=self.root_bg,fg=self.root_fg)
                self.convert_tflite_button.grid(row=10-7,column=13,sticky='se')
                self.convert_tflite_button_note=tk.Label(self.root,text='Convert Yolov4 to TFLITE',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
                self.convert_tflite_button_note.grid(row=11-7,column=13,sticky='ne')

    def calculate_epochs_yolov4(self):
        '''Calculates the number of epochs performed for training'''
        #self.max_batches
        #self.subdivisions
        #self.batch
        #self.train_list_path
        f=open(self.save_cfg_path_train,'r')
        cfg=f.readlines()
        f.close()
        self.get_header(cfg)
        f=open(self.train_list_path,'r')
        self.dataset=len(f.readlines()) # number of samples in training dataset
        f.close()
        print('number of training examples = {}'.format(self.dataset))
        self.iterations_per_epoch=self.dataset//int(self.batch.replace('\n','').strip().split('=')[1])
        self.iterations_per_epoch=max(1,self.iterations_per_epoch)
        print('number of iterations_per_epoch = {}'.format(self.iterations_per_epoch))
        print('self.max_batches={}'.format(self.max_batches))
        self.epochs=int(self.max_batches.replace('\n','').strip().split('=')[1])//self.iterations_per_epoch
        self.epochs_yolov4_output=os.path.join(os.path.dirname(self.best_weights_path),'EPOCHS_YoloV4.txt')
        f=open(self.epochs_yolov4_output,'w')
        f.writelines('self.train_list_path = \n {} \n'.format(self.train_list_path))
        f.writelines('self.dataset=# Training Samples in self.train_list_path\n')
        f.writelines('self.dataset={}\n'.format(self.dataset))
        f.writelines('self.batch & self.max_batches are in the training configuration file: \n {} \n'.format(self.save_cfg_path_train))
        f.writelines('self.batch = {}\n'.format(int(self.batch.replace('\n','').strip().split('=')[1])))
        f.writelines('self.max_batches = {}\n'.format(int(self.max_batches.replace('\n','').strip().split('=')[1])))
        f.writelines('self.iterations_per_epoch=self.dataset//self.batch\n')
        f.writelines('self.iterations_per_epoch={}//{}\n'.format(self.dataset,int(self.batch.replace('\n','').strip().split('=')[1])))
        f.writelines('self.iterations_per_epoch={}\n'.format(self.iterations_per_epoch))
        f.writelines('self.epochs=self.max_batches//self.iterations_per_epoch \n')
        f.writelines('self.epochs={}//{} \n'.format(int(self.max_batches.replace('\n','').strip().split('=')[1]),self.iterations_per_epoch))
        f.writelines('self.epochs={}\n'.format(self.epochs))
        f.close()
        try:
            self.EPOCH_entry.destroy()
            self.EPOCH_label.destroy()
        except:
            self.epochs_VAR=tk.StringVar()
            self.epochs_VAR.set(self.epochs)
            self.epochs_entry=tk.Entry(self.root,textvariable=self.epochs_VAR)
            self.epochs_entry.grid(row=21,column=0,sticky='se')
            self.epochs_label=tk.Label(self.root,text='epochs',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
            self.epochs_label.grid(row=22,column=0,sticky='ne')

    def popupWindow_RECORD_RAW(self):
        try:
            self.top.destroy()
        except:
            pass
        self.top=tk.Toplevel(self.root)
        self.top.geometry( "{}x{}".format(int(self.root.winfo_screenwidth()*0.95//1.5),int(self.root.winfo_screenheight()*0.95//1.5)) )
        self.top.title('Record Raw Video to File')
        self.top.configure(background = 'black')
        self.b=Button(self.top,text='Close',command=self.cleanup,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.b.grid(row=0,column=0,sticky='se')

        self.submit_RAW_VIDEO=Button(self.top,text='RECORD VIDEO',command=self.RECORD_RAW_VIDEO,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.submit_RAW_VIDEO.grid(row=0,column=1,sticky='se')


        # RAW VIDEO PATHS
        
        self.LOAD_RAW_VIDEO_PATHS()


        self.OPEN_RAW_VIDEO_Button=Button(self.top,text='OPEN RAW VIDEO PATHS',command=self.OPEN_RAW_VIDEO_PATHS,bg=DEFAULT_SETTINGS.root_fg,fg=DEFAULT_SETTINGS.root_bg)
        self.OPEN_RAW_VIDEO_Button.grid(row=1,column=3,sticky='sw')


        self.LOAD_RAW_VIDEO_Button=Button(self.top,text='LOAD RAW VIDEO PATHS',command=self.LOAD_RAW_VIDEO_PATHS,bg=DEFAULT_SETTINGS.root_fg,fg=DEFAULT_SETTINGS.root_bg)
        self.LOAD_RAW_VIDEO_Button.grid(row=1,column=4,sticky='sw')

        # imW
        self.RAW_VIDEO_imW_VAR=tk.StringVar()
        self.RAW_VIDEO_imW_VAR.set('1920')
        self.RAW_VIDEO_imW_entry=tk.Entry(self.top,textvariable=self.RAW_VIDEO_imW_VAR)
        self.RAW_VIDEO_imW_entry.grid(row=2,column=2,sticky='sw')
        self.RAW_VIDEO_imW_note=tk.Label(self.top,text='--imW',bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.RAW_VIDEO_imW_note.grid(row=2,column=1,sticky='se')

        # imH
        self.RAW_VIDEO_imH_VAR=tk.StringVar()
        self.RAW_VIDEO_imH_VAR.set('1080')
        self.RAW_VIDEO_imH_entry=tk.Entry(self.top,textvariable=self.RAW_VIDEO_imH_VAR)
        self.RAW_VIDEO_imH_entry.grid(row=3,column=2,sticky='sw')
        self.RAW_VIDEO_imH_note=tk.Label(self.top,text='--imH',bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.RAW_VIDEO_imH_note.grid(row=3,column=1,sticky='se')

        # fps
        self.RAW_VIDEO_fps_VAR=tk.StringVar()
        self.RAW_VIDEO_fps_VAR.set('30')
        self.RAW_VIDEO_fps_entry=tk.Entry(self.top,textvariable=self.RAW_VIDEO_fps_VAR)
        self.RAW_VIDEO_fps_entry.grid(row=4,column=2,sticky='sw')
        self.RAW_VIDEO_fps_note=tk.Label(self.top,text='--fps',bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.RAW_VIDEO_fps_note.grid(row=4,column=1,sticky='se')

        # UNIQUE DEVICE
        self.RAW_VIDEO_UNIQUE_DEVICE_VAR=tk.StringVar()
        self.RAW_VIDEO_UNIQUE_DEVICE_VAR.set('Jetson')
        self.RAW_VIDEO_UNIQUE_DEVICE_entry=tk.Entry(self.top,textvariable=self.RAW_VIDEO_UNIQUE_DEVICE_VAR)
        self.RAW_VIDEO_UNIQUE_DEVICE_entry.grid(row=5,column=2,columnspan=20,sticky='we')
        self.RAW_VIDEO_UNIQUE_DEVICE_note=tk.Label(self.top,text='--UNIQUE_DEVICE',bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.RAW_VIDEO_UNIQUE_DEVICE_note.grid(row=5,column=1,sticky='se')

        # UNIQUE PREFIX
        self.RAW_VIDEO_UNIQUE_PREFIX_VAR=tk.StringVar()
        self.RAW_VIDEO_UNIQUE_PREFIX_VAR.set('')
        self.RAW_VIDEO_UNIQUE_PREFIX_entry=tk.Entry(self.top,textvariable=self.RAW_VIDEO_UNIQUE_PREFIX_VAR)
        self.RAW_VIDEO_UNIQUE_PREFIX_entry.grid(row=6,column=2,columnspan=20,sticky='we')
        self.RAW_VIDEO_UNIQUE_PREFIX_note=tk.Label(self.top,text='--UNIQUE_PREFIX',bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.RAW_VIDEO_UNIQUE_PREFIX_note.grid(row=6,column=1,sticky='se')

        # main_path
        self.RAW_VIDEO_main_path_VAR=tk.StringVar()
        self.RAW_VIDEO_main_path_VAR.set(r"/media/steven/OneTouch4tb/Videos/")
        self.RAW_VIDEO_main_path_entry=tk.Entry(self.top,textvariable=self.RAW_VIDEO_main_path_VAR)
        self.RAW_VIDEO_main_path_entry.grid(row=7,column=2,columnspan=20,sticky='we')
        self.RAW_VIDEO_main_path_note=tk.Label(self.top,text='--main_path',bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.RAW_VIDEO_main_path_note.grid(row=7,column=1,sticky='se')

        # using_JETSON_NANO
        self.RAW_VIDEO_using_JETSON_NANO_VAR=tk.StringVar()
        self.using_JETSON_NANO_LIST=['False','True']
        self.RAW_VIDEO_using_JETSON_NANO_VAR.set('False')
        self.RAW_VIDEO_using_JETSON_NANO_dropdown=tk.OptionMenu(self.top,self.RAW_VIDEO_using_JETSON_NANO_VAR,*self.using_JETSON_NANO_LIST)
        self.RAW_VIDEO_using_JETSON_NANO_dropdown.grid(row=8,column=2,sticky='sw')
        self.RAW_VIDEO_using_JETSON_NANO_note=tk.Label(self.top,text='--using_JETSON_NANO',bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.RAW_VIDEO_using_JETSON_NANO_note.grid(row=8,column=1,sticky='se')

    def RECORD_RAW_VIDEO(self):
         # RAW VIDEO PATHS
        video=self.RAW_VIDEO_VAR.get()

        # imW
        imW=self.RAW_VIDEO_imW_VAR.get()
        try:
            int(imW)
        except:
            return print('imW is a bad value = {}, should be an int'.format(imW))
        # imH
        imH=self.RAW_VIDEO_imH_VAR.get()
        try:
            int(imW)
        except:
            return print('imH is a bad value = {}, should be an int'.format(imH))
        # fps
        fps=self.RAW_VIDEO_fps_VAR.get()
        try:
            int(fps)
        except:
            return print('fps is a bad value = {}, should be an int'.format(fps))

        # UNIQUE DEVICE
        UNIQUE_DEVICE=self.RAW_VIDEO_UNIQUE_DEVICE_VAR.get()
        UNIQUE_DEVICE=UNIQUE_DEVICE.replace(' ','_')


        # UNIQUE PREFIX
        UNIQUE_PREFIX=self.RAW_VIDEO_UNIQUE_PREFIX_VAR.get()
        UNIQUE_PREFIX=UNIQUE_PREFIX.replace(' ','_').replace('.','p').replace(':','_').replace(';','_')
        if UNIQUE_PREFIX=='':
            UNIQUE_PREFIX='""'

        # main_path
        main_path=self.RAW_VIDEO_main_path_VAR.get()
        if os.path.exists(main_path):
            main_path=main_path
        elif os.path.exists(os.path.dirname(main_path)):
            main_path=os.path.join(main_path,'Videos')
            try:
                os.makedirs(main_path)
            except:
                print('main_path exists at {}\n'.format(main_path))
        else:
            return print('Not a valid path at {}\n'.format(main_path))
        # using_JETSON_NANO
        using_JETSON_NANO=self.RAW_VIDEO_using_JETSON_NANO_VAR.get()

        if using_JETSON_NANO=='False':
            cmd_i=f'cd resources && python3 record_raw.py --video {video} --imW {imW} --imH {imH} --fps {fps} --UNIQUE_DEVICE {UNIQUE_DEVICE} --UNIQUE_PREFIX {UNIQUE_PREFIX} --main_path {main_path}'
        else:
            cmd_i=cmd_i+' --using_JETSON_NANO'
        print('cmd_i = {}\n'.format(cmd_i))
        self.run_cmd(cmd_i)

  
    def LOAD_RAW_VIDEO_PATHS(self):

        self.RAW_VIDEO_PATHS='libs/RAW_VIDEO_PATHS.txt'
        if os.path.exists(self.RAW_VIDEO_PATHS):
            f=open(self.RAW_VIDEO_PATHS,'r')
            f_read=f.readlines()
            f.close()
            self.RAW_VIDEO_LIST=[w.replace('\n','').strip() for w in f_read]
        else:
            self.RAW_VIDEO_LIST=['0']
            f=open(self.RAW_VIDEO_PATHS,'w')
            tmp=[f.writelines(w+'\n') for w in self.RAW_VIDEO_LIST]
            f.close()
        try:
           del self.RAW_VIDEO_dropdown
        except:
            pass
        self.RAW_VIDEO_VAR=tk.StringVar()
        self.RAW_VIDEO_VAR.set(self.RAW_VIDEO_LIST[0])
        self.RAW_VIDEO_dropdown=tk.OptionMenu(self.top,self.RAW_VIDEO_VAR,*self.RAW_VIDEO_LIST)
        self.RAW_VIDEO_dropdown.grid(row=1,column=2,sticky='sw')
        self.RAW_VIDEO_dropdown_note=tk.Label(self.top,text='--video',bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.RAW_VIDEO_dropdown_note.grid(row=1,column=1,sticky='se')
    
    def OPEN_RAW_VIDEO_PATHS(self):
        self.LOAD_RAW_VIDEO_PATHS()
        cmd_i=open_cmd+' {}'.format(self.RAW_VIDEO_PATHS)
        self.run_cmd(cmd_i)
        

    def popupWindow_ERROR(self,message):
        try:
            self.top.destroy()
        except:
            pass
        self.top=tk.Toplevel(self.root)
        self.top.geometry( "{}x{}".format(int(self.root.winfo_screenwidth()*0.95//1.5),int(self.root.winfo_screenheight()*0.95//1.5)) )
        self.top.title('ERROR')
        self.top.configure(background = 'black')
        self.b=Button(self.top,text='Close',command=self.cleanup,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.b.grid(row=0,column=0,sticky='se')
        self.label_Error=tk.Label(self.top,text=message,bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.label_Error.grid(row=1,column=1,sticky='s')


    def run_create_tflite_bash(self):
        if self.WIDTH_NUM!=self.HEIGHT_NUM:
            error_msg='WIDTH_NUM != HEIGHT_NUM.  THEY ARE REQUIRED TO BE EQUAL FOR THE CONVERSION.' 
            print(error_msg)
            self.popupWindow_ERROR(error_msg)
        else:
            self.create_tflite_bash()
            cmd_i=" bash '{}'".format(self.tensorflow_yolov4_tflite_bash_PATH)
            self.run_cmd(cmd_i)

    def create_train_bash_yolov7(self):
        self.epochs_yolov7=self.epochs_yolov7_VAR.get()
        self.TRAIN_YOLOV7=os.path.join(os.path.dirname(self.data_path),'train_custom_Yolov7-tiny.sh')
        f=open(self.TRAIN_YOLOV7,'w')
        f.writelines('cd {} \n'.format(self.yolov7_path))
        self.last_weights_path_yolov7=os.path.join(os.path.dirname(self.data_path),'yolov7-tiny/weights/last.pt')
        if os.path.exists(self.last_weights_path_yolov7)==False:
            f.writelines("python3 train.py --workers 8 --device 0 --batch-size 32 --data {} --img {} {} --cfg {} --weights '' --exist-ok --name {} --hyp {} --epochs {}\n".format(self.YAML_PATH,self.WIDTH_NUM,self.HEIGHT_NUM,self.yolov7_path_cfg,self.yolov7_path_name,self.yolov7_path_hyp,self.epochs_yolov7))
        else:
            f.writelines("python3 train.py --workers 8 --device 0 --batch-size 32 --data {} --img {} {} --cfg {} --weights {} --exist-ok --name {} --hyp {} --epochs {}\n".format(self.YAML_PATH,self.WIDTH_NUM,self.HEIGHT_NUM,self.yolov7_path_cfg,self.last_weights_path_yolov7,self.yolov7_path_name,self.yolov7_path_hyp,self.epochs_yolov7))           
        f.close()

    def create_train_bash_yolov7_e6e(self):
        self.epochs_yolov7_e6e=self.epochs_yolov7_e6e_VAR.get()
        #python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/coco.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights '' --name yolov7-w6 --hyp data/hyp.scratch.p6.yaml
        self.TRAIN_YOLOV7_e6e=os.path.join(os.path.dirname(self.data_path),'train_custom_Yolov7-e6e.sh')
        f=open(self.TRAIN_YOLOV7_e6e,'w')
        f.writelines('cd {} \n'.format(self.yolov7_path))
        self.last_weights_path_yolov7_e6e=os.path.join(os.path.dirname(self.data_path),'yolov7-e6e/weights/last.pt')
        if os.path.exists(self.last_weights_path_yolov7_e6e)==False:
            f.writelines("python3 train_aux.py --workers 8 --device 0 --batch-size 2 --data {} --img {} {} --cfg {} --weights '' --exist-ok --name {} --hyp {} --epochs {}\n".format(self.YAML_PATH,self.WIDTH_NUM,self.HEIGHT_NUM,self.yolov7_path_cfg_e6e,self.yolov7_path_name_e6e,self.yolov7_path_hyp_e6e,self.epochs_yolov7_e6e))
        else:
            f.writelines("python3 train_aux.py --workers 8 --device 0 --batch-size 2 --data {} --img {} {} --cfg {} --weights {} --exist-ok --name {} --hyp {} --epochs {}\n".format(self.YAML_PATH,self.WIDTH_NUM,self.HEIGHT_NUM,self.yolov7_path_cfg_e6e,self.last_weights_path_yolov7_e6e,self.yolov7_path_name_e6e,self.yolov7_path_hyp_e6e,self.epochs_yolov7_e6e))           

        f.close()

    def create_train_bash_yolov7_re(self):
        #python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/coco.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6.yaml --weights '' --name yolov7-w6 --hyp data/hyp.scratch.p6.yaml
        self.epochs_yolov7_re=self.epochs_yolov7_re_VAR.get()
        self.TRAIN_YOLOV7_re=os.path.join(os.path.dirname(self.data_path),'train_custom_Yolov7.sh')
        f=open(self.TRAIN_YOLOV7_re,'w')
        f.writelines('cd {} \n'.format(self.yolov7_path))
        self.last_weights_path_yolov7_re=os.path.join(os.path.dirname(self.data_path),'yolov7/weights/last.pt')
        if max(self.WIDTH_NUM,self.HEIGHT_NUM)>800:
            batch_size=2
        else:
            batch_size=8
        if os.path.exists(self.last_weights_path_yolov7_re)==False:
            f.writelines("python3 train.py --workers 8 --device 0 --batch-size {} --data {} --img {} {} --cfg {} --weights '' --exist-ok --name {} --hyp {} --epochs {}\n".format(batch_size,self.YAML_PATH,self.WIDTH_NUM,self.HEIGHT_NUM,self.yolov7_path_cfg_re,self.yolov7_path_name_re,self.yolov7_path_hyp_re,self.epochs_yolov7_re))
        else:
            f.writelines("python3 train.py --workers 8 --device 0 --batch-size {} --data {} --img {} {} --cfg {} --weights {} --exist-ok --name {} --hyp {} --epochs {}\n".format(batch_size,self.YAML_PATH,self.WIDTH_NUM,self.HEIGHT_NUM,self.yolov7_path_cfg_re,self.last_weights_path_yolov7_re,self.yolov7_path_name_re,self.yolov7_path_hyp_re,self.epochs_yolov7_re))           
        f.close()

    def create_test_bash_mp4_yolov7(self):
        self.TEST_MP4_YOLOV7=os.path.join(os.path.dirname(self.data_path),'test_MP4_custom_Yolov7-tiny.sh')
        f=open(self.TEST_MP4_YOLOV7,'w')
        f.writelines('cd {}\n'.format(self.yolov7_path))
        if self.RTSP_SERVER:
            if self.USE_RTSP_VAR.get()=="Yes":
                #f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_tiny, self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get()))
            #else:
            #    f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_tiny))
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_tiny, self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get())
            else:
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_tiny)
            if self.sec.get()=='y':
                self.destination_list_final=''
                for w_var in self.phone_dic_trigger_var.values():
                    var_i=w_var.get()
                    if var_i!='None':
                        self.destination_list_final=self.destination_list_final+";"+var_i
                self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
                cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)   
        elif self.sec.get()=='y':
            self.destination_list_final=''
            for w_var in self.phone_dic_trigger_var.values():
                var_i=w_var.get()
                if var_i!='None':
                    self.destination_list_final=self.destination_list_final+";"+var_i
            self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_tiny)
            cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        else:
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_tiny)
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        f.close()

    def create_test_bash_labelimg_yolov7(self):
        self.TEST_labelimg_YOLOV7=os.path.join(os.path.dirname(self.data_path),'test_labelimg_custom_Yolov7-tiny.sh')
        f=open(self.TEST_labelimg_YOLOV7,'w')
        f.writelines('cd {}\n'.format(self.yolov7_path))
        if self.RTSP_SERVER:
            if self.USE_RTSP_VAR.get()=="Yes":
                #f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_tiny, self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get()))
            #else:
            #    f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_tiny))
                cmd_i="python3 detect.py --use_socket_receive_imgs --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok  --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_tiny, self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get())
            else:
                cmd_i="python3 detect.py --use_socket_receive_imgs --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok \n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_tiny)
            if self.sec.get()=='y':
                self.destination_list_final=''
                for w_var in self.phone_dic_trigger_var.values():
                    var_i=w_var.get()
                    if var_i!='None':
                        self.destination_list_final=self.destination_list_final+";"+var_i
                self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
                cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)   
        elif self.sec.get()=='y':
            self.destination_list_final=''
            for w_var in self.phone_dic_trigger_var.values():
                var_i=w_var.get()
                if var_i!='None':
                    self.destination_list_final=self.destination_list_final+";"+var_i
            self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
            cmd_i="python3 detect.py --use_socket_receive_imgs --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok \n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_tiny)
            cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        else:
            cmd_i="python3 detect.py --use_socket_receive_imgs --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok \n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_tiny)
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        f.close()

    def create_test_bash_mp4_yolov7_e6e(self):
        self.TEST_MP4_YOLOV7_e6e=os.path.join(os.path.dirname(self.data_path),'test_MP4_custom_Yolov7-e6e.sh')
        f=open(self.TEST_MP4_YOLOV7_e6e,'w')
        f.writelines('cd {}\n'.format(self.yolov7_path))
        if self.RTSP_SERVER:
            #if self.USE_RTSP_VAR.get()=="Yes":
            #    f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_e6e, self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get()))
            #else:
            #    f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_e6e))
            if self.USE_RTSP_VAR.get()=="Yes":
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_e6e, self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get())
            else:
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_e6e)
            if self.sec.get()=='y':
                self.destination_list_final=''
                for w_var in self.phone_dic_trigger_var.values():
                    var_i=w_var.get()
                    if var_i!='None':
                        self.destination_list_final=self.destination_list_final+";"+var_i
                self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
                cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i) 
        elif self.sec.get()=='y':
            self.destination_list_final=''
            for w_var in self.phone_dic_trigger_var.values():
                var_i=w_var.get()
                if var_i!='None':
                    self.destination_list_final=self.destination_list_final+";"+var_i
            self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_e6e)
            cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        else:
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_e6e)
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        f.close()


    def create_test_bash_labelimg_yolov7_e6e(self):
        self.TEST_labelimg_YOLOV7_e6e=os.path.join(os.path.dirname(self.data_path),'test_labelimg_custom_Yolov7-e6e.sh')
        f=open(self.TEST_labelimg_YOLOV7_e6e,'w')
        f.writelines('cd {}\n'.format(self.yolov7_path))
        if self.RTSP_SERVER:
            #if self.USE_RTSP_VAR.get()=="Yes":
            #    f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_e6e, self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get()))
            #else:
            #    f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_e6e))
            if self.USE_RTSP_VAR.get()=="Yes":
                cmd_i="python3 detect.py --use_socket_receive_imgs --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_e6e, self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get())
            else:
                cmd_i="python3 detect.py --use_socket_receive_imgs --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok \n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_e6e)
            if self.sec.get()=='y':
                self.destination_list_final=''
                for w_var in self.phone_dic_trigger_var.values():
                    var_i=w_var.get()
                    if var_i!='None':
                        self.destination_list_final=self.destination_list_final+";"+var_i
                self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
                cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i) 
        elif self.sec.get()=='y':
            self.destination_list_final=''
            for w_var in self.phone_dic_trigger_var.values():
                var_i=w_var.get()
                if var_i!='None':
                    self.destination_list_final=self.destination_list_final+";"+var_i
            self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
            cmd_i="python3 detect.py --use_socket_receive_imgs --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok \n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_e6e)
            cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        else:
            cmd_i="python3 detect.py --use_socket_receive_imgs --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok \n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_e6e)
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        f.close()

    def create_test_bash_labelimg_yolov7_re(self):
        self.TEST_labelimg_YOLOV7_re=os.path.join(os.path.dirname(self.data_path),'test_labelimg_custom_Yolov7.sh')
        f=open(self.TEST_labelimg_YOLOV7_re,'w')
        f.writelines('cd {}\n'.format(self.yolov7_path))
        if self.RTSP_SERVER:
            #if self.USE_RTSP_VAR.get()=="Yes":
            #    f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_re,self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get()))
            #else:
            #    f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_re))
            if self.USE_RTSP_VAR.get()=="Yes":
                cmd_i="python3 detect.py --use_socket_receive_imgs --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_re,self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get())
            else:
                cmd_i="python3 detect.py --use_socket_receive_imgs --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok \n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_re)
            if self.sec.get()=='y':
                self.destination_list_final=''
                for w_var in self.phone_dic_trigger_var.values():
                    var_i=w_var.get()
                    if var_i!='None':
                        self.destination_list_final=self.destination_list_final+";"+var_i
                self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
                cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i) 
        elif self.sec.get()=='y':
            self.destination_list_final=''
            for w_var in self.phone_dic_trigger_var.values():
                var_i=w_var.get()
                if var_i!='None':
                    self.destination_list_final=self.destination_list_final+";"+var_i
            self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
            cmd_i="python3 detect.py --use_socket_receive_imgs --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok \n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_re)
            cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        
        else:
            cmd_i="python3 detect.py --use_socket_receive_imgs --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok \n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_re)
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        f.close()

    def create_test_bash_mp4_yolov7_re(self):
        self.TEST_MP4_YOLOV7_re=os.path.join(os.path.dirname(self.data_path),'test_MP4_custom_Yolov7.sh')
        f=open(self.TEST_MP4_YOLOV7_re,'w')
        f.writelines('cd {}\n'.format(self.yolov7_path))
        if self.RTSP_SERVER:
            #if self.USE_RTSP_VAR.get()=="Yes":
            #    f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_re,self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get()))
            #else:
            #    f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_re))
            if self.USE_RTSP_VAR.get()=="Yes":
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_re,self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get())
            else:
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_re)
            if self.sec.get()=='y':
                self.destination_list_final=''
                for w_var in self.phone_dic_trigger_var.values():
                    var_i=w_var.get()
                    if var_i!='None':
                        self.destination_list_final=self.destination_list_final+";"+var_i
                self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
                cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i) 
        elif self.sec.get()=='y':
            self.destination_list_final=''
            for w_var in self.phone_dic_trigger_var.values():
                var_i=w_var.get()
                if var_i!='None':
                    self.destination_list_final=self.destination_list_final+";"+var_i
            self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_re)
            cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        
        else:
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --source {} --project {} --exist-ok --view-img\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.mp4_video_path,self.yolov7_path_project_re)
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        f.close()

    def create_test_bash_webcam_yolov7(self):
        self.TEST_WEBCAM_YOLOV7=os.path.join(os.path.dirname(self.data_path),'test_webcam_custom_Yolov7-tiny.sh')
        f=open(self.TEST_WEBCAM_YOLOV7,'w')
        f.writelines('cd {}\n'.format(self.yolov7_path))

        if self.RTSP_SERVER:
            if self.USE_RTSP_VAR.get()=="Yes":
                #f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0 --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_tiny,self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get()))
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0 --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_tiny,self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get())
            else:
                #f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_tiny))
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_tiny)
            if self.sec.get()=='y':
                self.destination_list_final=''
                for w_var in self.phone_dic_trigger_var.values():
                    var_i=w_var.get()
                    if var_i!='None':
                        self.destination_list_final=self.destination_list_final+";"+var_i
                self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
                cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)


        elif self.sec.get()=='y':
            self.destination_list_final=''
            for w_var in self.phone_dic_trigger_var.values():
                var_i=w_var.get()
                if var_i!='None':
                    self.destination_list_final=self.destination_list_final+";"+var_i
            self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_tiny)
            cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        else:
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_tiny)
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        f.close()

    def create_test_bash_rtsp_yolov7(self):
        self.TEST_RTSP_YOLOV7=os.path.join(os.path.dirname(self.data_path),'test_rtsp_custom_Yolov7-tiny.sh')
        f=open(self.TEST_RTSP_YOLOV7,'w')
        f.writelines('cd {}\n'.format(self.yolov7_path))

        if self.RTSP_SERVER:
            if self.USE_RTSP_VAR.get()=="Yes":
                #f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0 --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_tiny,self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get()))
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {} --source {} \n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_tiny,self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get(),self.USE_RTSP_CLIENT_VAR.get())
            else:
                #f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_tiny))
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source {}\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_tiny,self.USE_RTSP_CLIENT_VAR.get())
            cmd_i=cmd_i.replace('\n',"")+ ' --PORT {} --HOST {} --socket_prefix {} \n'.format(self.USE_SOCKET_RTSP_PORT_VAR.get(),self.USE_SOCKET_RTSP_HOST_VAR.get(),self.SOCKET_PREFIX_VAR.get())
            if self.sec.get()=='y':
                self.destination_list_final=''
                for w_var in self.phone_dic_trigger_var.values():
                    var_i=w_var.get()
                    if var_i!='None':
                        self.destination_list_final=self.destination_list_final+";"+var_i
                self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
                cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
                cmd_i=cmd_i.replace('\n',"")+ ' --PORT {} --HOST {} --socket_prefix {} \n'.format(self.USE_SOCKET_RTSP_PORT_VAR.get(),self.USE_SOCKET_RTSP_HOST_VAR.get(),self.SOCKET_PREFIX_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)


        elif self.sec.get()=='y':
            self.destination_list_final=''
            for w_var in self.phone_dic_trigger_var.values():
                var_i=w_var.get()
                if var_i!='None':
                    self.destination_list_final=self.destination_list_final+";"+var_i
            self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source {}\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_tiny,self.USE_RTSP_CLIENT_VAR.get())
            cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            cmd_i=cmd_i.replace('\n',"")+ ' --PORT {} --HOST {} --socket_prefix {} \n'.format(self.USE_SOCKET_RTSP_PORT_VAR.get(),self.USE_SOCKET_RTSP_HOST_VAR.get(),self.SOCKET_PREFIX_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        else:
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source {}\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_tiny,self.USE_RTSP_CLIENT_VAR.get())
            cmd_i=cmd_i.replace('\n',"")+ ' --PORT {} --HOST {} --socket_prefix {} \n'.format(self.USE_SOCKET_RTSP_PORT_VAR.get(),self.USE_SOCKET_RTSP_HOST_VAR.get(),self.SOCKET_PREFIX_VAR.get())
            #f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source {}\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_tiny,self.USE_RTSP_CLIENT_VAR.get()))
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        f.close()

    def create_test_bash_webcam_yolov7_RTMP(self):
        self.TEST_WEBCAM_YOLOV7_RTMP=os.path.join(os.path.dirname(self.data_path),'test_webcam_custom_Yolov7-tiny_RTMP.sh')
        f=open(self.TEST_WEBCAM_YOLOV7_RTMP,'w')
        self.YOUTUBE_KEY=self.YOUTUBE_KEY_VAR.get()
        fo=open('YOUTUBE_KEY.txt','w')
        fo_read=fo.writelines(self.YOUTUBE_KEY+'\n')
        fo.close()
        f.writelines('YOUTUBE_RTMP={}\n'.format(self.YOUTUBE_KEY))
        f.writelines('YOUTUBE_STREAM_RES={}\n'.format(self.USER_SELECTION_yt.get()))
        f.writelines('cd {}\n'.format(self.yolov7_path))
        if self.RTSP_SERVER:
            if self.USE_RTSP_VAR.get()=="Yes":
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0 --YOUTUBE_RTMP=$YOUTUBE_RTMP --YOUTUBE_STREAM_RES=$YOUTUBE_STREAM_RES --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_tiny,self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get())
                if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                    cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
                f.writelines(cmd_i)
            else:
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0 --YOUTUBE_RTMP=$YOUTUBE_RTMP --YOUTUBE_STREAM_RES=$YOUTUBE_STREAM_RES\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_tiny)
                if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                    cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
                f.writelines(cmd_i)
        else:
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0 --YOUTUBE_RTMP=$YOUTUBE_RTMP --YOUTUBE_STREAM_RES=$YOUTUBE_STREAM_RES\n".format(self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_tiny)
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        f.close()

    def create_test_bash_webcam_yolov7_e6e(self):
        self.TEST_WEBCAM_YOLOV7_e6e=os.path.join(os.path.dirname(self.data_path),'test_webcam_custom_Yolov7-e6e.sh')
        f=open(self.TEST_WEBCAM_YOLOV7_e6e,'w')
        f.writelines('cd {}\n'.format(self.yolov7_path))
        if self.RTSP_SERVER:
            #if self.USE_RTSP_VAR.get()=="Yes":
            #    f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0 --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_e6e,self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get()))
            #else:
            #    f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_e6e))
            if self.USE_RTSP_VAR.get()=="Yes":
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0 --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_e6e,self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get())
            else:
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_e6e)
            if self.sec.get()=='y':
                self.destination_list_final=''
                for w_var in self.phone_dic_trigger_var.values():
                    var_i=w_var.get()
                    if var_i!='None':
                        self.destination_list_final=self.destination_list_final+";"+var_i
                self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
                cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)         
        elif self.sec.get()=='y':
            self.destination_list_final=''
            for w_var in self.phone_dic_trigger_var.values():
                var_i=w_var.get()
                if var_i!='None':
                    self.destination_list_final=self.destination_list_final+";"+var_i
            self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_e6e)
            cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        else:
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_e6e)
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        f.close()

    def create_test_bash_rtsp_yolov7_e6e(self):
        self.TEST_RTSP_YOLOV7_e6e=os.path.join(os.path.dirname(self.data_path),'test_rtsp_custom_Yolov7-e6e.sh')
        f=open(self.TEST_RTSP_YOLOV7_e6e,'w')
        f.writelines('cd {}\n'.format(self.yolov7_path))
        if self.RTSP_SERVER:
            #if self.USE_RTSP_VAR.get()=="Yes":
            #    f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0 --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_e6e,self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get()))
            #else:
            #    f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_e6e))
            if self.USE_RTSP_VAR.get()=="Yes":
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {} --source {} \n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_e6e,self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get(),self.USE_RTSP_CLIENT_VAR.get())
            else:
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source {}\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_e6e,self.USE_RTSP_CLIENT_VAR.get())
                cmd_i=cmd_i.replace('\n',"")+ ' --PORT {} --HOST {} --socket_prefix {} \n'.format(self.USE_SOCKET_RTSP_PORT_VAR.get(),self.USE_SOCKET_RTSP_HOST_VAR.get(),self.SOCKET_PREFIX_VAR.get())
            if self.sec.get()=='y':
                self.destination_list_final=''
                for w_var in self.phone_dic_trigger_var.values():
                    var_i=w_var.get()
                    if var_i!='None':
                        self.destination_list_final=self.destination_list_final+";"+var_i
                self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
                cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
                cmd_i=cmd_i.replace('\n',"")+ ' --PORT {} --HOST {} --socket_prefix {} \n'.format(self.USE_SOCKET_RTSP_PORT_VAR.get(),self.USE_SOCKET_RTSP_HOST_VAR.get(),self.SOCKET_PREFIX_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)         
        elif self.sec.get()=='y':
            self.destination_list_final=''
            for w_var in self.phone_dic_trigger_var.values():
                var_i=w_var.get()
                if var_i!='None':
                    self.destination_list_final=self.destination_list_final+";"+var_i
            self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source {}\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_e6e,self.USE_RTSP_CLIENT_VAR.get())
            cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            cmd_i=cmd_i.replace('\n',"")+ ' --PORT {} --HOST {} --socket_prefix {} \n'.format(self.USE_SOCKET_RTSP_PORT_VAR.get(),self.USE_SOCKET_RTSP_HOST_VAR.get(),self.SOCKET_PREFIX_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        else:
            #f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source {}\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_e6e,self.USE_RTSP_CLIENT_VAR.get()))
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source {}\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_e6e,self.USE_RTSP_CLIENT_VAR.get())
            #cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            cmd_i=cmd_i.replace('\n',"")+ ' --PORT {} --HOST {} --socket_prefix {} \n'.format(self.USE_SOCKET_RTSP_PORT_VAR.get(),self.USE_SOCKET_RTSP_HOST_VAR.get(),self.SOCKET_PREFIX_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        f.close()

    def create_test_bash_webcam_yolov7_e6e_RTMP(self):
        self.TEST_WEBCAM_YOLOV7_e6e_RTMP=os.path.join(os.path.dirname(self.data_path),'test_webcam_custom_Yolov7-e6e_RTMP.sh')
        f=open(self.TEST_WEBCAM_YOLOV7_e6e_RTMP,'w')
        self.YOUTUBE_KEY=self.YOUTUBE_KEY_VAR.get()
        fo=open('YOUTUBE_KEY.txt','w')
        fo_read=fo.writelines(self.YOUTUBE_KEY+'\n')
        fo.close()
        f.writelines('YOUTUBE_RTMP={}\n'.format(self.YOUTUBE_KEY))
        f.writelines('YOUTUBE_STREAM_RES={}\n'.format(self.USER_SELECTION_yt.get()))
        f.writelines('cd {}\n'.format(self.yolov7_path))
        if self.RTSP_SERVER:
            if self.USE_RTSP_VAR.get()=="Yes":
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0 --YOUTUBE_RTMP=$YOUTUBE_RTMP --YOUTUBE_STREAM_RES=$YOUTUBE_STREAM_RES --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_e6e,self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get())
                if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                    cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
                f.writelines(cmd_i)
            else:
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0 --YOUTUBE_RTMP=$YOUTUBE_RTMP --YOUTUBE_STREAM_RES=$YOUTUBE_STREAM_RES\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_e6e)
                if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                    cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
                f.writelines(cmd_i)
        else:
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0 --YOUTUBE_RTMP=$YOUTUBE_RTMP --YOUTUBE_STREAM_RES=$YOUTUBE_STREAM_RES\n".format(self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_e6e)
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        f.close()

    def create_test_bash_webcam_yolov7_re(self):
        self.TEST_WEBCAM_YOLOV7_re=os.path.join(os.path.dirname(self.data_path),'test_webcam_custom_Yolov7.sh')
        f=open(self.TEST_WEBCAM_YOLOV7_re,'w')
        f.writelines('cd {}\n'.format(self.yolov7_path))
        if self.RTSP_SERVER:
            #if self.USE_RTSP_VAR.get()=="Yes":
            #    f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0 --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_re,self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get()))
            #else:
            #    f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_re))
            if self.USE_RTSP_VAR.get()=="Yes":
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0 --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_re,self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get())
            else:
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_re)
            if self.sec.get()=='y':
                self.destination_list_final=''
                for w_var in self.phone_dic_trigger_var.values():
                    var_i=w_var.get()
                    if var_i!='None':
                        self.destination_list_final=self.destination_list_final+";"+var_i
                self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
                cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i) 
        elif self.sec.get()=='y':
            self.destination_list_final=''
            for w_var in self.phone_dic_trigger_var.values():
                var_i=w_var.get()
                if var_i!='None':
                    self.destination_list_final=self.destination_list_final+";"+var_i
            self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_re)
            cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        else:
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_re)
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        f.close()


    def create_test_bash_rtsp_yolov7_re(self):
        self.TEST_RTSP_YOLOV7_re=os.path.join(os.path.dirname(self.data_path),'test_rtsp_custom_Yolov7.sh')
        f=open(self.TEST_RTSP_YOLOV7_re,'w')
        f.writelines('cd {}\n'.format(self.yolov7_path))
        if self.RTSP_SERVER:
            #if self.USE_RTSP_VAR.get()=="Yes":
            #    f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0 --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_re,self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get()))
            #else:
            #    f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_re))
            if self.USE_RTSP_VAR.get()=="Yes":
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok  --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {} --source {}\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_re,self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get(),self.USE_RTSP_CLIENT_VAR.get())
            else:
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source {}\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_re,self.USE_RTSP_CLIENT_VAR.get())
                cmd_i=cmd_i.replace('\n',"")+ ' --PORT {} --HOST {} --socket_prefix {} \n'.format(self.USE_SOCKET_RTSP_PORT_VAR.get(),self.USE_SOCKET_RTSP_HOST_VAR.get(),self.SOCKET_PREFIX_VAR.get())
            if self.sec.get()=='y':
                self.destination_list_final=''
                for w_var in self.phone_dic_trigger_var.values():
                    var_i=w_var.get()
                    if var_i!='None':
                        self.destination_list_final=self.destination_list_final+";"+var_i
                self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
                cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
                cmd_i=cmd_i.replace('\n',"")+ ' --PORT {} --HOST {} --socket_prefix {} \n'.format(self.USE_SOCKET_RTSP_PORT_VAR.get(),self.USE_SOCKET_RTSP_HOST_VAR.get(),self.SOCKET_PREFIX_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i) 
        elif self.sec.get()=='y':
            self.destination_list_final=''
            for w_var in self.phone_dic_trigger_var.values():
                var_i=w_var.get()
                if var_i!='None':
                    self.destination_list_final=self.destination_list_final+";"+var_i
            self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source {}\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_re,self.USE_RTSP_CLIENT_VAR.get())
            cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            cmd_i=cmd_i.replace('\n',"")+ ' --PORT {} --HOST {} --socket_prefix {} \n'.format(self.USE_SOCKET_RTSP_PORT_VAR.get(),self.USE_SOCKET_RTSP_HOST_VAR.get(),self.SOCKET_PREFIX_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        else:
            #f.writelines("python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source {}\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_re,self.USE_RTSP_CLIENT_VAR.get()))
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source {}\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_re,self.USE_RTSP_CLIENT_VAR.get())
            cmd_i=cmd_i.replace('\n',"")+ ' --PORT {} --HOST {} --socket_prefix {} \n'.format(self.USE_SOCKET_RTSP_PORT_VAR.get(),self.USE_SOCKET_RTSP_HOST_VAR.get(),self.SOCKET_PREFIX_VAR.get())
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i) 
        f.close()

    def create_test_bash_webcam_yolov7_re_RTMP(self):
        self.TEST_WEBCAM_YOLOV7_re_RTMP=os.path.join(os.path.dirname(self.data_path),'test_webcam_custom_Yolov7_RTMP.sh')
        f=open(self.TEST_WEBCAM_YOLOV7_re_RTMP,'w')
        self.YOUTUBE_KEY=self.YOUTUBE_KEY_VAR.get()
        fo=open('YOUTUBE_KEY.txt','w')
        fo_read=fo.writelines(self.YOUTUBE_KEY+'\n')
        fo.close()
        f.writelines('YOUTUBE_RTMP={}\n'.format(self.YOUTUBE_KEY))
        f.writelines('YOUTUBE_STREAM_RES={}\n'.format(self.USER_SELECTION_yt.get()))
        f.writelines('cd {}\n'.format(self.yolov7_path))
        if self.RTSP_SERVER:
            if self.USE_RTSP_VAR.get()=="Yes":
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0 --YOUTUBE_RTMP=$YOUTUBE_RTMP --YOUTUBE_STREAM_RES=$YOUTUBE_STREAM_RES --RTSP_PATH Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_re,self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get())
                if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                    cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
                f.writelines(cmd_i)
            else:
                cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0 --YOUTUBE_RTMP=$YOUTUBE_RTMP --YOUTUBE_STREAM_RES=$YOUTUBE_STREAM_RES\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_re)
                if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                    cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
                f.writelines(cmd_i)
        else:
            cmd_i="python3 detect.py --weights {} --conf {} --img-size {} --project {} --exist-ok --source 0 --YOUTUBE_RTMP=$YOUTUBE_RTMP --YOUTUBE_STREAM_RES=$YOUTUBE_STREAM_RES\n".format(self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.yolov7_path_project_re)
            if self.CLASSIFY_CHIPS_LOGIC.get()=='Yes':
                cmd_i=cmd_i.replace('\n',"") + ' --INFERENCE_TENSORFLOW_path={} \n'.format(self.USE_CLASSIFY_CHIPS_VAR.get())
            f.writelines(cmd_i)
        f.close()

    def create_predict_bash_mAP_yolov7(self):
        self.TEST_PREDICT_YOLOV7=os.path.join(os.path.dirname(self.data_path),'predict_custom_mAP_IOU{}_CONF{}_Yolov7-tiny.sh'.format(str(self.IOU_THRESH).replace('.','p'),str(self.THRESH).replace('.','p')))
        f=open(self.TEST_PREDICT_YOLOV7,'w')
               
        self.create_YAML()  
        f.writelines('path_result_list_txt={}\n'.format(self.test_list_path))
        f.writelines('path_predictions_folder={}\n'.format(os.path.join(self.yolov7_path_project_tiny,'predictions')))
        f.writelines('path_objs_names={}\n'.format(self.names_path))
        f.writelines('cd {}\n'.format(self.yolov7_path))
        #python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val
        f.writelines("python3 test.py --data {} --weights {} --conf {} --img-size {} --batch 32 --iou {} --project {} --name predictions --exist-ok --save-txt --save-conf --verbose --task test --device 0\n".format(self.YAML_PATH, self.yolov7_path_weights,self.THRESH,self.WIDTH_NUM,self.IOU_THRESH,self.yolov7_path_project_tiny))
        f.writelines('cd {}\n'.format(self.CWD))
        f.writelines('python3 resources/convert_predictions_to_xml_yolov7.py --path_result_list_txt=$path_result_list_txt --path_predictions_folder=$path_predictions_folder --path_objs_names=$path_objs_names \n')
        #f.writelines('python3 resources/iou_chips.py --Prediction_xml $path_predictions_folder/Annotations --path_result_list_txt=$path_result_list_txt\n')
        f.close()

    def create_predict_bash_mAP_yolov7_e6e(self):
        self.TEST_PREDICT_YOLOV7_e6e=os.path.join(os.path.dirname(self.data_path),'predict_custom_mAP_IOU{}_CONF{}_Yolov7-e6e.sh'.format(str(self.IOU_THRESH).replace('.','p'),str(self.THRESH).replace('.','p')))
        f=open(self.TEST_PREDICT_YOLOV7_e6e,'w')
        
        self.create_YAML()  
        f.writelines('path_result_list_txt={}\n'.format(self.test_list_path))
        f.writelines('path_predictions_folder={}\n'.format(os.path.join(self.yolov7_path_project_e6e,'predictions')))
        f.writelines('path_objs_names={}\n'.format(self.names_path))
        f.writelines('cd {}\n'.format(self.yolov7_path))
        #python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val
        f.writelines("python3 test.py --data {} --weights {} --conf {} --img-size {} --batch 1 --iou {} --project {} --name predictions --exist-ok --save-txt --save-conf --verbose --task test --device 0\n".format(self.YAML_PATH, self.yolov7_path_weights_e6e,self.THRESH,self.WIDTH_NUM,self.IOU_THRESH,self.yolov7_path_project_e6e))
        f.writelines('cd {}\n'.format(self.CWD))
        f.writelines('python3 resources/convert_predictions_to_xml_yolov7.py --path_result_list_txt=$path_result_list_txt --path_predictions_folder=$path_predictions_folder --path_objs_names=$path_objs_names \n')
        #f.writelines('python3 resources/iou_chips.py --Prediction_xml $path_predictions_folder/Annotations --path_result_list_txt=$path_result_list_txt\n')
        f.close()

    def create_predict_bash_mAP_yolov7_re(self):
        self.TEST_PREDICT_YOLOV7_re=os.path.join(os.path.dirname(self.data_path),'predict_custom_mAP_IOU{}_CONF{}_Yolov7.sh'.format(str(self.IOU_THRESH).replace('.','p'),str(self.THRESH).replace('.','p')))
        f=open(self.TEST_PREDICT_YOLOV7_re,'w')
        self.create_YAML()  
        f.writelines('path_result_list_txt={}\n'.format(self.test_list_path))
        f.writelines('path_predictions_folder={}\n'.format(os.path.join(self.yolov7_path_project_re,'predictions')))
        f.writelines('path_objs_names={}\n'.format(self.names_path))
        f.writelines('cd {}\n'.format(self.yolov7_path))
        #python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val
        f.writelines("python3 test.py --data {} --weights {} --conf {} --img-size {} --batch 32 --iou {} --project {} --name predictions --exist-ok --save-txt --save-conf --verbose --task test --device 0\n".format(self.YAML_PATH, self.yolov7_path_weights_re,self.THRESH,self.WIDTH_NUM,self.IOU_THRESH,self.yolov7_path_project_re))
        f.writelines('cd {}\n'.format(self.CWD))
        f.writelines('python3 resources/convert_predictions_to_xml_yolov7.py --path_result_list_txt=$path_result_list_txt --path_predictions_folder=$path_predictions_folder --path_objs_names=$path_objs_names \n')
        #f.writelines('python3 resources/iou_chips.py --Prediction_xml $path_predictions_folder/Annotations --path_result_list_txt=$path_result_list_txt\n')
        f.close()

    def create_tflite_bash(self):
        if os.path.exists(os.path.join(self.CWD,'libs/tensorflow_yolov4_tflite_path.py')) and (self.WIDTH_NUM==self.HEIGHT_NUM) and os.path.exists(self.best_weights_path):
            from libs import tensorflow_yolov4_tflite_path
            self.tensorflow_yolov4_tflite_PATH=tensorflow_yolov4_tflite_path.path
            self.tensorflow_yolov4_tflite_bash_OG_PATH=os.path.join(self.CWD,'resources/tensorflow_yolov4_tflite_bash_OG.sh')
            self.convert_config_OG_path=os.path.join(self.CWD,'resources/convert_config_OG.py')
            self.tensorflow_yolov4_tflite_bash_PATH=os.path.join(os.path.dirname(self.data_path),'tensorflow_yolov4_tflite_bash.sh')
            OUTPUT_PATH=os.path.join(os.path.dirname(self.data_path),os.path.basename(self.best_weights_path.split('.')[0]))
            
            f=open(self.tensorflow_yolov4_tflite_bash_PATH,'w')
            f.writelines('BASEPATH={}\n'.format(self.tensorflow_yolov4_tflite_PATH))
            f.writelines('CONFIGPATH=$BASEPATH/core/config.py\n')
            f.writelines('CONFIGPATH_OG=$BASEPATH/core/config_OG.py\n')
            f.writelines('OBJ_NAMES={}\n'.format(self.names_path))
            f.writelines('WEIGHTS_PATH={}\n'.format(self.best_weights_path))
            f.writelines('OUTPUT_PATH={}\n'.format(OUTPUT_PATH))
            f.writelines('TFLITE_PATH={}\n'.format(os.path.join(OUTPUT_PATH,os.path.basename(OUTPUT_PATH)+'-fp32.tflite')))
            f.writelines('NEW_OBJ_NAMES={}\n'.format(os.path.join(OUTPUT_PATH,os.path.basename(OUTPUT_PATH)+'-fp32.txt')))
            f.writelines('INPUT_SIZE={}\n'.format(self.WIDTH_NUM)) #these have to be the same size
            f.writelines('CONVERT_CONFIG_OG_PATH={}\n'.format(self.convert_config_OG_path))
            f.writelines('echo "If this fails to export, try another libGLdispatch location"\n')
            f.writelines('export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0 \n')
            f.writelines('python3 $CONVERT_CONFIG_OG_PATH --CONFIGPATH=$CONFIGPATH --CONFIGPATH_OG=$CONFIGPATH_OG --OBJ_NAMES=$OBJ_NAMES \n')
            f.writelines('cd $BASEPATH \n')
            f.writelines('python3 save_model.py --weights $WEIGHTS_PATH --output $OUTPUT_PATH --input_size $INPUT_SIZE \n')
            f.writelines('python3 convert_tflite.py --weights $OUTPUT_PATH --output $TFLITE_PATH --input_size $INPUT_SIZE \n')
            f.writelines('cp $OBJ_NAMES $NEW_OBJ_NAMES \n')
            f.close()



    def create_YAML(self):
        self.YAML_PATH=os.path.join(os.path.dirname(self.data_path),'custom.yaml')
        f=open(self.YAML_PATH,'w')
        f.writelines('train: {}\n'.format(str(self.train_list_path)))
        f.writelines('val: {}\n'.format(str(self.valid_list_path)))
        if self.img_list_path==None:
            f.writelines('test: {}\n'.format(str(self.valid_list_path)))
            self.test_list_path=self.valid_list_path
        else:
            f.writelines('test: {}\n'.format(str(self.img_list_path)))
            self.test_list_path=self.img_list_path
        #f.writelines('test: {}\n'.format(str(self.valid_list_path)))
        f.writelines('# number of classes\n')
        f.writelines('nc: {}\n'.format(str(self.num_classes)))
        f.writelines('# class names\n')
        class_names_string=''
        if os.path.exists(self.names_path):
            fo=open(self.names_path,'r')
            fo_read=fo.readlines()
            fo.close()
            self.found_names={w.replace('\n',''):j for j,w in enumerate(fo_read)}
        else:
            self.found_names={}
            for i in range(self.num_classes):
                self.found_names[i]=i
        for k,i in self.found_names.items():
            class_names_string+=' {},'.format(k)
        f.writelines('names: [{}]'.format(class_names_string))
        f.close()
        if os.path.exists('libs/yolov7_path.py'):
            from libs import yolov7_path
            self.yolov7_path=yolov7_path.yolov7_path
            if os.path.exists(self.yolov7_path):
                pass
            else:
                self.yolov7_path=''
        else:
            self.yolov7_path=''
        self.yolov7_path_train=os.path.join(self.yolov7_path,'train.py')
        self.yolov7_path_cfg=os.path.join(self.yolov7_path,'cfg/training/yolov7-tiny.yaml')
        self.yolov7_path_cfg_e6e=os.path.join(self.yolov7_path,'cfg/training/yolov7-e6e.yaml')
        self.yolov7_path_cfg_re=os.path.join(self.yolov7_path,'cfg/training/yolov7.yaml')
        self.yolov7_path_name=os.path.join(os.path.dirname(self.data_path),'yolov7-tiny')
        self.yolov7_path_name_e6e=os.path.join(os.path.dirname(self.data_path),'yolov7-e6e')
        self.yolov7_path_name_re=os.path.join(os.path.dirname(self.data_path),'yolov7')
        self.yolov7_path_hyp=os.path.join(self.yolov7_path,'data/hyp.scratch.tiny.yaml')
        self.yolov7_path_hyp_re=os.path.join(self.yolov7_path,'data/hyp.scratch.p5.yaml')
        self.yolov7_path_hyp_e6e=os.path.join(self.yolov7_path,'data/hyp.scratch.p6.yaml')
        self.yolov7_path_project_tiny=os.path.join(self.yolov7_path,os.path.dirname(self.data_path),'yolov7-tiny_detections')
        self.yolov7_path_project_re=os.path.join(self.yolov7_path,os.path.dirname(self.data_path),'yolov7_detections')
        self.yolov7_path_project_e6e=os.path.join(self.yolov7_path,os.path.dirname(self.data_path),'yolov7-e6e_detections')
        self.create_train_bash_yolov7()
        self.create_train_bash_yolov7_e6e()
        self.create_train_bash_yolov7_re()
        self.TRAIN_YOLOV7=os.path.join(os.path.dirname(self.data_path),'train_custom_Yolov7-tiny.sh')
        self.TRAIN_YOLOV7_e6e=os.path.join(os.path.dirname(self.data_path),'train_custom_Yolov7-e6e.sh')
        #python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
        #python3 detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source 0
        self.yolov7_path_test=os.path.join(self.yolov7_path,'detect.py')
        self.yolov7_path_weights_main=os.path.join(self.yolov7_path_name,'weights')
        self.yolov7_path_weights_main_e6e=os.path.join(self.yolov7_path_name_e6e,'weights')
        self.yolov7_path_weights_main_re=os.path.join(self.yolov7_path_name_re,'weights')
        if os.path.exists(self.yolov7_path_weights_main):
            if 'best.pt' in os.listdir(self.yolov7_path_weights_main):
                self.yolov7_path_weights=os.path.join(self.yolov7_path_weights_main,'best.pt')
            else:
                self.yolov7_path_weights=os.path.join(self.yolov7_path_weights_main,'last.pt')
        else:
            self.yolov7_path_weights=os.path.join(self.yolov7_path_weights_main,'last.pt')
        if os.path.exists(self.yolov7_path_weights_main_e6e):
            if 'best.pt' in os.listdir(self.yolov7_path_weights_main_e6e):
                self.yolov7_path_weights_e6e=os.path.join(self.yolov7_path_weights_main_e6e,'best.pt')
            else:
                self.yolov7_path_weights_e6e=os.path.join(self.yolov7_path_weights_main_e6e,'last.pt')
        else:
            self.yolov7_path_weights_e6e=os.path.join(self.yolov7_path_weights_main_e6e,'last.pt')

        if os.path.exists(self.yolov7_path_weights_main_re):
            if 'best.pt' in os.listdir(self.yolov7_path_weights_main_re):
                self.yolov7_path_weights_re=os.path.join(self.yolov7_path_weights_main_re,'best.pt')
            else:
                self.yolov7_path_weights_re=os.path.join(self.yolov7_path_weights_main_re,'last.pt')
        else:
            self.yolov7_path_weights_re=os.path.join(self.yolov7_path_weights_main_re,'last.pt')

        self.create_test_bash_webcam_yolov7()
        self.create_test_bash_mp4_yolov7()
        self.create_test_bash_webcam_yolov7_e6e()
        self.create_test_bash_mp4_yolov7_e6e()
        self.create_test_bash_webcam_yolov7_re()
        self.create_test_bash_mp4_yolov7_re()

        self.create_test_bash_rtsp_yolov7()
        self.create_test_bash_rtsp_yolov7_re()



    def create_obj_data(self):
        try:
            os.makedirs(self.backup_path)
        except:
            pass
        self.check_backup_path_weights()
        f=open(self.data_path,'w')
        f.writelines('classes='+str(self.num_classes)+'\n')
        f.writelines('train='+str(self.train_list_path)+'\n')
        f.writelines('valid='+str(self.valid_list_path)+'\n')
        f.writelines('names='+str(self.names_path)+'\n')
        f.writelines('backup='+str(self.backup_path)+'\n')
        f.writelines('eval=coco\n')
        f.close()
        self.create_model_test()
        self.create_YAML()

    def create_model_test(self):
        try:
            os.makedirs(self.model_i_path)
        except:
            pass
        data_path_test=os.path.join(self.model_i_path,'obj_test.data')
        f=open(data_path_test,'w')
        f.writelines('classes='+str(self.num_classes)+'\n')
        f.writelines('names='+str(self.names_path)+'\n')
        f.close()

        cli_path_test=os.path.join(self.model_i_path,'cli_test.data')
        f=open(cli_path_test,'w')
        f.writelines('data_path='+str(data_path_test)+'\n')
        f.writelines('config_path_test='+str(self.save_cfg_path_test)+'\n')
        f.writelines('best_weights='+str(self.best_weights_path)+'\n')
        f.close()
    
    def read_model_test(self,model_path=None):
        if model_path:
            model_path_test=model_path
            model_path_test=os.path.join(self.MODEL_PATHS,model_path_test)
            print('model_path_test:\n',model_path_test)
        else:
            model_path_test=self.model_i_path
        data_path_test=os.path.join(model_path_test,'obj_test.data')
        f=open(data_path_test,'r')
        self.data_path_test_lines=f.readlines()
        f.close()
        #os.system('xdg-open {}'.format(data_path_test))

        cli_path_test=os.path.join(model_path_test,'cli_test.data')
        f=open(cli_path_test,'r')
        self.cli_path_test_lines=f.readlines()
        f.close()
    def check_backup_path_weights(self):
        if os.path.exists(self.backup_path):
            self.weight_paths=[os.path.join(self.backup_path,w) for w in os.listdir(self.backup_path) if w[0]!="." and w.find('.weights')!=-1]
            if len(self.weight_paths)==0:
                self.best_weights_path=None 
                self.tiny_conv29_path=self.tiny_conv29_path
            else:
                for weight_file in self.weight_paths:
                    if weight_file.find('best')!=-1:
                        self.best_weights_path=weight_file
                        self.tiny_conv29_path=weight_file
                        break
                    elif weight_file.find('last.weights')!=-1:
                        self.best_weights_path=weight_file
                        self.tiny_conv29_path=weight_file
                    else:
                        self.best_weights_path=None 
                        self.tiny_conv29_path=self.tiny_conv29_path                     


    def create_train_bash(self):
        self.check_backup_path_weights()
        f=open(self.save_cfg_path_train.replace('.cfg','.sh'),'w')
        f.writelines('darknet='+str(os.path.join(self.darknet_path,'darknet'))+'\n')
        f.writelines('data_path='+str(self.data_path)+'\n')
        f.writelines('config_path='+str(self.save_cfg_path_train)+'\n')
        f.writelines('tiny_weights='+str(self.tiny_conv29_path)+'\n')
        f.writelines('cd {}\n'.format(self.darknet_path))
        f.writelines('$darknet detector train $data_path $config_path $tiny_weights -map\n')
        f.close()
        #os.system('sudo chmod 777 "{}"'.format(self.save_cfg_path_train.replace('.cfg','.sh')))
    def create_test_bash_dnn(self):
        self.check_backup_path_weights()
        f=open(self.save_cfg_path_test.replace('.cfg','dnn.sh'),'w')
        f.writelines('config_path_test='+str(self.save_cfg_path_test)+'\n')
        f.writelines('obj_path='+str(self.names_path)+'\n')
        if self.best_weights_path==None:
            self.best_weights_path=os.path.join(self.backup_path,os.path.basename(self.save_cfg_path_test.replace('_test.cfg',''))+'_train_best.weights')
        f.writelines('best_weights='+str(self.best_weights_path)+'\n')
        f.writelines('imW='+str(self.WIDTH_NUM)+'\n')
        f.writelines('imH='+str(self.HEIGHT_NUM)+'\n')
        f.writelines('cd {}\n'.format(self.DNN_PATH.replace('yolo_dnn_multi_drone_hdmi.py','')))
        if self.RTSP_SERVER:
            if self.USE_RTSP_VAR.get()=="Yes":
                #f.writelines('python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No --RTSP_PATH=Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n'.format(self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get()))
                cmd_i='python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No --RTSP_PATH=Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n'.format(self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get())
            else:
                #f.writelines('python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No \n')
                cmd_i='python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No \n'
            if self.sec.get()=='y':
                self.destination_list_final=''
                for w_var in self.phone_dic_trigger_var.values():
                    var_i=w_var.get()
                    if var_i!='None':
                        self.destination_list_final=self.destination_list_final+";"+var_i
                self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
                cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            f.writelines(cmd_i)  
        elif self.sec.get()=='y':
            self.destination_list_final=''
            for w_var in self.phone_dic_trigger_var.values():
                var_i=w_var.get()
                if var_i!='None':
                    self.destination_list_final=self.destination_list_final+";"+var_i
            self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
            cmd_i='python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No \n'
            cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            f.writelines(cmd_i)        
        else:
            f.writelines('python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No \n')
        f.close()

    def create_test_bash_dnn_labelimg(self):
        self.check_backup_path_weights()
        f=open(self.save_cfg_path_test.replace('.cfg','dnn_labelimg.sh'),'w')
        f.writelines('config_path_test='+str(self.save_cfg_path_test)+'\n')
        f.writelines('obj_path='+str(self.names_path)+'\n')
        if self.best_weights_path==None:
            self.best_weights_path=os.path.join(self.backup_path,os.path.basename(self.save_cfg_path_test.replace('_test.cfg',''))+'_train_best.weights')
        f.writelines('best_weights='+str(self.best_weights_path)+'\n')
        f.writelines('image=going_to_labelimg\n')
        f.writelines('imW='+str(self.WIDTH_NUM)+'\n')
        f.writelines('imH='+str(self.HEIGHT_NUM)+'\n')
        f.writelines('cd {}\n'.format(self.DNN_PATH.replace('yolo_dnn_multi_drone_hdmi.py','')))
        f.writelines('python3 yolo_dnn_multi_drone_hdmi.py --image=$image --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No --use_socket_receive_imgs --noview \n')
        f.close()

    def create_test_bash_dnn_rtsp(self):
        self.check_backup_path_weights()
        f=open(self.save_cfg_path_test.replace('.cfg','dnn_rtsp.sh'),'w')
        f.writelines('config_path_test='+str(self.save_cfg_path_test)+'\n')
        f.writelines('obj_path='+str(self.names_path)+'\n')
        if self.best_weights_path==None:
            self.best_weights_path=os.path.join(self.backup_path,os.path.basename(self.save_cfg_path_test.replace('_test.cfg',''))+'_train_best.weights')
        f.writelines('best_weights='+str(self.best_weights_path)+'\n')
        f.writelines('imW='+str(self.WIDTH_NUM)+'\n')
        f.writelines('imH='+str(self.HEIGHT_NUM)+'\n')
        f.writelines('cd {}\n'.format(self.DNN_PATH.replace('yolo_dnn_multi_drone_hdmi.py','')))
        if self.RTSP_SERVER:
            if self.USE_RTSP_VAR.get()=="Yes":
                #f.writelines('python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No --RTSP_PATH=Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n'.format(self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get()))
                cmd_i='python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --save=No --RTSP_PATH=Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {} --video={}\n'.format(self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get(),self.USE_RTSP_CLIENT_VAR.get())
            else:
                #f.writelines('python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No \n')
                cmd_i='python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --save=No --video={}\n'.format(self.USE_RTSP_CLIENT_VAR.get())
            if self.sec.get()=='y':
                self.destination_list_final=''
                for w_var in self.phone_dic_trigger_var.values():
                    var_i=w_var.get()
                    if var_i!='None':
                        self.destination_list_final=self.destination_list_final+";"+var_i
                self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
                cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            f.writelines(cmd_i)  
        elif self.sec.get()=='y':
            self.destination_list_final=''
            for w_var in self.phone_dic_trigger_var.values():
                var_i=w_var.get()
                if var_i!='None':
                    self.destination_list_final=self.destination_list_final+";"+var_i
            self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
            cmd_i='python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video={} --save=No \n'.format(self.USE_RTSP_CLIENT_VAR.get())
            cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            f.writelines(cmd_i)        
        else:
            f.writelines('python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video={} --save=No \n'.format(self.USE_RTSP_CLIENT_VAR.get()))
        f.close()

    def TMP_create_test_dnn_bash(self):
        self.check_backup_path_weights()
        self.create_model_test()
        self.read_model_test()
        tmp_path=os.path.join(self.base_path_OG,'temp')
        self.tmp_test_path=os.path.join(tmp_path,'testdnn.sh')
        f=open(self.tmp_test_path,'w')
        [f.writelines(line) for line in self.cli_path_test_lines]
        f.writelines('darknet='+str(os.path.join(self.darknet_path,'darknet'))+'\n')
        f.writelines('cd {}\n'.format(self.darknet_path))
        if self.RTSP_SERVER:
            if self.USE_RTSP_VAR.get()=="Yes":
                #f.writelines('python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No --RTSP_PATH=Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n'.format(self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get()))
                cmd_i='python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No --RTSP_PATH=Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n'.format(self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get())
            else:
                #f.writelines('python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No \n')
                cmd_i='python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No \n'
            if self.sec.get()=='y':
                self.destination_list_final=''
                for w_var in self.phone_dic_trigger_var.values():
                    var_i=w_var.get()
                    if var_i!='None':
                        self.destination_list_final=self.destination_list_final+";"+var_i
                self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
                cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            f.writelines(cmd_i)  
        elif self.sec.get()=='y':
            self.destination_list_final=''
            for w_var in self.phone_dic_trigger_var.values():
                var_i=w_var.get()
                if var_i!='None':
                    self.destination_list_final=self.destination_list_final+";"+var_i
            self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
            cmd_i='python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No \n'
            cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            f.writelines(cmd_i)        
        else:
            f.writelines('python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No \n')
        f.close()

    def TMP_create_test_dnn_bash_rtsp(self):
        self.check_backup_path_weights()
        self.create_model_test()
        self.read_model_test()
        tmp_path=os.path.join(self.base_path_OG,'temp')
        self.tmp_test_path=os.path.join(tmp_path,'dnn_rtsp.sh')
        f=open(self.tmp_test_path,'w')
        [f.writelines(line) for line in self.cli_path_test_lines]
        f.writelines('darknet='+str(os.path.join(self.darknet_path,'darknet'))+'\n')
        f.writelines('cd {}\n'.format(self.darknet_path))
        if self.RTSP_SERVER:
            if self.USE_RTSP_VAR.get()=="Yes":
                #f.writelines('python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No --RTSP_PATH=Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {}\n'.format(self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get()))
                cmd_i='python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --save=No --RTSP_PATH=Custom --RTSP_SERVER_PATH {} --fps {} --port {} --stream_key {} --video={}\n'.format(self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get(),self.USE_RTSP_CLIENT_VAR.get())
            else:
                #f.writelines('python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No \n')
                cmd_i='python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --save=No --video={}\n'.format(self.USE_RTSP_CLIENT_VAR.get())
            if self.sec.get()=='y':
                self.destination_list_final=''
                for w_var in self.phone_dic_trigger_var.values():
                    var_i=w_var.get()
                    if var_i!='None':
                        self.destination_list_final=self.destination_list_final+";"+var_i
                self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
                cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            f.writelines(cmd_i)  
        elif self.sec.get()=='y':
            self.destination_list_final=''
            for w_var in self.phone_dic_trigger_var.values():
                var_i=w_var.get()
                if var_i!='None':
                    self.destination_list_final=self.destination_list_final+";"+var_i
            self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
            cmd_i='python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video={} --save=No \n'.format(self.use_RTSP_CLIENT_VAR.get())
            cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            f.writelines(cmd_i)        
        else:
            f.writelines('python3 yolo_dnn_multi_drone_hdmi.py --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video={} --save=No \n'.format(self.use_RTSP_CLIENT_VAR.get()))
        f.close()

    def create_test_bash_dnn_rtmp(self):
        self.check_backup_path_weights()
        self.YOUTUBE_KEY=self.YOUTUBE_KEY_VAR.get()
        fo=open('YOUTUBE_KEY.txt','w')
        fo_read=fo.writelines(self.YOUTUBE_KEY+'\n')
        fo.close()
        f=open(self.save_cfg_path_test.replace('.cfg','dnn_rtmp.sh'),'w')
        f.writelines('config_path_test='+str(self.save_cfg_path_test)+'\n')
        f.writelines('obj_path='+str(self.names_path)+'\n')
        if self.best_weights_path==None:
            self.best_weights_path=os.path.join(self.backup_path,os.path.basename(self.save_cfg_path_test.replace('_test.cfg',''))+'_train_best.weights')
        f.writelines('best_weights='+str(self.best_weights_path)+'\n')
        f.writelines('imW='+str(self.WIDTH_NUM)+'\n')
        f.writelines('imH='+str(self.HEIGHT_NUM)+'\n')

        f.writelines('YOUTUBE_RTMP={}\n'.format(self.YOUTUBE_KEY))
        f.writelines('YOUTUBE_STREAM_RES={}\n'.format(self.USER_SELECTION_yt.get()))
        f.writelines('cd {}\n'.format(self.DNN_PATH.replace('yolo_dnn_multi_drone_hdmi.py','')))
        if self.RTSP_SERVER:
            if self.USE_RTSP_VAR.get()=="Yes":
                cmd_i='python3 yolo_dnn_multi_drone_hdmi.py --YOUTUBE_RTMP=$YOUTUBE_RTMP --YOUTUBE_STREAM_RES=$YOUTUBE_STREAM_RES --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No --RTSP_PATH=Custom --RTSP_SERVER_PATH={} --fps={} --port={} --stream_key={}\n'.format(self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get())
            else:
                cmd_i='python3 yolo_dnn_multi_drone_hdmi.py --YOUTUBE_RTMP=$YOUTUBE_RTMP --YOUTUBE_STREAM_RES=$YOUTUBE_STREAM_RES --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No'
            if self.sec.get()=='y':
                self.destination_list_final=''
                for w_var in self.phone_dic_trigger_var.values():
                    var_i=w_var.get()
                    if var_i!='None':
                        self.destination_list_final=self.destination_list_final+";"+var_i
                self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
                cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            f.writelines(cmd_i) 
        elif self.sec.get()=='y':
            self.destination_list_final=''
            for w_var in self.phone_dic_trigger_var.values():
                var_i=w_var.get()
                if var_i!='None':
                    self.destination_list_final=self.destination_list_final+";"+var_i
            self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
            cmd_i='python3 yolo_dnn_multi_drone_hdmi.py --YOUTUBE_RTMP=$YOUTUBE_RTMP --YOUTUBE_STREAM_RES=$YOUTUBE_STREAM_RES --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No'
            cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            f.writelines(cmd_i)     
        else:
            f.writelines('python3 yolo_dnn_multi_drone_hdmi.py --YOUTUBE_RTMP=$YOUTUBE_RTMP --YOUTUBE_STREAM_RES=$YOUTUBE_STREAM_RES --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No')
        f.close()

    def TMP_create_test_dnn_rtmp_bash(self):
        self.check_backup_path_weights()
        self.create_model_test()
        self.read_model_test()
        tmp_path=os.path.join(self.base_path_OG,'temp')
        self.tmp_test_path=os.path.join(tmp_path,'testdnn_rtmp.sh')
        self.YOUTUBE_KEY=self.YOUTUBE_KEY_VAR.get()
        fo=open('YOUTUBE_KEY.txt','w')
        fo_read=fo.writelines(self.YOUTUBE_KEY+'\n')
        fo.close()
        f=open(self.tmp_test_path,'w')
        [f.writelines(line) for line in self.cli_path_test_lines]
        f.writelines('YOUTUBE_RTMP={}\n'.format(self.YOUTUBE_KEY))
        f.writelines('YOUTUBE_STREAM_RES={}\n'.format(self.USER_SELECTION_yt.get()))
        f.writelines('darknet='+str(os.path.join(self.darknet_path,'darknet'))+'\n')
        f.writelines('cd {}\n'.format(self.darknet_path))
        if self.RTSP_SERVER:
            if self.USE_RTSP_VAR.get()=="Yes":
                cmd_i='python3 yolo_dnn_multi_drone_hdmi.py --YOUTUBE_RTMP=$YOUTUBE_RTMP --YOUTUBE_STREAM_RES=$YOUTUBE_STREAM_RES --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No --RTSP_PATH=Custom --RTSP_SERVER_PATH={} --fps={} --port={} --stream_key={}\n'.format(self.RTSP_SERVER_PATH,self.FPS_VAR.get(),self.PORT_VAR.get(),self.STREAM_KEY_VAR.get())
            else:
                cmd_i='python3 yolo_dnn_multi_drone_hdmi.py --YOUTUBE_RTMP=$YOUTUBE_RTMP --YOUTUBE_STREAM_RES=$YOUTUBE_STREAM_RES --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No'
            if self.sec.get()=='y':
                self.destination_list_final=''
                for w_var in self.phone_dic_trigger_var.values():
                    var_i=w_var.get()
                    if var_i!='None':
                        self.destination_list_final=self.destination_list_final+";"+var_i
                self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
                cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            f.writelines(cmd_i) 
        elif self.sec.get()=='y':
            self.destination_list_final=''
            for w_var in self.phone_dic_trigger_var.values():
                var_i=w_var.get()
                if var_i!='None':
                    self.destination_list_final=self.destination_list_final+";"+var_i
            self.destination_list_final='"'+self.destination_list_final.lstrip(';')+'"' 
            cmd_i='python3 yolo_dnn_multi_drone_hdmi.py --YOUTUBE_RTMP=$YOUTUBE_RTMP --YOUTUBE_STREAM_RES=$YOUTUBE_STREAM_RES --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No'
            cmd_i=cmd_i.replace('\n',"") + ' --destinations={} --sleep_time_chips={} --send_image_to_cell \n'.format(self.destination_list_final,self.sleep_time_chips_VAR.get())
            f.writelines(cmd_i)     
        else:
            f.writelines('python3 yolo_dnn_multi_drone_hdmi.py --YOUTUBE_RTMP=$YOUTUBE_RTMP --YOUTUBE_STREAM_RES=$YOUTUBE_STREAM_RES --weightsPath=$best_weights --labelsPath=$obj_path --configPath=$config_path_test --imW=$imW --imH=$imH --video=0 --save=No')
        f.close()

    def create_test_bash(self):
        self.check_backup_path_weights()
        f=open(self.save_cfg_path_test.replace('.cfg','.sh'),'w')
        f.writelines('config_path_test='+str(self.save_cfg_path_test)+'\n')
        f.writelines('data_path='+str(self.data_path)+'\n')
        f.writelines('darknet='+str(os.path.join(self.darknet_path,'darknet'))+'\n')
        if self.best_weights_path==None:
            self.best_weights_path=os.path.join(self.backup_path,os.path.basename(self.save_cfg_path_test.replace('_test.cfg',''))+'_train_best.weights')
        f.writelines('best_weights='+str(self.best_weights_path)+'\n')
        f.writelines('threshold={}\n'.format(self.THRESH_VAR.get()))
        f.writelines('cd {}\n'.format(self.darknet_path))
        f.writelines('$darknet detector demo $data_path $config_path_test $best_weights -c 0 -thresh $threshold\n')
        f.close()
        #os.system('sudo chmod 777 "{}"'.format(self.save_cfg_path_test.replace('.cfg','.sh')))

    def TMP_create_test_bash(self):
        self.check_backup_path_weights()
        self.create_model_test()
        self.read_model_test()
        tmp_path=os.path.join(self.base_path_OG,'temp')
        self.tmp_test_path=os.path.join(tmp_path,'test.sh')
        f=open(self.tmp_test_path,'w')
        [f.writelines(line) for line in self.cli_path_test_lines]
        #f.writelines('config_path_test='+str(self.save_cfg_path_test)+'\n')
        #f.writelines('data_path='+str(self.data_path)+'\n')
        #f.writelines('best_weights='+str(self.best_weights_path)+'\n')
        f.writelines('darknet='+str(os.path.join(self.darknet_path,'darknet'))+'\n')
        f.writelines('cd {}\n'.format(self.darknet_path))
        f.writelines('$darknet detector demo $data_path $config_path_test $best_weights -c 0 -thresh $threshold\n')
        f.close()

    def create_test_bash_mp4(self):
        self.check_backup_path_weights()
        f=open(self.save_cfg_path_test.replace('.cfg','_mp4.sh'),'w')
        f.writelines('config_path_test='+str(self.save_cfg_path_test)+'\n')
        f.writelines('data_path='+str(self.data_path)+'\n')
        f.writelines('darknet='+str(os.path.join(self.darknet_path,'darknet'))+'\n')
        if self.best_weights_path==None:
            self.best_weights_path=os.path.join(self.backup_path,os.path.basename(self.save_cfg_path_test.replace('_test.cfg',''))+'_train_best.weights')
        f.writelines('best_weights='+str(self.best_weights_path)+'\n')
        f.writelines('mp4_video='+str(self.mp4_video_path)+'\n')
        f.writelines('result_mp4_file={}\n'.format(os.path.join(os.path.dirname(self.data_path),os.path.basename(self.mp4_video_path).split('.')[0]+'_results.txt')))
        f.writelines('cd {}\n'.format(self.darknet_path))

        f.writelines('$darknet detector demo $data_path $config_path_test $best_weights $mp4_video -i 0 -thresh {} > $result_mp4_file \n'.format(str(round(float(self.THRESH_VAR.get()),2))))
        f.close()
        #os.system('sudo chmod 777 "{}"'.format(self.save_cfg_path_test.replace('.cfg','_mp4.sh')))

    def TMP_create_test_bash_mp4(self):
        self.check_backup_path_weights()
        self.create_model_test()
        self.read_model_test()
        tmp_path=os.path.join(self.base_path_OG,'temp')
        self.tmp_test_path=os.path.join(tmp_path,'test.sh')
        f=open(self.tmp_test_path,'w')
        [f.writelines(line) for line in self.cli_path_test_lines]
        #f.writelines('config_path_test='+str(self.save_cfg_path_test)+'\n')
        #f.writelines('data_path='+str(self.data_path)+'\n')
        #f.writelines('best_weights='+str(self.best_weights_path)+'\n')
        f.writelines('darknet='+str(os.path.join(self.darknet_path,'darknet'))+'\n')
        f.writelines('mp4_video='+str(self.mp4_video_path)+'\n')
        f.writelines('result_mp4_file={}\n'.format(os.path.join(os.path.dirname(self.data_path),os.path.basename(self.mp4_video_path).split('.')[0]+'_results.txt')))
        f.writelines('cd {}\n'.format(self.darknet_path))
        f.writelines('$darknet detector demo $data_path $config_path_test $best_weights $mp4_video -i 0 -thresh {} > $result_mp4_file\n'.format(str(round(float(self.THRESH_VAR.get()),2))))
        f.close()
        
    def create_test_bash_mp4_record(self):
        self.check_backup_path_weights()
        f=open(self.save_cfg_path_test.replace('.cfg','_mp4_record.sh'),'w')
        f.writelines('config_path_test='+str(self.save_cfg_path_test)+'\n')
        f.writelines('data_path='+str(self.data_path)+'\n')
        f.writelines('avi_output='+str(os.path.join(self.base_path,os.path.basename(self.mp4_video_path).replace('.mp4','.avi')))+'\n')
        f.writelines('darknet='+str(os.path.join(self.darknet_path,'darknet'))+'\n')
        if self.best_weights_path==None:
            self.best_weights_path=os.path.join(self.backup_path,os.path.basename(self.save_cfg_path_test.replace('_test.cfg',''))+'_train_best.weights')
        f.writelines('best_weights='+str(self.best_weights_path)+'\n')
        f.writelines('mp4_video='+str(self.mp4_video_path)+'\n')

        f.writelines('cd {}\n'.format(self.darknet_path))
        f.writelines('$darknet detector demo $data_path $config_path_test $best_weights $mp4_video -i 0 -thresh {} -out_filename $avi_output\n'.format(str(round(float(self.THRESH_VAR.get()),2))))
        f.close()
        #os.system('sudo chmod 777 "{}"'.format(self.save_cfg_path_test.replace('.cfg','_mp4_record.sh')))
        
    def TMP_create_test_bash_mp4_record(self):
        self.check_backup_path_weights()
        self.create_model_test()
        self.read_model_test()
        tmp_path=os.path.join(self.base_path_OG,'temp')
        self.tmp_test_path=os.path.join(tmp_path,'test.sh')
        f=open(self.tmp_test_path,'w')
        [f.writelines(line) for line in self.cli_path_test_lines]
        #f.writelines('config_path_test='+str(self.save_cfg_path_test)+'\n')
        #f.writelines('data_path='+str(self.data_path)+'\n')
        #f.writelines('best_weights='+str(self.best_weights_path)+'\n')
        f.writelines('darknet='+str(os.path.join(self.darknet_path,'darknet'))+'\n')
        f.writelines('mp4_video='+str(self.mp4_video_path)+'\n')
        f.writelines('avi_output='+str(os.path.join(self.base_path,os.path.basename(self.mp4_video_path).replace('.mp4','')+'__PREDICTED_WITH-'+os.path.basename(self.model_path_test)+'.avi'))+'\n')
        f.writelines('cd {}\n'.format(self.darknet_path))
        f.writelines('$darknet detector demo $data_path $config_path_test $best_weights $mp4_video -i 0 -thresh {} -out_filename $avi_output\n'.format(str(round(float(self.THRESH_VAR.get()),2))))
        f.close()


    def create_test_bash_images_with_predictions(self):
        self.check_backup_path_weights()
        f=open(self.save_cfg_path_test.replace('.cfg','_images_with_predictions.sh'),'w')
        f.writelines('config_path_test='+str(self.save_cfg_path_test)+'\n')
        f.writelines('data_path='+str(self.data_path)+'\n')
        f.writelines('darknet='+str(os.path.join(self.darknet_path,'darknet'))+'\n')
        if self.best_weights_path==None:
            self.best_weights_path=os.path.join(self.backup_path,os.path.basename(self.save_cfg_path_test.replace('_test.cfg',''))+'_train_best.weights')
        f.writelines('best_weights='+str(self.best_weights_path)+'\n')
        if self.img_list_path==None:
            f.writelines('path_test_list_txt='+str(self.valid_list_path)+'\n')
        else:
            f.writelines('path_test_list_txt='+str(self.img_list_path)+'\n')
        self.prediction_list_path=os.path.join(self.base_path,'predictions.txt')
        print('self.prediction_list_path',self.prediction_list_path)
        f.writelines('path_result_list_txt='+str(self.prediction_list_path)+'\n')
        f.writelines('cd {}\n'.format(self.darknet_path))
        f.writelines('$darknet detector test $data_path $config_path_test $best_weights -thresh {} -dont_show -ext_output < $path_test_list_txt > $path_result_list_txt\n'.format(str(round(float(self.THRESH_VAR.get()),2))))
        create_predictions=os.path.join(os.getcwd(),'resources/convert_predictions_to_xml.py')
        f.writelines('path_compute_mAP={}\n'.format(os.path.abspath(os.path.join(os.getcwd(),'resources/compute_mAP.py'))))
        f.writelines('python3 {} --path_result_list_txt {} --path_compute_mAP=$path_compute_mAP\n'.format(create_predictions, self.prediction_list_path))
        #create_chips=os.path.join(os.getcwd(),'resources/iou_chips.py')
        #f.writelines('python3 {} --Prediction_xml {}'.format(create_chips,os.path.join(self.prediction_list_path.split('.txt')[0],'Annotations')))
        #f.writelines('python3 resources/iou_chips.py --Prediction_xml $path_predictions_folder/Annotations --path_result_list_txt=$path_result_list_txt\n')
        f.close()
        #os.system('sudo chmod 777 "{}"'.format(self.save_cfg_path_test.replace('.cfg','_images_with_predictions.sh')))

    def create_test_bash_images_with_predictions_mAP(self):
        self.check_backup_path_weights()
        self.THRESH=self.THRESH_VAR.get()
        self.IOU_THRESH=self.IOU_THRESH_VAR.get()
        self.POINTS=self.POINTS_VAR.get().split(':')[0]
        f=open(self.save_cfg_path_test.replace('.cfg','_images_with_predictions_mAP.sh'),'w')
        f.writelines('config_path_test='+str(self.save_cfg_path_test)+'\n')
        f.writelines('data_path='+str(self.data_path)+'\n')
        f.writelines('darknet='+str(os.path.join(self.darknet_path,'darknet'))+'\n')
        if self.best_weights_path==None:
            self.best_weights_path=os.path.join(self.backup_path,os.path.basename(self.save_cfg_path_test.replace('_test.cfg',''))+'_train_best.weights')
        f.writelines('best_weights='+str(self.best_weights_path)+'\n')
        if self.img_list_path==None:
            f.writelines('path_test_list_txt='+str(self.valid_list_path)+'\n')
            sample_fo=open(self.valid_list_path,'r')
            sample_fo_read=sample_fo.readlines()
            sample_fo.close()


        else:
            f.writelines('path_test_list_txt='+str(self.img_list_path)+'\n')
            print('self.img_list_path',self.img_list_path)
            sample_fo=open(self.img_list_path,'r')
            sample_fo_read=sample_fo.readlines()
            sample_fo.close()
        print("sample_fo_read[0]",sample_fo_read[0])
        self.result_list_path=os.path.join(self.base_path,'prediction_images_with_mAP.sh')
        #self.result_list_path=os.path.join(self.base_path,'obj'+os.path.basename(sample_fo_read[0].split('.')[0].replace('\n','').strip())).replace('/','_').rstrip('/').lstrip('/')+'_THRESH{}__IOU{}__POINTS{}_results.txt'.format(str(self.THRESH).replace('.','p'),str(self.IOU_THRESH).replace('.','p'),str(self.POINTS).replace('.','p'))
        f.writelines('path_result_list_txt='+str(self.result_list_path)+'\n')
        f.writelines('thresh='+str(self.THRESH)+'\n')
        f.writelines('iou_thresh='+str(self.IOU_THRESH)+'\n')
        f.writelines('points='+str(self.POINTS)+'\n')
        filter_path=os.path.join(os.getcwd(),'resources/filter_results.py')
        f.writelines('filter_path={}\n'.format(filter_path))
        create_yolov4_metrics=os.path.join(os.getcwd(),'resources/yolov4_metrics.py')
        f.writelines('python3 {} --points=$points --filter_path=$filter_path --thresh=$thresh --iou_thresh=$iou_thresh --config_path_test=$config_path_test --data_path=$data_path --darknet=$darknet --best_weights=$best_weights --path_test_list_txt=$path_test_list_txt --path_result_list_txt=$path_result_list_txt\n'.format(create_yolov4_metrics))
        f.close()
        #os.system('sudo chmod 777 "{}"'.format(self.save_cfg_path_test.replace('.cfg','_images_with_predictions.sh')))

    def close(self,event):
        self.root.destroy()

    def save_settings(self,save_root='libs'):
        if os.path.exists(os.path.join('libs','DEFAULT_SETTINGS.py')):
            f=open(os.path.join('libs','DEFAULT_SETTINGS.py'),'r')
            f_read=f.readlines()
            f.close()
            from libs import DEFAULT_SETTINGS as DS
            all_variables = dir(DS)
            all_real_variables=[]
            # Iterate over the whole list where dir( )
            # is stored.
            for name in all_variables:
                # Print the item if it doesn't start with '__'
                if not name.startswith('__'):
                    if name.find('path_prefix_volumes_one')==-1 and name.find('path_prefix_elements')==-1 and name.find('path_prefix_mount_mac')==-1 and name!='os':
                        all_real_variables.append(name)
            f_new=[]
            for prefix_i in all_real_variables:
                try:
                    prefix_i_comb="self."+prefix_i
                    prefix_i_comb=prefix_i_comb.strip()
                    print(prefix_i_comb)
                    prefix_i_value=eval(prefix_i_comb)
                except:
                    pass
                if prefix_i=='path_prefix':
                    pass
                elif (prefix_i.lower().find('path')!=-1 or prefix_i.lower().find('background')!=-1):
                    prefix_i_value="r'"+prefix_i_value+"'"
                elif type(prefix_i_value).__name__.find('int')!=-1:
                    pass
                elif type(prefix_i_value).__name__.find('str')!=-1:
                    prefix_i_value="'"+prefix_i_value+"'"
                f_new.append(prefix_i+"="+str(prefix_i_value)+'\n')
                           
            prefix_save=_platform+'_'+self.prefix_foldername+'_SAVED_SETTINGS'
            self.YOLO_MODEL_PATH=os.path.join(self.base_path_OG,self.prefix_foldername)
            f_new.append('YOLO_MODEL_PATH=r"{}"\n'.format(os.path.join(self.base_path_OG,self.prefix_foldername)))
            f_new.append('ITERATION_NUM={}\n'.format(self.ITERATION_NUM_VAR.get()))
            f_new.append('epochs_yolov7={}\n'.format(self.epochs_yolov7_VAR.get()))
            f_new.append('epochs_yolov7_re={}\n'.format(self.epochs_yolov7_re_VAR.get()))        
            f_new.append('epochs_yolov7_e6e={}\n'.format(self.epochs_yolov7_e6e_VAR.get()))   
            f=open('{}.py'.format(os.path.join(save_root,prefix_save.replace('-','_'))),'w')
            wrote=[f.writelines(w) for w in f_new]
            f.close()
            self.SAVED_SETTINGS_PATH='{}.py'.format(os.path.join(save_root,prefix_save.replace('-','_')))
            self.root.title(os.path.basename(self.SAVED_SETTINGS_PATH))
    def save_settings_CUSTOM(self,save_root='libs'):
        if os.path.exists(os.path.join('libs','DEFAULT_SETTINGS.py')):
            f=open(os.path.join('libs','DEFAULT_SETTINGS.py'),'r')
            f_read=f.readlines()
            f.close()
            from libs import DEFAULT_SETTINGS as DS
            all_variables = dir(DS)
            all_real_variables=[]
            # Iterate over the whole list where dir( )
            # is stored.
            for name in all_variables:
                # Print the item if it doesn't start with '__'
                if not name.startswith('__'):
                    if name.find('path_prefix_volumes_one')==-1 and name.find('path_prefix_elements')==-1 and name.find('path_prefix_mount_mac')==-1 and name!='os':
                        all_real_variables.append(name)
            f_new=[]
            custom_time=str(time.time()).split('.')[0]
            self.path_Yolo_CUSTOM=os.path.join(os.path.dirname(self.path_Annotations_CUSTOM),'Yolo_Objs')
            if not(os.path.exists(self.path_Yolo_CUSTOM)):
                os.makedirs(self.path_Yolo_CUSTOM)
            shutil.copy(os.path.join(self.path_Yolo,os.path.basename(self.names_path)),self.path_Yolo_CUSTOM)
            for prefix_i in all_real_variables:
                try:
                    if prefix_i.find('PREFIX')!=-1:
                        self.PREFIX_CUSTOM=self.PREFIX+"_TXLEARN_"+custom_time
                        prefix_i_value=self.PREFIX_CUSTOM
                    elif prefix_i.find('path_Annotations')!=-1:
                        prefix_i_value=self.path_Annotations_CUSTOM
                    elif prefix_i.find('path_JPEGImages')!=-1:
                        prefix_i_value=self.path_JPEGImages_CUSTOM
                    elif prefix_i.find('path_Yolo')!=-1:
                        prefix_i_value=self.path_Yolo_CUSTOM
                    elif prefix_i.find('path_prefix')!=-1:
                        prefix_i_value='r"'+str(os.path.abspath(self.path_Yolo_CUSTOM))+'"'
                    else: 
                        prefix_i_comb="self."+prefix_i
                        prefix_i_comb=prefix_i_comb.strip()
                        print(prefix_i_comb)
                        prefix_i_value=eval(prefix_i_comb)
                except:
                    pass
                if prefix_i=='path_prefix':
                    pass
                elif (prefix_i.lower().find('path')!=-1 or prefix_i.lower().find('background')!=-1):
                    prefix_i_value="r'"+prefix_i_value+"'"
                elif type(prefix_i_value).__name__.find('int')!=-1:
                    pass
                elif type(prefix_i_value).__name__.find('str')!=-1:
                    prefix_i_value="'"+prefix_i_value+"'"
                f_new.append(prefix_i+"="+str(prefix_i_value)+'\n')  
            if self.random=='0':
                self.prefix_foldername_CUSTOM='{}_w{}_h{}_d{}_c{}'.format(self.PREFIX_CUSTOM,self.WIDTH_NUM,self.HEIGHT_NUM,self.num_div,self.num_classes)
            else:
                self.prefix_foldername_CUSTOM='{}_w{}_h{}_d{}_c{}_r{}'.format(self.PREFIX_CUSTOM,self.WIDTH_NUM,self.HEIGHT_NUM,self.num_div,self.num_classes,self.random)          
            prefix_save=_platform+'_'+self.prefix_foldername_CUSTOM+'_SAVED_SETTINGS'
            existing_weight_dirs=os.listdir(self.YOLO_MODEL_PATH)
            existing_weight_dirs=[os.path.join(self.YOLO_MODEL_PATH,w) for w in existing_weight_dirs]
            existing_weight_dirs=[w for w in existing_weight_dirs if os.path.isdir(w) and w.find('detections')==-1]
            YOLO_MODEL_PATH_CUSTOM=os.path.join(self.base_path_OG,self.prefix_foldername_CUSTOM)
            if not(os.path.exists(YOLO_MODEL_PATH_CUSTOM)):
                os.makedirs(YOLO_MODEL_PATH_CUSTOM)
            shutil.copy(os.path.join(self.path_Yolo,os.path.basename(self.names_path)),YOLO_MODEL_PATH_CUSTOM)
            for dir_i in tqdm(existing_weight_dirs):
                new_dirs_in_custom_model_path=os.listdir(YOLO_MODEL_PATH_CUSTOM)
                new_dir_i_base=os.path.basename(dir_i)
                if new_dir_i_base not in new_dirs_in_custom_model_path:
                    print(f'SHUTIL COPYING dir_i={dir_i} to YOLO_MODEL_PATH_CUSTOM={YOLO_MODEL_PATH_CUSTOM}')
                    shutil.copytree(dir_i,os.path.join(YOLO_MODEL_PATH_CUSTOM,new_dir_i_base))
                else:
                    os.system(f'rm -rf {os.path.join(YOLO_MODEL_PATH_CUSTOM,new_dir_i_base)}')
                    print(f'SHUTIL COPYING dir_i={dir_i} to YOLO_MODEL_PATH_CUSTOM={YOLO_MODEL_PATH_CUSTOM}')
                    shutil.copytree(dir_i,os.path.join(YOLO_MODEL_PATH_CUSTOM,new_dir_i_base))
            f_new.append('YOLO_MODEL_PATH=r"{}"\n'.format(YOLO_MODEL_PATH_CUSTOM))
            f_new.append('ITERATION_NUM={}\n'.format(self.ITERATION_NUM_VAR.get()))
            f_new.append('epochs_yolov7={}\n'.format(self.epochs_yolov7_VAR.get()))
            f_new.append('epochs_yolov7_re={}\n'.format(self.epochs_yolov7_re_VAR.get()))        
            f_new.append('epochs_yolov7_e6e={}\n'.format(self.epochs_yolov7_e6e_VAR.get()))   
            f=open('{}.py'.format(os.path.join(save_root,prefix_save.replace('-','_'))),'w')
            wrote=[f.writelines(w) for w in f_new]
            f.close()
            self.SAVED_SETTINGS_PATH_CUSTOM='{}.py'.format(os.path.join(save_root,prefix_save.replace('-','_')))
            self.return_to_main()
            #self.root.title(os.path.basename(self.SAVED_SETTINGS_PATH_CUSTOM))
    def get_update_background_img(self):
        self.image=Image.open(self.root_background_img)
        self.image=self.image.resize((self.root_W,self.root_H),Image.ANTIALIAS)
        self.bg=ImageTk.PhotoImage(self.image)
        self.canvas=tk.Canvas(self.root,width=self.root_W,height=self.root_H)
        self.canvas.grid(row=0,column=0,columnspan=self.canvas_columnspan,rowspan=self.canvas_rowspan,sticky='nw')
        self.canvas.create_image(0,0,image=self.bg,anchor='nw')

    def run_cmd(self,cmd_i):
        os.system(cmd_i)
    def BndBox2Yolo(self,xmin,xmax,ymin,ymax,imgSize,classIndex):
        #print(imgSize)
        xcen=float((xmin+xmax))/2/imgSize[1]
        ycen=float((ymin+ymax))/2/imgSize[0]
        w=float((xmax-xmin))/imgSize[1]
        h=float((ymax-ymin))/imgSize[0]
        return classIndex,xcen,ycen,w,h
    def get_all_annos(self):
        self.counts=0
        all_annos_new=os.listdir(self.path_Annotations)
        all_annos_old=os.listdir(self.path_Yolo)
        annos_new=[w for w in tqdm(all_annos_new) if w.find('.xml')!=-1 and w[0]!='.']
        annos_old=[w for w in tqdm(all_annos_old) if w.find('.xml')!=-1 and w[0]!='.']
        if self.var_overwrite.get()!='Add':
            self.total_annos_list=[]
            for anno in tqdm(annos_new):
                if anno[0]!='.' and anno.find('.xml')!=-1:
                    img_i_name=anno.split('.xml')[0]
                    self.total_annos_list.append(os.path.join(self.path_Annotations,img_i_name+'.xml'))
        if self.var_overwrite.get()=='No':
            self.yolo_files=[os.path.join(self.path_Yolo,w) for w in os.listdir(self.path_Yolo) if w.find('df_YOLO.pkl')!=-1]
            self.yolo_ints=[int(os.path.basename(w).split('_df_YOLO.pkl')[0]) for w in self.yolo_files if w[0]!='.']
            self.max_ints=max(self.yolo_ints)
            self.counts=self.max_ints
        elif self.var_overwrite.get()=='Add':
            self.yolo_files=[os.path.join(self.path_Yolo,w) for w in os.listdir(self.path_Yolo) if w.find('df_YOLO.pkl')!=-1]
            if len(self.yolo_files)>0:
                self.yolo_ints=[int(os.path.basename(w).split('_df_YOLO.pkl')[0]) for w in self.yolo_files if w[0]!='.']
                self.max_ints=max(self.yolo_ints)
                self.counts=self.max_ints+self.increment
            annos_combined=list(set(annos_new))+list(set(annos_old)-set(annos_new))
            self.total_annos_list=[]
            for anno in tqdm(annos_combined):
                if anno[0]!='.' and anno.find('.xml')!=-1:
                    img_i_name=anno.split('.xml')[0]
                    if anno in os.listdir(self.path_Annotations):
                        self.total_annos_list.append(os.path.join(self.path_Annotations,img_i_name+'.xml'))
                    else:
                        pass
                        #self.total_annos_list.append(os.path.join(self.path_Yolo,img_i_name+'.xml'))

        self.total_annos=len(self.total_annos_list)
    def pad(self,input_i,pad_len=8):
        input_i=str(input_i)
        while len(input_i)!=pad_len:
            input_i='0'+input_i
        return input_i
    def copy_files(self,file_source,file_dest):
        shutil.copy(file_source,file_dest)
    def write_Yolo(self,xmin,xmax,ymin,ymax,imgSize,className,path_anno_dest_i):
        classIndex,xcen,ycen,w,h=self.BndBox2Yolo(xmin,xmax,ymin,ymax,imgSize,className)
        yolo_i=" ".join([str(yolo) for yolo in (int(classIndex),xcen,ycen,w,h)])
        f=open(path_anno_dest_i,'a')
        f.writelines(yolo_i+'\n')
        f.close()
    def open_popupwindow_labels(self):
        self.popupWindow_objnames()
        if os.path.exists(self.names_path):
            fo=open(self.names_path,'r')
            fo_read=fo.readlines()
            fo.close()
            self.found_names={w.replace('\n',''):j for j,w in enumerate(fo_read)}
            f=open(self.names_path,'w')
            f.writelines([k+'\n' for k, v in sorted(self.found_names.items(), key=lambda item: item[1])])
            f.close()
            shutil.copy(self.names_path,self.path_Yolo)
    def grep_annos_labels(self):
        '''FIXED OBJ.NAMES for testing'''
        self.get_all_annos()
        count_str=self.pad(self.counts)
        self.df_filename=os.path.join(self.path_Yolo,"{}_df_YOLO.pkl".format(count_str))
        self.df=pd.DataFrame(columns=['label_i','path_jpeg_dest_i','path_anno_i'])
        if self.var_overwrite.get()!='No':
            
            self.grep_result_file=os.path.join(os.path.dirname(self.path_Annotations),"grep_results_for_df.txt")
            cmd_i=f'grep -r "<name>" {self.path_Annotations} > {self.grep_result_file}'
            print('GREP STARTING')
            print(cmd_i)
            self.run_cmd(cmd_i)         
            f=open(self.grep_result_file,'r')
            f_read=f.readlines()
            f.close()
            print('GREP FINISHED')
            self.df_gr=pd.DataFrame(columns=['path','label','grep_line'])
            self.df_gr['grep_line']=f_read
            self.df_gr['label_i']=[w.split('<name>')[1].split('</name>')[0] for w in self.df_gr['grep_line']]
            self.df_gr['path_anno_i']=[os.path.abspath(w.split(':')[0].strip(' ')) for w in self.df_gr['grep_line']]

            self.df_gr['img_i_name']=[os.path.basename(w).split('.')[0] for w in self.df_gr['path_anno_i']]
            self.df_gr['path_jpeg_i']=[os.path.join(self.path_JPEGImages,w+'.jpg') for w in self.df_gr['img_i_name']]
            self.df_gr['path_anno_dest_i']=[os.path.join(self.path_Yolo,w+'.xml') for w in self.df_gr['img_i_name']]
            self.df_gr['path_jpeg_dest_i']=[os.path.join(self.path_Yolo,w+'.jpg') for w in self.df_gr['img_i_name']]
            self.found_names={name:i for i,name in enumerate(self.df_gr['label_i'].unique())}
            self.found_obj_names=os.path.join(os.path.dirname(self.names_path),'found_obj.names')
            f=open(self.found_obj_names,'w')
            f.writelines([k+'\n' for k, v in sorted(self.found_names.items(), key=lambda item: item[1])])
            f.close()
            if os.path.exists(self.names_path):
                from multiprocessing import Queue,Process
                f=open(self.names_path,'r')
                f_read=f.readlines()
                f.close()
                self.current_names={w.replace('\n',''):j for j,w in enumerate(f_read)}

            
                self.open_popupwindow_labels()
            self.df_gr=self.df_gr.drop_duplicates(subset=['path_anno_i']).reset_index().drop('index',axis=1)
            self.df_gr_filename=self.grep_result_file.replace('.txt','.csv')
            self.df_gr.to_csv(self.df_gr_filename,index=None)
            self.df=self.df_gr[['label_i','path_jpeg_dest_i','path_anno_i']]
            self.df_filename=os.path.join(self.path_Yolo,"{}_df_YOLO.pkl".format(count_str))
            self.df_filename_csv=self.df_filename.replace('.pkl','.csv')
            self.df.to_pickle(self.df_filename,protocol=2)
            self.df.to_csv(self.df_filename_csv,index=None)
        if os.path.exists(self.df_filename) and self.var_overwrite.get()=='No':
            print(self.df_filename)
            print('found')
            keep_columns=list(self.df.columns)
            df_pkls=os.listdir(self.path_Yolo)
            df_pkls=[os.path.join(self.path_Yolo,w) for w in df_pkls if w.find('_df_YOLO.pkl')!=-1]
            for pkl_i in tqdm(df_pkls):
                self.df_filename_csv=pkl_i.replace('.pkl','.csv')
                if os.path.exists(self.df_filename_csv)==False:
                    print('Creating: \n {}'.format(self.df_filename_csv))
                    try:
                        self.df_pkl=pd.read_pickle(pkl_i)
                        self.df_pkl.to_csv(self.df_filename_csv,index=None)
                    except:
                        self.df_pkl=pd.read_csv(self.df_filename_csv,index_col=None)
            self.df_filename_csv=self.df_filename.replace('.pkl','.csv')
            try:
                self.df=pd.read_pickle(self.df_filename)
                self.df.to_csv(self.df_filename_csv,index=None)
            except:
                self.df=pd.read_csv(self.df_filename_csv,index_col=None)
            drop_columns=[col for col in self.df.columns if col not in keep_columns]
            if len(drop_columns)>0:
                self.df.drop(drop_columns,axis=1,inplace=True)
            #self.found_names={w:i for i,w in enumerate(self.df['label_i'].unique())}
            if os.path.exists(self.names_path):
                f=open(self.names_path,'r')
                f_read=f.readlines()
                f.close()
                self.found_names={w.replace('\n',''):j for j,w in enumerate(f_read)}
            else:
                print('READING self.path_Yolo again')
                if os.path.exists(os.path.join(self.path_Yolo,'obj.names')):
                    print('I exist')
                    shutil.copy(os.path.join(self.path_Yolo,os.path.basename(self.names_path)),self.names_path)
                    f=open(self.names_path,'r')
                    f_read=f.readlines()
                    f.close()
                    self.found_names={w.replace('\n',''):j for j,w in enumerate(f_read)}
                else:
                    #input('ARE YOU SURE?')
                    df_pkls=os.listdir(self.path_Yolo)
                    df_pkls=[os.path.join(self.path_Yolo,w) for w in df_pkls if w.find('NEW_df_YOLO.pkl')!=-1]
                    for pkl_i in tqdm(df_pkls):
                        self.df_filename_csv=pkl_i.replace('.pkl','.csv')
                        try:
                            self.df_pkl=pd.read_pickle(pkl_i)
                            self.df_pkl.to_csv(self.df_filename_csv,index=None)
                        except:
                            self.df_pkl=pd.read_csv(self.df_filename_csv,index_col=None)

                        #self.df_pkl=pd.read_pickle(pkl_i)
                        names_possible=list(self.df_pkl['label_i'].unique())
                        for name in names_possible:
                            if name not in self.found_names.keys():
                                self.found_names[name]=len(self.found_names.keys())+0
                    f=open(self.names_path,'w')
                    f.writelines([k+'\n' for k, v in sorted(self.found_names.items(), key=lambda item: item[1])])
                    f.close()
                #break
        else:
            if os.path.exists(self.df_filename) and self.var_overwrite.get()=='Add' and self.var_overwite.get()=='No':
                keep_columns=list(self.df.columns)
                self.df_filename_csv=self.df_filename.replace('.pkl','.csv')
                try:
                    self.df=pd.read_pickle(self.df_filename)
                    self.df.to_csv(self.df_filename_csv,index=None)
                except:
                    self.df=pd.read_csv(self.df_filename_csv,index_col=None)
                #self.df=pd.read_pickle(self.df_filename)
                drop_columns=[col for col in self.df.columns if col not in keep_columns]
                if len(drop_columns)>0:
                    self.df.drop(drop_columns,axis=1,inplace=True)
                #self.found_names={w:i for i,w in enumerate(self.df['label_i'].unique())}
                if os.path.exists(self.names_path):
                    f=open(self.names_path,'r')
                    f_read=f.readlines()
                    f.close()
                    self.found_names={w.replace('\n',''):j for j,w in enumerate(f_read)}
                    if os.path.exists(os.path.join(self.path_Yolo,os.path.basename(self.names_path)))==False:
                        shutil.copy(self.names_path,self.path_Yolo)
                else:
                    if os.path.exists(os.path.join(self.path_Yolo,os.path.basename(self.names_path))):
                        shutil.copy(os.path.join(self.path_Yolo,os.path.basename(self.names_path)),self.names_path)
                        f=open(self.names_path,'r')
                        f_read=f.readlines()
                        f.close()
                        self.found_names={w.replace('\n',''):j for j,w in enumerate(f_read)}
                    else:
                        df_pkls=os.listdir(self.path_Yolo)
                        df_pkls=[os.path.join(self.path_Yolo,w) for w in df_pkls if w.find('_df_YOLO.pkl')!=-1]
                        for pkl_i in tqdm(df_pkls):
                            self.df_filename_csv=pkl_i.replace('.pkl','.csv')
                            try:
                                self.df_pkl=pd.read_pickle(pkl_i)
                                self.df_pkl.to_csv(self.df_filename_csv,index=None)
                            except:
                                self.df_pkl=pd.read_csv(self.df_filename_csv,index_col=None)
                            #self.df_pkl=pd.read_pickle(pkl_i)
                            names_possible=list(self.df_pkl['label_i'].unique())
                            for name in names_possible:
                                if name not in self.found_names.keys():
                                    self.found_names[name]=len(self.found_names.keys())+0
                        f=open(self.names_path,'w')
                        f.writelines([k+'\n' for k, v in sorted(self.found_names.items(), key=lambda item: item[1])])
                        f.close()
                i=len(self.df)
            else:

                if os.path.exists(os.path.join(os.path.dirname(self.path_Annotations),'Yolo_Objs'))==False:
                    self.path_Yolo=os.path.join(os.path.dirname(self.path_Annotations),'Yolo_Objs')
                    print(f'Generating new self.path_Yolo here instead: {self.path_Yolo}')
                    os.makedirs(self.path_Yolo)
                else:
                    print('Moving self.path_Yolo to BACKUP_OLD_MOVED: \n {}'.format(self.path_Yolo))
                    #os.system('rm -rf {}'.format(self.path_Yolo))
                    os.system('mv {} {}'.format(self.path_Yolo,self.path_Yolo+'_BACKUP_OLD_MOVED_at_'+str(time.time()).split('.')[0]))
                pass
            if os.path.exists(self.path_Yolo)==False:
                print('Creating self.path_Yolo: \n {}'.format(self.path_Yolo))
                os.makedirs(self.path_Yolo)
        if self.var_overwrite.get()!='No':
            print("STARTING PARALLEL PROCESSING FOR READING xml Annotations and WRITING txt Yolo Objs")
            time_start=time.time()
            
            import multiprocessing
            from multiprocessing import Process,Queue
            if multiprocessing.cpu_count()>1:
                NUM_PROCESS=multiprocessing.cpu_count()-1
            else:
                NUM_PROCESS=1
            i=0
            processes={}
            df_queues={}
            path_annos=[]

            expected_count=len(list(self.df['path_jpeg_dest_i'].unique()))
            print('expected_count=',expected_count)
            CHUNK_NUM=125
            time_start=time.time()
            #for j,(path_anno_i,path_jpeg_i) in tqdm(enumerate(zip(self.Annotations,self.JPEGs))):
            for j in range(0,len(self.df),CHUNK_NUM):
                #print('j=',j)
                #print('len(processes)=',len(processes))
                path_annos=self.df['path_anno_i'].loc[j:j+CHUNK_NUM]
                df_queues[len(processes)]=Queue()
                df_queues[len(processes)].put(self.df.copy())
                p=Process(target=self.read_multiple_XML,args=(path_annos,df_queues[len(processes)]))
                PROCESS_COUNT=len(processes)
                processes[PROCESS_COUNT]=p
                p.start()
                if (j%NUM_PROCESS==0 and j!=0 or j+CHUNK_NUM>expected_count):
                    print('\Started {} New Processes for Reading {} xml Annotations of {} & Writing txt Yolo Objs \n'.format(len(processes),min(j+CHUNK_NUM,len(self.df)),len(self.df)))
                    for (p_i,process_i),queue_i in zip(processes.items(),df_queues.values()):

                    #for queue_i in df_queues.values():
                        #print(f'Getting queue_i')
                        queue_i.get()
                        #print(f'Joining process loop {p_i}')
                        process_i.join()
                        #print('Joined')
                    df_queues={}
                    processes={}
                if j+CHUNK_NUM>expected_count:
                    print('FINISHED')
                    break
            try:
                for process_i in processes.values():
                    print('Joining remaining processes')
                    process_i.join()
                for queue_i in df_queues.values():
                    if i==0 and queue_i.empty()==False:
                        queue_i.get()
                        i+=1
                    elif queue_i.empty()==False:
                        pass
            except:
                pass
            time_finished=time.time()
            print(f'STARTED at {time_start}')
            print(f'ENDED at {time_finished}')
            print(f'TOTAL TIME took {round(time_finished-time_start,2)} seconds.')
    
    def read_multiple_XML(self,df_i,queue_i):
        for anno in tqdm(df_i):
            self.read_XML(os.path.basename(anno))
        queue_i.put('ready')

    def read_XML(self,anno):
        #print('LENGTH OF DF: ',len(self.df))
        #for anno in tqdm(os.listdir(self.path_Annotations)):
        label='None'
        if anno[0]!='.' and anno.find('.xml')!=-1:
            img_i_name=anno.split('.xml')[0]
            path_anno_i=os.path.join(self.path_Annotations,img_i_name+'.xml')
            path_jpeg_i=os.path.join(self.path_JPEGImages,img_i_name+'.jpg')
            path_anno_dest_xml_i=os.path.join(self.path_Yolo,img_i_name+'.xml')
            path_anno_dest_i=os.path.join(self.path_Yolo,img_i_name+'.txt')
            path_jpeg_dest_i=os.path.join(self.path_Yolo,img_i_name+'.jpg')
            f=open(path_anno_i,'r')
            f_read=f.readlines()
            f.close()
            f=open(path_anno_dest_i,'w')
            f.close()
            parser = etree.XMLParser(encoding=ENCODE_METHOD)
            xmltree = ElementTree.parse(path_anno_i, parser=parser).getroot()
            filename = xmltree.find('filename').text
            for size_iter in xmltree.findall('size'):
                width_i=int(size_iter.find('width').text)
                height_i=int(size_iter.find('height').text)
                depth_i=int(size_iter.find('depth').text)
                imgSize=tuple([height_i,width_i,depth_i])
            num_objs=[w for w in f_read if w.find('object')!=-1]
            num_objs=len(num_objs)
            if num_objs==0:
                print('No objects found')
            else:
                for object_iter in xmltree.findall('object'):
                    bndbox = object_iter.find("bndbox")
                    label = object_iter.find('name').text
                    if label not in self.found_names.keys():
                        self.found_names[label]=len(self.found_names.keys())+0
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                    self.write_Yolo(xmin,xmax,ymin,ymax,imgSize,self.found_names[label],path_anno_dest_i,)

                
            Thread(target=self.copy_files,args=(path_jpeg_i,path_jpeg_dest_i,)).start()
            Thread(target=self.copy_files,args=(path_anno_i,path_anno_dest_xml_i,)).start()
            return True


    def read_XML_VALID(self,anno):
        label='None'
        if anno[0]!='.' and anno.find('.xml')!=-1:
            img_i_name=os.path.basename(anno).split('.xml')[0]
            path_anno_i=os.path.join(self.path_predAnnotations,img_i_name+'.xml')
            path_jpeg_i=os.path.join(self.path_predJPEGImages,img_i_name+'.jpg')
            if os.path.exists(path_anno_i) and os.path.exists(path_jpeg_i):
                path_anno_dest_xml_i=os.path.join(self.path_predYolo,img_i_name+'.xml')
                path_anno_dest_i=os.path.join(self.path_predYolo,img_i_name+'.txt')
                path_jpeg_dest_i=os.path.join(self.path_predYolo,img_i_name+'.jpg')
                f=open(path_anno_i,'r')
                f_read=f.readlines()
                f.close()
                f=open(path_anno_dest_i,'w')
                f.close()

                parser = etree.XMLParser(encoding=ENCODE_METHOD)
                xmltree = ElementTree.parse(path_anno_i, parser=parser).getroot()
                filename = xmltree.find('filename').text
                for size_iter in xmltree.findall('size'):
                    width_i=int(size_iter.find('width').text)
                    height_i=int(size_iter.find('height').text)
                    depth_i=int(size_iter.find('depth').text)
                    imgSize=tuple([height_i,width_i,depth_i])
                num_objs=[w for w in f_read if w.find('object')!=-1]
                num_objs=len(num_objs)
                if num_objs==0:
                    print('No objects found')
                else:
                    for object_iter in xmltree.findall('object'):
                        bndbox = object_iter.find("bndbox")
                        label = object_iter.find('name').text
                        if label not in self.found_names.keys():
                            print('Did not see this label {}\n'.format(label))
                            if label.replace('augmentation_','') in self.found_names.keys():
                                print('Removing the augmentation_ from the label found')
                                label=label.replace('augmentation_','')
                            else:
                                print('ERROR FOUND WITH LABEL')
                                self.ERROR_FOUND=True
                            
                        xmin = int(float(bndbox.find('xmin').text))
                        ymin = int(float(bndbox.find('ymin').text))
                        xmax = int(float(bndbox.find('xmax').text))
                        ymax = int(float(bndbox.find('ymax').text))
                        if self.ERROR_FOUND==False:
                            self.write_Yolo(xmin,xmax,ymin,ymax,imgSize,self.found_names[label],path_anno_dest_i,)

                if os.path.exists(path_jpeg_dest_i)==False:
                    Thread(target=self.copy_files,args=(path_jpeg_i,path_jpeg_dest_i,)).start()
                Thread(target=self.copy_files,args=(path_anno_i,path_anno_dest_xml_i,)).start()

    def loop_train_load(self,train_cmd,madness='None'):
        count=0
        while True:

            if madness.find('train_yolov4_madness')!=-1:
                    self.epochs=int(self.epochs_VAR.get())
                    self.epochs+=1
                    self.epochs=str(self.epochs)
                    self.epochs_VAR.set(self.epochs)
                    self.train_yolov4_madness()
            if madness.find('train_yolov7_madness')!=-1:
                    self.epochs_yolov7=int(self.epochs_yolov7_VAR.get())
                    self.epochs_yolov7+=1
                    self.epochs_yolov7=str(self.epochs_yolov7)
                    self.epochs_yolov7_VAR.set(self.epochs_yolov7)
                    self.train_yolov7_madness()
            elif madness.find('train_yolov7_re_madness')!=-1:
                    self.epochs_yolov7_re=int(self.epochs_yolov7_re_VAR.get())
                    self.epochs_yolov7_re+=1
                    self.epochs_yolov7_re=str(self.epochs_yolov7_re)
                    self.epochs_yolov7_re_VAR.set(self.epochs_yolov7_re)
                    self.train_yolov7_re_madness()
            elif madness.find('train_yolov7_e6e_madness')!=-1:
                    self.epochs_yolov7_e6e=int(self.epochs_yolov7_e6e_VAR.get())
                    self.epochs_yolov7_e6e+=1
                    self.epochs_yolov7_e6e=str(self.epochs_yolov7_e6e)
                    self.epochs_yolov7_e6e_VAR.set(self.epochs_yolov7_e6e)
                    self.train_yolov7_e6e_madness()
            else:
                pass

            self.run_cmd(train_cmd)
            print('Finished Training Loop # {} at {}'.format(count,time.time()))
            count+=1
            self.var_overwrite.set('Yes')
            self.convert_PascalVOC_to_YOLO()
            self.split_objs()

    def popupWindow_TRAIN(self):
        try:
            self.top.destroy()
        except:
            pass
        self.top=tk.Toplevel(self.root)
        self.top.geometry( "{}x{}".format(int(self.root.winfo_screenwidth()*0.95//1.5),int(self.root.winfo_screenheight()*0.95//1.5)) )
        self.top.configure(background = 'black')
        self.b=Button(self.top,text='Close',command=self.cleanup,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.b.grid(row=0,column=0,sticky='se')
        self.test_yolov4_note=tk.Label(self.top,text='{}'.format(self.var_yolo_choice.get().replace('-','\n-')),bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.test_yolov4_note.grid(row=2,column=2,sticky='s')

        #TD Loop Loading Objects
        self.train_load_repeat=tk.StringVar()
        self.train_load_repeat.set('None')
        self.loop_load_check=tk.Checkbutton(self.top,text='{}'.format('Train/Load Repeat?'),variable=self.train_load_repeat,onvalue="On",offvalue='None',bg='white',fg='blue')
        self.loop_load_check.grid(row=1,column=2,sticky='s')

        #TD TRAIN yolov4
        self.train_yolo()

        self.test_yolov7_note=tk.Label(self.top,text='Yolov7\n-tiny',bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.test_yolov7_note.grid(row=2,column=3,sticky='se')
        #TD TRAIN yolov7 tiny
        self.train_yolov7()

        self.test_yolov7_re_note=tk.Label(self.top,text='Yolov7\n',bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.test_yolov7_re_note.grid(row=2,column=5,sticky='se')
        #TD TRAIN yolov7 re
        self.train_yolov7_re()

        self.test_yolov7_E6E_note=tk.Label(self.top,text='Yolov7\n-E6E',bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.test_yolov7_E6E_note.grid(row=2,column=4,sticky='se')
        #TD TRAIN yolov7 E6E
        self.train_yolov7_e6e()

    def popupWindow_objnames(self):
            try:
                self.top.destroy()
            except:
                pass
            self.top=tk.Toplevel(self.root)
            self.top.geometry( "{}x{}".format(int(self.root.winfo_screenwidth()*0.95//1.0),int(self.root.winfo_screenheight()*0.95//1.5)) )
            self.top.configure(background = 'black')
            self.b=Button(self.top,text='Close',command=self.cleanup,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
            self.b.grid(row=0,column=1,sticky='se')

            #current obj_names_path
            try:
                self.option_obj_names_path_var.get()
                if self.option_obj_names_path_var.get()=='None':
                 try:
                    self.option_obj_names_path_var.set(self.names_path)
                 except:
                    self.option_obj_names_path_var.set('None')               
            except:
                self.option_obj_names_path_var=tk.StringVar()
                try:
                    self.option_obj_names_path_var.set(self.names_path)
                except:
                    self.option_obj_names_path_var.set('None')
            self.option_obj_names_path_button=Button(self.top,image=self.icon_folder,command=self.select_file_optionobjnames,bg=self.root_bg,fg=self.root_fg)
            self.option_obj_names_path_button.grid(row=1,column=1,sticky='se')   
            self.option_obj_names_path_label=Button(self.top,textvariable=self.option_obj_names_path_var,command=partial(self.open_something,self.option_obj_names_path_var),bg=self.root_bg,fg=self.root_fg)
            self.option_obj_names_path_label.grid(row=1,column=2,sticky='sw')  
            self.option_obj_names_note=tk.Label(self.top,text='path to current obj.names',bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
            self.option_obj_names_note.grid(row=1,column=0,sticky='se') 


            #found obj_names_path
            self.FOUND_NAMES=False
            try:
                self.found_obj_names_var.get()
                if self.found_obj_names_var.get()==self.option_obj_names_path_var.get():
                    self.FOUND_NAMES=False
                    try:
                        self.found_obj_names_var.set(self.found_obj_names)
                    except:
                        fo=open(self.names_path,'r')
                        fo_read=fo.readlines()
                        fo.close()
                        self.found_names={w.replace('\n',''):j for j,w in enumerate(fo_read)}
                        self.found_obj_names=os.path.join(os.path.dirname(self.names_path),'found_obj.names')
                        f=open(self.found_obj_names,'w')
                        f.writelines([k+'\n' for k, v in sorted(self.found_names.items(), key=lambda item: item[1])])
                        f.close()
                else:
                    self.found_obj_names_var.set(self.found_obj_names)
                    self.FOUND_NAMES=True
           
            except:
                self.found_obj_names_var=tk.StringVar()
                try:
                    self.found_obj_names_var.set(self.found_obj_names)
                except:
                    fo=open(self.names_path,'r')
                    fo_read=fo.readlines()
                    fo.close()
                    self.found_names={w.replace('\n',''):j for j,w in enumerate(fo_read)}
                    self.found_obj_names=os.path.join(os.path.dirname(self.names_path),'found_obj.names')
                    f=open(self.found_obj_names,'w')
                    f.writelines([k+'\n' for k, v in sorted(self.found_names.items(), key=lambda item: item[1])])
                    f.close()
                    self.found_obj_names_var.set(self.found_obj_names)
            
            
            if self.FOUND_NAMES:
                self.TEXT_FOUND_OBJ_NAMES='path to newly found obj.names'
                self.TEXT_FOUND_OPTION='found obj.names'
            else:
                self.TEXT_FOUND_OBJ_NAMES='path to a custom obj.names you can alter and select'
                self.TEXT_FOUND_OPTION='changed obj.names'
            self.found_obj_names_path_label=Button(self.top,textvariable=self.found_obj_names_var,command=partial(self.open_something,self.found_obj_names_var),bg=self.root_bg,fg=self.root_fg)
            self.found_obj_names_path_label.grid(row=3,column=2,sticky='sw')  
            self.found_obj_names_note=tk.Label(self.top,text=self.TEXT_FOUND_OBJ_NAMES,bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
            self.found_obj_names_note.grid(row=3,column=0,sticky='se') 


            #path_obj_namses used if starting over
            self.style3=ttk.Style()
            self.style3.configure('Normal.TRadiobutton',
                                background='green',
                                foreground='black')
            self.var_choice=tk.StringVar()
            self.var_choice.set(self.TEXT_FOUND_OPTION)
            self.obj_choice_current=ttk.Radiobutton(self.top,text='current obj.names',style='Normal.TRadiobutton',variable=self.var_choice,value='Current')
            self.obj_choice_current.grid(row=2,column=2,stick='sw')

            self.obj_choice_found=ttk.Radiobutton(self.top,text=self.TEXT_FOUND_OPTION,style='Normal.TRadiobutton',variable=self.var_choice,value=self.TEXT_FOUND_OPTION)
            self.obj_choice_found.grid(row=4,column=2,stick='sw')

    
            self.submit_choice_button=Button(self.top,text='Submit Choice for obj.names?',command=self.SUBMIT_objnames,bg='green', fg=DEFAULT_SETTINGS.root_bg)
            self.submit_choice_button.grid(row=0,column=2,sticky='sw')
            self.root.wait_window(self.top)
    
    def SUBMIT_objnames(self):
        if self.var_choice.get()==self.TEXT_FOUND_OPTION:
            print(f'Using {self.TEXT_FOUND_OPTION}')
            os.remove(self.option_obj_names_path_var.get())
            shutil.move(self.found_obj_names_var.get(),self.option_obj_names_path_var.get())
        else:
            print('Using Current')
            if os.path.exists(self.found_obj_names_var.get()):
                os.remove(self.found_obj_names_var.get())
        self.top.destroy()


    def popupWindow_mAP(self):
        try:
            self.top.destroy()
        except:
            pass
        self.top=tk.Toplevel(self.root)
        self.top.geometry( "{}x{}".format(int(self.root.winfo_screenwidth()*0.95//1.0),int(self.root.winfo_screenheight()*0.95//1.5)) )
        self.top.configure(background = 'black')
        self.b=Button(self.top,text='Close',command=self.cleanup,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.b.grid(row=0,column=1,sticky='se')

        #path_Anno_Pred
        try:
            self.path_Anno_Pred_var.get()
            if self.path_Anno_Pred_var.get()=='None':
                try:
                    self.path_Anno_Pred_var.set(self.YOLO_MODEL_PATH)
                except:
                    self.path_Anno_Pred_var.set('None')
        except:
            self.path_Anno_Pred_var=tk.StringVar()
            try:
                self.path_Anno_Pred_var.set(self.YOLO_MODEL_PATH)
            except:
                self.path_Anno_Pred_var.set('None')
        self.open_path_Anno_Pred_button=Button(self.top,image=self.icon_folder,command=partial(self.select_folder_mAP,self.path_Anno_Pred_var.get(),'path to Annotation Predictions',self.path_Anno_Pred_var),bg=self.root_bg,fg=self.root_fg)
        self.open_path_Anno_Pred_button.grid(row=3,column=1,sticky='se')
        self.open_path_Anno_Pred_label=Button(self.top,textvariable=self.path_Anno_Pred_var,command=partial(self.open_something,self.path_Anno_Pred_var),bg=self.root_bg,fg=self.root_fg)
        self.open_path_Anno_Pred_label.grid(row=3,column=2,sticky='sw')    
        self.path_Anno_Pred_note=tk.Label(self.top,text='path to Annotation Predictions',bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.path_Anno_Pred_note.grid(row=3,column=0,sticky='se')

        #path_Anno_GT
        try:
            self.path_Anno_GT_var.get()
        except:
            self.path_Anno_GT_var=tk.StringVar()
            self.path_Anno_GT_var.set(self.path_Annotations)
        self.open_path_Anno_GT_button=Button(self.top,image=self.icon_folder,command=partial(self.select_folder_mAP,self.path_Anno_GT_var.get(),'path to Annotation Ground Truth',self.path_Anno_GT_var),bg=self.root_bg,fg=self.root_fg)
        self.open_path_Anno_GT_button.grid(row=5,column=1,sticky='se')   
        self.open_path_Anno_GT_label=Button(self.top,textvariable=self.path_Anno_GT_var,command=partial(self.open_something,self.path_Anno_GT_var),bg=self.root_bg,fg=self.root_fg)
        self.open_path_Anno_GT_label.grid(row=5,column=2,sticky='sw')       
        self.path_Anno_GT_note=tk.Label(self.top,text='path to Annotation Ground Truth',bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.path_Anno_GT_note.grid(row=5,column=0,sticky='se')

        #path_JPEGS_GT
        try:
            self.path_JPEGS_GT_var.get()
        except:
            self.path_JPEGS_GT_var=tk.StringVar()
            self.path_JPEGS_GT_var.set(self.path_JPEGImages)
        self.open_path_JPEGS_GT_button=Button(self.top,image=self.icon_folder,command=partial(self.select_folder_mAP,self.path_JPEGS_GT_var.get(),'path to JPEGImages Ground Truth',self.path_JPEGS_GT_var),bg=self.root_bg,fg=self.root_fg)
        self.open_path_JPEGS_GT_button.grid(row=7,column=1,sticky='se')    
        self.open_path_JPEGS_GT_label=Button(self.top,textvariable=self.path_JPEGS_GT_var,command=partial(self.open_something,self.path_JPEGS_GT_var),bg=self.root_bg,fg=self.root_fg)
        self.open_path_JPEGS_GT_label.grid(row=7,column=2,sticky='sw')  
        self.path_JPEGS_GT_note=tk.Label(self.top,text='path to JPEGImages Ground Truth',bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.path_JPEGS_GT_note.grid(row=7,column=0,sticky='se') 

        #obj_names_path
        try:
            self.obj_names_path_var.get()
            if self.obj_names_path_var.get()=='None':
             try:
                self.obj_names_path_var.set(self.names_path)
             except:
                self.obj_names_path_var.set('None')               
        except:
            self.obj_names_path_var=tk.StringVar()
            try:
                self.obj_names_path_var.set(self.names_path)
            except:
                self.obj_names_path_var.set('None')
        self.obj_names_path_button=Button(self.top,image=self.icon_folder,command=self.select_file_objnames,bg=self.root_bg,fg=self.root_fg)
        self.obj_names_path_button.grid(row=9,column=1,sticky='se')   
        self.obj_names_path_label=Button(self.top,textvariable=self.obj_names_path_var,command=partial(self.open_something,self.obj_names_path_var),bg=self.root_bg,fg=self.root_fg)
        self.obj_names_path_label.grid(row=9,column=2,sticky='sw')  
        self.obj_names_note=tk.Label(self.top,text='path to obj.names',bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.obj_names_note.grid(row=9,column=0,sticky='se') 

        #valid_list
        try:
            self.valid_list_var.get()
            if self.valid_list_var.get()=='None':
                try:
                    self.valid_list_var.set(self.valid_list_path)
                except:
                    self.valid_list_var.set('None')

        except:
            self.valid_list_var=tk.StringVar()
            try:
                self.valid_list_var.set(self.valid_list_path)
            except:
                self.valid_list_var.set('None')
        self.valid_list_button=Button(self.top,image=self.icon_single_file,command=self.select_file_validlist,bg=self.root_bg,fg=self.root_fg)
        self.valid_list_button.grid(row=11,column=1,sticky='se')   
        self.valid_list_label=Button(self.top,textvariable=self.valid_list_var,command=partial(self.open_something,self.valid_list_var),bg=self.root_bg,fg=self.root_fg)
        self.valid_list_label.grid(row=11,column=2,sticky='sw')  
        self.valid_list_note=tk.Label(self.top,text='path to valid_list.txt/img_list.txt etc',bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.valid_list_note.grid(row=11,column=0,sticky='se') 

        self.create_combine_jpeg_pred_vars=tk.IntVar()
        self.create_combine_jpeg_pred_vars.set(0)
        self.create_combine_jpeg_pred_buttons=ttk.Checkbutton(self.top, style='Normal.TCheckbutton',text="Create JPEGImages/Annotations of Combined Ground Truth & Predictions Sorted by Accuracy",variable=self.create_combine_jpeg_pred_vars,onvalue=1, offvalue=0)
        self.create_combine_jpeg_pred_buttons.grid(row=12,column=2,sticky='sw')

        self.submit_mAP_button=Button(self.top,text='Compute mAP metrics',command=self.SUBMIT_mAP,bg='green', fg=DEFAULT_SETTINGS.root_bg)
        self.submit_mAP_button.grid(row=0,column=2,sticky='sw')

    def open_something(self,var_to_open):
        cmd_i=open_cmd+" "+var_to_open.get()
        print('OPENING')
        print(cmd_i)
        self.run_cmd(cmd_i)

    def select_folder_mAP(self,folder_i,title_i,var_i=None):
            filetypes=(('All files','*.*'))
            if var_i:
                folder_i=var_i.get() 
                initialdir=folder_i 
                           
                if os.path.exists(folder_i):
                    self.foldername=fd.askdirectory(title=title_i,
                                                initialdir=initialdir)
                else:
                    self.foldername=fd.askdirectory(title=title_i)
                if self.foldername=='' or len(self.foldername)==0:
                    showinfo(title='NOT FOUND! Using previous path',
                            message=self.foldername)
                elif self.foldername!='' and len(self.foldername)!=0:
                    showinfo(title='Selected Folder',
                        message=self.foldername)
                    folder_i=self.foldername
                    var_i.set(folder_i)


    def SUBMIT_mAP(self):
        cmd_i=""
        path_Anno_Pred=self.path_Anno_Pred_var.get()    
        path_Anno_GT=self.path_Anno_GT_var.get()
        path_JPEGS_GT=self.path_JPEGS_GT_var.get()
        obj_names_path=self.obj_names_path_var.get()
        valid_list=self.valid_list_var.get()
        path_compute_mAP=os.path.abspath('resources/compute_mAP.py')
        create_combine_jpeg_pred_vars=self.create_combine_jpeg_pred_vars.get()
        
        if os.path.exists(path_Anno_Pred):
            bash_mAP=os.path.join(os.path.dirname(path_Anno_Pred),'bash_mAP.sh')
            f=open(bash_mAP,'w')
            f.writelines(f'path_Anno_Pred={path_Anno_Pred}\n')
            f.writelines(f'path_Anno_GT={path_Anno_GT}\n')
            f.writelines(f'path_JPEGS_GT={path_JPEGS_GT}\n')
            f.writelines(f'obj_names_path={obj_names_path}\n')
            f.writelines(f'valid_list={valid_list}\n')
            f.writelines(f'path_compute_mAP={path_compute_mAP}\n')
            if create_combine_jpeg_pred_vars==0:
                f.writelines(f'python3 $path_compute_mAP --valid_list=$valid_list --path_Anno_Pred=$path_Anno_Pred --path_JPEGS_GT=$path_JPEGS_GT --path_Anno_GT=$path_Anno_GT --obj_names_path=$obj_names_path --show_results\n')
            else:
                f.writelines(f'python3 $path_compute_mAP --valid_list=$valid_list --path_Anno_Pred=$path_Anno_Pred --path_JPEGS_GT=$path_JPEGS_GT --path_Anno_GT=$path_Anno_GT --obj_names_path=$obj_names_path --show_results --create_combine_gt_pred\n')
            f.close()
            print(f'FINISHED WRITING bash_mAP at: {bash_mAP}')
            cmd_i=f'bash {bash_mAP}'
            print(f'STARTING bash_mAP')
            self.run_cmd(cmd_i)
            print('FINISHED bash_mAP')
        else:
            print(f'path_Anno_Pred does not exist.  Curent location is, \n{path_Anno_Pred}\n')
        


        
    
    def popupWindow_showtable(self):
        try:
            self.top.destroy()
        except:
            pass
        self.top=tk.Toplevel(self.root)
        self.top.geometry( "{}x{}".format(int(self.root.winfo_screenwidth()*0.95//1.5),int(self.root.winfo_screenheight()*0.95//1.5)) )
        self.top.configure(background = 'black')
        self.b=Button(self.top,text='Close',command=self.cleanup_show,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.b.pack()
        self.show_table()
    def cleanup_show(self):
        self.app.table.saveAs(self.df_filename_csv)
        print('self.df_filename')
        print(self.df_filename_csv)
        df_i=pd.read_csv(self.df_filename_csv,index_col=None)
        columns=df_i.columns
        drop_columns=[w for w in columns if w.find('Unnamed')!=-1]
        df_i.drop(drop_columns,axis=1,inplace=True)
        df_i.to_pickle(self.df_filename,protocol=2)
        df_i.to_csv(self.df_filename_csv,index=None)
        self.top.destroy()
        
    def popupWindow_TEST(self):
        try:
            self.top.destroy()
        except:
            pass
        self.top=tk.Toplevel(self.root)
        self.top.geometry( "{}x{}".format(int(self.root.winfo_screenwidth()*0.95//1.5),int(self.root.winfo_screenheight()*0.95//1.5)) )
        self.top.configure(background = 'black')
        self.b=Button(self.top,text='Close',command=self.cleanup,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.b.grid(row=0,column=0,sticky='se')
        self.test_yolov4_note=tk.Label(self.top,text='{}'.format(self.var_yolo_choice.get().replace('-','\n-')),bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.test_yolov4_note.grid(row=2,column=2,sticky='se')
        self.test_yolo()
        self.test_yolo_predict()
        self.test_yolo_predict_mAP()
        self.test_yolodnn()
        self.test_yolodnn_rtmp()
        self.test_yolo_mp4()
        self.test_yolodnn_labelimg()
        self.test_yolodnn_rtsp()

        self.test_yolov7_note=tk.Label(self.top,text='Yolov7\n-tiny',bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.test_yolov7_note.grid(row=2,column=3,sticky='se')
        self.test_yolov7_mp4()
        self.test_yolov7_webcam()
        self.test_yolov7_webcam_RTMP()
        self.test_yolov7_mAP()
        self.test_yolov7_rtsp()
        self.test_yolov7_labelimg()

        self.test_yolov7_re_note=tk.Label(self.top,text='Yolov7\n',bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.test_yolov7_re_note.grid(row=2,column=5,sticky='se')
        self.test_yolov7_mp4_re()
        self.test_yolov7_webcam_re()
        self.test_yolov7_webcam_re_RTMP()
        self.test_yolov7_mAP_re()
        self.test_yolov7_rtsp_re()
        self.test_yolov7_labelimg_re()

        self.test_yolov7_E6E_note=tk.Label(self.top,text='Yolov7\n-E6E',bg=self.root_fg,fg=self.root_bg,font=("Arial", 9))
        self.test_yolov7_E6E_note.grid(row=2,column=4,sticky='se')
        self.test_yolov7_mp4_e6e()
        self.test_yolov7_webcam_e6e()
        self.test_yolov7_webcam_e6e_RTMP()
        self.test_yolov7_mAP_e6e()
        self.test_yolov7_rtsp_e6e()
        self.test_yolov7_labelimg_e6e()

    def cleanup(self):
        self.ITERATION_NUM_VAR.get()
        self.epochs_yolov7_VAR.get()
        self.epochs_yolov7_re_VAR.get()      
        self.epochs_yolov7_e6e_VAR.get()
        self.top.destroy()
        try:
            sleep_time_i=self.sleep_time_chips_VAR.get()
            float(sleep_time_i) #prove it is able to be floating point
        except:
            print('Invalid sleep time of {}\n should be int/float'.format(sleep_time_i))
            self.sleep_time_chips_VAR.set('30')
    def cleanup_RTSP(self):
        self.USE_RTSP_CLIENT_VAR.set(self.USE_RTSP_CLIENT_VAR.get())
        self.SOCKET_PREFIX=self.SOCKET_PREFIX_VAR.get()
        self.top.destroy()

    def convert_PascalVOC_to_YOLO_VALID(self):
        self.ERROR_FOUND=False
        if os.path.exists(str(self.path_predJPEGImages)):
            self.path_predAnnotations=self.path_predJPEGImages.replace('JPEGImages','Annotations')
            if os.path.exists(self.path_predAnnotations):
                print('FOUND PREDICTION ANNOTATIONS DIRECTORY')
                self.path_predYolo=os.path.join(os.path.dirname(self.path_predAnnotations),'Yolo_Objs')
                if os.path.exists(self.path_predYolo)==False:
                    os.makedirs(self.path_predYolo)
                #self.Yolo_pred_stuff=os.listdir(self.path_Yolo_pred)
                self.predAnnos=os.listdir(self.path_predAnnotations)
                self.predAnnos=[os.path.join(self.path_predAnnotations,w) for w in self.predAnnos]
                self.predJPEGs=os.listdir(self.path_predJPEGImages)
                self.predJPEGs=[os.path.join(self.path_predJPEGImages,w) for w in self.predJPEGs]
                #tmp=[shutil.copy(os.path.join(self.path_predAnnotations,w),self.path_predYolo) for w in self.predAnnos]
                #tmp=[shutil.copy(os.path.join(self.path_predJPEGImages,w),self.path_predYolo) for w in self.predJPEGs]
                #self.found_names
                for jpg_i in tqdm(self.predJPEGs):
                    shutil.copy(jpg_i,self.path_predYolo)
                for anno in tqdm(self.predAnnos):
                    self.read_XML_VALID(anno)
                    if self.ERROR_FOUND==True:
                        break

                if self.ERROR_FOUND==False:                
                    self.predYolo=os.listdir(self.path_predYolo)

                    self.VAL_LIST=[os.path.join(self.path_predYolo,w) for w in self.predYolo if w.find('.jpg')!=-1]
                    f=open(self.valid_list_path,'w')
                    done=[f.writelines(line+'\n') for line in self.VAL_LIST]
                    f.close()
                    self.img_list_path=self.valid_list_path
                else:
                    print('ERROR was found with given label in annotation, you can only predict off the images without metrics')
            else:
                print('No Annotations directory found with JPEGImages directory')
        else:
            print('ERROR, JPEGImages directory does NOT exist')
    def convert_PascalVOC_to_YOLO(self):
        self.grep_annos_labels()
        if len(self.df)>0:
            self.df=self.df.drop_duplicates(keep='last').reset_index().drop('index',axis=1)
            self.df.to_pickle(self.df_filename,protocol=2)  
            self.df_filename_csv=self.df_filename.replace('.pkl','.csv')
            self.df.to_csv(self.df_filename_csv,index=None)
        if self.num_classes!=len(self.found_names.items()):
            self.num_classes=len(self.found_names.items())
            self.num_classes_VAR.set(self.num_classes)
            self.existing_weight_dirs=os.listdir(self.base_path)
            self.existing_weight_dirs=[os.path.join(self.base_path,w) for w in self.existing_weight_dirs]
            self.existing_weight_dirs=[w for w in self.existing_weight_dirs if os.path.isdir(w) and w.find('detections')==-1]
            if len(self.existing_weight_dirs)>0:
                moved_list=[]
                tmp_backup=os.path.join(os.path.dirname(os.getcwd()),'tmp_'+str(time.time()).split('.')[0])
                if not(os.path.exists(tmp_backup)):
                    os.makedirs(tmp_backup)
                for item in tqdm(self.existing_weight_dirs):
                    moved_list.append(os.path.join(tmp_backup,os.path.basename(item)))
                    shutil.move(item,os.path.join(tmp_backup,os.path.basename(item)))
                os.system('rm -rf {}'.format(self.base_path))
                for item in tqdm(moved_list):   
                    shutil.move(item,os.path.join(self.base_path,os.path.basename(item)))
                os.system('rm -rf {}'.format(tmp_backup))
            else:
                os.system('rm -rf "{}"'.format(self.base_path))
            self.generate_cfg()
        if os.path.exists(self.names_path):
            fo=open(self.names_path,'r')
            fo_read=fo.readlines()
            fo.close()
            self.found_names={w.replace('\n',''):j for j,w in enumerate(fo_read)}
        f=open(self.names_path,'w')
        f.writelines([k+'\n' for k, v in sorted(self.found_names.items(), key=lambda item: item[1])])
        f.close()
        shutil.copy(self.names_path,self.path_Yolo)

        
        
        self.TRAIN_SPLIT_VAR=tk.StringVar()
        self.TRAIN_SPLIT_VAR.set(self.TRAIN_SPLIT)
        self.TRAIN_SPLIT_entry=tk.Entry(self.root,textvariable=self.TRAIN_SPLIT_VAR)
        self.TRAIN_SPLIT_entry.grid(row=4,column=2,sticky='sw')
        self.TRAIN_SPLIT_label=tk.Label(self.root,text='TRAIN SPLIT',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.TRAIN_SPLIT_label.grid(row=5,column=2,sticky='nw')

        self.split_yolo_objs_button=Button(self.root,image=self.icon_divide,command=self.split_objs,bg=self.root_bg,fg=self.root_fg)
        self.split_yolo_objs_button.grid(row=4,column=1,sticky='se')
        self.split_yolo_objs_button_note=tk.Label(self.root,text='2.b \n Split Train/Test Yolo \n Objects (.jpg/.txt)',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.split_yolo_objs_button_note.grid(row=5,column=1,sticky='ne')
       

    def split_objs(self):
        self.TRAIN_SPLIT=int(self.TRAIN_SPLIT_VAR.get())
        if self.TRAIN_SPLIT>99:
            self.TRAIN_SPLIT=99
            self.TRAIN_SPLIT_VAR.set(self.TRAIN_SPLIT)
        elif self.TRAIN_SPLIT<0:
            self.TRAIN_SPLIT=1
            self.TRAIN_SPLIT_VAR.set(self.TRAIN_SPLIT)           
        self.TRAIN_LIST=[]
        self.VAL_LIST=[]
        f=open(self.names_path,'r')
        f_read=f.readlines()
        f.close()
        self.unique_names=[w.replace('\n','').strip() for w in f_read]
        self.yolo_files=[os.path.join(self.path_Yolo,w) for w in os.listdir(self.path_Yolo) if w.find('df_YOLO.pkl')!=-1 and w[0]!='.']
        pprint(self.yolo_files)
        self.label_counter={}
        for unique_label in tqdm(self.unique_names):
            unique_label_count_train=0
            unique_label_count_val=0
            for yolo_file_i in tqdm(self.yolo_files):
                try:
                    self.df=pd.read_pickle(yolo_file_i)
                    self.df.to_csv(yolo_file_i.replace('.pkl','.csv'),index=None)
                except:
                    self.df=pd.read_csv(yolo_file_i.replace('.pkl','.csv'),index_col=None)
                self.df=self.df.reset_index().drop('index',axis=1)
                #pprint(self.df)
                self.df_i=self.df[self.df['label_i']==unique_label].copy()
                self.df_i=self.df_i.drop_duplicates().reset_index().drop('index',axis=1)
                if len(self.df_i)>0:
                    self.df_i=self.df_i.sample(frac=1,random_state=42) #shuffle all rows 
                self.df_i=self.df_i.sort_values(by='path_jpeg_dest_i')
                total_list_i=list(self.df_i['path_jpeg_dest_i'])
                train_list_i=total_list_i[:int(self.TRAIN_SPLIT*len(self.df_i)/100.)]
                val_list_i=total_list_i[int(self.TRAIN_SPLIT*len(self.df_i)/100.):]
                self.TRAIN_LIST+=train_list_i
                self.VAL_LIST+=val_list_i
                unique_label_count_train+=len(train_list_i)
                unique_label_count_val+=len(val_list_i)
            self.label_counter[unique_label]=[unique_label_count_train,unique_label_count_val]
        self.VAL_LIST=list(pd.DataFrame(self.VAL_LIST)[0].drop_duplicates())
        self.TRAIN_LIST=list(pd.DataFrame(self.TRAIN_LIST)[0].drop_duplicates())
        print(len(self.VAL_LIST)+len(self.TRAIN_LIST))
        #self.VAL_LIST=set(self.VAL_LIST)-set(self.TRAIN_LIST)
        if self.TRAIN_SPLIT<50:
            self.TRAIN_LIST=set(self.TRAIN_LIST)-set(self.VAL_LIST)
        else:
            self.TRAIN_LIST=set(self.TRAIN_LIST)-set(self.VAL_LIST)
            #self.VAL_LIST=set(self.VAL_LIST)-set(self.TRAIN_LIST)

        print('\nIMAGE COUNTS')
        print('len(self.VAL_LIST) =',len(self.VAL_LIST))
        print('len(self.TRAIN_LIST) =',len(self.TRAIN_LIST))
        print('len(self.VAL_LIST)+len(self.TRAIN_LIST) = ',len(self.VAL_LIST)+len(self.TRAIN_LIST))
        #print('len(self.df["path_jpeg_dest_i"].unique()) =',len(self.df["path_jpeg_dest_i"].unique()))
        f=open(self.train_list_path,'w')
        done=[f.writelines(line+'\n') for line in self.TRAIN_LIST]
        f.close()
        f=open(self.valid_list_path,'w')
        done=[f.writelines(line+'\n') for line in self.VAL_LIST]
        f.close()
        print('\nOBJECT COUNTS')
        for label,(count_train,count_val) in self.label_counter.items():
            print("LABEL={}; TRAIN={}; VALID={}".format(label,count_train,count_val))

        
        self.THRESH_VAR=tk.StringVar()
        self.THRESH_VAR.set(self.THRESH)
        self.THRESH_entry=tk.Entry(self.root,textvariable=self.THRESH_VAR)
        self.THRESH_entry.grid(row=8,column=2,sticky='sw')
        self.THRESH_label=tk.Label(self.root,text='Threshold',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.THRESH_label.grid(row=9,column=2,sticky='nw')
        self.IOU_THRESH_VAR=tk.StringVar()
        self.IOU_THRESH_VAR.set(self.IOU_THRESH)
        self.IOU_THRESH_entry=tk.Entry(self.root,textvariable=self.IOU_THRESH_VAR)
        self.IOU_THRESH_entry.grid(row=8,column=3,sticky='sw')
        self.IOU_THRESH_label=tk.Label(self.root,text='IOU Threshold',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.IOU_THRESH_label.grid(row=9,column=3,sticky='nw')
        self.POINTS_VAR=tk.StringVar()
        self.POINTS_LIST=['0: Custom','11: PascalVOC 2007','101: MS COCO']
        self.POINTS_VAR.set('0: Custom')
        self.POINTS=self.POINTS_VAR.get().split(':')[0]
        self.POINTS_dropdown=tk.OptionMenu(self.root,self.POINTS_VAR,*self.POINTS_LIST)
        self.POINTS_dropdown.grid(row=8,column=4,sticky='sw')
        self.POINTS_label=tk.Label(self.root,text='POINTS',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.POINTS_label.grid(row=9,column=4,sticky='nw')
        self.create_yolo_scripts_buttons()
        self.load_yolo_scripts_buttons()
        self.change_obj_names_buttons()

    def change_obj_names_buttons(self):
        self.change_obj_names_button=Button(self.root,text='Change obj.names',command=self.open_popupwindow_labels,bg=self.root_fg,fg=self.root_bg)
        self.change_obj_names_button.grid(row=0,column=9,sticky='sw')

    def load_yolo_scripts_buttons(self):
        self.load_yolo_files_button=Button(self.root,image=self.icon_scripts,command=self.remaining_buttons,bg=self.root_bg,fg=self.root_fg)
        self.load_yolo_files_button.grid(row=6,column=1,sticky='se')
        self.load_yolo_files_button_note=tk.Label(self.root,text='3.a \n Load Yolo \n Scripts (.sh)',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.load_yolo_files_button_note.grid(row=7,column=1,sticky='ne')
        self.create_darknet_buttons()


    def create_yolo_scripts_buttons(self):
        self.create_yolo_files_button=Button(self.root,image=self.icon_scripts,command=self.create_yolo_files,bg=self.root_bg,fg=self.root_fg)
        self.create_yolo_files_button.grid(row=8,column=1,sticky='se')
        self.create_yolo_files_button_note=tk.Label(self.root,text='3.b \n Create Yolo \n Scripts (.sh)',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.create_yolo_files_button_note.grid(row=9,column=1,sticky='ne')
        self.create_darknet_buttons()
    def create_darknet_buttons(self):
        if self.darknet_selected==True:
            self.open_darknet_label.destroy()
            del self.open_darknet_label

        self.open_darknet_label_var=tk.StringVar()
        self.open_darknet_label_var.set(self.darknet_path)
        self.open_darknet_button=Button(self.root,image=self.icon_folder,command=partial(self.select_folder,self.darknet_path,'path to Darknet',self.open_darknet_label_var),bg=self.root_bg,fg=self.root_fg)
        self.open_darknet_button.grid(row=17,column=4,sticky='se')
        self.open_darknet_note=tk.Label(self.root,text="darknet_path dir",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_darknet_note.grid(row=18,column=4,sticky='ne')
        cmd_i=open_cmd+" '{}'".format(self.open_darknet_label_var.get())
        self.open_darknet_label=Button(self.root,textvariable=self.open_darknet_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.open_darknet_label.grid(row=17,column=5,columnspan=50,sticky='sw')


        if self.backup_models_selected==True:
            self.open_backup_note.destroy()
            self.open_backup_button.destroy()
            del self.open_backup_note

        self.open_backup_models_label_var=tk.StringVar()
        self.open_backup_models_label_var.set(self.backup_path)
        cmd_i=open_cmd+" '{}'".format(self.open_backup_models_label_var.get())
        self.open_backup_button=Button(self.root,image=self.icon_open,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
        self.open_backup_button.grid(row=3,column=5,sticky='se')
        self.open_backup_note=tk.Label(self.root,text=os.path.basename(self.open_backup_models_label_var.get()),bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_backup_note.grid(row=4,column=5,sticky='ne')
        # cmd_i=open_cmd+" '{}'".format(self.open_backup_models_label_var.get())
        # self.open_backup_models_label=Button(self.root,textvariable=self.open_backup_models_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        # self.open_backup_models_label.grid(row=19,column=5,columnspan=50,sticky='sw')
        self.backup_models_selected=True




            

if __name__=='__main__':
    while return_to_main==True:
        return_to_main=False
        root_tk=tk.Tk()
        main_yolo=main_entry(root_tk)
        main_yolo.root.mainloop()
        del main_yolo
        if PROCEED==True:
            root_tk=tk.Tk()
            my_yolo=yolo_cfg(root_tk,SAVED_SETTINGS_PATH)
            my_yolo.root.mainloop()
            PROCEED=False
            del my_yolo


    





