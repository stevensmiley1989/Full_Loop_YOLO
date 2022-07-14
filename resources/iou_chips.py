from lib2to3.pgen2 import grammar
import os
import argparse
import pandas as pd
import shutil
import matplotlib.pyplot as plt
import sys
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
from tqdm import tqdm
import cv2
XML_EXT = '.xml'
DEFAULT_ENCODING = 'utf-8'
ENCODE_METHOD = DEFAULT_ENCODING
class PascalVocReader:
    def __init__(self, file_path,EXT='.jpg',gt_path=None):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]]
        self.shapes = []
        self.file_path = file_path
        self.verified = False
        self.gt_path=gt_path
        self.EXT=EXT
        try:
            self.parse_xml()
        except:
            pass

    def get_shapes(self):
        return self.shapes

    def add_shape(self, label, bnd_box,confidence):
        x_min = int(float(bnd_box.find('xmin').text))
        y_min = int(float(bnd_box.find('ymin').text))
        x_max = int(float(bnd_box.find('xmax').text))
        y_max = int(float(bnd_box.find('ymax').text))
        points = (x_min, y_min,x_max, y_max)
        self.shapes.append((label, points,confidence))

    def parse_xml(self):
        assert self.file_path.endswith(XML_EXT), "Unsupported file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xml_tree = ElementTree.parse(self.file_path, parser=parser).getroot()
        filename = xml_tree.find('filename').text
        self.img_path=xml_tree.find('path').text
        if self.img_path.find(self.EXT)==-1:
            self.img_path=os.path.join(self.img_path,filename)
        if self.img_path.find(self.EXT)!=-1 and self.gt_path==None:
            if os.path.exists(self.img_path.replace(self.EXT,'.xml').replace('JPEGImages','Annotations')) and self.gt_path==None:
                self.gt_path=self.img_path.replace(self.EXT,'.xml').replace('JPEGImages','Annotations')
            else:
                pass
        for object_iter in xml_tree.findall('object'):
            bnd_box = object_iter.find("bndbox")
            label = object_iter.find('name').text
            try:
                confidence=object_iter.find('confidence').text
            except:
                confidence=0
            self.add_shape(label, bnd_box,confidence)
        return True

def calc_precision(TP,FP):
    if TP+FP==0:
        return 'NA'
    precision_i=100*TP/(TP+FP)
    print('Precision = {}'.format(precision_i))
    return precision_i

def calc_recall(TP,FN):
    if TP+FN==0:
        return 'NA'
    recall_i=100*TP/(TP+FN)
    print('Recall = {}'.format(recall_i))
    return recall_i

Prediction_xml="/home/steven/Elements/Drone_Images/Yolo/tiny_yolo-Elements_novs_5_18_w640_h640_d0_c1/predictions/Annotations"
GT_xml='None'
if os.path.exists(Prediction_xml):
    pass
else:
    Prediction_xml=None
if os.path.exists(GT_xml):
    pass
else:
    GT_xml=None
ap = argparse.ArgumentParser()
ap.add_argument("--GT_xml",type=str,default=GT_xml,help='The path to Ground Truth Annotation files')
ap.add_argument("--Prediction_xml",type=str,default=Prediction_xml,help="The path to Prediction Annotation files")
ap.add_argument("--Threshold",type=float,default=0.0,help="The iou threshold for False Positive")
args=ap.parse_args()
if args.GT_xml!=None:
    GT_xml=args.GT_xml
    print("--GT_xml == {}".format(GT_xml))
else:
    print('WARNING! \n \t --GT_xml \t None specified')
if args.Prediction_xml!=None:
    Prediction_xml=args.Prediction_xml
    print("--Prediction_xml == {}".format(Prediction_xml))
else:
    print('WARNING! \n \t --Prediction_xml \t None specified')

Threshold=args.Threshold
Chips_pathbase=os.path.join(os.path.dirname(Prediction_xml),'Prediction_Chips')
if os.path.exists(Chips_pathbase)==False:
    os.makedirs(Chips_pathbase)
else:
    os.system('rm -rf {}'.format(Chips_pathbase))
    os.makedirs(Chips_pathbase)
def bb_intersection(boxA,boxB):
    xA=max(boxA[0],boxB[0])
    yA=max(boxA[1],boxB[1])
    xB=min(boxA[2],boxB[2])
    yB=min(boxA[3],boxB[3])
    interArea=max(0,xB-xA+1)*max(0,yB-yA+1)
    boxAArea=(boxA[2]-boxA[0]+1)*(boxA[3]-boxA[1]+1)
    boxBArea=(boxB[2]-boxB[0]+1)*(boxB[3]-boxB[1]+1)
    iou=interArea/float(boxAArea+boxBArea-interArea)
    return iou
def pad(str_i,min_len=8):
    while len(str_i)<min_len:
        str_i='0'+str_i
    return str_i

Chips_path=Chips_pathbase
Chips_path_iou=Chips_path
if os.path.exists(Chips_path)==False:
    os.makedirs(Chips_path)
else:
    os.system('rm -rf {}'.format(Chips_path))
    os.makedirs(Chips_path)
Annotations_path_i=os.path.join(Chips_path,'Annotations')
if os.path.exists(Annotations_path_i)==False:
    os.makedirs(Annotations_path_i)
else:
    os.system('rm -rf {}'.format(Annotations_path_i))
    os.makedirs(Annotations_path_i)
os.system('cp -r {} {}'.format(Prediction_xml,Chips_path))
Chips_path=os.path.join(Chips_pathbase,'chips')

Predictions=os.listdir(Annotations_path_i)
Predictions=[os.path.join(Annotations_path_i,w) for w in Predictions if w.find('.xml')!=-1]
if os.path.exists(Chips_path)==False:
    os.makedirs(Chips_path)
else:
    os.system('rm -rf {}'.format(Chips_path))
    os.makedirs(Chips_path)
chipA_dic={}
for PRED_i in tqdm(Predictions):
    PRED=PascalVocReader(PRED_i)
    prediction_i=PRED.get_shapes()
    #print(prediction_i)
    GT=PascalVocReader(PRED.gt_path)
    gt_i=GT.get_shapes()
    #print(PRED_i)
    imgPRED=cv2.imread(PRED.img_path)
    if prediction_i:
        for prediction_j in prediction_i:
            #print(gt_i)
            boxA=prediction_j[1]
            nameA=prediction_j[0]
            confidenceA=str(prediction_j[2]).replace('.','p')
            imageA=os.path.basename(PRED.img_path).split('.')[0]
            chipA=imgPRED[boxA[1]:boxA[3],boxA[0]:boxA[2]]
            Chips_pathA=os.path.join(Chips_path,os.path.join('PREDICTIONS',nameA))
            if os.path.exists(Chips_pathA)==False:
                os.makedirs(Chips_pathA)
            cv2.imwrite(os.path.join(Chips_pathA,'chip_IMAGE{}_NAME{}_XMIN{}_YMIN{}_XMAX{}_YMAX{}_CONFIDENCE{}.jpg'.format(imageA,nameA,boxA[0],boxA[1],boxA[2],boxA[3],confidenceA)),chipA)

    if gt_i:
        for j,gt_j in tqdm(enumerate(gt_i)):
            boxB=gt_j[1]
            nameB=gt_j[0]
            imageB=os.path.basename(PRED.img_path).split('.')[0]
            chipB=imgPRED[boxB[1]:boxB[3],boxB[0]:boxB[2]]
            Chips_pathB=os.path.join(Chips_path,os.path.join('GROUND_TRUTH',nameB))
            if os.path.exists(Chips_pathB)==False:
                os.makedirs(Chips_pathB)
            cv2.imwrite(os.path.join(Chips_pathB,'chip_IMAGE{}_NAME{}_XMIN{}_YMIN{}_XMAX{}_YMAX{}.jpg'.format(imageB,nameB,boxB[0],boxB[1],boxB[2],boxB[3])),chipB)

os.system('xdg-open {}'.format(Chips_pathbase))
