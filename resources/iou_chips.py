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

    def add_shape(self, label, bnd_box):
        x_min = int(float(bnd_box.find('xmin').text))
        y_min = int(float(bnd_box.find('ymin').text))
        x_max = int(float(bnd_box.find('xmax').text))
        y_max = int(float(bnd_box.find('ymax').text))
        points = (x_min, y_min,x_max, y_max)
        self.shapes.append((label, points))

    def parse_xml(self):
        assert self.file_path.endswith(XML_EXT), "Unsupported file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xml_tree = ElementTree.parse(self.file_path, parser=parser).getroot()
        filename = xml_tree.find('filename').text
        self.img_path=xml_tree.find('path').text
        if self.img_path.find(self.EXT)==-1:
            self.img_path=os.path.join(self.img_path,filename)
        #print(self.img_path)
        if self.img_path.find(self.EXT)!=-1 and self.gt_path==None:
            if os.path.exists(self.img_path.replace(self.EXT,'.xml').replace('JPEGImages','Annotations')) and self.gt_path==None:
                self.gt_path=self.img_path.replace(self.EXT,'.xml').replace('JPEGImages','Annotations')
        for object_iter in xml_tree.findall('object'):
            bnd_box = object_iter.find("bndbox")
            label = object_iter.find('name').text
            self.add_shape(label, bnd_box)
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
path_result_list_txt='None'
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
ap.add_argument("--Threshold",type=float,default=0.1,help="The iou threshold for False Positive")
ap.add_argument("--useCOCO",action="store_true",default=True,help="use COCO for metrics")
ap.add_argument("--path_result_list_txt",type=str,default=path_result_list_txt,help='path to valid.txt')
args=ap.parse_args()
if args.GT_xml!=None:
    GT_xml=args.GT_xml
    print("--GT_xml == {}".format(GT_xml))
else:
    print('WARNING! \n \t --GT_xml \t None specified')
    if args.path_result_list_txt!='None':
        path_result_list_txt=args.path_result_list_txt
        if os.path.exists(path_result_list_txt):
            f=open(path_result_list_txt,'r')
            f_read=f.readlines()
            f.close()
            GT_xml=os.path.dirname(f_read[0])
            print('FOUND GT_xml at {}'.format(GT_xml))

if args.Prediction_xml!=None:
    Prediction_xml=args.Prediction_xml
    print("--Prediction_xml == {}".format(Prediction_xml))
else:
    print('WARNING! \n \t --Prediction_xml \t None specified')
useCOCO=args.useCOCO
print('useCOCO',useCOCO)
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
# if useCOCO:
#     COCO_iOUs=[0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
# else:
COCO_iOUs=[Threshold]
df=pd.DataFrame(columns=['iou','name','TP','FP','FN','GT','Precision','Recall'])
df_filename_main=os.path.join(Chips_pathbase,'df_chips_metrics.csv')
i=0
for Threshold in tqdm(COCO_iOUs):
    Chips_path=os.path.join(Chips_pathbase,'iOU_{}'.format(Threshold).replace('.','p'))
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
    

    Predictions=os.listdir(Annotations_path_i)
    Predictions=[os.path.join(Annotations_path_i,w) for w in Predictions if w.find('.xml')!=-1]
    Chips_path=os.path.join(Chips_path,'chips')
    if os.path.exists(Chips_path)==False:
        os.makedirs(Chips_path)
    else:
        os.system('rm -rf {}'.format(Chips_path))
        os.makedirs(Chips_path)
    chipA_dic={}
    if path_result_list_txt!='None':
        GT_xml_paths=os.listdir(GT_xml)
        GT_xml_paths=[os.path.join(GT_xml,w) for w in GT_xml_paths]
    else:
        GT_xml_paths=['None']
    for PRED_i in tqdm(Predictions):
        PRED=PascalVocReader(PRED_i)
        prediction_i=PRED.get_shapes()
        #print(prediction_i)
        pred_img=os.path.basename(PRED.img_path)
        if path_result_list_txt!='None':
            GT_path_i=[w for w in GT_xml_paths if pred_img.replace('.jpg','.xml')==os.path.basename(w)]
            try:
                GT=PascalVocReader(GT_path_i[0])
            except:
                print("GT_path_i not found, using GT=PascalVocReader(PRED.gt_path)")
                GT=PascalVocReader(PRED.gt_path)
            #input('Did you get this GT?')
        else:
            GT=PascalVocReader(PRED.gt_path)
        #GT=PascalVocReader(PRED.gt_path)
        gt_i=GT.get_shapes()
        imgPRED=cv2.imread(PRED.img_path)
        # print('')
        # print(PRED.img_path)
        # print('')
        # print(GT_path_i[0])
        # print('')
        # input('Does this look right?')
        GT_list=[]
        FN_list=[]
        # print('gt_i',gt_i)
        # print('prediction_i',prediction_i)
        # input('Look good?')
        if prediction_i and gt_i:
            df_i=pd.DataFrame(columns=['nameA','nameB','boxA','boxB','iou'])
            jj=0
            for prediction_j in prediction_i:
                #print(gt_i)
                boxA=prediction_j[1]
                nameA=prediction_j[0]
                chipA=imgPRED[boxA[1]:boxA[3],boxA[0]:boxA[2]]
                Chips_pathA=os.path.join(Chips_path,nameA)
                if os.path.exists(Chips_pathA)==False:
                    os.makedirs(Chips_pathA)
                Chips_pathA_FP=os.path.join(Chips_pathA,'FP')
                Chips_pathA_TP=os.path.join(Chips_pathA,'TP')
                Chips_pathA_GT=os.path.join(Chips_pathA,'GT')

                if os.path.exists(Chips_pathA_FP)==False:
                    os.makedirs(Chips_pathA_FP)
                if os.path.exists(Chips_pathA_TP)==False:
                    os.makedirs(Chips_pathA_TP)
                if os.path.exists(Chips_pathA_GT)==False:
                    os.makedirs(Chips_pathA_GT)     
        
                #plt.imshow(imgA)
                #plt.show()
                for j,gt_j in enumerate(gt_i):
                    boxB=gt_j[1]
                    nameB=gt_j[0]

                    #if nameA==nameB:
                    df_i.at[jj,'iou']=bb_intersection(boxA,boxB)
                    #else:
                        #break
                        #df_i.at[jj,'iou']=0.0
                    df_i.at[jj,'nameA']=nameA
                    df_i.at[jj,'nameB']=nameB
                    df_i.at[jj,'boxA']=boxA
                    df_i.at[jj,'boxB']=boxB
                    jj+=1
                    # if len(boxB)>0:
                    #     if boxB not in GT_list:
                    #         GT_list.append(boxB)
                    #         chipB=imgPRED[boxB[1]:boxB[3],boxB[0]:boxB[2]]
                    #         chip_count_GT=str(len(os.listdir(Chips_pathA_GT))+1)
                    #         try:
                    #             cv2.imwrite(os.path.join(Chips_pathA_GT,'chip_{}_{}'.format(pad(str(chip_count_GT)),PRED.img_path.split('/')[-1])),chipB)
                    #         except:
                    #             print('ERROR with chipB')
            df_i=df_i.sort_values(by=['boxB','boxA','iou'],ascending=False).reset_index().drop('index',axis=1)
            df_i['box_A_B_iou']=df_i['boxA'].astype(str)+df_i['boxB'].astype(str)+df_i['iou'].astype(str)+df_i['nameA']+df_i['nameB']
            df_i=df_i.drop_duplicates(subset=['box_A_B_iou'],keep='first').reset_index().drop('index',axis=1)
            df_i=df_i.sort_values(by=['boxB','iou'],ascending=False).drop_duplicates(subset='boxB',keep='first').reset_index().drop('index',axis=1)

            df_i_TP_index=list(df_i[(df_i.iou>=Threshold) & (df_i.nameA==df_i.nameB)].index)
            df_i_FP_index=list(df_i[((df_i.iou>0) & (df_i.iou<Threshold) & (df_i.nameA!=df_i.nameB))].index)
            if len(df_i_FP_index)>0 and len(df_i_TP_index)>0:
                df_i_FN_index=list(df_i.drop(df_i_TP_index+df_i_FP_index,axis=0).index)
            elif len(df_i_FP_index)>0:
                df_i_FN_index=list(df_i.drop(df_i_FP_index,axis=0).index)
            elif len(df_i_TP_index)>0:
                df_i_FN_index=list(df_i.drop(df_i_TP_index,axis=0).index)
            else:
                df_i_FN_index=[]
            if len(df_i_TP_index)>0:
                #print('df_i_TP_index')
                for TP_i in tqdm(df_i_TP_index):
                    boxA=df_i['boxA'][TP_i]
                    boxB=df_i['boxB'][TP_i]
                    nameA=df_i['nameA'][TP_i]
                    nameB=df_i['nameB'][TP_i]
                    chipA=imgPRED[boxA[1]:boxA[3],boxA[0]:boxA[2]]
                    Chips_pathA=os.path.join(Chips_path,nameA)
                    if os.path.exists(Chips_pathA)==False:
                        os.makedirs(Chips_pathA)
                    Chips_pathA_FP=os.path.join(Chips_pathA,'FP')
                    Chips_pathA_TP=os.path.join(Chips_pathA,'TP')
                    Chips_pathA_GT=os.path.join(Chips_pathA,'GT')

                    if os.path.exists(Chips_pathA_FP)==False:
                        os.makedirs(Chips_pathA_FP)
                    if os.path.exists(Chips_pathA_TP)==False:
                        os.makedirs(Chips_pathA_TP)
                    if os.path.exists(Chips_pathA_GT)==False:
                        os.makedirs(Chips_pathA_GT)    

                    chip_count_i=TP_i  
                    cv2.imwrite(os.path.join(Chips_pathA_TP,'chip_{}_of_{}_{}'.format(pad(str(chip_count_i)),pad(str(len(df_i))),PRED.img_path.split('/')[-1])),chipA)
                    # plt.title('chipA')
                    # plt.imshow(chipA)
                    # plt.show()
                    # plt.title('chipB')
                    # plt.imshow(chipB)
                    # plt.show()
                    f=open(PRED.file_path,'r')
                    f_read=f.readlines()
                    f.close()
                    f_new=[]
                    for j,line in enumerate(f_read):
                        if line.find(nameA)!=-1 and line.find(nameA+'_')==-1:
                            for k,line_k in enumerate(f_read[j:]):
                                if line_k.find(str(boxB[0])[0])!=-1 and f_read[j+k+1].find(str(boxB[1])[0])!=-1 and f_read[j+k+2].find(str(boxB[2])[0])!=-1 and f_read[j+k+3].find(str(boxB[3])[0])!=-1:
                                    line=line.replace(nameA,nameA+'_TP')
                                    #print(line)
                                    break
                                #elif line_k.find('object')!=-1:
                                #    break
                        f_new.append(line)
                    f=open(PRED.file_path,'w')
                    tmp=[f.writelines(w) for w in f_new]
                    f.close()
            if len(df_i_FP_index)>0:
                #print('df_i_FP_index')
                for FP_i in tqdm(df_i_FP_index):
                    boxA=df_i['boxA'][FP_i]
                    boxB=df_i['boxB'][FP_i]
                    nameA=df_i['nameA'][FP_i]
                    nameB=df_i['nameB'][FP_i]
                    chipA=imgPRED[boxA[1]:boxA[3],boxA[0]:boxA[2]]
                    Chips_pathA=os.path.join(Chips_path,nameA)
                    if os.path.exists(Chips_pathA)==False:
                        os.makedirs(Chips_pathA)
                    Chips_pathA_FP=os.path.join(Chips_pathA,'FP')
                    Chips_pathA_TP=os.path.join(Chips_pathA,'TP')
                    Chips_pathA_FP=os.path.join(Chips_pathA,'FP')
                    Chips_pathA_GT=os.path.join(Chips_pathA,'GT')

                    if os.path.exists(Chips_pathA_FP)==False:
                        os.makedirs(Chips_pathA_FP)
                    if os.path.exists(Chips_pathA_TP)==False:
                        os.makedirs(Chips_pathA_TP)
                    if os.path.exists(Chips_pathA_FP)==False:
                        os.makedirs(Chips_pathA_FP)     
                    if os.path.exists(Chips_pathA_GT)==False:
                        os.makedirs(Chips_pathA_GT)    

                    chip_count_i=FP_i  
                    cv2.imwrite(os.path.join(Chips_pathA_FP,'chip_{}_of_{}_{}'.format(pad(str(chip_count_i)),pad(str(len(df_i))),PRED.img_path.split('/')[-1])),chipA)
                    f=open(PRED.file_path,'r')
                    f_read=f.readlines()
                    f.close()
                    f_new=[w for w in f_read if w.find('</annotation>')==-1]
                    f=open(PRED.file_path,'w')
                    [f.writelines(w) for w in f_new]
                    f.close()
                    f=open(PRED.file_path,'a')
                    name=nameA+'_FP'
                    pose='Unspecified'
                    truncated='0' 
                    difficult='0'
                    xmin=boxA[0]
                    ymin=boxA[1]
                    xmax=boxA[2]
                    ymax=boxA[3]
                    f.writelines('\t<object>\n')
                    f.writelines('\t\t<name>{}</name>\n'.format(name))
                    f.writelines('\t\t<pose>{}</pose>\n'.format(pose))       
                    f.writelines('\t\t<truncated>{}</truncated>\n'.format(truncated))
                    f.writelines('\t\t<difficult>{}</difficult>\n'.format(difficult))
                    f.writelines('\t\t<bndbox>\n')
                    f.writelines('\t\t\t<xmin>{}</xmin>\n'.format(xmin))
                    f.writelines('\t\t\t<ymin>{}</ymin>\n'.format(ymin))
                    f.writelines('\t\t\t<xmax>{}</xmax>\n'.format(xmax))
                    f.writelines('\t\t\t<ymax>{}</ymax>\n'.format(ymax))
                    f.writelines('\t\t</bndbox>\n')
                    f.writelines('\t</object>\n')
                    f.writelines('</annotation>\n')
                    f.close()

            if len(df_i_FN_index)>0:
                #print('df_i_FN_index')
                for FN_i in tqdm(df_i_FN_index):
                    boxA=df_i['boxA'][FN_i]
                    boxB=df_i['boxB'][FN_i]
                    nameA=df_i['nameA'][FN_i]
                    nameB=df_i['nameB'][FN_i]
                    chipB=imgPRED[boxB[1]:boxB[3],boxB[0]:boxB[2]]
                    Chips_pathB=os.path.join(Chips_path,nameB)
                    if os.path.exists(Chips_pathB)==False:
                        os.makedirs(Chips_pathB)
                    Chips_pathB_FN=os.path.join(Chips_pathB,'FN')
                    if os.path.exists(Chips_pathB_FN)==False:
                        os.makedirs(Chips_pathB_FN)     
                    if boxB not in FN_list:
                        chip_count_i=FN_i  
                        FN_list.append(boxB)
                        try:
                            cv2.imwrite(os.path.join(Chips_pathB_FN,'chip_{}_of_{}_{}'.format(pad(str(chip_count_i)),pad(str(len(df_i))),PRED.img_path.split('/')[-1])),chipB)
                        except:
                            print('ERROR with chipB')
                        # f=open(PRED.file_path,'r')
                        # f_read=f.readlines()
                        # f.close()
                        # f_new=[w for w in f_read if w.find('</annotation>')==-1]
                        # f=open(PRED.file_path,'w')
                        # [f.writelines(w) for w in f_new]
                        # f.close()
                        # f=open(PRED.file_path,'a')
                        # name=nameB+'_FN'
                        # pose='Unspecified'
                        # truncated='0' 
                        # difficult='0'
                        # xmin=boxB[0]
                        # ymin=boxB[1]
                        # xmax=boxB[2]
                        # ymax=boxB[3]
                        # f.writelines('\t<object>\n')
                        # f.writelines('\t\t<name>{}</name>\n'.format(name))
                        # f.writelines('\t\t<pose>{}</pose>\n'.format(pose))       
                        # f.writelines('\t\t<truncated>{}</truncated>\n'.format(truncated))
                        # f.writelines('\t\t<difficult>{}</difficult>\n'.format(difficult))
                        # f.writelines('\t\t<bndbox>\n')
                        # f.writelines('\t\t\t<xmin>{}</xmin>\n'.format(xmin))
                        # f.writelines('\t\t\t<ymin>{}</ymin>\n'.format(ymin))
                        # f.writelines('\t\t\t<xmax>{}</xmax>\n'.format(xmax))
                        # f.writelines('\t\t\t<ymax>{}</ymax>\n'.format(ymax))
                        # f.writelines('\t\t</bndbox>\n')
                        # f.writelines('\t</object>\n')
                        # f.writelines('</annotation>\n')
                        # f.close()


        elif prediction_i:
            df_i=pd.DataFrame(columns=['nameA','boxA'])
            jj=0
            for prediction_j in prediction_i:
                #print(gt_i)
                boxA=prediction_j[1]
                nameA=prediction_j[0]
                chipA=imgPRED[boxA[1]:boxA[3],boxA[0]:boxA[2]]
                Chips_pathA=os.path.join(Chips_path,nameA)
                if os.path.exists(Chips_pathA)==False:
                    os.makedirs(Chips_pathA)
                Chips_pathA_FP=os.path.join(Chips_pathA,'FP')
                if os.path.exists(Chips_pathA_FP)==False:
                    os.makedirs(Chips_pathA_FP)          
                #plt.imshow(imgA)
                #plt.show()
                #for j,gt_j in enumerate(gt_i):
                    #boxB=gt_j[1]
                    #nameB=gt_j[0]
                df_i.at[jj,'nameA']=nameA
                df_i.at[jj,'boxA']=boxA
                jj+=1
            df_i=df_i.drop_duplicates(subset='boxA',keep='first').reset_index().drop('index',axis=1)
            df_i_FP_index=list(df_i.index)
            if len(df_i_FP_index)>0:
                #print('df_i_FP_index')
                for FP_i in tqdm(df_i_FP_index):
                    #print(FP_i)
                    boxA=df_i['boxA'][FP_i]
                    nameA=df_i['nameA'][FP_i]
                    chipA=imgPRED[boxA[1]:boxA[3],boxA[0]:boxA[2]]
                    Chips_pathA=os.path.join(Chips_path,nameA)
                    if os.path.exists(Chips_pathA)==False:
                        os.makedirs(Chips_pathA)
                    Chips_pathA_FP=os.path.join(Chips_pathA,'FP')
                    if os.path.exists(Chips_pathA_FP)==False:
                        os.makedirs(Chips_pathA_FP)     
                    chip_count_i=FP_i  
                    cv2.imwrite(os.path.join(Chips_pathA_FP,'chip_{}_of_{}_{}'.format(pad(str(chip_count_i)),pad(str(len(df_i))),PRED.img_path.split('/')[-1])),chipA)
                    f=open(PRED.file_path,'r')
                    f_read=f.readlines()
                    f.close()
                    f_new=[]
                    for j,line in enumerate(f_read):
                        if line.find(nameA)!=-1 and line.find(nameA+'_')==-1:
                            for k,line_k in enumerate(f_read[j:]):
                                if line_k.find(str(boxA[0]))!=-1 and f_read[j+k+1].find(str(boxA[1]))!=-1 and f_read[j+k+2].find(str(boxA[2]))!=-1 and f_read[j+k+3].find(str(boxA[3]))!=-1:
                                    line=line.replace(nameA,nameA+'_FP')
                                    #print(line)
                                    break
                                elif line_k.find('object')!=-1:
                                    break
                        f_new.append(line)
                    f=open(PRED.file_path,'w')
                    tmp=[f.writelines(w) for w in f_new]
                    f.close()
        if gt_i:
            df_i=pd.DataFrame(columns=['nameB','boxB'])
            jj=0
            GT_list=[]
            for j,gt_j in tqdm(enumerate(gt_i)):
                boxB=gt_j[1]
                nameB=gt_j[0]
                chipB=imgPRED[boxB[1]:boxB[3],boxB[0]:boxB[2]]
                Chips_pathB=os.path.join(Chips_path,nameB)
                if os.path.exists(Chips_pathB)==False:
                    os.makedirs(Chips_pathB)
                Chips_pathB_FN=os.path.join(Chips_pathB,'FN')
                Chips_pathB_GT=os.path.join(Chips_pathB,'GT')
                if os.path.exists(Chips_pathB_GT)==False:
                    os.makedirs(Chips_pathB_GT)  
                if os.path.exists(Chips_pathB_FN)==False:
                    os.makedirs(Chips_pathB_FN) 
                chip_count_GT=str(len(os.listdir(Chips_pathB_GT))+1)
                if boxB not in GT_list:
                    GT_list.append(boxB)
                    chipB=imgPRED[boxB[1]:boxB[3],boxB[0]:boxB[2]]
                    chip_count_GT=str(len(os.listdir(Chips_pathB_GT))+1)
                    try:
                        cv2.imwrite(os.path.join(Chips_pathB_GT,'chip_{}_{}'.format(pad(str(chip_count_GT)),PRED.img_path.split('/')[-1])),chipB)
                    except:
                        print('ERROR writing chip')
                    #cv2.imwrite(os.path.join(Chips_pathB_GT,'chip_{}_{}'.format(pad(str(chip_count_GT)),PRED.img_path.split('/')[-1])),chipB)
                # if boxB not in FN_list:
                #     FN_list.append(boxB)
                #     cv2.imwrite(os.path.join(Chips_pathB_FN,'chip_{}_{}'.format(pad(str(chip_count_GT)),PRED.img_path.split('/')[-1])),chipB)
                    # f=open(PRED.file_path,'r')
                    # f_read=f.readlines()
                    # f.close()
                    # f_new=[w for w in f_read if w.find('</annotation>')==-1]
                    # f=open(PRED.file_path,'w')
                    # [f.writelines(w) for w in f_new]
                    # f.close()
                    # f=open(PRED.file_path,'a')
                    # name=nameB+'_FN'

                    # pose='Unspecified'
                    # truncated='0' 
                    # difficult='0'
                    # xmin=boxB[0]
                    # ymin=boxB[1]
                    # xmax=boxB[2]
                    # ymax=boxB[3]
                    # f.writelines('\t<object>\n')
                    # f.writelines('\t\t<name>{}</name>\n'.format(name))
                    # f.writelines('\t\t<pose>{}</pose>\n'.format(pose))       
                    # f.writelines('\t\t<truncated>{}</truncated>\n'.format(truncated))
                    # f.writelines('\t\t<difficult>{}</difficult>\n'.format(difficult))
                    # f.writelines('\t\t<bndbox>\n')
                    # f.writelines('\t\t\t<xmin>{}</xmin>\n'.format(xmin))
                    # f.writelines('\t\t\t<ymin>{}</ymin>\n'.format(ymin))
                    # f.writelines('\t\t\t<xmax>{}</xmax>\n'.format(xmax))
                    # f.writelines('\t\t\t<ymax>{}</ymax>\n'.format(ymax))
                    # f.writelines('\t\t</bndbox>\n')
                    # f.writelines('\t</object>\n')
                    # f.writelines('</annotation>\n')
                    # f.close()
    #count TP,FP,FN in each directory
    for dir_i in tqdm(os.listdir(Chips_path)):
        subdirs_i=os.listdir(os.path.join(Chips_path,dir_i))
        name=dir_i
        dir_i=os.path.join(Chips_path,dir_i)
        df_filename=os.path.join(Chips_path_iou,dir_i+'.csv')
        df_dir_i=pd.DataFrame(columns=['TP','FP','FN','GT','Precision','Recall'])
        if 'TP' in subdirs_i:
            TP_list=os.listdir(os.path.join(dir_i,'TP'))
            TP_list=[w for w in TP_list if w.find('.jpg')!=-1]
            TP=len(TP_list)
        else:
            TP=0.0
        if 'FP' in subdirs_i:
            FP_list=os.listdir(os.path.join(dir_i,'FP'))
            FP_list=[w for w in FP_list if w.find('.jpg')!=-1]
            FP=len(FP_list)
        else:
            FP=0.0       
        if 'FN' in subdirs_i:
            FN_list=os.listdir(os.path.join(dir_i,'FN'))
            FN_list=[w for w in FN_list if w.find('.jpg')!=-1]
            FN=len(FN_list)
        else:
            FN=0.0    
        if 'GT' in subdirs_i:
            GT_list=os.listdir(os.path.join(dir_i,'GT'))
            GT_list=[w for w in GT_list if w.find('.jpg')!=-1]
            GT=len(GT_list)
        else:
            GT=0.0   
        precision=calc_precision(TP,FP)
        recall=calc_recall(TP,FN)
        df_dir_i['TP']=[TP]
        df_dir_i['FP']=[FP]
        df_dir_i['FN']=[FN]
        df_dir_i['GT']=[GT]
        df_dir_i['Precision']=[precision]
        df_dir_i['Recall']=[recall]
        df_dir_i.to_csv(df_filename,index=None)
        
        df.at[i,'iou']=Threshold
        df.at[i,'name']=name
        df.at[i,'TP']=TP
        df.at[i,'FP']=FP
        df.at[i,'FN']=FN
        df.at[i,'GT']=GT
        df.at[i,'Precision']=precision
        df.at[i,'Recall']=recall
        i+=1
        df.to_csv(df_filename_main,index=None)

for name in list(df['name'].unique()):
    df['mAP_'+name]=df['Precision'].copy()
    df['mAP_'+name]=''
    try:
        df['mAP_'+name][0]=[df[df['name']==name]['Precision'].mean()]
    except:
        print(f'ERROR with {name}')
        pass
df['mAP']=df['Precision'].copy()
df['mAP']=''
try:
    if df['Precision'].astype(str).str.contains('NA').any()==True:
        df['mAP'][0]=[df['Precision'].drop(df['Precision'].str.contains('NA').dropna().index).mean()]
    else:
        df['mAP'][0]=[df['Precision'].mean()]
except:
    print('invalid mAP')
    pass
df.to_csv(df_filename_main,index=None)
print(df)