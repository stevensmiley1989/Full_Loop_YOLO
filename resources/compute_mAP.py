# import the necessary packages

from tqdm import tqdm

import pickle
import json
import cv2
import os
import datetime
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import pandas as pd
# import the necessary packages
from pycocotools.cocoeval import COCOeval
from pycocotools.cocoeval import Params
from pycocotools.coco import COCO
import numpy as np
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import time
def setDetParams(self):
    '''
    Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met: 

    1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer. 
    2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution. 

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    The views and conclusions contained in the software and documentation are those
    of the authors and should not be interpreted as representing official policies, 
    either expressed or implied, of the FreeBSD Project.
    '''
    self.imgIds = []
    self.catIds = []
    # np.arange causes trouble.  the data point on arange is slightly larger than the true value
    self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05) + 1), endpoint=True) #issue with numpy so fixing it here with in
    self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01) + 1), endpoint=True)
    self.maxDets = [1, 10, 100]
    self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    self.areaRngLbl = ['all', 'small', 'medium', 'large']
    self.useCats = 1
Params.setDetParams=setDetParams #issue with numpy so fixing it here
def summarize(self,names={'person':0},result_file='results.txt'):
    '''

    Copyright (c) 2014, Piotr Dollar and Tsung-Yi Lin
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met: 

    1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer. 
    2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution. 

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    The views and conclusions contained in the software and documentation are those
    of the authors and should not be interpreted as representing official policies, 
    either expressed or implied, of the FreeBSD Project.
    '''

    '''
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    '''
    f=open(result_file,'w')
    time_generated=str(datetime.datetime.now())
    HEADER_TEXT=f"######################################\nTHESE mAP METRICS WERE GENERATED ON {time_generated}\n######################################\n"
    f.writelines(HEADER_TEXT)
    SPACER="\n######################################\n"
    f.close()
    def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100,names=names,result_file=result_file ):
        my_results=[]
        p = self.params
        
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap==1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]
            s = s[:,:,:,aind,mind]
        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']


            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            s = s[:,:,aind,mind]
        if len(s[s>-1])==0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s>-1])

            #cacluate AP(average precision) for each category
            #num_classes = 80
            avg_ap = 0.0
            if ap == 1:
                num_classes=len(names)
                rev_names={v:k for k,v in names.items()}
                remove_negative=0
                for i in range(0, num_classes):
                    line_i='category : {0} : {1}'.format(rev_names[i],np.mean(s[:,:,i,:]))
                    print(line_i)
                    my_results.append(line_i)
                    if np.mean(s[:,:,i,:]>0):
                        avg_ap +=np.mean(s[:,:,i,:])
                    else:
                        remove_negative+=1
                if num_classes-remove_negative==0:
                    remove_negative=0 #prevent dividing by zero
                line_i='(all categories) mAP : {}'.format(avg_ap / (num_classes-remove_negative))
                print(line_i)
                my_results.append(line_i)
        line_i=iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        print(line_i)
        my_results.append(line_i)
        f=open(result_file,'a')
        f.writelines(SPACER)
        [f.writelines(w+'\n') for w in my_results]
        f.close()
        
        return mean_s
    def _summarizeDets():
        stats = np.zeros((12,))
        stats[0] = _summarize(1)
        stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
        stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
        stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
        stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
        stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
        stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
        stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
        stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
        stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
        stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
        return stats
    if not self.eval:
        raise Exception('Please run accumulate() first')
    iouType = self.params.iouType
    if iouType == 'bbox':
        summarize = _summarizeDets
    self.stats = summarize()
    #os.system('xdg-open {}'.format(result_file))

def __str__(self):
    self.summarize()
def read_XML_quick(path):
    parser = etree.XMLParser(encoding='utf-8')
    f=open(path,'r')
    f_read=f.read()
    f.close()
    if f_read.find('/>')!=-1:
        f_read=f_read.splitlines()
        f_new=[]
        for line in f_read:
            if line.find('/>')==-1:
                f_new.append(line)
            else:
                print(f'bad line found for {line}')
        if f_new!=f_read:
            f=open(path,'w')
            [f.writelines(w) for w in f_new]
            f.close()
    try:
        xmltree = ElementTree.parse(path, parser=parser).getroot()
    except:
        #print(path)
        f=open(path,'r')
        f_read=f.readlines()
        f.close()
        f_new=[]
        if f_read[0].find('annotation')==-1:
            f_new.append('<annotation>\n')
            for line in f_read:
                f_new.append(line)
            f=open(path,'w')
            [f.writelines(w) for w in f_new]
            f.close()
        xmltree = ElementTree.parse(path, parser=parser).getroot()
    filename = xmltree.find('filename').text
    width_i=xmltree.find('size').find('width').text
    height_i=xmltree.find('size').find('height').text
    for object_iter in xmltree.findall('object'):
        bndbox = object_iter.find("bndbox")
        label = object_iter.find('name').text
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        (x,y,w,h)=(int(xmin),int(ymin),int(xmax-xmin),int(ymax-ymin))
    return xmltree   

def read_XML(path,result_list,names,images_found,id_i,result_list_plots,gt=False):
    single_batch=[]
    imageId=os.path.basename(path).split('.')[0]
    imageId=images_found[imageId]
    #print(imageId)
    parser = etree.XMLParser(encoding='utf-8')
    #print(path_anno_i)
    f=open(path,'r')
    f_read=f.read()
    f.close()
    if f_read.find('/>')!=-1:
        f_read=f_read.splitlines()
        f_new=[]
        for line in f_read:
            if line.find('/>')==-1:
                f_new.append(line)
            else:
                print(f'bad line found for {line}')
        if f_new!=f_read:
            f=open(path,'w')
            [f.writelines(w) for w in f_new]
            f.close()
    try:
        xmltree = ElementTree.parse(path, parser=parser).getroot()
    except:
        #print(path)
        f=open(path,'r')
        f_read=f.readlines()
        f.close()
        f_new=[]
        if f_read[0].find('annotation')==-1:
            f_new.append('<annotation>\n')
            for line in f_read:
                f_new.append(line)
            f=open(path,'w')
            [f.writelines(w) for w in f_new]
            f.close()
        xmltree = ElementTree.parse(path, parser=parser).getroot()
    filename = xmltree.find('filename').text
    width_i=xmltree.find('size').find('width').text
    height_i=xmltree.find('size').find('height').text
    for object_iter in xmltree.findall('object'):
        bndbox = object_iter.find("bndbox")
        label = object_iter.find('name').text
        label=names[label]
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        (x,y,w,h)=(int(xmin),int(ymin),int(xmax-xmin),int(ymax-ymin)) 
        if gt==False:
            conf=object_iter.find('confidence').text
            result_list.append({"id":int(id_i),
                                "image_id": imageId,
                                "category_id": label,
                                "bbox": [x, y, w, h],
                                "score": float(conf),
                                "iscrowd":0,
                                "area":int(int(width_i)*int(height_i))})
            #result_list_plots.append([xmin,ymin,xmax,ymax,float(conf),label])
            single_batch.append([xmin,ymin,xmax,ymax,float(conf),label])
        else:
            conf='1'
            result_list.append({"id":int(id_i),
                                "image_id": imageId,
                                "category_id": label,
                                "bbox": [x, y, w, h],
                                "iscrowd":0,
                                "area":int(int(width_i)*int(height_i))})
            single_batch.append([label,xmin,ymin,xmax,ymax])


        id_i+=1

    result_list_plots[imageId]=np.array(single_batch)       
    return result_list,id_i,result_list_plots
def create_groundtruths_json(path_Annotations,obj_names,images_found,coco_output,path_JSON='groundtruths.json',id_i=0):   
    fo=open(obj_names,'r')
    fo_read=fo.readlines()
    fo.close()
    names={w.replace('\n',''):j for j,w in enumerate(fo_read)}
    gt_results = []
    result_list_plot_gt={}
    #coco_output
    path_Annotation_OG=path_Annotations
    path_Annotations=os.listdir(path_Annotations)
    path_Annotations=[os.path.abspath(os.path.join(path_Annotation_OG,w)) for w in path_Annotations if w.find('.xml')!=-1 and os.path.basename(w).split('.')[0] in images_found.keys()]
    for _,path_anno_i in tqdm(enumerate(path_Annotations)):
        try:
        #print(path_anno_i)
            gt_results,id_i,result_list_plot_gt=read_XML(path_anno_i,gt_results,names,images_found,id_i,result_list_plot_gt,gt=True)
        except:
            print(f'issue with {path_anno_i}')
            pass
    #print(gt_results)
    coco_output['annotations']=gt_results
    # save the results on disk in a JSON format
    with open(path_JSON, "w") as f:
        json.dump(coco_output, f, indent=4)
    return path_JSON,id_i,result_list_plot_gt
def create_predictions_json(path_Annotations,obj_names,images_found,coco_output,path_JSON='predictions.json',id_i=0):   
    print('STARTED')
    fo=open(obj_names,'r')
    fo_read=fo.readlines()
    fo.close()
    #print('images_found',images_found)
    names={w.replace('\n',''):j for j,w in enumerate(fo_read)}
    pred_results = []
    result_list_plot_pred={}
    path_Annotation_OG=path_Annotations
    path_Annotations=os.listdir(path_Annotations)
    path_Annotations=[os.path.abspath(os.path.join(path_Annotation_OG,w)) for w in path_Annotations if w.find('.xml')!=-1]
    for _,path_anno_i in tqdm(enumerate(path_Annotations)):
        try:
        #print(path_anno_i)
            pred_results,id_i,result_list_plot_pred=read_XML(path_anno_i,pred_results,names,images_found,id_i,result_list_plot_pred,gt=False)
        except:
            print(f'issue with {path_anno_i}')
            pass

    coco_output['annotations']=pred_results
    # save the results on disk in a JSON format
    with open(path_JSON, "w") as f:
        json.dump(coco_output, f, indent=4)
    #print(pred_results)

    return path_JSON,result_list_plot_pred




def compute_map(path_JPEGS_GT,path_Anno_GT,path_Anno_Pred,obj_names_path,result_file,valid_list,show_results,create_combine_gt_pred):
    INFO = {
        "description": "Example Dataset",
        "url": "NA",
        "version": "0.1.0",
        "year": 2022,
        "contributor": "SPACENUT",
        "date_created": datetime.datetime.utcnow().isoformat(' ')
    }

    LICENSES = [
        {
            "id": 1,
            "name": "NA",
            "url": "NA"
        }
    ]
    f=open(valid_list,'r')
    f_read=f.readlines()
    f.close()
    images_found=[w.rstrip('\n').replace(' ','') for w in f_read]
    #images_found=os.listdir(path_Anno_GT)
    images_found=[os.path.basename(w).split('.')[0] for w in images_found]
    images_found={w:i for i,w in enumerate(images_found)}
    #CHECK FOR BAD PATHS
    bad_list=[]
    for w,i in tqdm(images_found.items()):
        if os.path.exists(os.path.join(path_JPEGS_GT,w+'.jpg'))==False:
            print(os.path.join(path_JPEGS_GT,w+'.jpg'))
            bad_list.append(w)
        elif os.path.exists(os.path.join(path_Anno_GT,w+'.xml'))==False:
            print(os.path.join(path_Anno_GT,w+'.xml'))
            bad_list.append(w)
    if len(bad_list)>0:
        print(f'FOUND {len(bad_list)} bad images to remove from metrics')
        for bad_item in bad_list:
            print(f'removing: {bad_item}')
            images_found.pop(bad_item)
        images_found={w:i for i,w in enumerate(images_found)}

    IMAGES=[]
    for w,i in tqdm(images_found.items()):
        img_i=os.path.join(path_JPEGS_GT,w+'.jpg')
        anno_i=os.path.join(path_Anno_GT,w+'.xml')
        try:
            assert os.path.exists(anno_i)
        except:
            print('Failed for:',anno_i)
        try:
            assert os.path.exists(img_i)
        except:
            print('Failed for:',img_i)
        tree_i=read_XML_quick(anno_i)
        W=int(tree_i.find('size').find('width').text)
        H=int(tree_i.find('size').find('height').text)

        #print(img_i)
        #image_i=cv2.imread(img_i)
        
        #W,H=image_i.shape[:2]
        IMAGES.append({
            "id"
            : i,
            "width"
            : W,
            "height"
            : H,
            "file_name"
            : img_i,
            })
    if os.path.exists(obj_names_path)==False:
            grep_result_file=os.path.join(os.path.dirname(path_Anno_GT),"grep_results_for_df.txt")
            cmd_i=f'grep -r "<name>" {path_Anno_GT} > {grep_result_file}'
            print('GREP STARTING')
            print(cmd_i)
            os.system(cmd_i)         
            f=open(grep_result_file,'r')
            f_read=f.readlines()
            f.close()
            print('GREP FINISHED')
            df_gr=pd.DataFrame(columns=['path','label','grep_line'])
            df_gr['grep_line']=f_read
            df_gr['label_i']=[w.split('<name>')[1].split('</name>')[0] for w in df_gr['grep_line']]

            found_names={name:i for i,name in enumerate(df_gr['label_i'].unique())}
            f=open(obj_names_path,'w')
            f.writelines([k+'\n' for k, v in sorted(found_names.items(), key=lambda item: item[1])])
            f.close()
    fo=open(obj_names_path,'r')
    fo_read=fo.readlines()
    fo.close()
    names={w.replace('\n',''):j for j,w in enumerate(fo_read)}
    print('names:',names)
    CATEGORIES=[]
    for k,v in names.items():
        CATEGORIES.append(        {
            'id': v,
            'name': k,
            'supercategory': k,
        })
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": IMAGES,
        "annotations": []
    }
	# use the COCO class to load and read the ground-truth annotations
    GT_ANNOTATION,id_i,result_lists_plot_gt=create_groundtruths_json(path_Anno_GT,obj_names_path,images_found,coco_output,path_JSON=os.path.join(os.path.dirname(path_Anno_Pred),'groundtruths.json'))
    print(GT_ANNOTATION)
    cocoAnnotation= COCO(annotation_file=GT_ANNOTATION)
    PRED_ANNOTATION,result_lists_plot_pred=create_predictions_json(path_Anno_Pred,obj_names_path,images_found,coco_output,path_JSON=os.path.join(os.path.dirname(path_Anno_Pred),'predictions.json'),id_i=id_i)
    print(PRED_ANNOTATION)
    cocovalPrediction = COCO(annotation_file=PRED_ANNOTATION)	

	# initialize the COCOeval object by passing the coco object with
	# ground truth annotations, coco object with detection results
    
    cocoEval = COCOeval(cocoAnnotation, cocovalPrediction, "bbox")
	
	# run evaluation for each image, accumulates per image results
	# display the summary metrics of the evaluation
    save_dir=os.path.dirname(path_Anno_Pred)
    save_dir=os.path.join(save_dir,'METRICS')
   
    if os.path.exists(save_dir)==False:
        os.makedirs(save_dir)
    else:
        print('MOVING previous METRICS')
        os.rename(save_dir,save_dir+'_OLDER_BACKEDUP_AT_'+str(time.time()).split('.')[0])
        os.system(f'rm -rf {save_dir}')
        os.makedirs(save_dir)
    shutil.copy(valid_list,save_dir)

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize=summarize
    cocoEval.summarize(cocoEval,names=names,result_file=os.path.join(save_dir,'mAP_metrics.txt'))
    shutil.move(GT_ANNOTATION,save_dir)
    shutil.move(PRED_ANNOTATION,save_dir)
    cm=ConfusionMatrix(len(names))
    df=pd.DataFrame(columns=['path_JPEGS_GT','path_Anno_GT','path_Anno_Pred','acc','tp','fp','fn','tn','prediction_box','ground_truth_box','matrix'])
    mdim=len(names)+1
    for k,v in tqdm(images_found.items()):
        #print(k,v)
        img_i=os.path.join(path_JPEGS_GT,k+'.jpg')
        anno_i=os.path.join(path_Anno_GT,k+'.xml')
        pred_i=os.path.join(path_Anno_Pred,k+'.xml')
        try:
            if v in result_lists_plot_pred.keys() and v in result_lists_plot_gt.keys():
                #print(result_lists_plot_pred[v])
                #print(result_lists_plot_gt[v])
                cm.process_batch(result_lists_plot_pred[v],result_lists_plot_gt[v])
                matrix_i=cm.process_batch_for_analysis(result_lists_plot_pred[v],result_lists_plot_gt[v])
                df.at[v,'path_JPEGS_GT']=img_i
                df.at[v,'path_Anno_GT']=anno_i
                df.at[v,'path_Anno_Pred']=pred_i
                df.at[v,'prediction_box']=result_lists_plot_pred[v]
                df.at[v,'ground_truth_box']=result_lists_plot_gt[v]
                df.at[v,'matrix']=matrix_i
                #print(matrix_i.shape)
                try:
                    matrix_j=np.matrix(matrix_i).reshape(mdim,mdim)
                except:
                    matrix_j=np.zeros((len(names.keys()) + 1, len(names.keys()) + 1)).reshape(mdim,mdim)
                df.at[v,'acc']=100.0*(matrix_j.diagonal().sum()/matrix_j.sum())
                df.at[v,'tp']=matrix_j.diagonal().sum()
                df.at[v,'fp']=matrix_j.T[-1].sum()
                df.at[v,'fn']=matrix_j[-1].sum()
                df.at[v,'tn']=matrix_j.sum()-matrix_j.diagonal().sum()-matrix_j.T[-1].sum()-matrix_j[-1].sum()
            elif v in result_lists_plot_gt.keys():
                matrix_i=np.zeros((len(names.keys()) + 1, len(names.keys()) + 1))
                df.at[v,'path_JPEGS_GT']=img_i
                df.at[v,'path_Anno_GT']=anno_i
                df.at[v,'path_Anno_Pred']='None'
                df.at[v,'prediction_box']=[]
                df.at[v,'ground_truth_box']=result_lists_plot_gt[v]
                df.at[v,'matrix']=matrix_i
                df.at[v,'acc']=0.0
                df.at[v,'tp']=0.0
                df.at[v,'fp']=0.0
                df.at[v,'fn']=np.matrix(result_lists_plot_gt[v]).shape[-1]
                df.at[v,'tn']=0.0
                pass
                #print('NOT FOUND in result_lists_plot_pred.keys(), but found in result_lits_gt.keys()',k)
                #cm.process_batch(np.zeros_like(np.arange(5)),result_lists_plot_gt[v])
            else:
                matrix_i=np.zeros((len(names.keys()) + 1, len(names.keys()) + 1))
                df.at[v,'path_JPEGS_GT']=img_i
                df.at[v,'path_Anno_GT']='None'
                df.at[v,'path_Anno_Pred']=pred_i
                df.at[v,'prediction_box']=result_lists_plot_pred[v]
                df.at[v,'ground_truth_box']=[]
                df.at[v,'matrix']=matrix_i
                df.at[v,'acc']=0.0
                df.at[v,'tp']=0.0
                df.at[v,'fp']=0.0
                df.at[v,'fn']=0.0
                df.at[v,'tn']=0.0
                pass
                #print('NOT FOUND in result_lists_plot_gt.keys(), but found in result_lits_pred.keys()',k)
                #cm.process_batch(result_lists_plot_pred[v],np.zeros_like(np.arange(6)))
        except:
            print(f'Issue with {k} {v}')
    df_filename=os.path.join(save_dir,'df_results.csv')
    df.to_csv(df_filename,index=None)

            

    cm.plot(save_dir=save_dir, names=list(names.keys()))
    #cm.print_matrix()
    cm.plot_full(save_dir=save_dir, names=list(names.keys()))
    cm.get_probability_of_detection(save_dir=save_dir, names=list(names.keys()))
    if show_results:
        try:
            os.system(f'xdg-open {save_dir}')
        except:
            pass
        # try:
        #     os.system(f'xdg-open {os.path.join(save_dir,"mAP_metrics.txt")}')
        # except:
        #     print('Trouble opening the mAP_metrics.txt')
        # try:
        #     os.system(f'xdg-open {os.path.join(save_dir,"Confusion_Matrix_Fractions.png")}')
        # except:
        #     print('Trouble opening the Confusion_Matrix_Fractions.png')
        # try:
        #     os.system(f'xdg-open {os.path.join(save_dir,"Confusion_Matrix_Numbers.png")}')
        # except:
        #     print('Trouble opening the Confusion_Matrix_Numbers.png')
        # try:
        #     os.system(f'xdg-open {os.path.join(save_dir,"probability_of_detection_metrics.txt")}')
        # except:
        #     print('Trouble opening the probability_of_detection_metrics.txt')
    if create_combine_gt_pred:
        COMBINED_GT_PRED=os.path.join(save_dir,'Annotations')
        os.makedirs(COMBINED_GT_PRED)
        COMBINED_GT_PRED_JPEGS=os.path.join(save_dir,'JPEGImages')
        os.makedirs(COMBINED_GT_PRED_JPEGS)
        GT_ANNOTATIONS=os.path.join(save_dir,'GROUND_TRUTH_Annotations')
        os.makedirs(GT_ANNOTATIONS)
        for row in tqdm(range(len(df))):
            anno_i=df['path_Anno_GT'].loc[row]
            anno_p=df['path_Anno_Pred'].loc[row]
            jpeg_i=df['path_JPEGS_GT'].loc[row]
            acc_i=df['acc'].loc[row]
            if os.path.exists(anno_i):
                shutil.copy(anno_i,COMBINED_GT_PRED)
                anno_j=os.path.join(COMBINED_GT_PRED,os.path.basename(anno_i))

                f=open(anno_j,'r')
                f_read=f.readlines()
                f.close()
                f_read_og=f_read
                f_read=[w for w in f_read if w.find('</annotation>')==-1]
                f_read=[w.replace('<name>','<name>GROUND_TRUTH__') for w in f_read]
                if os.path.exists(anno_p):
                    f=open(anno_p,'r')
                    f_pred=f.readlines()
                    f.close()
                    for j,line in enumerate(f_pred):
                        if line.find('<object>')!=-1:
                            break
                    f_pred=f_pred[j:]
                    f_comb=f_read+f_pred
                else:
                    f_comb=f_read+['</annotation>']
                f=open(anno_j,'w')
                f.writelines(f_comb)
                f.close()
                new_anno_j=os.path.join(COMBINED_GT_PRED,f'ACCURACY{np.round(acc_i,2)}__'.replace('.','p')+os.path.basename(anno_i))
                os.rename(anno_j,new_anno_j)
                shutil.copy(anno_i,os.path.join(GT_ANNOTATIONS,f'ACCURACY{np.round(acc_i,2)}__'.replace('.','p')+os.path.basename(anno_i)))
                if os.path.exists(jpeg_i):
                    shutil.copy(jpeg_i,COMBINED_GT_PRED_JPEGS)
                    jpeg_j=os.path.join(COMBINED_GT_PRED_JPEGS,os.path.basename(jpeg_i))
                    new_jpeg_j=os.path.join(COMBINED_GT_PRED_JPEGS,f'ACCURACY{np.round(acc_i,2)}__'.replace('.','p')+os.path.basename(jpeg_i))
                    os.rename(jpeg_j,new_jpeg_j)
        
    return cocoEval

def box_iou_calc(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, num_classes: int, CONF_THRESHOLD=0.25, IOU_THRESHOLD=0.45):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD

    def process_batch(self, detections, labels: np.ndarray):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        gt_classes = labels[:, 0].astype(np.int16)

        try:
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        except IndexError or TypeError:
            #print('detections are empty, end of process')
            for i, label in enumerate(labels):
                gt_class = gt_classes[i]
                self.matrix[self.num_classes, gt_class] += 1 #FN
            return

        detection_classes = detections[:, 5].astype(np.int16)

        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)

        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                       for i in range(want_idx[0].shape[0])]

        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        for i, label in enumerate(labels):
            gt_class = gt_classes[i]
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[detection_class, gt_class] += 1 #TP
            else:
                self.matrix[self.num_classes, gt_class] += 1 #background FP

        for i, detection in enumerate(detections):
            if not all_matches.shape[0] or ( all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0 ):
                detection_class = detection_classes[i]
                self.matrix[detection_class, self.num_classes] += 1 #background FN
    def process_batch_for_analysis(self, detections, labels: np.ndarray):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        matrix = np.zeros((self.num_classes + 1, self.num_classes + 1))
        gt_classes = labels[:, 0].astype(np.int16)

        try:
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        except IndexError or TypeError:
            #print('detections are empty, end of process')
            for i, label in enumerate(labels):
                gt_class = gt_classes[i]
                matrix[self.num_classes, gt_class] += 1 #FN
            return

        detection_classes = detections[:, 5].astype(np.int16)

        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)

        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                       for i in range(want_idx[0].shape[0])]

        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        for i, label in enumerate(labels):
            gt_class = gt_classes[i]
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                matrix[detection_class, gt_class] += 1 #TP
            else:
                matrix[self.num_classes, gt_class] += 1 #background FP

        for i, detection in enumerate(detections):
            if not all_matches.shape[0] or ( all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0 ):
                detection_class = detection_classes[i]
                matrix[detection_class, self.num_classes] += 1 #background FN
        return matrix

    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))
    def print_array(self,array):
        for i in range(self.num_classes+1):
            print(' '.join(map(str,array[i])))
    def plot(self, save_dir='', names=()):
        try:
            import seaborn as sn
            array = self.matrix / (self.matrix.sum(0).reshape(1, self.num_classes + 1) + 1E-6)  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.num_classes < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.num_classes  # apply names to ticklabels
            sn.heatmap(array, annot=self.num_classes < 30, annot_kws={"size": 8}, cmap='Greens', fmt='.2f', square=True,
                        xticklabels=names + ['background FP'] if labels else "auto",
                        yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'Confusion_Matrix_Fractions.png', dpi=250)
        except Exception as e:
            pass
    def get_probability_of_detection(self, save_dir='', names=()):
        try:
            array = self.matrix / (self.matrix.sum(0).reshape(1, self.num_classes + 1) + 1E-6)  # normalize
            array=list(np.diag(array))
            #print(array)
            text_summary="Probability of Detection per Class metrics\n"
            for i in range(self.num_classes):
                #print(i,text_summary)
                text_summary=text_summary+f'{names[i]}: {np.round(100*array[i],2)}% \n'
            print(text_summary)
            filename=os.path.join(save_dir,'probability_of_detection_metrics.txt')
            f=open(filename,'w')
            f.writelines(text_summary)
            f.close()
        except Exception as e:
            pass
    def plot_full(self, save_dir='', names=()):
        try:
            import seaborn as sn

            array = self.matrix.copy()
            array[array < 1] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.num_classes < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.num_classes  # apply names to ticklabels
            colormap=sn.color_palette('Greens')
            sn.heatmap(array, annot=self.num_classes < 30, annot_kws={"size": 8}, cmap=colormap, fmt='0.0f', square=True,
                        xticklabels=names + ['background FP'] if labels else "auto",
                        yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(Path(save_dir) / 'Confusion_Matrix_Numbers.png', dpi=250)
        except Exception as e:
            pass
    def print(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_Anno_Pred', type=str, default='None', help='path to prediction annos')
    parser.add_argument('--path_Anno_GT', type=str, default='None', help='path to ground truth annos')
    parser.add_argument('--path_JPEGS_GT', type=str, default='None', help='path to ground truth jpegs')
    parser.add_argument('--obj_names_path', type=str, default='obj.names', help='path to obj.names')
    parser.add_argument('--result_file', type=str, default='metric_results.txt', help='path to dumping mAP results')
    parser.add_argument('--valid_list',type=str,default='valid.txt',help='location of the validation list')
    parser.add_argument('--show_results',action='store_true',help='show results or not')
    parser.add_argument('--create_combine_gt_pred',action='store_true',help='create JPEGImages/Annotations that merge the predictions with ground truth and sort by accuracy')

    opt = parser.parse_args()
    path_Anno_Pred=opt.path_Anno_Pred
    path_Anno_GT=opt.path_Anno_GT
    path_JPEGS_GT=opt.path_JPEGS_GT
    obj_names_path=opt.obj_names_path
    result_file=opt.result_file
    valid_list=opt.valid_list
    show_results=opt.show_results
    create_combine_gt_pred=opt.create_combine_gt_pred

    if os.path.exists(path_Anno_Pred) and os.path.exists(path_Anno_GT) and os.path.exists(path_JPEGS_GT) and os.path.exists(valid_list):
        print('path_JPEGS_GT:',path_JPEGS_GT)
        print('path_Anno_GT:',path_Anno_GT)
        print('path_Anno_Pred:',path_Anno_Pred)
        print('obj_names_path:',obj_names_path)
        print('result_file',result_file)
        print('valid_list',valid_list)
        mycocoEval=compute_map(path_JPEGS_GT,path_Anno_GT,path_Anno_Pred,obj_names_path,result_file,valid_list,show_results,create_combine_gt_pred)
    else:
        print(path_Anno_Pred, path_Anno_GT,path_JPEGS_GT)
        print('os.path.exists(path_Anno_Pred),os.path.exists(path_Anno_GT),os.path.exists(path_JPEGS_GT)')
        print(os.path.exists(path_Anno_Pred), os.path.exists(path_Anno_GT),os.path.exists(path_JPEGS_GT))