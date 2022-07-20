import os
import argparse
import shutil
from tqdm import tqdm
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
def BndBox2Yolo(xmin,xmax,ymin,ymax,imgSize,classIndex):
    #print(imgSize)
    xcen=float((xmin+xmax))/2/imgSize[1]
    ycen=float((ymin+ymax))/2/imgSize[0]
    w=float((xmax-xmin))/imgSize[1]
    h=float((ymax-ymin))/imgSize[0]
    return classIndex,xcen,ycen,w,h
def create_img_list(path_change=None):
    '''path_change should be in the JPEGImages directory'''
    if path_change:
        return_dir=os.getcwd()
        os.chdir(path_change)
    else:
        return_dir=os.getcwd()
    f=open('img_list.txt','w')
    imgs=os.listdir()
    imgs=[w for w in imgs if w.find('.jpg')!=-1]
    cwd=os.getcwd()
    img_list=[os.path.join(cwd,w) for w in imgs]
    tmp=[f.writelines(w+'\n') for w in img_list]
    f.close()
    os.chdir(return_dir)
DEFAULT_ENCODING = 'utf-8'
XML_EXT = '.xml'
ENCODE_METHOD = DEFAULT_ENCODING
ap = argparse.ArgumentParser()
ap.add_argument("--config_path_test",type=str,default=None,help='config_path_test.cfg')
ap.add_argument("--data_path",type=str,default=None,help="obj.data")
ap.add_argument("--darknet",type=str,default=None,help="darknet path")
ap.add_argument("--best_weights",type=str,default=None,help="best_weights path")
ap.add_argument("--path_test_list_txt",type=str,default=None,help='path of .txt/.xml files for inference with yolo')
ap.add_argument("--path_result_list_txt",type=str,default=None,help="path for output of results")
ap.add_argument("--iou_thresh",type=str,default=None,help='iou threshold')
ap.add_argument("--thresh",type=str,default=None,help='threshold')
ap.add_argument("--filter_path",type=str,default=None,help='filter path')
ap.add_argument("--points",type=str,default=None,help='points for IoU')
args=ap.parse_args()

#config_path_test
if args.config_path_test!=None:
    config_path_test=args.config_path_test
    print("--config_path_test == {}".format(config_path_test))
else:
    print('WARNING! \n \t --config_path_test\t None specified')
#data_path
if args.data_path!=None:
    data_path=args.data_path
    print("--data_path == {}".format(data_path))
else:
    print('WARNING! \n \t --data_path\t None specified')
#darknet
if args.darknet!=None:
    darknet=args.darknet
    print("--darknet == {}".format(darknet))
    darknet_dir=os.path.dirname(darknet)
else:
    print('WARNING! \n \t --darknet\t None specified')

#best_weights
if args.best_weights!=None:
    best_weights=args.best_weights
    print("--best_weights == {}".format(best_weights))
else:
    print('WARNING! \n \t --best_weights\t None specified')
#thresh
if args.thresh!=None:
    thresh=args.thresh
    print("--thresh == {}".format(thresh))
else:
    print('WARNING! \n \t --thresh\t None specified')
#iou_thresh
if args.iou_thresh!=None:
    iou_thresh=args.iou_thresh
    print("--iou_thresh == {}".format(iou_thresh))
else:
    print('WARNING! \n \t --thresh\t None specified')
#filter_path
if args.filter_path!=None:
    filter_path=args.filter_path
    print("--filter_path == {}".format(filter_path))
else:
    print('WARNING! \n \t --filter_path\t None specified')
#points
if args.filter_path!=None:
    points=args.points
    print("--points == {}".format(points))
else:
    print('WARNING! \n \t --points\t None specified')
#path_test_list_txt
valid_list=False #check to make sure valid list of yolo .txt/.jpg files in same directory
jpg_list=True #check to make sure at least jpg list exists
if args.path_test_list_txt!=None:
    path_test_list_txt=args.path_test_list_txt
    print("--path_test_list_txt == {}".format(path_test_list_txt))
    f=open(path_test_list_txt,'r')
    f_read=f.readlines()
    f.close()
    for line in tqdm(f_read):
        if os.path.exists(line.replace('\n','').strip()):
            if os.path.exists(line.replace('.jpg','.txt').replace('\n','').strip()):
                valid_list=True
                pass
            else:
                print('Missing YOLO .txt file for:\n {}'.format(line))
                valid_list=False
                break
        else:
            print('This path does not exist:\n{}'.format(line))
            valid_list=False
            jpg_list=False
            break   
    if jpg_list and valid_list==False and os.path.exists(f_read[0].replace('JPEGImages','Annotations').replace('.jpg','.xml').replace('\n','').strip()):
        Yolo_Objs_path=os.path.join(os.path.dirname(os.path.dirname(f_read[0].replace('\n','').strip())),'Yolo_Objs')
        if  os.path.exists(Yolo_Objs_path)==False:
            os.makedirs(Yolo_Objs_path)
        print('Copying JPEGImages to Yolo_Objs')
        for line in tqdm(f_read):
            shutil.copy(line.replace('\n','').strip(),Yolo_Objs_path)
        print('Copying Annotations to Yolo_Objs')
        for line in tqdm(f_read):
            shutil.copy(line.replace('JPEGImages','Annotations').replace('.jpg','.xml').replace('\n','').strip(),Yolo_Objs_path)
        print('Creating YOLO txt files')
        xml_files=os.listdir(Yolo_Objs_path)
        xml_files=[os.path.join(Yolo_Objs_path,w) for w in xml_files if w.find('.xml')!=-1]
        found_names={}
        for path_anno_i in tqdm(xml_files):
            f=open(path_anno_i,'r')
            f_read=f.readlines()
            f.close()
            path_anno_dest_i=path_anno_i.replace('.xml','.txt')
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
                    if label not in found_names.keys():
                        found_names[label]=len(found_names.keys())+0
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))                   
                    classIndex,xcen,ycen,w,h=BndBox2Yolo(xmin,xmax,ymin,ymax,imgSize,found_names[label])
                    yolo_i=" ".join([str(yolo) for yolo in (int(classIndex),xcen,ycen,w,h)])
                    f=open(path_anno_dest_i,'a')
                    f.writelines(yolo_i+'\n')
                    f.close()
        create_img_list(Yolo_Objs_path)
        valid_list=True
        path_test_list_txt=os.path.join(Yolo_Objs_path,'img_list.txt')
        
else:
    print('WARNING! \n \t --path_result_list_txt\t None specified')
#path_result_list_txt
if args.path_result_list_txt!=None:
    path_result_list_txt=args.path_result_list_txt
    print("--path_result_list_txt == {}".format(path_result_list_txt))
else:
    print('WARNING! \n \t --path_result_list_txt\t None specified')

if path_result_list_txt and best_weights and darknet and darknet_dir and data_path and config_path_test and path_test_list_txt and valid_list and iou_thresh and thresh and points:
    f=open(data_path,'r')
    f_read=f.readlines()
    f.close()
    f_new=[]
    for line in tqdm(f_read):
        if line.find('valid=')!=-1:
            line="valid={}\n".format(path_test_list_txt)
        f_new.append(line)
    subdirectory_valid=os.path.dirname(os.path.dirname(path_test_list_txt))
    subdirectory_valid=subdirectory_valid.replace('/','_').rstrip('_').lstrip('_')
    new_data_path=data_path.replace('obj.data','obj_{}.data'.format(subdirectory_valid))
    f=open(new_data_path,'w')
    tmp=[f.writelines(w) for w in f_new]
    f.close()
    new_prediction_path=new_data_path.replace('.data','_prediction_mAP.sh')
    f=open(new_prediction_path,'w')
    f.writelines('config_path_test={}\n'.format(config_path_test))
    f.writelines('data_path={}\n'.format(new_data_path))
    f.writelines('darknet={}\n'.format(darknet))
    f.writelines('best_weights={}\n'.format(best_weights))
    f.writelines('path_result_list_txt={}\n'.format(path_result_list_txt))
    f.writelines('thresh={}\n'.format(thresh))
    f.writelines('iou_thresh={}\n'.format(iou_thresh))
    f.writelines('points={}\n'.format(points))
    f.writelines('cd {}\n'.format(darknet_dir))
    f.writelines('$darknet detector map $data_path $config_path_test $best_weights -thresh $thresh -iou_thresh $iou_thresh -points $points -dont_show -ext_output > $path_result_list_txt \n')
    f.writelines('filter_path={}\n'.format(filter_path))
    f.writelines('python3 $filter_path --path_result_list_txt=$path_result_list_txt \n')
    f.writelines('echo $(cat "{}") >> $path_result_list_txt\n'.format(new_prediction_path))
    f.close()
    os.system('bash {}'.format(new_prediction_path))
    os.system('xdg-open {}'.format(path_result_list_txt))

