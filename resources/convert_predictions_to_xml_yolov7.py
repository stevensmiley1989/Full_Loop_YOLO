import os
import pandas as pd
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse 
ap = argparse.ArgumentParser()
ap.add_argument("--path_result_list_txt",type=str,default=None,help='The path to your Yolo predictions.txt list')
ap.add_argument("--path_predictions_folder",type=str,default=None,help="The path to your Yolo predictions folder to output Annotations/JPEGImages")
ap.add_argument("--path_objs_names",type=str,default=None,help="path to the obj.names file")
args=ap.parse_args()
if args.path_result_list_txt!=None:
    path_result_list_txt=args.path_result_list_txt
    print("--path_result_list_txt == {}".format(path_result_list_txt))
else:
    print('WARNING! \n \t --path_result_list_txt \t None specified')
if args.path_predictions_folder!=None:
    path_predictions_folder=args.path_predictions_folder
    print("--path_predictions_folder == {}".format(path_predictions_folder))
else:
    print('WARNING! \n \t --path_predictions_folder \t None specified')
    if args.path_result_list_txt!=None:
        path_predictions_folder=os.path.join(os.path.dirname(path_result_list_txt),'predictions')
        print("Using --path_predictions_folder == {}".format(path_predictions_folder))
if args.path_objs_names!=None:
    path_objs_names=args.path_objs_names
    print("--path_objs_names == {}".format(path_objs_names))
else:
    print('WARNING! \n \t --path_objs_names \t None specified')

f=open(path_objs_names,'r')
f_read=f.readlines()
f.close()
obj_names={k:v.replace('\n','').strip() for k,v in enumerate(f_read)}
print('obj_names = ',obj_names)
if os.path.exists(path_predictions_folder)==False:
    os.makedirs(path_predictions_folder)
else:
    #os.system('rm -rf {}'.format(path_predictions_folder))
    #os.makedirs(path_predictions_folder)
    pass
if os.path.exists(path_result_list_txt) and os.path.exists(path_predictions_folder):
    shutil.copy(path_result_list_txt,path_predictions_folder)
path_anno=os.path.join(path_predictions_folder,'Annotations')
path_jpegs=os.path.join(path_predictions_folder,'JPEGImages')
if os.path.exists(path_anno)==False:
    os.makedirs(path_anno)
else:
    os.system('rm -rf {}'.format(path_anno))
    os.makedirs(path_anno)
if os.path.exists(path_jpegs)==False:
    os.makedirs(path_jpegs)
else:
    os.system('rm -rf {}'.format(path_jpegs))
    os.makedirs(path_jpegs)

f=open(path_result_list_txt,'r')
f_read=f.readlines()
f.close()
start=0
end=0
j=0
first_sample=os.path.dirname(f_read[0].rstrip('/n').strip(' '))
PATH_JPEG_GT_DIR=first_sample#os.path.join(os.path.dirname(first_sample),"JPEGImages")
print(PATH_JPEG_GT_DIR)
print(os.path.exists(PATH_JPEG_GT_DIR))
df=pd.DataFrame(columns=['path_jpeg'])
for i,line in enumerate(f_read):
    jpg_i=line.rstrip('/n').strip()
    
    path_txt_i=os.path.basename(jpg_i).replace('.jpg','.txt')
    path_pred_txt_i=os.path.join(path_predictions_folder,os.path.join('labels',path_txt_i))
    if os.path.exists(path_pred_txt_i):
        shutil.copy(jpg_i,path_jpegs)
        df.at[j,'path_jpeg']=os.path.join(path_jpegs,os.path.basename(jpg_i))
        df.at[j,'path_txt']=path_pred_txt_i
        df.at[j,'path_anno']=os.path.join(path_anno,os.path.basename(jpg_i).replace('.jpg','.xml'))
        j+=1
df=df.dropna(axis=0).reset_index().drop('index',axis=1)
def BndBox2Yolo(xmin,xmax,ymin,ymax,imgSize,classIndex):
    '''converts PascalVOC to YOLO'''
    xcen=float((xmin+xmax))/2/imgSize[1]
    ycen=float((ymin+ymax))/2/imgSize[0]
    w=float((xmax-xmin))/imgSize[1]
    h=float((ymax-ymin))/imgSize[0]
    return classIndex,xcen,ycen,w,h

def BndBoxYolo2XML(xcen,ycen,w,h,width,height,classIndex):
    '''converts YOLO to PascalVOC'''
    xmax = int((xcen*width) + (w * width)/2.0)
    xmin = int((xcen*width) - (w * width)/2.0)
    ymax = int((ycen*height) + (h * height)/2.0)
    ymin = int((ycen*height) - (h * height)/2.0)
    return classIndex, xmin, xmax, ymin, ymax

for i in tqdm(range(len(df))):
    path_jpg=df['path_jpeg'][i]
    path_pred=df['path_txt'][i]
    path_xml=df['path_anno'][i]
    f=open(path_pred,'r')
    f_read=f.readlines()
    f.close()

    img_data=plt.imread(path_jpg)
    try:
        height,width,depth=img_data.shape
    except:
        print('EXCEPTION for {}'.format(path_pred))
        height,width=img_data.shape
        depth=2
    #print(height,width,depth)
    folder='JPEGImages'
    filename=os.path.basename(path_jpg)
    path=os.path.join(path_jpegs,filename)
    database='Unknown'
    
    f=open(path_xml,'w')
    f.writelines('<annotation>\n')
    f.writelines('\t <folder>{}</folder>\n'.format(folder))
    f.writelines('\t <filename>{}</filename>\n'.format(filename))
    f.writelines('\t <path>{}</path>\n'.format(path))
    f.writelines('\t<source>\n')
    f.writelines('\t\t<database>{}</database>\n'.format(database))
    f.writelines('\t</source>\n')
    f.writelines('\t<size>\n')
    f.writelines('\t\t<width>{}</width>\n'.format(width))
    f.writelines('\t\t<height>{}</height>\n'.format(height))
    f.writelines('\t\t<depth>{}</depth>\n'.format(depth))  
    f.writelines('\t</size>\n')
    f.writelines('\t<segmented>0</segmented>\n')

    
    for line in f_read:
        if len(line.split(' '))>5:
            name=int(line.split(' ')[0].strip())
            xcen=float(line.split(' ')[1])
            ycen=float(line.split(' ')[2])
            w=float(line.split(' ')[3])
            h=float(line.split(' ')[4])
            conf_i_f=float(line.split(' ')[5])

            name=obj_names[name]
            name, xmin, xmax, ymin, ymax=BndBoxYolo2XML(xcen,ycen,w,h,width,height,name)

            confidence=conf_i_f

            pose='Unspecified'
            truncated='0' 
            difficult='0'
            xmin=max(0,xmin)
            ymin=max(0,ymin)
            xmax=min(width,xmax)
            ymax=min(height,ymax)  
            f.writelines('\t<object>\n')
            f.writelines('\t\t<name>{}</name>\n'.format(name))
            f.writelines('\t\t<confidence>{}</confidence>\n'.format(confidence))
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


RAN_CUSTOM_METRICS=False
if os.path.exists(PATH_JPEG_GT_DIR):
    path_JPEGS_GT=os.path.abspath(PATH_JPEG_GT_DIR)
    print("SUCCESS:",path_JPEGS_GT)
    if os.path.exists(path_JPEGS_GT):
        print("SUCCESS:",path_JPEGS_GT)
        if os.path.exists(path_JPEGS_GT):
            path_Anno_GT=path_JPEGS_GT
            print("SUCCESS:",path_Anno_GT)
            if os.path.exists(path_anno):
                print("SUCCESS:",path_anno)
                path_Anno_Pred=path_anno
                result_file=os.path.abspath(os.path.join(os.path.dirname(path_Anno_Pred),'metric_results.txt'))
                if os.path.exists(result_file):
                    os.remove(result_file)
                if os.path.exists('resources/compute_mAP.py'):
                    print("SUCCESS:",'resources/compute_mAP.py')
                    cmd_i=f'python3 resources/compute_mAP.py --valid_list="{path_result_list_txt}" --path_Anno_Pred="{path_Anno_Pred}" --path_JPEGS_GT="{path_JPEGS_GT}" --path_Anno_GT="{path_Anno_GT}" --obj_names_path="{path_objs_names}" --result_file="{result_file}"'
                    os.system(cmd_i)
                    RAN_CUSTOM_METRICS=True
                else:
                    print('FAILED:',os.listdir())
                    print('cwd:',os.getcwd())
if RAN_CUSTOM_METRICS==False:
    print('Not able to generate custom metrics.')
    


    
    