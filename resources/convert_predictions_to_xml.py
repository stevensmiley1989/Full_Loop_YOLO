import os
import pandas as pd
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse 
ap = argparse.ArgumentParser()
ap.add_argument("--path_result_list_txt",type=str,default=None,help='The path to your Yolo predictions.txt list')
ap.add_argument("--path_predictions_folder",type=str,default=None,help="The path to your Yolo predictions folder to output Annotations/JPEGImages")
ap.add_argument("--path_compute_mAP",type=str,default="None")
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
#path_result_list_txt="/media/steven/Elements//Drone_Images/Yolo/tiny_yolo-Elements_upto_5_15_single_w1920_h1056_d4_c1/predictions.txt"
#path_predictions_folder=os.path.join(os.path.dirname(path_result_list_txt),'predictions')
#answer1=input('Does this look good? Yes or No')
#if answer1[0].lower().strip()=='y':
if os.path.exists(path_predictions_folder)==False:
    os.makedirs(path_predictions_folder)
else:
    os.system('rm -rf {}'.format(path_predictions_folder))
    os.makedirs(path_predictions_folder)
path_anno=os.path.join(path_predictions_folder,'Annotations')
path_jpegs=os.path.join(path_predictions_folder,'JPEGImages')
if os.path.exists(path_anno)==False:
    os.makedirs(path_anno)
if os.path.exists(path_jpegs)==False:
    os.makedirs(path_jpegs)

f=open(path_result_list_txt,'r')
f_read=f.readlines()
f.close()
start=0
end=0
j=0

df=pd.DataFrame(columns=['start','end','path_jpeg'])
for i,line in enumerate(f_read[:-1]):
    if line.find('Enter Image Path')!=-1:
        start=i
        if start!=0:
            end=start
            df.at[j-1,'end']=end
        df.at[j,'start']=start
        if f_read[i+1].find('.jpg')!=-1:
            df.at[j,'path_jpeg']=f_read[i+1].split('.jpg')[0]+'.jpg' #Finds where the jpg is
        elif f_read[i+2].find('.jpg')!=-1:
            df.at[j,'path_jpeg']=f_read[i+2].split('.jpg')[0]+'.jpg' #TD 
        j+=1
df=df.dropna(axis=0).reset_index().drop('index',axis=1)
for i in tqdm(range(len(df))):
    start=df['start'][i]
    end=df['end'][i]
    path_jpg=df['path_jpeg'][i]
    first_sample=os.path.dirname(path_jpg)
    PATH_JPEG_GT_DIR=os.path.join(os.path.dirname(first_sample),"JPEGImages")
    #print(PATH_JPEG_GT_DIR)
    #print(os.path.exists(PATH_JPEG_GT_DIR))
    img_data=plt.imread(path_jpg)
    if len(img_data.shape)==3:
        height,width,depth=img_data.shape
    elif len(img_data.shape)==2:
        height,width=img_data.shape
        depth=1
    #print(height,width,depth)
    folder='JPEGImages'
    filename=path_jpg.split('/')[-1]
    path=os.path.join(path_jpegs,filename)
    shutil.copy(path_jpg,path)
    path=path_jpg
    database='Unknown'
    path_xml=os.path.join(path_anno,filename.replace('.jpg','.xml'))
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
    for line in f_read[start:end]:
        if line.find('left_x')!=-1:
            name=line.split(':')[0].strip()
            left_x=int(line.split('left_x:')[1].split('top_y')[0].strip())
            top_y=int(line.split('top_y:')[1].split('width:')[0].strip())
            width_i=int(line.split('width:')[1].split('height:')[0].strip())
            height_i=int(line.split('height:')[1].split(')')[0].strip())  
            confidence=line.split(':')[1].split('%')[0].strip()
            pose='Unspecified'
            truncated='0' 
            difficult='0'
            xmin=max(0,left_x)
            ymin=max(0,top_y)
            xmax=min(width,left_x+width_i)
            ymax=min(height,top_y+height_i)  
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
        if os.path.exists(path_JPEGS_GT.replace('JPEGImages','Annotations')):
            path_Anno_GT=os.path.abspath(path_JPEGS_GT.replace('JPEGImages','Annotations'))
            print("SUCCESS:",path_Anno_GT)
            if os.path.exists(path_anno):
                path_Anno_Pred=path_anno
                result_file=os.path.abspath(os.path.join(os.path.dirname(path_Anno_Pred),'metric_results.txt'))
                print("SUCCESS:",path_Anno_Pred)
                if os.path.exists(result_file):
                    os.remove(result_file)
                if os.path.exists(args.path_compute_mAP):
                    print("SUCCESS:",args.path_compute_mAP)
                    cmd_i=f'python3 {args.path_compute_mAP} --path_Anno_Pred="{path_Anno_Pred}"  --path_JPEGS_GT="{path_JPEGS_GT}" --path_Anno_GT="{path_Anno_GT}" --result_file="{result_file}"'
                    os.system(cmd_i)
                    RAN_CUSTOM_METRICS=True
                else:
                    print('FAILED:',os.listdir())
                    print('cwd:',os.getcwd())                    
if RAN_CUSTOM_METRICS==False:
    print('Not able to generate custom metrics.')

    
    