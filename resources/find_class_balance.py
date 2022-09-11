import os
import pandas as pd
import time
import shutil
from tqdm import tqdm
from multiprocessing import Process
import argparse
import time
parser = argparse.ArgumentParser()

parser.add_argument('--path_Annotations', type=str, default=r"/media/steven/OneTouch4tb/coco2017/val2017/Annotations", help='path to annotatinos searching for')  # file/folder, 0 for webcam
parser.add_argument('--path_JPEGImages', type=str, default=None, help='path to jpegs searching for')
parser.add_argument('--path_Desired', type=str, default='Results', help='path to dump results')
parser.add_argument('--CLASS_I',type=str,default='car',help='object to find in annotation files')
parser.add_argument('--MAX_PER_CLASS',type=int,default=500,help='max number of files to pull from given directory if objects found')
opt = parser.parse_args()
path_Annotations=opt.path_Annotations
path_JPEGImages=opt.path_JPEGImages
path_Desired=opt.path_Desired
CLASS_I=opt.CLASS_I
MAX_PER_CLASS=opt.MAX_PER_CLASS
class GENERATE_SEARCH:
    def __init__(self,path_Annotations,path_JPEGImages,path_Desired,CLASS_I,MAX_PER_CLASS):
        self.path_Annotations=path_Annotations
        self.path_JPEGImages=path_JPEGImages
        
        if not(os.path.exists(path_Desired)):
            os.makedirs(path_Desired)
        self.path_Desired=os.path.abspath(os.path.join(path_Desired,CLASS_I))
        #print('self.path_Desired',self.path_Desired)
        if os.path.exists(self.path_Desired)==False:
            os.makedirs(self.path_Desired)
        self.CLASS_I=CLASS_I
        self.MAX_PER_CLASS=MAX_PER_CLASS
        self.WAIT_TIME=10
        self.search()
    def run_cmd(self,cmd_i):
        os.system(cmd_i)
    def check_length(self,file):
        if os.path.exists(file):
            f=open(file,'r')
            f_read=f.readlines()
            f.close()
            return len(f_read)
        else:
            return 0
    def kill_if_length(self,MAX_PER_CLASS,file,process_to_kill,path_search):
        terminated=False
        time_checked=time.time()
        time_interval=self.WAIT_TIME #seconds to check changes
        previous_length=-1000
        while terminated==False:
            current_length=self.check_length(file)
            if current_length>MAX_PER_CLASS:
                terminated=True
            if time.time()-time_checked>time_interval:
                time_checked=time.time()
                print(f'checking: previous_length={previous_length}, current_length={current_length}')
                if previous_length==current_length:
                    terminated=True
            previous_length=current_length
        process_to_kill.terminate()
        f=open(file,'r')
        f_read=f.readlines()
        f.close()
        if current_length<MAX_PER_CLASS:
            MAX_PER_CLASS=current_length
        prefix=path_search.replace('/','').replace('-','').replace('_','')
        self.search_results=file.replace('.txt',f'_MAX_PER_CLASS_{MAX_PER_CLASS}_checked_{prefix}.txt')
        f=open(self.search_results,'w')
        [f.writelines(w) for w in f_read[:MAX_PER_CLASS]]
        f.close()
        os.remove(file)

    def search(self):
        self.CLASS_FILE=f"{os.path.join(self.path_Desired,self.CLASS_I)}_Annotation_List.txt"
        cmd_i=f"cd '{self.path_Desired}' && grep -r -l {self.CLASS_I} {self.path_Annotations} > '{self.CLASS_FILE}'"
        #print(cmd_i)
        self.p1=Process(target=self.run_cmd,args=(cmd_i,))
        self.p1.start()
        self.kill_if_length(self.MAX_PER_CLASS,self.CLASS_FILE,self.p1,self.path_Annotations)
        self.create_csv()
    def create_csv(self):
        #df=pd.DataFrame(columns=['CLASS_I','path_Annotations','path_JPEGImages'])
        f=open(self.search_results,'r')
        f_read=f.readlines()
        f.close()
        #print(len(f_read))
        f_read=[w.replace('\n','').replace(' ','') for w in f_read]
        df=pd.DataFrame(f_read,columns=['path_Annotations'])
        df['path_JPEGImages']=df['path_Annotations'].copy()
        
        df['CLASS_I']=df['path_Annotations'].copy()
        if str(type(self.path_JPEGImages)).find('str')==-1:
            self.path_JPEGImages=self.path_Annotations.replace('Annotations','JPEGImages')
        df['path_JPEGImages']=[os.path.join(self.path_JPEGImages,os.path.basename(w).replace('.xml','.jpg')) for w in df['path_Annotations']]
        df['CLASS_I']=self.CLASS_I
        df['basename']=[os.path.basename(w) for w in df['path_Annotations']]
        self.df_filename=self.search_results.replace('.txt','.csv')
        bad_list=[]
        for row in range(len(df)):
            jpeg_i=df['path_JPEGImages'].loc[row]
            if not(os.path.exists(jpeg_i)):
                if row not in bad_list:
                    bad_list.append(row)
        for row in range(len(df)):
            anno_i=df['path_Annotations'].loc[row]
            if not(os.path.exists(anno_i)):
                if row not in bad_list:
                    bad_list.append(row)    
        self.df=df.copy()
        self.df.drop(bad_list,inplace=True)
        self.df=self.df.reset_index().drop('index',axis=1)
        self.df.to_csv(self.df_filename) 
def SEARCH_LISTS(dic_anno_jpeg_path,list_targets,MAX_PER_CLASS,path_Desired):
    print('SEARCHING_LISTS')
    mysearches=[]
    for i,(anno_i,jpeg_i) in enumerate(dic_anno_jpeg_path.items()):
        for j,target_j in enumerate(list_targets):
            mysearch=GENERATE_SEARCH(anno_i,jpeg_i,path_Desired,target_j,MAX_PER_CLASS)
            print(mysearch.df_filename)
            mysearches.append(mysearch)
            if len(mysearches)==1:
                df_all_results=mysearch.df
            else:
                df_all_results=pd.concat([df_all_results,mysearch.df],ignore_index=True)
                df_all_results=df_all_results.drop_duplicates('basename').reset_index().drop('index',axis=1)
    return df_all_results
def FILTER_LISTS(df,MAX_PER_CLASS):
    print('FILTERING LISTS')
    unique_classes=df['CLASS_I'].unique()
    df_new=pd.DataFrame()
    for i,class_i in enumerate(unique_classes):
        df_i=df[df['CLASS_I']==class_i].copy()
        len_df_i=len(df_i)
        df_i=df_i.sample(n=min(len_df_i,MAX_PER_CLASS),random_state=42)
        print(f'CLASS_I={class_i} found in {len(df_i)} FILES to be used.')
        if i==0:
            df_new=df_i
        else:
            df_new=pd.concat([df_new,df_i],ignore_index=True)
    df_new=df_new.reset_index().drop('index',axis=1)
    return df_new
def CREATE_NEW(df,path_Desired):
    print('CREATING NEW FILES')
    if len(df)==0:
        return print('Nothing Found')
    if os.path.exists(path_Desired)==False:
        os.makedirs(path_Desired)
    else:
        #os.system(f'rm -rf {path_Desired}')
        path_Desired=path_Desired+"_"+str(time.time()).split('.')[0]
        os.makedirs(path_Desired)
    path_Annotations_new=os.path.join(path_Desired,'Annotations')
    os.makedirs(path_Annotations_new)
    path_JPEGImages_new=os.path.join(path_Desired,'JPEGImages')
    os.makedirs(path_JPEGImages_new)

    for row in tqdm(range(len(df))):
        anno_i=df['path_Annotations'][row]
        jpeg_i=df['path_JPEGImages'][row]
        shutil.copy(anno_i,path_Annotations_new)
        shutil.copy(jpeg_i,path_JPEGImages_new)
    df_new_filename=os.path.join(path_Desired,'df_results.csv')
    df.to_csv(df_new_filename)
    return df,os.path.abspath(path_Annotations_new),os.path.abspath(path_JPEGImages_new)
def JUST_DO_IT(dic_anno_jpeg_path,list_targets,MAX_PER_CLASS,path_Desired):
    myresults,path_Annotations_new,path_JPEGImages_new=CREATE_NEW(FILTER_LISTS(SEARCH_LISTS(dic_anno_jpeg_path,list_targets,MAX_PER_CLASS,path_Desired),MAX_PER_CLASS),path_Desired)
    return myresults,path_Annotations_new,path_JPEGImages_new
if __name__=='__main__':
    dic_anno_jpeg_paths={r"/media/steven/OneTouch4tb/coco2017/val2017/Annotations":None,
    r"/media/steven/OneTouch4tb/coco2017/train2017/Annotations":None,
    r'/media/steven/Elements/coco2017_VOC2012/custom_with_VOC2012_coco2017val/Augmentations/train/Annotations':None}
    list_targets=['people','person','car']
    myresults,path_Annotations_new,path_JPEGImages_new=JUST_DO_IT(dic_anno_jpeg_paths,list_targets,MAX_PER_CLASS,path_Desired)
    print(myresults)
    print(path_Annotations_new)
    print(path_JPEGImages_new)
