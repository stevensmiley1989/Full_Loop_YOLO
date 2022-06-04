import os
import pandas as pd
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from targets_dic import targets_dic
from targets_dic import path_annotations

def FIND_REPLACE_TARGETS(targets_dic,path_annotations,path_annotations_new=None):
	targets_keep=targets_dic.values()
	#targets_keep=['transporter9']
	#path_annotations="/media/steven/Elements/Drone_Videos/Combined_Transporter9_withVisDrone_noanno/Annotations"
	if path_annotations_new==None:
		path_annotations_new=path_annotations+"_filtered"
	if os.path.exists(path_annotations_new)==False:
		os.makedirs(path_annotations_new)
	else:
		os.system('sudo rm -rf {}'.format(path_annotations_new))
		os.makedirs(path_annotations_new)

	annotation_files=os.listdir(path_annotations)
	annotation_files=[w for w in annotation_files if w.find('.xml')!=-1]
	start=False
	keep=True
	for file in tqdm(annotation_files):
		file_og=os.path.join(path_annotations,file)
		file_new=os.path.join(path_annotations_new,file)
		f=open(file_og,'r')
		f_read=f.readlines()
		f.close()
		f_new=[]
		for i,line in enumerate(f_read):
			if line.find('<object>')!=-1:
				start=True
				name=f_read[i+1].split('<name>')[1].split('</name>')[0].replace('\n','').strip()
				name_og=name.strip()
				if name in list(targets_dic.keys()):
					name=targets_dic[name]
				if name in list(targets_keep):
					#print('BEFORE: \t',f_read[i+1])
					keep=True
					f_read[i+1]=f_read[i+1].replace(name_og.strip(),name.strip()) #find/replace label
					#print('AFTER: \t',f_read[i+1])
				else:
					keep=False
			if line.find('</object>')!=-1:
				start=False
				if keep:
					f_new.append(line)
				else:
					pass
			elif start and keep:
				f_new.append(line)
			elif start and keep==False:
				pass
			else:
				f_new.append(line)
		f=open(file_new,'w')
		tmp=[f.writelines(w) for w in f_new]
		f.close()
if __name__=='__main__':
	FIND_REPLACE_TARGETS(targets_dic,path_annotations)
