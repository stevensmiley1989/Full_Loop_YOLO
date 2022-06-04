import os
from pprint import pprint
#from tqdm import tqdm
def switch_paths(path_of_interest,basePath_bad,basePath_good):
	cwd=os.getcwd()
	dirs_found=[]
	if os.path.exists(path_of_interest):
		#print("path_of_interest: \n {}".format(path_of_interest))
		#print("basePath_bad:  \n {}".format(basePath_bad))
		#print("basePath_good: \n {} \n".format(basePath_good))
		os.chdir(path_of_interest)
		#print("SEARCHING at {}".format(path_of_interest))
		path_of_interest_files=list(os.listdir(path_of_interest))
		#pprint(path_of_interest_files)
		#[print(w) for w in path_of_interest_files]
		for file_i in path_of_interest_files:
			#print(file_i)
			file=os.path.join(path_of_interest,file_i)
			if os.path.isdir(file) and file.find('pycache')==-1 and file.find('predictions')==-1:
				#print(file)
				dirs_found.append(file)
				#break
			elif file.find('.weights')!=-1 or file.find('pycache')!=-1 or file.find('.avi')!=-1 or file.find('predictions')!=-1 or file.find('.MP4')!=-1 or file.find('.mp4')!=-1:
				pass
				#print(file)
				#break
			else:
				#print(file)
				f=open(file,'r')
				f_read=f.readlines()
				f.close()
				f_new=[]

				for line in f_read:
					if line.find(basePath_bad)!=-1:
						#print('bad line: ',line,'\t for file: ',file,'\n')
						#break
						pass
					line=line.replace(basePath_bad,basePath_good)
					f_new.append(line)
				f=open(file,'w')
				tmp=[f.writelines(line) for line in f_new]
				f.close()
		if len(dirs_found)!=0:
			for dir in dirs_found:
				#print('SEARCHING at {}'.format(dir))
				dir_list=os.listdir(dir)
				for file in dir_list:
					if file.find('.weights')!=-1 or file.find('pycache')!=-1 or file.find('predictions')!=-1:
						pass
					else:
						#print('file found',file)
						file=os.path.join(dir,file)
						#print('combined file',file)
						f=open(file,'r')
						f_read=f.readlines()
						f.close()
						f_new=[]
						for line in f_read:
							if line.find(basePath_bad)!=-1:
								#print('bad line: ',line,'\t for file: ',file,'\n')
								pass
							line=line.replace(basePath_bad,basePath_good)
							f_new.append(line)
						f=open(file,'w')
						tmp=[f.writelines(line) for line in f_new]
						f.close()
	
	else:
		print('path_of_interest does not exist')
	os.chdir(cwd)
	return True
def check_current():
	good_case=os.getcwd()
	good_case=good_case.split('Elements')[0]
	good_case=good_case+'Elements'
	#print('good_case',good_case)
	path_of_interest=os.path.join(os.getcwd(),'libs')
	saved_settings_files=[os.path.join(path_of_interest,w) for w in os.listdir(path_of_interest) if w.find('cache')==-1 and w.find('SAVED')!=-1 and w.find('.py')!=-1]
	if len(saved_settings_files)!=0:
		saved_settings_sample=saved_settings_files[0]
		f=open(saved_settings_sample,'r')
		f_read=f.read()
		f.close()
		if f_read.find('/home/steven/Elements')!=-1:
			bad_case='/home/steven/Elements'
		elif f_read.find('/media/pi/Elements')!=-1:
			bad_case='/media/pi/Elements'
		elif f_read.find('/media/steven/Elements')!=-1:
			bad_case='/media/steven/Elements'
		else:
			bad_case='UNKNOWN'
	else:
		bad_case=good_case


	#bad_case='/home/steven/Elements'
	#bad_case='/media/steven/Elements'
	#bad_case='/media/pi/Elements'
	#print('bad_case',bad_case)
	return good_case,bad_case

def switch_config(config_path):
	good_case,bad_case=check_current()
	if good_case=='/media/pi/Elements':
		#print('OPTION1')
		basePath_bad='/home/steven/Elements'
		basePath_good='/media/pi/Elements'
		path_of_interest=basePath_good+'/Drone_Images/Yolo/'+config_path
		darknetPath_bad='/home/steven/darknet'
		darknetPath_good='/home/pi/YOLOv4_darknet/darknet'
		conf_bad='/home/steven/Elements/create_yolo_config'
		conf_good='/media/pi/Elements/create_yolo_config'
		switch_paths(path_of_interest,basePath_bad,basePath_good)
		switch_paths(path_of_interest,darknetPath_bad,darknetPath_good)
		switch_paths(path_of_interest,conf_bad,conf_good)
		#print('OPTION5')
		basePath_bad='/media/steven/Elements'
		basePath_good='/media/pi/Elements'
		path_of_interest=basePath_good+'/Drone_Images/Yolo/'+config_path
		darknetPath_bad='/home/steven/darknet'
		darknetPath_good='/home/pi/YOLOv4_darknet/darknet'
		conf_bad='/media/steven/Elements/create_yolo_config'
		conf_good='/media/pi/Elements/create_yolo_config'
		switch_paths(path_of_interest,basePath_bad,basePath_good)
		switch_paths(path_of_interest,darknetPath_bad,darknetPath_good)
		switch_paths(path_of_interest,conf_bad,conf_good) 
	elif good_case=='/home/steven/Elements':
		#print('OPTION2')
		basePath_good='/home/steven/Elements'
		basePath_bad='/media/pi/Elements'
		path_of_interest=basePath_good+'/Drone_Images/Yolo/'+config_path
		darknetPath_good='/home/steven/darknet'
		darknetPath_bad='/home/pi/YOLOv4_darknet/darknet'
		conf_good='/home/steven/Elements/create_yolo_config'
		conf_bad='/media/pi/Elements/create_yolo_config'
		switch_paths(path_of_interest,basePath_bad,basePath_good)
		switch_paths(path_of_interest,darknetPath_bad,darknetPath_good)
		switch_paths(path_of_interest,conf_bad,conf_good)  
		#print('OPTION6')
		basePath_good='/home/steven/Elements'
		basePath_bad='/media/steven/Elements'
		path_of_interest=basePath_good+'/Drone_Images/Yolo/'+config_path
		darknetPath_good='/home/steven/darknet'
		darknetPath_bad='/home/steven/darknet'
		conf_good='/home/steven/Elements/create_yolo_config'
		conf_bad='/media/steven/Elements/create_yolo_config'
		switch_paths(path_of_interest,basePath_bad,basePath_good)
		switch_paths(path_of_interest,darknetPath_bad,darknetPath_good)
		switch_paths(path_of_interest,conf_bad,conf_good) 
	elif good_case=='/media/steven/Elements':
		#print('OPTION3')
		basePath_good='/media/steven/Elements'
		basePath_bad='/media/pi/Elements'
		path_of_interest=basePath_good+'/Drone_Images/Yolo/'+config_path
		darknetPath_good='/home/steven/darknet'
		darknetPath_bad='/home/pi/YOLOv4_darknet/darknet'
		conf_good='/media/steven/Elements/create_yolo_config'
		conf_bad='/media/pi/Elements/create_yolo_config'
		switch_paths(path_of_interest,basePath_bad,basePath_good)
		switch_paths(path_of_interest,darknetPath_bad,darknetPath_good)
		switch_paths(path_of_interest,conf_bad,conf_good) 
		#print('OPTION4')
		basePath_bad='/home/steven/Elements'
		basePath_good='/media/steven/Elements'
		path_of_interest=basePath_good+'/Drone_Images/Yolo/'+config_path
		darknetPath_bad='/home/steven/darknet'
		darknetPath_good='/home/steven/darknet'
		conf_bad='/home/steven/Elements/create_yolo_config'
		conf_good='/media/steven/Elements/create_yolo_config'
		switch_paths(path_of_interest,basePath_bad,basePath_good)
		switch_paths(path_of_interest,darknetPath_bad,darknetPath_good)
		switch_paths(path_of_interest,conf_bad,conf_good) 


def switch_scripts():
	good_case,bad_case=check_current()
	if good_case=='/media/pi/Elements':
		#print('OPTION1')
		basePath_bad='/home/steven/Elements'
		basePath_good='/media/pi/Elements'
		path_of_interest=os.path.join(os.getcwd(),'libs')
		darknetPath_bad='/home/steven/darknet'
		darknetPath_good='/home/pi/YOLOv4_darknet/darknet'
		conf_bad='/home/steven/Elements/create_yolo_config'
		conf_good='/media/pi/Elements/create_yolo_config'
		switch_paths(path_of_interest,basePath_bad,basePath_good)
		switch_paths(path_of_interest,darknetPath_bad,darknetPath_good)
		switch_paths(path_of_interest,conf_bad,conf_good)
		#print('OPTION3')
		basePath_good='/media/steven/Elements'
		basePath_bad='/media/pi/Elements'
		path_of_interest=os.path.join(os.getcwd(),'libs')
		darknetPath_good='/home/steven/darknet'
		darknetPath_bad='/home/pi/YOLOv4_darknet/darknet'
		conf_good='/media/steven/Elements/create_yolo_config'
		conf_bad='/media/pi/Elements/create_yolo_config'
		switch_paths(path_of_interest,basePath_bad,basePath_good)
		switch_paths(path_of_interest,darknetPath_bad,darknetPath_good)
		switch_paths(path_of_interest,conf_bad,conf_good) 
	elif good_case=='/home/steven/Elements':
		#print('OPTION2')
		basePath_good='/home/steven/Elements'
		basePath_bad='/media/pi/Elements'
		path_of_interest=os.path.join(os.getcwd(),'libs')
		darknetPath_good='/home/steven/darknet'
		darknetPath_bad='/home/pi/YOLOv4_darknet/darknet'
		conf_good='/home/steven/Elements/create_yolo_config'
		conf_bad='/media/pi/Elements/create_yolo_config'
		switch_paths(path_of_interest,basePath_bad,basePath_good)
		switch_paths(path_of_interest,darknetPath_bad,darknetPath_good)
		switch_paths(path_of_interest,conf_bad,conf_good) 
		#print('OPTION5')
		basePath_good='/home/steven/Elements'
		basePath_bad='/media/steven/Elements'
		path_of_interest=os.path.join(os.getcwd(),'libs')
		darknetPath_good='/home/steven/darknet'
		darknetPath_bad='/home/steven/darknet'
		conf_good='/home/steven/Elements/create_yolo_config'
		conf_bad='/media/steven/Elements/create_yolo_config'
		switch_paths(path_of_interest,basePath_bad,basePath_good)
		switch_paths(path_of_interest,darknetPath_bad,darknetPath_good)
		switch_paths(path_of_interest,conf_bad,conf_good)  	
	elif good_case=='/media/steven/Elements':
		#print('OPTION3')
		basePath_good='/media/steven/Elements'
		basePath_bad='/media/pi/Elements'
		path_of_interest=os.path.join(os.getcwd(),'libs')
		darknetPath_good='/home/steven/darknet'
		darknetPath_bad='/home/pi/YOLOv4_darknet/darknet'
		conf_good='/media/steven/Elements/create_yolo_config'
		conf_bad='/media/pi/Elements/create_yolo_config'
		switch_paths(path_of_interest,basePath_bad,basePath_good)
		switch_paths(path_of_interest,darknetPath_bad,darknetPath_good)
		switch_paths(path_of_interest,conf_bad,conf_good) 
		#print('OPTION4')
		basePath_bad='/home/steven/Elements'
		basePath_good='/media/steven/Elements'
		path_of_interest=os.path.join(os.getcwd(),'libs')
		darknetPath_bad='/home/steven/darknet'
		darknetPath_good='/home/steven/darknet'
		conf_bad='/home/steven/Elements/create_yolo_config'
		conf_good='/media/steven/Elements/create_yolo_config'
		switch_paths(path_of_interest,basePath_bad,basePath_good)
		switch_paths(path_of_interest,darknetPath_bad,darknetPath_good)
		switch_paths(path_of_interest,conf_bad,conf_good) 

    
if __name__=='__main__':
	config_path='tiny_yolo-Elements_transporter9_w640_h640_d0_c1'
	switch_config(config_path)
