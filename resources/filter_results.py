import os
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--path_result_list_txt",type=str,default=None,help="path for output of results")
args=ap.parse_args()
if args.path_result_list_txt!=None:
    path_result_list_txt=args.path_result_list_txt
    print("--path_result_list_txt == {}".format(path_result_list_txt))
else:
    print('WARNING! \n \t --path_result_list_txt\t None specified')
if path_result_list_txt:
    f=open(path_result_list_txt,'r')
    f_read=f.readlines()
    f.close()
    f_new=[]
    for line in f_read:
        if line.find('rank')==-1:
            f_new.append(line)
    f=open(path_result_list_txt,'w')
    tmp=[f.writelines(w) for w in f_new]
    f.close()