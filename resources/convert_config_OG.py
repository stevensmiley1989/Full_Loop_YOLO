import os
import shutil
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--CONFIGPATH",type=str,default=None,help='CONFIGPATH')
ap.add_argument("--CONFIGPATH_OG",type=str,default=None,help="CONFIGPATH_OG")
ap.add_argument("--OBJ_NAMES",type=str,default=None,help="path to the obj.names file")
args=ap.parse_args()
if args.CONFIGPATH!=None:
    CONFIGPATH=args.CONFIGPATH
    print("--CONFIGPATH == {}".format(CONFIGPATH))
else:
    print('WARNING! \n \t --CONFIGPATH \t None specified')
if args.CONFIGPATH_OG!=None:
    CONFIGPATH_OG=args.CONFIGPATH_OG
    print("--CONFIGPATH_OG == {}".format(CONFIGPATH_OG))
else:
    print('WARNING! \n \t --CONFIGPATH_OG \t None specified')
if args.OBJ_NAMES!=None:
    OBJ_NAMES=args.OBJ_NAMES
    print("--OBJ_NAMES == {}".format(OBJ_NAMES))
else:
    print('WARNING! \n \t --OBJ_NAMES\t None specified')
if os.path.exists(CONFIGPATH_OG)==False and os.path.exists(CONFIGPATH):
    print('THE ORIGINAL DID NOT EXIST FOR CONFIGPATH_OG=\n {}'.format(CONFIGPATH_OG))
    print('COPYING CONFIGPATH TO CONFIGPATH_OG')
    shutil.copy(CONFIGPATH,CONFIGPATH_OG)
    print("SUCCESS")
f=open(CONFIGPATH_OG,'r')
f_read=f.readlines()
f.close()
f_new=[]
for line in f_read:
    if line.find('YOLO.CLASSES')!=-1:
        print('CHANGING old line: \n {}'.format(line))
        line='__C.YOLO.CLASSES              = "{}"\n'.format(OBJ_NAMES)
        print('SUCCESSFULLY changed to new line: \n {}'.format(line))
    f_new.append(line)
f=open(CONFIGPATH,'w')
tmp=[f.writelines(w) for w in f_new]
f.close()
print('SUCCESSFULLY wrote new CONFIGPATH at {}'.format(CONFIGPATH))