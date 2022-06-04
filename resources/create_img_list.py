import os
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
