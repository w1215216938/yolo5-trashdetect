import os


def rename():
    path = r"D:\xianniudownload\yolov5-master\data\images"  # 文件位置
    filelist = os.listdir(path)
    for files in filelist:
        olddir = os.path.join(path, files)
        if os.path.isdir(olddir):
            continue
        filename = files.split('.')[0]  # 根据自己的文件格式进行分割
        filetype1 = files.split('.')[1]
        newdir = os.path.join(path, filename + "." + filetype1.lower())
        os.rename(olddir, newdir)


rename()