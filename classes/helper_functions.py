import numpy as np
import os
import pickle as pkl

class helper_functions:
    def __init__(self):
        init=0

    def read_coordinate_file(self,file_name):
        f = open(file_name, "r+")
        arr = []
        index = 0
        for lines in f:
            dat = lines.split(",")
            linearr = []
            for i in range(0, len(dat) - 1):
                dat2 = dat[i].split(":")
                linearr.append((float(dat2[0]), float(dat2[1])))
            arr.append(linearr)
            index = index + 1
        f.close
        return np.asarray(arr)  # ,np.asarray(pix)

    def read_coordinate_file_w_timestamps(self,file_name):
        f = open(file_name, "r+")
        arr = []
        time_arr=[]
        index = 0
        for lines in f:
            timstamp=lines[0:29]
            time_arr.append(timstamp)
            line=lines[29:]
            dat = line.split(":")
            linearr = []
            for i in range(0, len(dat) - 1):
                dat2 = dat[i].split(",")
                try:
                    linearr.append((float(dat2[0]), float(dat2[1])))
                except:
                    pass
            arr.append(linearr)
            index = index + 1
        f.close
        return np.asarray(time_arr),np.asarray(arr)  # ,np.asarray(pix)
    def read_coordinate_file_w_timestamps(self,file_name):
        f = open(file_name, "r+")
        arr = []
        time_arr=[]
        index = 0
        for lines in f:
            timstamp=lines[0:29]
            time_arr.append(timstamp)
            line=lines[29:]
            dat = line.split(":")
            linearr = []
            for i in range(0, len(dat) - 1):
                dat2 = dat[i].split(",")
                try:
                    linearr.append((float(dat2[0]), float(dat2[1])))
                except:
                    pass
            arr.append(linearr)
            index = index + 1
        f.close
        return np.asarray(time_arr),np.asarray(arr)  # ,np.asarray(pix)

    def map_to_pixel(self,pixel):
        pixelx = 960 * pixel[0]
        pixely = 540 * pixel[1]
        return (pixelx, pixely)

    def create_output_folder(self,folder_path):
        output = os.path.join(folder_path, "world")
        try:
            os.mkdir(output)
        except OSError:
            pass
        return output
    
    def read_pkl_file(self, path_pkl_file):
        f = open(path_pkl_file, 'rb')
        data = pkl.load(f)
        return data
