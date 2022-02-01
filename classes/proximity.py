import numpy as np
import os
import pickle as pkl
from classes.undistort_models import undistortion
from math import hypot
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as md


class Proximity(undistortion):
    def __init__(self, location, data_base_path = 'data'):
        # Also update the undistortion class
        undistortion.__init__(self, location)
        self.base_path = 'data'
        self.base_path_undist = os.path.join(data_base_path, 'undistorted', location)
        # Create Proximity Folder
        self.base_path_prox = os.path.join(data_base_path, 'prox', location)
        Path(self.base_path_prox).mkdir(parents=True, exist_ok=True)    

        # Possibly make location changeable
        self.location = location

        self.DISTANCE_CATEGORIES = 'TODO'
    
    def change_name(self, path):
        """ Used to change the name of the pkl files in path"""
        file_list = self.list_pkl_files(path)
        for file in file_list:
            old_name = os.path.join(path, file)
            file = file.split('.')[0]
            new_name_list = []
            for item in file.split('-'):
                if len(item) == 1:
                    new_name_list.append('0' + str(item))
                else:
                    new_name_list.append(item)
            new_name = '-'.join(new_name_list)
            new_name += '.pkl'
            new_name = os.path.join(path, new_name)
            os.rename(old_name, new_name)

    # Helper functions
    def get_path(self, file_name):
        """ Deprecated """
        return os.path.join(self.base_path, self.location, file_name)

    def list_pkl_files(self, p):
        """ List pickle files in a folder """
        pkl_files = [f for f in os.listdir(p) if (os.path.isfile(os.path.join(p, f))  and f.split('.')[-1] == 'pkl')]
        pkl_files.sort()
        return pkl_files

    def check_if_analyzed_file_exists(self, file):
        path = os.path.join(self.base_path_prox, file)
        return os.path.isfile(path)

    # Distance/Proximity functions
    def calc_proximity_folder(self):
        """ Calculates and saves avg distance proximty for location"""
        files = self.list_pkl_files(self.base_path_undist)
        for file in files:
            print(file)
            # Skip if file is already analyzed
            if self.check_if_analyzed_file_exists(file):
                continue
            path = os.path.join(self.base_path_undist, file)
            avg_dist_list = self.iterate_single_pkl(path)
            out_file = os.path.join(self.base_path_prox, file)
            with open(out_file, 'wb') as f:
                pkl.dump(avg_dist_list, f)

    def iterate_single_pkl(self, path):
        data = self.read_pkl_file(path)
        avg_distances = []
        for index, row in data.iterrows():
            try:
                avg_distances = avg_distances + self.get_distance(row['data'])
            except:
                pass
        return  avg_distances

    def get_distance(self, data):
        """ for each coordinate return the average distance to the
            three closes points """
        data = np.asarray(data, dtype=np.float64)
        world_dat = self.get_world_coordinate_arr(data)
        number_of_closest_points = 3
        avg_dist_arr = []
        numb_people = len(world_dat)
        if numb_people > 1:
            for i in range(numb_people):
                temp_arr = np.delete(world_dat, i, axis=0)
                dist = [self.distance(world_dat[i], ele) for ele in temp_arr]
                dist.sort()
                dist = dist[0:number_of_closest_points] # <=3 closest points
                avg_distance = sum(dist) / len(dist[0:number_of_closest_points])
                avg_dist_arr.append(avg_distance)
        return avg_dist_arr

    def distance(self, w1, w2):
        """ Returns Eucledian distance between two world coordinates """
        x1,y1 = w1
        x2,y2 = w2
        return hypot(x2 - x1, y2 - y1)

    # plotting / data presentation function.
    def scatter_plot_proximity(self):
        proximity_files = self.list_pkl_files(self.base_path_prox)
        K_list = []
        ts_list = []
        for file in proximity_files:
            proximity_list = self.read_proximity_pkl(os.path.join(self.base_path_prox, file))
            if len(proximity_list) != 0:
                proximity_K = sum(proximity_list)/len(proximity_list)
                K_list.append(proximity_K)
            else:
                K_list.append(0)
            time = file.split('.')[0].split('-')
            ts_list.append(self.create_time_stamp(time)) 
        datenums=md.date2num(ts_list)
        plt.subplots_adjust(bottom=0.2)
        plt.xticks( rotation=25 )
        ax=plt.gca()
        xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
        ax.xaxis.set_major_formatter(xfmt)
        plt.plot(datenums,K_list)
        plt.show()

    def create_time_stamp(self, time):
        return '2020-' + str(time[0]) + '-' + str(time[1])+ ' ' + str(time[2]) +':00:00'

    def read_proximity_pkl(self, p):
        """ Reads and returns list of single pkl file"""
        with open(p, 'rb') as f:
            return pkl.load(f)

