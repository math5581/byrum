import numpy as np
import os
import pickle as pkl
from classes.undistort_models import undistortion
from math import hypot
from pathlib import Path

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

        self.pkl_files = self.list_pkl_files()
        self.DISTANCE_CATEGORIES = 'TODO'
        
    def get_path(self,fileName):
        return os.path.join(self.dataBasePath, self.location, fileName)

    def list_pkl_files(self):
        p = self.base_path_undist
        print(os.listdir(p))
        pkl_files = [f for f in os.listdir(p) if (os.path.isfile(os.path.join(p, f))  and f.split('.')[-1] == 'pkl')]
        pkl_files.sort()
        return pkl_files

    def calc_proximity_folder(self):
        """ Calculates and saves avg distance proximty for location"""
        for file in self.pkl_files:
            print(file)
            # Skip if file is already analyzed
            if self.check_if_analyzed_file_exists(file):
                continue
            path = os.path.join(self.base_path_undist, file)
            avg_dist_list = self.iterate_single_pkl(path)
            out_file = os.path.join(self.base_path_prox, file)
            with open(out_file, 'wb') as f:
                pkl.dump(avg_dist_list, f)

    def check_if_analyzed_file_exists(self, file):
        path = os.path.join(self.base_path_prox, file)
        return os.path.isfile(path)

    def iterate_single_pkl(self, path):
        data = self.read_pkl_file(path)
        avg_distances = []
        for index, row in data.iterrows():
            avg_distances = avg_distances + self.get_distance(row['data'])
        sum(avg_distances)/len(avg_distances)
        return  avg_distances

    def get_distance(self, data):
        """ for each coordinate return the average distance to the
            three closes points """
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

