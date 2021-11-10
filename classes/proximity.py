import numpy as np
import os
import pickle as pkl
from classes.undistort_models import undistortion

class Proximity(undistortion):
    def __init__(self, location, dataBasePath = 'data'):
        # Also update the undistortion class
        undistortion.__init__(self, location)
        self.dataBasePath = dataBasePath
        # Possibly make location changeable
        self.location = location

        self.DISTANCE_CATEGORIES = 'TODO'
        
    def get_path(self,fileName):
        return os.path.join(self.dataBasePath, self.location, fileName)

    def loop_through_data(self, data):
        for index, row in data.iterrows():
            self.get_distance(row['data'])
            if index >200:
                 break

    def get_distance(self, data):
        for point in data:
            ## Make these two arr.
            print('Undistorted Point ',self.undistort_point(point))
            if self.check_if_dist_or_undist(point):
                self.get_world_coordinate_distorted(point[0], point[1])
            else:
                self.get_world_coordinate(point[0], point[1])
            print('dist ',abs(point[0]-int(point[0])) + abs(point[1]-int(point[1])))
            # if type(unpack) is not bool:
            #world_coordinates = self.get_world_coordinate_arr(data)
            #    print('x ', x)
            #    print('y ', y)
        # For each point in data calculate distances to other array and pop it from the list
        # return array of ditances
