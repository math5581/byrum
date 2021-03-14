# This is a sample Python script.
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '..')
import numpy as np
from classes.undistort_models import undistortion
from classes.roi import roi as ROI
np.set_printoptions(suppress=True)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #Example Usage
    ##setup, specify location:
    undist_class=undistortion("JAG10")

    ##Undistort a single point
    point_distorted=(383, 330)
    ### This function also checks the corresponding ROI...
    point_undistorted = undist_class.undistort_point(point_distorted)

    ## get world coordinate from undistorted point
    ##Input is an undistorted point!
    coord = undist_class.get_world_coordinate(383,330)

    ##probably just use this funtion....
    #get world coordinate from distorted point
    ### Input is a distorted point!
    #### This function also checks the corresponding ROI.
    coord1 = undist_class.get_world_coordinate_distorted(383,330)


    ##### checking ROI
    ##### has only set up kennedy, we need to agree on the two others...
    roi_Kennedy = ROI(location="Kennedy")
    list_points = [(200, 200)]
    print(roi_Kennedy.check_roi(list_points))
