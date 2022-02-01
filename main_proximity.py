from classes.proximity import Proximity

LOCATION = 'JFK'

prox = Proximity(LOCATION)

prox.calc_proximity_folder()

#plot the stuff
# prox.scatter_plot_proximity()

# prox.change_name('/Users/mathiaspoulsen/Desktop/work2/byrum/data/prox/JAG7')

# Old 
# Undistort JAG7 again by checking ROI.
# Check ROIS
# JAG10 is probably already undistorted.
# prox.iterate_through_pkls('data/'+LOCATION, 'data/undistorted/'+LOCATION)

# print(type(data))
# prox.loop_through_data(data)