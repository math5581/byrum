import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, 'classes') # A bit of a hack...
from classes.proximity import Proximity

LOCATION = 'Nytorv'

prox = Proximity(LOCATION)

### Iterate through 1 pkl and save a pkl of "distances for each person"
prox.calc_proximity_folder()
# Undistort JAG7 again by checking ROI.
# Check ROIS
# JAG10 is probably already undistorted.
# prox.iterate_through_pkls('data/'+LOCATION, 'data/undistorted/'+LOCATION)

# print(type(data))
# prox.loop_through_data(data)