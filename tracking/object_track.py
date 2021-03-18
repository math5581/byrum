import numpy as np 
import cv2
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '..')
from classes.tracker import Tracker
import time
from classes.undistort_models import undistortion
from classes.helper_functions import helper_functions
images = []

transform=undistortion("Kennedy")
hel=helper_functions()

file="/Users/mathiaspoulsen/Desktop/work2/byrum/tracking/example_data/world_2020-06-06 12.00.00.txt"
input_scene="scenes//Kennedy_roi_l.bmp"

def main():
	timestamps,data = hel.read_coordinate_file_w_timestamps(file)
	tracker = Tracker(0.8, 5, 1)
	skip_frame_count = 0

	frame_src = cv2.imread(input_scene)
	for i in range(0,len(data)):
		if i>2000:
			break
		#The detections:
		centers=np.asarray(data[i])

		#scene to visualize on
		frame=frame_src.copy()

		if (len(centers) > 0):
			#Updating the trackers
			tracker.update(centers)

			#print(len(centers))
			current_tracks=tracker.get_current_trackers()
			for i in range(0,len(current_tracks)):
				#print(current_tracks[i].trackId)
				if (len(current_tracks[i].trace) > 1):
					print("id:", current_tracks[i].trackId,
						  " x ", current_tracks[i].trace[-1][0, 0],
						  " y ", current_tracks[i].trace[-1][0, 1])

					##It is also possible to get the previous points from the trace.

			#Visualizing trackers on a scene:
			frame=tracker.visualize_on_frame(frame,transform)

			#Visualizing detections:
			for j in range(0,len(centers)):
				x,y=transform.get_image_coordinates(centers[j][0],centers[j][1])
				cv2.circle(frame,(int(x),int(y)), 5, (0,0,0),-1)
			cv2.imshow('image',frame)

			time.sleep(0.25)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break
			#out.write(frame)
	#out.release()

if __name__ == '__main__':
	#tracking_example()
	main()