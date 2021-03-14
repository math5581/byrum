import numpy as np 
import cv2
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '..')
from classes.scrap.tracker import Tracker
import time
from classes.undistort_models import undistortion
from classes.helper_functions import helper_functions
images = []

def createimage(w,h):
	size = (w, h, 1)
	img = np.ones((w,h,3),np.uint8)*255
	return img
transform=undistortion("Kennedy")
hel=helper_functions()

file="/Users/mathiaspoulsen/Desktop/work2/byrum/tracking/example_data/world_2020-06-06 12.00.00.txt"
input_scene="scenes//Kennedy_roi_l.bmp"

def main():
	timestamps,data = hel.read_coordinate_file_w_timestamps(file)
	tracker = Tracker(4, 3, 1)
	skip_frame_count = 0
	track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
					(127, 127, 255), (255, 0, 255), (255, 127, 255),
					(127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127)]
	frame_src = cv2.imread(input_scene)
	#shap=frame_src.shape
	#fps=4
	#out = cv2.VideoWriter('tracking_example.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (shap[1], shap[0]))

	for i in range(0,len(data)):
		if i>1000:
			break
		centers=np.asarray(data[i])
		print(centers)
		temp_del=[]
		for k in range(0,len(centers)):
			if centers[k][0]<-24 and centers[k][1]<-14:
				temp_del.append(k)
		centers=np.delete(centers,temp_del,axis=0)
		frame=frame_src.copy()
		#print(centers)
		if (len(centers) > 0):
			tracker.update(centers)
			print(len(centers))
			for j in range(len(tracker.tracks)):
				if (len(tracker.tracks[j].trace) > 1):
					x = tracker.tracks[j].trace[-1][0,0]
					y = tracker.tracks[j].trace[-1][0,1]
					x,y=transform.get_image_coordinates(x,y)
					tl = (x-10,y-10)
					br = (x+10,y+10)
					cv2.rectangle(frame,tl,br,track_colors[j],1)
					cv2.putText(frame,str(tracker.tracks[j].trackId), (x-10,y-20),0, 0.5, track_colors[j],2)
					for k in range(len(tracker.tracks[j].trace)):
						x = tracker.tracks[j].trace[k][0,0]
						y = tracker.tracks[j].trace[k][0,1]
						x, y = transform.get_image_coordinates(x, y)
						cv2.circle(frame,(x,y), 3, track_colors[j],-1)
					cv2.circle(frame,(x,y), 6, track_colors[j],-1)

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



def tracking_example():
	tracker = Tracker(150, 30, 5)
	skip_frame_count = 0
	track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
					(127, 127, 255), (255, 0, 255), (255, 127, 255),
					(127, 0, 255), (127, 0, 127),(127, 10, 255), (0,255, 127)]
	data = np.array(np.load('Detections.npy'))[0:10,0:150,0:150]
	for i in range(data.shape[1]):
		centers = data[:,i,:]
		print(centers)
		frame = createimage(512,512)
		if (len(centers) > 0):
			tracker.update(centers)
			for j in range(len(tracker.tracks)):
				if (len(tracker.tracks[j].trace) > 1):
					x = int(tracker.tracks[j].trace[-1][0,0])
					y = int(tracker.tracks[j].trace[-1][0,1])
					tl = (x-10,y-10)
					br = (x+10,y+10)
					cv2.rectangle(frame,tl,br,track_colors[j],1)
					cv2.putText(frame,str(tracker.tracks[j].trackId), (x-10,y-20),0, 0.5, track_colors[j],2)
					for k in range(len(tracker.tracks[j].trace)):
						x = int(tracker.tracks[j].trace[k][0,0])
						y = int(tracker.tracks[j].trace[k][0,1])
						cv2.circle(frame,(x,y), 3, track_colors[j],-1)
					cv2.circle(frame,(x,y), 6, track_colors[j],-1)
				cv2.circle(frame,(int(data[j,i,0]),int(data[j,i,1])), 6, (0,0,0),-1)
			cv2.imshow('image',frame)
			# cv2.imwrite("image"+str(i)+".jpg", frame)
			# images.append(imageio.imread("image"+str(i)+".jpg"))
			time.sleep(0.1)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				cv2.destroyAllWindows()
				break

	# imageio.mimsave('Multi-Object-Tracking.gif', images, duration=0.08)
			
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

if __name__ == '__main__':
	#tracking_example()
	main()