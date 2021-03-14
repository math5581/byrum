'''
    File name         : objTracking.py
    Description       : Main file for object tracking
    Author            : Rahmad Sadli
    Date created      : 20/02/2020
    Python Version    : 3.7
'''
import cv2
from crap.Detector import detect
from classes.KalmanFilter import KalmanFilter
from classes.helper_functions import helper_functions
from classes.Tracking import tracking
import os
import numpy as np
from classes.undistort_models import undistortion

folder_path= "../example_data"
input_scene="scenes//Kennedy_roi_l.bmp"

def istantiate_new_tracker(id):
    # KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
    return KalmanFilter(0.25, 5, 5, 1, 0.1, 0.1,id)

def associate_trackers_detections(trackers,detection,detection_id,associations):
    smallest_dist=10000
    for k in range(0,len(trackers)):
        state=trackers[k].get_current_state()
        dist=np.sqrt(np.power(state[0,0]-detection[0,0],2)+np.power(state[1,0]-detection[1,0],2))

        if dist<smallest_dist:
            smallest_dist=dist
            associations[k]=detection_id



def main():

    hel=helper_functions()
    #Variable used to control the speed of reading the video
    ControlSpeedVar = 1  #Lowest: 1 - Highest:100

    HiSpeed = 200
    transform=undistortion("Kennedy")
    tracker=tracking()
    for filename in os.listdir(folder_path):
        timestamps, coordinates = hel.read_coordinate_file_w_timestamps(os.path.join(folder_path, filename))
        frame = cv2.imread(input_scene)
        for i in range(0, coordinates.shape[0]):
            if i<340:
                continue
            copy_frame = frame.copy()
            detection_array_temp=[]

            for j in range(0,len(coordinates[i])):
                #draw_detections
                detection_points=np.asarray(coordinates[i][j]).reshape((2,1))
                u,v=transform.get_image_coordinates(detection_points[0,0],detection_points[1,0])
                cv2.circle(copy_frame, (u, v), 2, (0, 191, 255), 2)
                #cv2.putText(copy_frame,  str(j), (u + 3, v + 30), 0, 0.5, (0, 191, 255), 1)
                detection_array_temp.append(detection_points)

            #tracker.update_trackers(detections)
            #looping through. association could be done here:
            trackin_list=tracker.update_trackers(detection_array_temp)



            #tracker_id_to_update=associate_trackers_detections(KF,detections,j,associations)
            #istantiate new if nothing fits:

            # Predict
            #cv2.rectangle(copy_frame, (x - 15, y - 15), (x + 15, y + 15), (255, 0, 0), 1)
            #print(len(trackin_list))
            for i in range(0,len(trackin_list)):

                point=trackin_list[i].get_current_state()
                id=trackin_list[i].get_tracker_id()
                covar=trackin_list[i].get_covariance_sum()
                ## predicted state:
                #p_state=trackin_list[i].get_predicted_state()
                #u_p, v_p = transform.get_image_coordinates(p_state[0,0],p_state[1,0])
                #cv2.circle(copy_frame, (u_p, v_p), 2, (0, 255, 0), 2)
                #cv2.putText(copy_frame, str(id), (u_p-8, v_p + 10), 0, 0.5, (0, 255, 0), 1)

                #print(point)
                # Draw a rectangle as the estimated object position
                u, v = transform.get_image_coordinates(point[0,0],point[1,0])
                cv2.putText(copy_frame, str(covar), (u + 3, v + 30), 0, 0.5, (255, 0, 255), 1)
                cv2.rectangle(copy_frame, (u - 2, v - 2), (int(u) + 2, int(v) + 2), (0, 0, 255), 1)

                cv2.putText(copy_frame, str(id), (u-8, v + 30), 0, 0.5, (0, 0, 255), 1)

            cv2.imshow('image', copy_frame)

            if cv2.waitKey(2) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
            #cv2.waitKey(0)
            cv2.waitKey(250)


def example():
    # Create opencv video capture object
    # Create opencv video capture object
    VideoCap = cv2.VideoCapture('video/randomball_Trim.mp4')

    #Variable used to control the speed of reading the video
    ControlSpeedVar = 100  #Lowest: 1 - Highest:100

    HiSpeed = 100

    #Create KalmanFilter object KF
    #KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)

    KF = KalmanFilter(0.1, 1, 1, 1, 0.1,0.1)
    ret, frame = VideoCap.read()

    debugMode=0

    while(True):
        # Read frame
        ret, frame = VideoCap.read()

        # Detect object
        centers = detect(frame,debugMode)
        print(centers[0].shape)
        # If centroids are detected then track them
        if (len(centers) > 0):

            # Draw the detected circle
            cv2.circle(frame, (int(centers[0][0]), int(centers[0][1])), 10, (0, 191, 255), 2)

            # Predict
            (x, y) = KF.predict()
            # Draw a rectangle as the predicted object position
            cv2.rectangle(frame, (x - 15, y - 15), (x + 15, y + 15), (255, 0, 0), 2)

            # Update
            (x1, y1) = KF.update(centers[0])

            # Draw a rectangle as the estimated object position
            cv2.rectangle(frame, (x1 - 15, y1 - 15), (x1 + 15, y1 + 15), (0, 0, 255), 2)

            cv2.putText(frame, "Estimated Position", (x1 + 15, y1 + 10), 0, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, "Predicted Position", (x + 15, y), 0, 0.5, (255, 0, 0), 2)
            cv2.putText(frame, "Measured Position", (centers[0][0] + 15, centers[0][1] - 15), 0, 0.5, (0,191,255), 2)

        cv2.imshow('image', frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            VideoCap.release()
            cv2.destroyAllWindows()
            break

        cv2.waitKey(HiSpeed-ControlSpeedVar+1)


if __name__ == "__main__":
    # execute main
    main()
    #example()
