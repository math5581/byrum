from classes.KalmanFilter import KalmanFilter
import numpy as np
from itertools import count, filterfalse # ifilterfalse on py2

class tracking:
    def __init__(self):
        i=0
        self.max_number_trackers=20
        self.trackin_list=np.asarray([])
        self.trackin_list_used=np.asarray([])
        self.tracking_id_list=np.asarray([])
        self.tracker_remove_count_frames=2

        ### This needs to be based on the covariance matrix.
        self.tracking_threshold=1

    def find_distance(self,state,detection):
        dist=np.sqrt(np.power(state[0,0]-detection[0,0],2)+np.power(state[1,0]-detection[1,0],2))
        return dist

    #### THis funtion returns the distance to the closest tracker and the tracker id, for the given detection()
    #If no tracker is below the threshold, it returns -1. This indicates, that a new tracker is required.

    def predict_filters(self):
        temp_ids_to_remove=[]
        if len(self.trackin_list)==0:
            return 0
        #print("length",len(self.trackin_list))
        for i in range(0,len(self.trackin_list)):
            self.trackin_list[i].predict()
            self.trackin_list_used[i]+=1
            #checking if trackers should be removed:
            if self.trackin_list_used[i]>=self.tracker_remove_count_frames:# or self.trackin_list[i].get_covariance_sum()>4:
                temp_ids_to_remove.append(i)
        ##removing trackers
        self.trackin_list=np.delete(self.trackin_list,temp_ids_to_remove)
        self.trackin_list_used=np.delete(self.trackin_list_used,temp_ids_to_remove)
        self.tracking_id_list=np.delete(self.tracking_id_list,temp_ids_to_remove)

    def find_closest_tracker(self, detection, numb_trackers):
        lowest_dist = 10
        tracker_id = -1
        for j in range(0, numb_trackers):
            # print(self.trackin_list[j])
            temp_dist = self.find_distance(self.trackin_list[j].get_current_state(), detection)
            covar_dist = self.trackin_list[j].get_covariance_sum()
            if covar_dist > self.tracking_threshold:
                discard = covar_dist

            if temp_dist < lowest_dist:
                lowest_dist = temp_dist
                tracker_id = j

        print("tracler_id", tracker_id, "lowest_dist", lowest_dist)
        covar_dist = self.trackin_list[tracker_id].get_covariance_sum()
        if lowest_dist >= covar_dist and lowest_dist >= self.tracking_threshold:
            lowest_dist = -1  ### This means, that no tracker fits for this

        return lowest_dist, tracker_id
    def find_distance_detections(self, detections,tracker_state,covar_dist):
        lowest_dist = 100
        temp_dist=self.find_distance(tracker_state, detections)
        if temp_dist >= covar_dist and lowest_dist >= self.tracking_threshold:
            return  1000
        else:
            return temp_dist

    def update_trackers(self,detections):
        #Delete tracker after not used for 4 frames.
        self.predict_filters()
        association_array=[]
        number_of_trackers=len(self.trackin_list)

        #creating matrix
        print("trackers",number_of_trackers)
        number_of_detections=len(detections)
        print("detections",number_of_detections)
        self.distance_closest_detection_list=np.empty((number_of_trackers,number_of_detections))
        #self.detection_id_list=np.empty(number_of_trackers)
        #if no trackers are active, we initialize one for each.
        if number_of_trackers==0:
            for i in range(0,len(detections)):
                self.istantiate_tracker(detections[i])

        #for each tracker find the distance to all other detections
        for i in range(0,number_of_trackers):
            for j in range(0,number_of_detections):
                self.distance_closest_detection_list[i,j]=self.find_distance_detections(detections[j],self.trackin_list[i].get_current_state(),self.trackin_list[i].get_covariance_sum())

        print(self.distance_closest_detection_list)


        if number_of_trackers and number_of_detections:

            detections_copy=np.asarray(detections)
            #print("forst",detections_copy)
            temp_detections_delete=[]
            while number_of_trackers>0:
                val = self.distance_closest_detection_list.min()
                if val<500:
                    index_of_minimum = np.where(self.distance_closest_detection_list == val)
                    #print(self.distance_closest_detection_list)
                    row=index_of_minimum[0][0]
                    col=index_of_minimum[1][0]
                    self.trackin_list[row].update(detections[col])
                    self.distance_closest_detection_list = np.delete(self.distance_closest_detection_list, (row), axis=0)
                    temp_detections_delete.append(col)
                    self.trackin_list_used[row] = 0
                    number_of_trackers=number_of_trackers - 1
                    number_of_detections = number_of_detections - 1
                else:
                    break

            detections_copy=np.delete(detections_copy,temp_detections_delete,axis=0)
            print(len(detections_copy))
            for i in range(0,len(detections_copy)):
                self.istantiate_tracker(detections[i])

        return self.trackin_list

    def assign_tracker_id(self):
        #always assign with the lowest id available.
        n=len(self.tracking_id_list)
        id_available=next(filterfalse(set(self.tracking_id_list).__contains__, count(0)))
        self.tracking_id_list = np.append(self.tracking_id_list, id_available)
        return id_available

    #How to structure this
    def istantiate_tracker(self,detection):#detection has to be ID
        #Assign a tracking ID Here. Assign the lowest ID available.
        available_id=self.assign_tracker_id()
                                                    # KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
        self.trackin_list=np.append(self.trackin_list,KalmanFilter(0.25, 0.5, 0.5, 0, 0.1, 0.1,available_id))
        self.trackin_list[-1].update(detection)
        self.trackin_list_used=np.append(self.trackin_list_used,0)


    def remove_tracker(self,id):
        self.tracking_list
