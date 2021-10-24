from shapely.geometry import Point,Polygon
import cv2 as cv
import numpy as np


class roi:

    def __init__(self,location):
        #The ordering of the polygons is important!
        #setting up roi
        self.location=location
        if location=="Kennedy":
            self.set_roi([(106,315), (246,110),(828,116),(908,380)])
        elif location=="Nytorv":
            self.set_roi([(20, 268), (146, 534), (436, 470), (894, 226),(568, 102)])
            #self.set_roi(0)
        elif location=="JAG7":
            self.set_roi([(265,505),(634,520),(632,137),(522,133)])#(522,133)
        elif location =="JAG10":
            self.set_roi([(68,508),(510,515),(777,87),(654,90)])


    def set_roi(self,roi):
        self.roi_points=roi
        self.roi = Polygon(roi)

    #This is changed...
    def check_roi(self,point):
        p = Point(point)
        temp=p.within(self.roi)
        #temp=[]
        #for i in range(0,len(point)):
        #    p = Point(point[i])
        #    temp.append(p.within(self.roi))
        return temp

    def draw_roi(self):
        if self.location== "Kennedy":
            img=cv.imread("files/JFK_corrected.bmp")
        elif self.location=="JAG7":
            img=cv.imread("files/JAG7_corrected.bmp")
        elif self.location=="JAG10":
            img=cv.imread("files/JAG10_Dist.bmp")
        elif self.location=="Nytorv":
            print("update nytorv")
            img=cv.imread("files/file.bmp")
        color = (0, 0, 255)
        for i in range(0,len(self.roi_points)):
            cv.circle(img, (self.roi_points[i][0], self.roi_points[i][1]), 3, color, -1)
        cv.imwrite("files/"+self.location+"_roi.bmp",img)

    def draw_roi_lines(self):
        if self.location == "Kennedy":
            img = cv.imread("files/JFK_corrected.bmp")
        elif self.location == "JAG7":
            img = cv.imread("files/JAG7_corrected.bmp")
        elif self.location == "JAG10":
            img = cv.imread("files/JAG10_Dist.bmp")
        elif self.location == "Nytorv":
            img = cv.imread("files/NYT_dist.bmp")
        color = (0, 0, 255)
        pts = np.array(self.roi_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv.polylines(img, [pts], True, (0, 0, 255))
        #for i in range(0, len(self.roi_points)):
        #    cv.circle(img, (self.roi_points[i][0], self.roi_points[i][1]), 3, color, -1)
        cv.imwrite("files/" + self.location + "_roi_l.bmp", img)