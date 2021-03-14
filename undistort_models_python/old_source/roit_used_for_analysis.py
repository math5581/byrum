import os
import sys
sys.path.append("..")
from classes.undistort_models import undistortion
import cv2 as cv
from shapely.geometry import Point,Polygon
import numpy as np
import matplotlib.pyplot as plt
from classes.roi import roi as ROI
os.chdir("..")
def check_if_within_roi(roi,p):
    points=[]
    #for i in range(0,len(roi)):
    #    points.append(Point(roi[i]))
    poly = Polygon(roi)
    p = Point(p)
    return p.within(poly)


def read_coordinate_file(file_name):
    f = open(file_name, "r+")
    arr = []
    print(arr)
    index = 0
    for lines in f:
        dat = lines.split(",")
        linearr = []
        for i in range(0, len(dat)-1):
            dat2 = dat[i].split(":")
            linearr.append((float(dat2[0]),float(dat2[1])))
        arr.append(linearr)
        index = index + 1
    f.close
    return np.asarray(arr)  # ,np.asarray(pix)

def map_to_pixel(pixel):
    pixelx=960*pixel[0]
    pixely=540*pixel[1]
    return (pixelx,pixely)

color=(0,0,255)
img_roi=cv.imread("files/JFK_corrected.bmp")
roi=[(106,315),
     (244,113),
    (827,123),
    (853,376)]


#coordinates=read_coordinate_file("coordinates_full.txt")

helper=undistortion("Kennedy")
def write_video_analysis():
    out = cv.VideoWriter('output.avi',cv.VideoWriter_fourcc('M','J','P','G'), 24, (960,540))
    shape=int(coordinates.shape[0]/3)

    for i in range(shape*2,shape*3):
        frame=img_roi.copy()
        for j in range(0,len(coordinates[i])):
            point=map_to_pixel(coordinates[i][j])
            undistorted_point=helper.undistort_point(point)
            if isinstance(undistorted_point,str):
                excpet=0
            else:
                cv.circle(frame, (undistorted_point[0],undistorted_point[1]), 3, color, -1)
        print(i)
        out.write(frame)

def write_world_coordinates():
    ##### Functions to write world coordinates
    file_name="world_coordinates.txt"
    #Convert to world_coordinates:
    file=open(file_name,"w+")
    for i in range(0,coordinates.shape[0]):
        frame=img_roi.copy()
        string = ""
        for j in range(0,len(coordinates[i])):
            point=map_to_pixel(coordinates[i][j])
            undistorted_point=helper.undistort_point(point)
            if isinstance(undistorted_point,str):
                print("string")
            else:
                world_point = helper.get_world_coordinate(undistorted_point[0], undistorted_point[1])
                string += str(world_point[0])+","+str(world_point[1])
                if j != len(coordinates[i])-1:
                    string += ":"
        ##DELETING LAST :
        if len(string)>0:
            if string[-1]==":":
                string=string[:-1]
        string += "\n"
        print(i)
        file.write(string)


def draw_world_roi():
    print(helper.get_world_coordinate(483,331))
    print(helper.get_world_coordinate(441,364))
    for i in range(0, len(roi)):
        cv.circle(img_roi, roi[i], 3, color, -1)

    #cv.circle(img_roi, (483,331), 3, color, -1)
    #cv.circle(img_roi, (550,350), 3, color, -1)
    #cv.circle(img_roi, (441,364), 3, color, -1)
    cv.line(img_roi, (483,331), (550,350), (0, 0, 255), thickness=2)
    cv.line(img_roi, (483,331), (441,364), (0, 0, 255), thickness=2)
    image = cv.putText(img_roi, '(1,0)', (441-30,364+20), cv.FONT_HERSHEY_SIMPLEX,
                       0.7, (0,0,0), 1, cv.LINE_AA)
    image = cv.putText(img_roi, '(0,1)', (550-20,350+25), cv.FONT_HERSHEY_SIMPLEX,
                       0.7, (0,0,0), 1, cv.LINE_AA)
    helper.show_frame(img_roi)

    cv.imwrite("JFK_corrected_ROI.bmp",img_roi)


def write_video_avg_5():
    #out = cv.VideoWriter('output.avi',cv.VideoWriter_fourcc('M','J','P','G'), 24, (960,540))
    roi = [(106, 315),
           (244, 113),
           (827, 123),
           (853, 376)]

    count=0
    k = 0
    avg=[]
    for i in range(0,coordinates.shape[0]):
        if k>=7200 or i>=coordinates.shape[0]-1:
            avg.append(count/k)
            count=0
            k=0
        for j in range(0,len(coordinates[i])):
            point=map_to_pixel(coordinates[i][j])
            undistorted_point=helper.undistort_point(point)
            if isinstance(undistorted_point,str):
                excpet=0
            else:
                #IS a point(within the big image)
                if check_if_within_roi(roi, undistorted_point):
                    #Within ROI
                    count = count +1
        k=k+1
    print(avg)
    plt.bar(range(5,25,5),avg,width=2.5)  # `density=False` would make counts
    plt.ylabel('Average detections per frame')
    plt.xlabel('Time [min]')
    plt.show()
    file_name = "avg_5_minutes.txt"
    # Convert to world_coordinates:
    #file = open(file_name, "w+")
    #for i in range(0,len(avg)):
    #    string=str(avg[i])+"\n"
    #    file.write(string)
    #file.close
frame=cv.imread("files/JFK_corrected.bmp")
helper.show_frame(frame)
roi_Kennedy=ROI(location="Kennedy")
roi_Kennedy.draw_roi_lines()
#list_points=[(106,339),(158,293),(808,159),(925,122)]
#print(roi_Kennedy.check_roi(list_points))

##testing of ROI
#write_video_avg_5()