import numpy as np
import pickle
import cv2 as cv
from classes.roi import roi

class undistortion:
    def __init__(self,location):
        #setup locations
        self.location=location
        self.setup=False

        if location == "Kennedy":
            width = 960
            height = 540
            ### param setup here.
            undistortion_params=[1.000000000000000e+00, 1.265587606625624e-06, 2.980798171945931e-13]
            center=[]
            center.append(width/2)
            center.append(height/2)

            fname = "dat/JFK_mtx.pkl"
            with open(fname, 'rb') as f:
                cameraParams = pickle.load(f)
            self.cameraMatrix = cameraParams["cameraMatrix"]
            self.tvecs = cameraParams["tvecs"]
            self.rvecs = cameraParams["rvecs"]
            self.setProjectionMatrix()

        elif location == "Nytorv":
            width = 960
            height = 540
            ### param setup here.
            undistortion_params = [1, 0, 0]
            center = []
            center.append(width / 2)
            center.append(height / 2)

            fname = "dat/NYT_mtx.pkl"
            with open(fname, 'rb') as f:
                cameraParams = pickle.load(f)
            self.cameraMatrix = cameraParams["cameraMatrix"]
            self.tvecs = cameraParams["tvecs"]
            self.rvecs = cameraParams["rvecs"]
            self.setProjectionMatrix()

        elif location == "JAG7":
            width = 960
            height = 540
            ### param setup here.
            undistortion_params=[9.980773242242400e-01, 2.135646775941647e-08, -4.334852717377035e-15]
            center=[]
            center.append(width/2)
            center.append(height/2)

            fname = "dat/JAG7_mtx.pkl"
            with open(fname, 'rb') as f:
                cameraParams = pickle.load(f)
            self.cameraMatrix = cameraParams["cameraMatrix"]
            self.tvecs = cameraParams["tvecs"]
            self.rvecs = cameraParams["rvecs"]
            self.setProjectionMatrix()

        elif location == "JAG10":
            width = 960
            height = 540
            ### param setup here.
            undistortion_params=[1, 0, 0]
            center=[]
            center.append(width/2)
            center.append(height/2)

            fname = "dat/JAG10_mtx.pkl"
            with open(fname, 'rb') as f:
                cameraParams = pickle.load(f)
            self.cameraMatrix = cameraParams["cameraMatrix"]
            self.tvecs = cameraParams["tvecs"]
            self.rvecs = cameraParams["rvecs"]
            self.setProjectionMatrix()
        elif "setup":
            self.setup=True
        #default setup
        if self.setup!=True:
            self.set_Na(4)
            self.set_a(undistortion_params,center)

        self.ROI=roi(location)

    def set_Na(self,Na):
        self.Na=4

    def map_to_pixel(self,pixel):
        if self.location == "Kennedy":
            pixelx = int(pixel[0])
            pixely = int(pixel[1])
        elif self.location == "JAG7":
            pixelx = int(pixel[0])
            pixely = int(pixel[1])
        elif self.location =="JAG10":
            pixelx = int(pixel[0])
            pixely = int(pixel[1])
        elif self.location =="Nytorv":
            pixelx = int(pixel[0])
            pixely = int(pixel[1])
        return (pixelx, pixely)



    def set_a(self,params,center):
        self.a=[params[0],0,params[1],
                0,params[2],center[0],center[1]]

    def poly_eval(self,eval_point):
        poly_coef=self.a
        sol=poly_coef[self.Na];
        for i in range(self.Na-1,-1,-1):
            sol=sol*eval_point+poly_coef[i];
        #print(sol)
        return sol

    def undistort_point(self,input_point):
        ### Do not perform undistortion on JAG10
        ### checking ROI
        temp=self.ROI.check_roi(input_point)

        #Undistortion is currently not working for Nytorv and JAG10
        if self.location == "JAG10" or self.location == "Nytorv":
            if temp:
                return input_point
            else:
                return temp

        if temp:
            cx=self.a[5]
            cy=self.a[6]
            #finding distance to center of the model.
            dist_cent = np.sqrt(np.power((input_point[0] - cx),2)+np.power((input_point[1] - cy),2) )
            #evaluating against the model:
            temp = self.poly_eval(dist_cent)
            undistorted_point=[]
            #print(temp)
            undistorted_point.append(int(cx + (input_point[0] - cx) * temp))
            undistorted_point.append(int(cy + (input_point[1] - cy) * temp))
            if undistorted_point[0]<0 or undistorted_point[1]<0 or undistorted_point[0]>cx*2 or undistorted_point[1]>cy*2:
                return "Error, the chosen point is outside of the undistorted image"
            return undistorted_point
        else:
            return temp

    def setProjectionMatrix(self):
        self.Lcam=self.cameraMatrix.dot(np.hstack((cv.Rodrigues(self.rvecs[0])[0], self.tvecs[0])))

    def get_world_coordinate_arr(self,arr,Z=0):
        """ Numpy arrary of shape (N,2)"""
        temp = []
        for point in arr:
            x, y = self.get_world_coordinate(point[0],point[1])[:2]
            temp.append([x,y])
        return np.asarray(temp)

    def get_world_coordinate(self,px,py,Z=0):
        #rotMat = R.from_euler('zyx', (self.rvecs[0][0][0], self.rvecs[0][1][0], self.rvecs[0][1][0]), degrees=False)
        #rotMat=rotMat.as_matrix()

        X = np.linalg.inv(np.hstack((self.Lcam[:, 0:2], np.array([[-1 * px], [-1 * py], [-1]])))).dot(
            (-Z * self.Lcam[:, 2] - self.Lcam[:, 3]))
        return np.round(X[0],2), np.round(X[1],2), Z

    def get_pixels(self,px,py,Z=0):
        mat=np.hstack((self.Lcam[:, 0:2], np.array([[-1 * px], [-1 * py], [-1]])))
        vec=(-Z * self.Lcam[:, 2] - self.Lcam[:, 3])
        print(mat)
        print(vec)
        X=vec
        X = np.linalg.inv(mat).dot(vec)
        return X[0], X[1], Z


    def get_world_coordinate_distorted(self,x,y):
        ### loop
        und_poi=self.undistort_point((x,y))
        if isinstance(und_poi,str):
            #### point error
            return False
        elif isinstance(und_poi,bool):
            #### not inside ROI
            return False
        else:
            return self.get_world_coordinate(und_poi[0],und_poi[1])
        #return points

    #functions used for setup.

    def callback_clik(self,event,x,y,flags,params):
        if event == cv.EVENT_LBUTTONDBLCLK:
            #cv.circle(img,(x,y),100,(255,0,0),-1)
            print(x,y)
            mouseX,mouseY = x,y

    def show_frame(self,frame):
        while True:
            cv.imshow("show", frame)
            cv.setMouseCallback("show", self.callback_clik)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    def read_file(self,file_name):
        f = open(file_name, "r+")
        arr = []
        for lines in f:
            dat = lines.split(",")
            arr.append((float(dat[0]), float(dat[1]), int(dat[2]), int(dat[3].split('\n')[0])))
            # arr.append((float(dat[0]),float(dat[1])))
        f.close
        return np.asarray(arr)  # ,np.asarray(pix)

    def remap_points_JFK(self,img_size, xnew, ynew, points):
        ymax = img_size[0]
        xmax = img_size[1]
        pixels = []
        for i in range(0, len(points)):
            xpixel = int(int(points[i][2]) / xmax * xnew)
            ypixel = int(int(points[i][3]) / ymax * ynew)
            pixels.append((xpixel, ypixel))
        return pixels

    def remap_points(self,img_size, xnew, ynew, points):
        ymax = img_size[0]
        xmax = img_size[1]
        pixels = []
        for i in range(0, len(points[0])):
            xpixel = int(int(points[0][i][0]) / xmax * xnew)
            ypixel = int(int(points[0][i][1]) / ymax * ynew)
            pixels.append((xpixel, ypixel))
        return pixels

    def getUVPts(self,camera=""):

        if camera == "kamera7":
            uv = np.array([[528, 1009], [626, 868], [723, 724], [766, 663], [814, 592], [949, 394], [1012, 303],
                           [1271, 798], [1274, 660], [1263, 414]], dtype=np.float32)

        elif camera == "kamera10":
            uv = np.array([[412, 824], [638, 663], [909, 466], [1068, 349], [1306, 186],
                           [1265, 844], [1379, 593], [1423, 478]], dtype=np.float32)

        elif camera == "nytorv":
            uv = np.array([[3547, 878], [2363, 1587], [1357, 2079], [2564, 1775], [935, 1612], [198, 1931], [2109, 461],
                           [1808, 532], [1442, 632], [1011, 757], [543, 912], [67, 1093]], dtype=np.float32)

        else:
            print("Specify a camera")

        uv = np.reshape(uv, (1, -1, 2))
        return uv

    def getObjPts(self,camera=""):

        if camera == "kamera7":
            xyz = np.array([[0.0, 0, 0], [0.0, 1.63, 0], [0, 3.83, 0], [0, 5.18, 0], [0, 6.68, 0],
                            [0, 15.0, 0], [0, 21.0, 0], [4, 3.8, 0], [4, 6.42, 0], [4.0, 14.84, 0]], dtype=np.float32)

        elif camera == "kamera10":
            xyz = np.array([[0., 0., 0.], [0, 2.33, 0.], [0, 6.79, 0.], [0, 10.79, 0.], [0, 22.48, 0.],
                            [4, 1.72, 0.], [4, 5.39, 0], [4, 6.9, 0]], dtype=np.float32)

        elif camera == "nytorv":
            xyz = np.array([[0.0, 0, 0], [0.0, -6.44, 0], [0, -10.0, 0], [0.732, -7.5, 0], [-1.74, -10.0, 0],
                            [-1.74, -12.0, 0], [-6.07, -1.52, 0], [-6.07, -3.52, 0], [-6.07, -5.52, 0],
                            [-6.07, -7.52, 0], [-6.07, -9.52, 0], [-6.07, -11.52, 0]], dtype=np.float32)

        else:
            print("Specify a camera")

        xyz = np.reshape(xyz, (1, -1, 3))
        return xyz

    def calibration_setup_kennedy(self):
        point = (1080, 20)
        color = (0, 0, 255)
        full_shape = (2160, 3840, 3)
        coordinates = self.read_file('../JFK.txt')
        pixels = []
        remapped_pixels = self.remap_points_JFK(full_shape, 960, 540, coordinates)
        undist_class = undistortion("Kennedy")
        undistorted_pixels = []
        # for i in range(0,len(remapped_pixels)):
        #    undistorted_pixels.append(undist_class.undistort_point(remapped_pixels[i]))
        img = cv.imread("files/JFK_Undist.bmp")
        img_calib = cv.imread("../files/JFK_corrected.bmp")
        # show_frame(img)
        # cv.imwrite('pixels.jpg',img)
        # undist=undistortion("Kennedy")
        # undist_point=undist.undistort_point(point)'
        world_coord = []
        print("j")
        index = []
        for i in range(0, len(remapped_pixels)):
            undistorted_pixels.append(undist_class.undistort_point(remapped_pixels[i]))
            if isinstance(undistorted_pixels[i], str):
                index.append(i)
                pass
            else:
                # cv.circle(img,(int(remapped_pixels[i][0]),int(remapped_pixels[i][1])),2,color,-1)
                cv.circle(img_calib, (int(undistorted_pixels[i][0]), int(undistorted_pixels[i][1])), 2, color, -1)
                world_coord.append([coordinates[i][0] - coordinates[0][0], coordinates[i][1] - coordinates[0][1], 0])
        undistorted_pixels = np.delete(np.asarray(undistorted_pixels), index)
        print(world_coord)
        print(remapped_pixels)
        print(undistorted_pixels)
        # world_coord=np.asarray(world_coord,np.float32)
        # undistorted_pixels=np.asarray(undistorted_pixels,np.float32)
        self.show_frame(img_calib)

        world_coord = np.asarray(world_coord)
        objectPointsOneBoard = np.zeros((world_coord.shape[0], 3), np.float32)
        for i in range(0, world_coord.shape[0]):
            objectPointsOneBoard[i, 0] = world_coord[i][0]
            objectPointsOneBoard[i, 1] = world_coord[i][1]
            objectPointsOneBoard[i, 2] = world_coord[i][2]
        objectPoints = [objectPointsOneBoard]

        undistorted_pixels = np.asarray(undistorted_pixels)
        shap = undistorted_pixels.shape
        imagePointsOneBoard = np.zeros((shap[0], 1, 2), np.float32)
        for i in range(0, shap[0]):
            imagePointsOneBoard[i, 0, 0] = undistorted_pixels[i][0]
            imagePointsOneBoard[i, 0, 1] = undistorted_pixels[i][1]
        imagePoints = [imagePointsOneBoard]

        # Perform calibration
        camera_matrix = cv.initCameraMatrix2D(objectPoints, imagePoints, (img_calib.shape[1], img_calib.shape[0]))
        RMS, cameraMatrix, distCoeff, rvec, tvec = cv.calibrateCamera(objectPoints, imagePoints,
                                                                      (img_calib.shape[1], img_calib.shape[0]),
                                                                      camera_matrix, None,
                                                                      flags=cv.CALIB_USE_INTRINSIC_GUESS)

        # print(RMS)
        # print(cameraMatrix)

        """
        previous
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objectPoints, imagePoints, (img_calib.shape[1], img_calib.shape[0]), None, None)
        print(ret)
        """
        #save camera matrix
        #data = {"cameraMatrix": cameraMatrix, "rvecs": rvec, "tvecs": tvec}
        #fname = "dat/JFK_mtx.pkl"

        #with open(fname, 'wb') as f:
        #    pickle.dump(data, f)

    def calibration_setup_JAG10(self):
        full_shape=(1080, 1920, 3)##size on calibration image.
        color=(0,0,255)
        coordinates=self.getUVPts("kamera10")
        remapped_pixels=self.remap_points(full_shape, 960, 540,coordinates)
        undistorted_pixels=[]

        #img=cv.imread("files/JAG10_Undist.bmp")
        img_calib=cv.imread("C://Users//Mathias Poulsen//Desktop//Byrum//undistort_models_python//files//JAG10_Dist.bmp")
        objt=self.getObjPts("kamera10")
        print(remapped_pixels)
        self.show_frame(img_calib)
        exit()
        undistorted_pixels=np.asarray(remapped_pixels)
        shap=undistorted_pixels.shape
        imagePointsOneBoard = np.zeros((shap[0],1, 2), np.float32)
        for i in range(0,shap[0]):
            imagePointsOneBoard[i,0,0]=undistorted_pixels[i][0]
            imagePointsOneBoard[i,0,1]=undistorted_pixels[i][1]
        imagePoints=[imagePointsOneBoard]
        print(objt)
        # Perform calibration
        objectPoints=[objt]
        camera_matrix = cv.initCameraMatrix2D(objectPoints,imagePoints, (img_calib.shape[1], img_calib.shape[0]))
        RMS, cameraMatrix, distCoeff, rvec, tvec = cv.calibrateCamera(objectPoints, imagePoints, (img_calib.shape[1], img_calib.shape[0]), camera_matrix, None, flags=cv.CALIB_USE_INTRINSIC_GUESS)
        print(RMS)
        print(cameraMatrix)
        """
        previous
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objectPoints, imagePoints, (img_calib.shape[1], img_calib.shape[0]), None, None)
        print(ret)
        """
        #save camera matrix
        #data = {"cameraMatrix": cameraMatrix,"rvecs":rvec,"tvecs":tvec}
        #fname = "dat/JAG10_mtx.pkl"

        #with open(fname, 'wb') as f:
        #    pickle.dump(data, f)


    def calibration_setup_JAG7(self):
        full_shape=(1080, 1920, 3)##size on calibration image.
        color=(0,0,255)
        coordinates=self.getUVPts("kamera7")
        remapped_pixels=self.remap_points(full_shape, 960, 540,coordinates)
        undistorted_pixels=[]

        img=cv.imread("files/JAG7_Undist.bmp")
        img_calib=cv.imread("C://Users//Mathias Poulsen//Desktop//Byrum//undistort_models_python//files//JAG7_Dist.bmp")
        undist_class=undistortion("JAG7")
        objt=self.getObjPts("kamera7")
        print(remapped_pixels)
        index=[]
        print(img_calib)
        self.show_frame(img_calib)

        for i in range(0,len(remapped_pixels)):
            undistorted_pixels.append(undist_class.undistort_point(remapped_pixels[i]))
            if isinstance(undistorted_pixels[i], str):
                index.append(i)
                pass
            #else:
                #cv.circle(img,(int(remapped_pixels[i][0]),int(remapped_pixels[i][1])),2,color,-1)
            #    cv.circle(img_calib,(int(undistorted_pixels[i][0]),int(undistorted_pixels[i][1])),2,color,-1)
        print(undistorted_pixels)
        #print(objt)
        print(undistorted_pixels)
        #world_coord=np.asarray(world_coord,np.float32)
        #undistorted_pixels=np.asarray(undistorted_pixels,np.float32)
        self.show_frame(img_calib)
        undistorted_pixels=np.asarray(undistorted_pixels)
        shap=undistorted_pixels.shape
        imagePointsOneBoard = np.zeros((shap[0],1, 2), np.float32)
        for i in range(0,shap[0]):
            imagePointsOneBoard[i,0,0]=undistorted_pixels[i][0]
            imagePointsOneBoard[i,0,1]=undistorted_pixels[i][1]
        imagePoints=[imagePointsOneBoard]

        # Perform calibration
        objectPoints=[objt]
        camera_matrix = cv.initCameraMatrix2D(objectPoints,imagePoints, (img_calib.shape[1], img_calib.shape[0]))
        RMS, cameraMatrix, distCoeff, rvec, tvec = cv.calibrateCamera(objectPoints, imagePoints, (img_calib.shape[1], img_calib.shape[0]), camera_matrix, None, flags=cv.CALIB_USE_INTRINSIC_GUESS)
        print(RMS)
        print(cameraMatrix)
        """
        previous
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objectPoints, imagePoints, (img_calib.shape[1], img_calib.shape[0]), None, None)
        print(ret)
        """
        #save camera matrix
        #data = {"cameraMatrix": cameraMatrix,"rvecs":rvec,"tvecs":tvec}
        #fname = "dat/JAG7_mtx.pkl"

        #with open(fname, 'wb') as f:
        #    pickle.dump(data, f)


    def calibration_setup_NYT(self):
        full_shape = (2160, 3840, 3)
        color=(0,0,255)
        coordinates=self.getUVPts("nytorv")
        remapped_pixels=self.remap_points(full_shape, 960, 540,coordinates)
        undistorted_pixels=[]

        #img=cv.imread("files/JAG10_Undist.bmp")
        img_calib=cv.imread("C://Users//Mathias Poulsen//Desktop//Byrum//undistort_models_python//files//NYT_dist.bmp")
        objt=self.getObjPts("nytorv")
        print(remapped_pixels)
        print(objt)
        self.show_frame(img_calib)
        #exit()
        #undistorted_pixels=np.asarray(remapped_pixels)
        remapped_pixels=np.asarray(remapped_pixels)
        shap=remapped_pixels.shape
        imagePointsOneBoard = np.zeros((shap[0],1, 2), np.float32)
        for i in range(0,shap[0]):
            imagePointsOneBoard[i,0,0]=remapped_pixels[i][0]
            imagePointsOneBoard[i,0,1]=remapped_pixels[i][1]
        imagePoints=[imagePointsOneBoard]
        print(objt)
        # Perform calibration
        objectPoints=[objt]
        camera_matrix = cv.initCameraMatrix2D(objectPoints,imagePoints, (img_calib.shape[1], img_calib.shape[0]))
        RMS, cameraMatrix, distCoeff, rvec, tvec = cv.calibrateCamera(objectPoints, imagePoints, (img_calib.shape[1], img_calib.shape[0]), camera_matrix, None, flags=cv.CALIB_USE_INTRINSIC_GUESS)
        print(RMS)
        print(cameraMatrix)


        """
        previous
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objectPoints, imagePoints, (img_calib.shape[1], img_calib.shape[0]), None, None)
        print(ret)
        """
        #save camera matrix
        #data = {"cameraMatrix": cameraMatrix,"rvecs":rvec,"tvecs":tvec}
        #fname = "dat/NYT_mtx.pkl"

        #with open(fname, 'wb') as f:
        #    pickle.dump(data, f)

