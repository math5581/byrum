import numpy as np
import cv2
import io
import imageio

def drawPlane(img, pts):
    pts = pts.reshape(-1,2)
    for i in range(pts.shape[0]):
        pts[i][0] = round(pts[i][0], 0)
        pts[i][1] = round(pts[i][1], 0)
    pts = pts.astype(np.int32)
    vertices = []
    for i in range(pts.shape[0]):
        vertices.append((pts[i][0], pts[i][1]))

    img = cv2.fillPoly(img, np.array([vertices], dtype=np.int32), (255,255,0))
    return img

def getObjPts(camera = ""):

    if camera == "kamera7":
        xyz = np.array([[0.0, 0, 0], [0.0, 1.63, 0], [0, 3.83, 0], [0, 5.18, 0], [0, 6.68, 0],
        [0, 15.0, 0], [0, 21.0, 0], [4, 3.8, 0], [4, 6.42, 0], [4.0, 14.84, 0]], dtype=np.float32)

    elif camera == "kamera10":
        xyz = np.array([[0.,0.,0.], [0,2.33,0.], [0,6.79,0.], [0,10.79,0.], [0,22.48,0.],
        [4,1.72, 0.], [4,5.39, 0], [4,7.28,0]], dtype=np.float32)

    elif camera == "nytorv":
        xyz = np.array([[0.0, 0, 0], [0.0, -6.44, 0], [0, -10.0, 0], [0.732, -7.5, 0], [-1.74, -10.0, 0],
        [-1.74, -12.0, 0], [-6.07, -1.52, 0], [-6.07, -3.52, 0], [-6.07, -5.52, 0], [-6.07, -7.52, 0],
        [-6.07, -9.52, 0], [-6.07, -11.52, 0]], dtype=np.float32)

    else:
        print("Specify a camera")

    xyz = np.reshape(xyz, (1,-1,3))
    return xyz

def getUVPts(camera = ""):

    if camera == "kamera7":
        uv = np.array([[528, 1009], [626, 868], [723, 724], [766, 663], [814, 592], [949, 394], [1012, 303],
                [1271, 798], [1274, 660], [1263, 414]], dtype=np.float32)

    elif camera == "kamera10":
        uv = np.array([[412, 824], [638,663], [909, 466], [1068,349], [1306,186],
        [1265,844], [1379,593], [1413,478]], dtype=np.float32)

    elif camera == "nytorv":
        uv = np.array([[3547, 878], [2363, 1587], [1357, 2079], [2564, 1775], [935, 1612], [198, 1931], [2109, 461],
                [1808,532], [1442, 632], [1011, 757], [543, 912], [67, 1093]], dtype=np.float32)

    else:
        print("Specify a camera")

    uv = np.reshape(uv, (1,-1,2))
    return uv

def getImgSize(camera=""):

    if camera == "kamera7" or camera == "kamera10":
        imageSize = (1920, 1080) # kamera7 and kamera10

    elif camera == "nytorv":
        imageSize = (3840, 2160) # nytorv

    else:
        print("Specify a camera")

    return imageSize

def getPlane(i,j, camera=""):

    if camera == "kamera7":
        plane = np.array([[0.+j, 1.5+i, 0.],
                        [0.+j, 2.5+i, 0.],
                        [4.+j, 2.5+i, 0],
                        [4+j, 1.5+i, 0.]], np.float32)

    elif camera == "kamera10":
        plane = np.array([[0.+j, 0.0+i, 0.],
                        [0.+j, 1.0+i, 0.],
                        [4.+j, 1.0+i, 0],
                        [4+j, 0.0+i, 0.]], np.float32)

    elif camera == "nytorv":
        plane = np.array([[0.-j, 0-i, 0.],
                        [0.-j, -1-i, 0.],
                        [-1.-j, -1-i, 0],
                        [-1-j, 0-i, 0.]], np.float32)


    return plane

def getProjectionMatrix(cameraMatrix, rvec, tvec):
    return cameraMatrix.dot(np.hstack((cv2.Rodrigues(rvec[0])[0],tvec[0])))

def project2DTo3D(projectionMatrix, px, py, Z):
    X=np.linalg.inv(np.hstack((Lcam[:,0:2],np.array([[-1*px],[-1*py],[-1]])))).dot((-Z*Lcam[:,2]-Lcam[:,3]))
    return X[0], X[1], Z


cam = "kamera7"
xyz = getObjPts(camera=cam)
uv = getUVPts(camera=cam)
imageSize = getImgSize(camera=cam)

# Perform calibration
camera_matrix = cv2.initCameraMatrix2D([xyz],[uv], imageSize)
RMS, cameraMatrix, distCoeff, rvec, tvec = cv2.calibrateCamera(xyz, uv, imageSize, camera_matrix, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

print("RMS: ", RMS)

# projection matrix
Lcam=cameraMatrix.dot(np.hstack((cv2.Rodrigues(rvec[0])[0],tvec[0])))
px = 723
py = 724
Z = 0
X=np.linalg.inv(np.hstack((Lcam[:,0:2],np.array([[-1*px],[-1*py],[-1]])))).dot((-Z*Lcam[:,2]-Lcam[:,3]))
print(Lcam)
print(X)
"""
out_video = imageio.get_writer(cam+".mp4", fps=3)

for i in range(10):
    for j in range(6):


        plane = getPlane(i, j, camera=cam)
        imgPts, _ = cv2.projectPoints(plane, rvec[0], tvec[0], cameraMatrix, distCoeff)

        #img = cv2.imread("kamera7.jpg")
        img = cv2.imread("nytorv.jpg")
        #img = cv2.imread("kamera10.jpg")
        drawPlane(img, imgPts)
        img = cv2.resize(img, (960, 540))
        font = cv2.FONT_HERSHEY_SIMPLEX
        image = cv2.putText(img, "RMS: " + str(round(RMS,3)), (50,50), font,
                   1, (0,0,255), 2, cv2.LINE_AA)
        out_video.append_data(img[:, :, ::-1])

        cv2.imshow("title", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

out_video.close()
cv2.destroyAllWindows()
"""
