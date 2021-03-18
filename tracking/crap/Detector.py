'''
    File name         : detectors.py
    Description       : Object detector used for detecting the objects in a video /image
    Author            : Rahmad Sadli
    Date created      : 20/02/2020
    Python Version    : 3.7
'''

# Import python libraries
import numpy as np
import cv2


def detect(frame,debugMode):
    # Convert frame from BGR to GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if (debugMode):
        cv2.imshow('gray', gray)

    # Edge detection using Canny function
    img_edges = cv2.Canny(gray,  50, 190, 3)
    if (debugMode):
        cv2.imshow('img_edges', img_edges)

    # Convert to black and white image
    ret, img_thresh = cv2.threshold(img_edges, 254, 255,cv2.THRESH_BINARY)
    if (debugMode):
        cv2.imshow('img_thresh', img_thresh)

    # Find contours
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Set the accepted minimum & maximum radius of a detected object
    min_radius_thresh= 3
    max_radius_thresh= 30

    centers=[]
    for c in contours:
        # ref: https://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html
        (x, y), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)

        #Take only the valid circle(s)
        if (radius > min_radius_thresh) and (radius < max_radius_thresh):
            centers.append(np.array([[x], [y]]))
    cv2.imshow('contours', img_thresh)
    return centers

#old function
def tracking_example():
    tracker = Tracker(150, 30, 5)
    skip_frame_count = 0
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (127, 127, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127), (127, 10, 255), (0, 255, 127)]
    data = np.array(np.load('Detections.npy'))[0:10, 0:150, 0:150]
    for i in range(data.shape[1]):
        centers = data[:, i, :]
        print(centers)
        frame = createimage(512, 512)
        if (len(centers) > 0):
            tracker.update(centers)
            for j in range(len(tracker.tracks)):
                if (len(tracker.tracks[j].trace) > 1):
                    x = int(tracker.tracks[j].trace[-1][0, 0])
                    y = int(tracker.tracks[j].trace[-1][0, 1])
                    tl = (x - 10, y - 10)
                    br = (x + 10, y + 10)
                    cv2.rectangle(frame, tl, br, track_colors[j], 1)
                    cv2.putText(frame, str(tracker.tracks[j].trackId), (x - 10, y - 20), 0, 0.5, track_colors[j], 2)
                    for k in range(len(tracker.tracks[j].trace)):
                        x = int(tracker.tracks[j].trace[k][0, 0])
                        y = int(tracker.tracks[j].trace[k][0, 1])
                        cv2.circle(frame, (x, y), 3, track_colors[j], -1)
                    cv2.circle(frame, (x, y), 6, track_colors[j], -1)
                cv2.circle(frame, (int(data[j, i, 0]), int(data[j, i, 1])), 6, (0, 0, 0), -1)
            cv2.imshow('image', frame)
            # cv2.imwrite("image"+str(i)+".jpg", frame)
            # images.append(imageio.imread("image"+str(i)+".jpg"))
            time.sleep(0.1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

# imageio.mimsave('Multi-Object-Tracking.gif', images, duration=0.08)

