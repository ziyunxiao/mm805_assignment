import numpy as np
import cv2

# org video: https://www.youtube.com/watch?v=zPx5N6Lh3sw&t=1276s
# ffmpeg -i BillGates.mp4 -ss 00:00:05 -t 00:10:00 -async 1 cut.mp4
# ffmpeg -i cut.mp4 -vf scale=320:240 small.mp4

def harr_casscade():
    harr_cascade = cv2.CascadeClassifier('./data/cv2/haarcascade_frontalface_default.xml')


    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        heads = harr_cascade.detectMultiScale(gray,scaleFactor=2,minNeighbors=2)

        for (x,y,w,h) in heads:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.imshow('img',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def harr_cascade_detect(harr_cascade,frame):
    # detection, inside the detection the frame data is changed with 
    # bounding box and returned
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    a = 2
    gray = cv2.resize(gray, dsize=(0, 0), fx=1/a, fy=1/a)
    heads = harr_cascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)
    
    for (x,y,w,h) in heads:
        cv2.rectangle(frame,(a*x,a*y),(a*x+a*w,a*y+a*h),(255,0,0),2)
    
    return frame


def contour_detect(frame):
    # detection, inside the detection the frame data is changed with 
    # bounding box and returned
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # # Setup SimpleBlobDetector parameters.
    # params = cv2.SimpleBlobDetector_Params()
  	# # Filter by Circularity
    # params.filterByCircularity = True
    # params.minCircularity = 0.1

    # detector = cv2.SimpleBlobDetector(params)

    # # Detect blobs.
    # keypoints = detector.detect(gray)

    # # Draw detected blobs as red circles.
    # # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # frame = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


    ret, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
                                     
    # draw contours on the original image
    cv2.drawContours(image=frame, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                
    # see the results
    cv2.imshow('binary', thresh)
    
    return frame

def harr_casscade_mp4():
    xml = 'haarcascade_frontalface_default.xml'
    # xml = 'haarcascade_upperbody.xml'

    harr_cascade = cv2.CascadeClassifier(f'./data/cv2/{xml}')
    video_name = "cut1.mp4"
    # cut2 does not work. 
    vid_capture = cv2.VideoCapture(f'./data/{video_name}')

    if (vid_capture.isOpened() == False):
        print("Error opening the video file")
        return

    fps = vid_capture.get(5)
    frame_count = vid_capture.get(7)
    frame_width = int(vid_capture.get(3))
    frame_height = int(vid_capture.get(4))
    frame_size = (frame_width,frame_height)
    # fps = 20
    print(f"Video Info:")
    print(f"width:{frame_width} height:{frame_height} FPS: {fps} Count: {frame_count}")

    # output video
    # output = cv2.VideoWriter(f'./data/output_{video_name}', \
    #     cv2.VideoWriter_fourcc('M','J','P','G'), 20, frame_size)

    output = cv2.VideoWriter(f'./data/output_{video_name}', \
        cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)


    while(vid_capture.isOpened()):
        # vid_capture.read() methods returns a tuple, first element is a bool
        # and the second is frame
        ret, frame = vid_capture.read()
        if ret == True:
            
            # detection, inside the detection the frame data is changed with 
            # bounding box and returned
            frame = harr_cascade_detect(harr_cascade,frame)
            # frame = contour_detect(frame)

            cv2.imshow('Frame',frame)
            output.write(frame)

            # press q to quit
            key = cv2.waitKey(20)
            if key == ord('q'):
                break
        else:
            break

    vid_capture.release()
    output.release()
    cv2.destroyAllWindows()

harr_casscade_mp4()