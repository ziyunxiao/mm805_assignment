import numpy as np
import cv2

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

def harr_casscade_mp4():

    harr_cascade = cv2.CascadeClassifier('./data/cv2/haarcascade_frontalface_default.xml')
    vid_capture = cv2.VideoCapture('./data/BillGates.mp4')

    if (vid_capture.isOpened() == False):
        print("Error opening the video file")
    else:
        fps = vid_capture.get(5)
        print('Frames per second : ', fps,'FPS')

        frame_count = vid_capture.get(7)
        print('Frame count : ', frame_count)

    while(vid_capture.isOpened()):
        # vid_capture.read() methods returns a tuple, first element is a bool 
        # and the second is frame
        ret, frame = vid_capture.read()
        if ret == True:
            cv2.imshow('Frame',frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            heads = harr_cascade.detectMultiScale(gray,scaleFactor=2,minNeighbors=2)
            
            for (x,y,w,h) in heads:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            # # press q to quit
            # key = cv2.waitKey(20)            
            # if key == ord('q'):
            #     break
        else:
            break

    vid_capture.release()
    cv2.destroyAllWindows()

harr_casscade_mp4()