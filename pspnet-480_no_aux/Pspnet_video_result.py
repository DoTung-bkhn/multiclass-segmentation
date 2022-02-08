import time
import cv2 as cv


if __name__ =='__main__':

    #calculate frame per second
    prev_frame_time = 0
    new_frame_time = 0

    #create video writter object
    Video_writter = cv.VideoWriter(filename='0005VD_output.avi', fourcc=cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   fps=30, frameSize=(960, 960))

    #read color map
    image=cv.imread('./Color_map.png')

    #video 1
    cap1 = cv.VideoCapture('./Videos/Vid1/0005VD_Seg.mp4')
    if not cap1.isOpened():
        print("Cannot find videos 1")
        exit()

    #videos 2
    cap2 = cv.VideoCapture('./Videos/Vid1/0005VD.mp4')
    if not cap2.isOpened():
        print("Cannot find videos 2")
        exit()

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        frame1 = cv.resize(frame1, (480, 480), cv.INTER_CUBIC)
        frame2 = cv.resize(frame2, (480, 480), cv.INTER_CUBIC)
        overlay = cv.addWeighted(frame1, 0.7, frame2, 0.3, 0)
        pad=cv.hconcat([overlay,image])
        merge = cv.hconcat([frame1, frame2])
        total=cv.vconcat([merge,pad])

        #calculate fps
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv.putText(total, 'Frame per second:' + fps, (7, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv.LINE_AA)

        cv.imshow('Segmentation', total)
        if cv.waitKey(20) == ord('q'):
            break
        Video_writter.write(total)
    cap1.release()
    cap2.release()
    cv.destroyAllWindows()