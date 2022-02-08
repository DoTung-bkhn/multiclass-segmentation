from DatasetLabel import*
from Utils import*
import cv2 as cv
from Resnet import*
from Segmentation_models import*

if __name__ =='__main__':
    #load model
    model=Psp_net(input_shape=(480,480,3),numb_class=32,encoder=Resnet50,resize_factor=8,bin_size=[1,2,3,6],
                  weights='./psp_net480_train.h5')

    #create video writter object
    Video_writter =cv.VideoWriter(filename='./Videos/Vid1/0005VD_output.avi',fourcc=cv.VideoWriter_fourcc('M','J','P','G'),fps=30,frameSize=(480,480))

    #reading frame and predict corresponding annotation
    cap = cv.VideoCapture('./Videos/Vid1/0005VD.mp4')
    count=0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.resize(frame, (480, 480), cv.INTER_CUBIC)
        RGB = Predict(model, frame, Camvid_labels)
        if cv.waitKey(20) == ord('q'):
            break
        Video_writter.write(cv.cvtColor(RGB,cv.COLOR_BGR2RGB))
        print('%d frame finished'%(count+1))
        count+=1
    cap.release()
    cv.destroyAllWindows()

