#python -m streamlit run Posture_correction.py
#pip freeze > requirements.txt
import cv2
import mediapipe as mp
import time
import math
import numpy as np
import streamlit as st

def show():

    st.title('Posture correction tools')
    st.write("""By Victor&Gusto""")
    st.info("This software is used to improve rounded shoulders and hunchback, "
            "and the effect will be better with daily use."
            )
    st.info("More features are in development!")
    placeholder = st.empty()




class poseDetector():

    def __init__(self, mode=False, complexity=1, smooth=True,
               esegmentation=False,
               ssegmentation=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.complexity = complexity
        self.smooth = smooth
        self.esegmentation= esegmentation
        self.ssegmentation= ssegmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth,
           self.esegmentation,self.ssegmentation,
             self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #cv2.cvtColor()
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,
                                           self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # print(angle)

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

class Task():
    def __init__(self, video):
        self.video = video
        self.frame_counter = 0
        self.cap = cv2.VideoCapture(self.video)
        #self.task_count = task_count
        #self.img = cv2.resize(self.img, (640, 480))

    def video_loop(self):
        self.frame_counter += 1
        if self.frame_counter == int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            self.frame_counter = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


def main():
    global i,totaltime,imgall
    global placeholder
    cap = cv2.VideoCapture(1)
    cap = cv2.VideoCapture("1.mp4")
    rand = 20
    pTime = 0
    imgex = 0
    task_count =0
    #frame_counter =0
    detector = poseDetector()
    task1 = Task("1.mp4")
    task2 = Task("2.mp4")
    task3 = Task("3.mp4")
    task4 = Task("4.mp4")
    task5 = Task("5.mp4")
    task6 = Task("6.mp4")
    task7 = Task("7.mp4")
    task8 = Task("8.mp4")
    imgover = cv2.imread("over.jpg")
    while True:
        try:
            success, img = cap.read()
            if(task1_count !=rand):
                _, img1 = task1.cap.read()
                task1.video_loop()
                img1 = cv2.resize(img1, (640, 480))
                cv2.putText(img1, "Follow up and repeat 20 times", (70, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 0, 255), 2)
                imgex = img1
            elif(task2_count !=rand):
                _, img2 = task2.cap.read()
                task2.video_loop()
                img2 = cv2.resize(img2, (640, 480))
                cv2.putText(img2, "Follow up and repeat 20 times", (70, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 0, 255), 2)
                imgex = img2
            elif (task3_count != rand):
                _, img3 = task3.cap.read()
                task3.video_loop()
                img3 = cv2.resize(img3, (640, 480))
                cv2.putText(img3, "Follow up and repeat 20 times", (70, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 0, 255), 2)
                imgex = img3
            elif (task4_count != rand):
                _, img4 = task4.cap.read()
                task4.video_loop()
                img4 = cv2.resize(img4, (640, 480))
                cv2.putText(img4, "Follow up and repeat 20 times", (70, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 0, 255), 2)
                imgex = img4
            elif (task5_count != rand):
                _, img5 = task5.cap.read()
                task5.video_loop()
                img5 = cv2.resize(img5, (640, 480))
                cv2.putText(img5, "Follow up and repeat 20 times", (70, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 0, 255), 2)
                imgex = img5
            elif (task6_count != rand):
                _, img6 = task6.cap.read()
                task6.video_loop()
                img6 = cv2.resize(img6, (640, 480))
                cv2.putText(img6, "Follow up and repeat 20 times", (70, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 0, 255), 2)
                imgex = img6
            elif (task7_count != rand):
                _, img7 = task7.cap.read()
                task7.video_loop()
                img7 = cv2.resize(img7, (640, 480))
                cv2.putText(img7, "Follow up and repeat 20 times", (70, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 0, 255), 2)
                imgex = img7
            elif (task8_count != rand):
                _, img8 = task8.cap.read()
                task8.video_loop()
                img8 = cv2.resize(img8, (640, 480))
                cv2.putText(img8, "Follow up and repeat 20 times", (70, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                            (0, 0, 255), 2)
                imgex = img8
            else:
                if(i==2):
                    #print(totaltime)
                    totaltime = time.time()-totaltime
                    #print(totaltime)
                    totaltime = round(totaltime/60,2)
                    i = 0
                    #print(totaltime)
                cv2.putText(imgover, "Congratulations", (70, 100), cv2.FONT_HERSHEY_PLAIN, 4,
                            (255, 255, 255), 2)
                cv2.putText(imgover, "Total time spent: "+str(totaltime)+" minutes", (50, 250), cv2.FONT_HERSHEY_PLAIN, 2,
                            (255, 255, 255), 2)
                imgex = imgover
            #frame_counter += 1

            img = detector.findPose(img)
            lmList = detector.findPosition(img, draw=False)
            #angle = detector.findAngle(img,24,12,14)
            #print(angle)



            if len(lmList) != 0:
                if (task1_count !=rand):
                    task_pose_check1(lmList[14][2], lmList[13][2], lmList[12][2],lmList[16][1],lmList[14][1])
                    task_count =task1_count
                elif (task2_count !=rand):
                    task_pose_check2(lmList[16][2], lmList[15][2], lmList[0][2],lmList[12][2])
                    task_count = task2_count
                elif (task3_count != rand):
                    task_pose_check3(lmList[16][2],lmList[14][2],lmList[11][2],lmList[13][2],
                     lmList[12][2],lmList[24][2],lmList[14][2])
                    task_count = task3_count
                elif (task4_count != rand):
                    task_pose_check4(lmList[15][2], lmList[16][2], lmList[0][2], lmList[24][2])
                    task_count = task4_count
                elif (task5_count != rand):
                    task_pose_check5(lmList[16][2], lmList[15][2],lmList[16][1], lmList[15][1],
                                     lmList[0][2], lmList[24][2])
                    task_count = task5_count
                elif (task6_count != rand):
                    task_pose_check6(lmList[16][2], lmList[15][2], lmList[0][2],
                                     lmList[14][1], lmList[12][1], lmList[11][1], lmList[13][1],
                                     lmList[24][2])
                    task_count = task6_count
                elif (task7_count != rand):
                    task_pose_check7(lmList[14][2], lmList[12][2], lmList[13][2], lmList[11][2],
                                     lmList[14][1], lmList[12][1], lmList[13][1], lmList[11][1])
                    task_count = task7_count
                elif (task8_count != rand):
                    task_pose_check8(lmList[16][1], lmList[24][1], lmList[15][1], lmList[23][1],
                                     lmList[12][1], lmList[11][1])
                    task_count = task8_count
                else:
                    pass
                #print(state1)
                #print(lmList[14][2])
                #print(lmList[12][2])
                #cv2.circle(img, (lmList[0][1], lmList[0][2]), 15, (255, 255, 255), cv2.FILLED)
                #cv2.rectangle(img, (lmList[8][1], lmList[8][2]), (lmList[7][1], lmList[7][2]),(0, 0, 0),10)

            #显示帧率
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                        (0, 255, 255), 1)
            cv2.putText(img, "System automatic counting:"+str(task_count), (100, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                        (255, 0, 255), 2)

            img = cv2.resize(img, (640, 480))

            #test()

            imgall = np.hstack((imgex,img))
            #cv2.imshow("Image", img1)
            #cv2.imshow("Image", imgall)
            imgall = cv2.cvtColor(imgall,cv2.COLOR_BGR2RGB)
            #cv2.waitKey(1)
            placeholder.empty()
            #cv2.waitKey(1)
            placeholder.image(imgall, caption='If the right display is black, please adjust your camera')


        except:
            imgblack = np.zeros((480, 640, 3), dtype=np.uint8)
            #imgwhite = 255 * img
            imgall = np.hstack((imgex, imgblack))
            #cv2.imshow("Image", imgall)
            imgall = cv2.cvtColor(imgall, cv2.COLOR_BGR2RGB)
            #cv2.waitKey(10)
            placeholder.empty()
            placeholder.image(imgall, caption='If the right display is black, please adjust your camera')


def task_pose_check1(checkpoint1y, checkpoint2y, holdpointy,checkpoint3x,checkpoint4x):
    global state1,task1_count,totaltime,i
    #print("16"+checkpoint3x)
    #print("14" + checkpoint4x)
    if(checkpoint3x>checkpoint4x):

        if(holdpointy-checkpoint1y>50 and holdpointy-checkpoint2y>50 and checkpoint3x):
            state1=1
            #print("aaa")
        if(holdpointy - checkpoint1y <0 and holdpointy - checkpoint1y <0 and state1==1):
            task1_count +=1
            state1 = 0
            if(i==0):
                totaltime = time.time()
                i=2


def task_pose_check2(checkpoint1y,checkpoint2y, holdpoint1y,holdpoint2y):
    global state2,task2_count

    if(holdpoint1y-checkpoint1y>50 and holdpoint1y-checkpoint2y>50):
        state2=1
        #print("aaa")
    if(holdpoint2y - checkpoint1y <0 and holdpoint2y - checkpoint1y <0 and state2==1):
        task2_count +=1
        state2 = 0
#task_pose_check2(lmList[16][2], lmList[15][2], lmList[0][2],lmList[12][2])
def task_pose_check3(checkpoint1y,checkpoint2y,checkpoint3y,checkpoint4y,
                     holdpoint1y,holdpoint2y,checkpoint5y):
    global state3,task3_count

    if(abs(checkpoint1y-checkpoint2y)<10 and abs(checkpoint1y-checkpoint2y)<10):
        dis = holdpoint2y-holdpoint1y
        if(checkpoint5y-holdpoint1y< dis/3):
            state3=1
        if(checkpoint5y-holdpoint1y> dis/3*2 and state3==1):
            task3_count +=1
            state3 = 0
        #print("bbb")
#lmList[15][2], lmList[16][2], lmList[0][2], lmList[24][2]
def task_pose_check4(checkpoint1y,checkpoint2y, holdpoint1y,holdpoint2y):
    global state4,task4_count

    if(holdpoint1y-checkpoint1y>50 and checkpoint2y-holdpoint2y>50):
        state4=1
        #print("aaa")
    if(holdpoint2y-checkpoint1y>50 and checkpoint1y-holdpoint2y>50 and state4==1):
        task4_count +=1
        state4 = 0
#lmList[16][2], lmList[15][2], lmList[0][2], lmList[24][2]
def task_pose_check5(checkpoint1y,checkpoint2y, checkpoint1x,checkpoint2x,
                     holdpoint1y,holdpoint2y):
    global state5,task5_count

    if(holdpoint1y-checkpoint1y>50 and holdpoint1y-checkpoint2y>50):
        if(checkpoint1x-checkpoint2x<5):
            state5=1
        #print("aaa")
    if(holdpoint2y - checkpoint1y <0 and holdpoint2y - checkpoint2y <0 and state5==1):
        task5_count +=1
        state5 = 0
#lmList[16][2], lmList[15][2], lmList[0][2],
#lmList[14][1], lmList[12][1], lmList[11][1], lmList[13][1],
#lmList[24][2]
def task_pose_check6(checkpoint1y,checkpoint2y, holdpoint1y,
                     checkpoint3x,checkpoint4x,holdpoint2x,holdpoint3x,holdpoint4y):
    global state6,task6_count

    if(holdpoint1y-checkpoint1y>50 and holdpoint1y-checkpoint2y>50):
        state6=1
        #print("aaa")
    if(abs(checkpoint3x - holdpoint3x)<1 and abs(checkpoint4x - holdpoint2x)<1 and state6==1):
        state6 = 2
    if(checkpoint1y-holdpoint4y>50 and checkpoint2y-holdpoint4y>50 and state6==2):
        state6 = 0
        task6_count += 1

def task_pose_check7(checkpoint1y,holdpoint1y,checkpoint2y,holdpoint2y,
                     checkpoint1x,holdpoint1x,checkpoint2x,holdpoint2x):
    global state7,task7_count

    if(abs(holdpoint1y-checkpoint1y<1) and abs(holdpoint2y-checkpoint2y<1)):
        state7=1
        #print("aaa")
    if(abs(holdpoint1x-checkpoint1x<1) and abs(holdpoint2x-checkpoint2x<1) and state7==1):
        task7_count +=1
        state7 = 0
#lmList[16][1], lmList[24][1], lmList[15][1], lmList[23][1],
#lmList[12][1], lmList[11][1]
def task_pose_check8(checkpoint1x,holdpoint1x,checkpoint2x,holdpoint2x,
                     holdpoint3x,holdpoint4x):
    global state8,task8_count

    if(holdpoint1x-checkpoint1x>50 and checkpoint2x-holdpoint2x>50):
        state8=1
        #print("aaa")
    if(abs(holdpoint3x - checkpoint1x)<1 and abs(holdpoint4x - checkpoint2x)<1 and state8==1):
        task8_count +=1
        state8 = 0

def test():
    imgblack = np.zeros((480, 640, 3), dtype=np.uint8)
    # imgwhite = 255 * img
    imgall = np.hstack((imgex, imgblack))
    cv2.imshow("Image", imgall)
    imgall = cv2.cvtColor(imgall, cv2.COLOR_BGR2RGB)
    # cv2.waitKey(10)
    placeholder.empty()
    placeholder.image(imgall, caption='If the right display is black, please adjust your camera')


if __name__ == "__main__":
    state1 = 0
    task1_count = 0
    state2 = 0
    task2_count = 0
    state3 = 0
    task3_count = 0
    state4 = 0
    task4_count = 0
    state5 = 0
    task5_count = 0
    state6 = 0
    task6_count = 0
    state7 = 0
    task7_count = 0
    state8 = 0
    task8_count = 0
    i = 0
    totaltime =0
    imgall =0
    #
    show()
    placeholder = st.empty()
    main()

    st.image(imgall, caption='If the right display is black, please adjust your camera')
