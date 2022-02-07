import cv2
import mediapipe as mp
import time
import os
import handtrackingmodule as htm

cap = cv2.VideoCapture(0)
folderPath = "fingerImages"
myList = os.listdir(folderPath)
# print(myList)
overLay = []
pTime = 0
cTime = 0
detector = htm.handDetector()
tipId = [4,8,12,16,20]
for myPath in myList:
    image = cv2.imread(f'{folderPath}/{myPath}')
    # print(f'{folderPath}/{myPath}')
    overLay.append(image)

print(len(overLay))

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    fingerList = detector.findPosition(img, draw=False)
    img = cv2.flip(img, 1)

    # print(fingerList)
    if len(fingerList) != 0:
        fingers = []
        # Thumb
        if fingerList[tipId[0]][1] > fingerList[tipId[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

            # 4 Fingers
        for id in range(1, 5):
            if fingerList[tipId[id]][2] < fingerList[tipId[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

    # print(fingers)
        totalFingers = fingers.count(1)
        # print(totalFingers)
        h, w, c = overLay[totalFingers].shape
        # img[0:h, 0:w] = overLay[totalFingers]
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN,
                    10, (255, 0, 0), 25)



        # if fingerList[8][2] < fingerList[6][2]:
        #     print("index finger open")
    # img[0:200, 0:200] = overLay[0]
    # print(success);
    # imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # results = hands.process(imgRgb)

    # if results.multi_hand_landmarks:
    #     for handLms in results.multi_hand_landmarks:
    #         mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("image", img)
    cv2.waitKey(1)
