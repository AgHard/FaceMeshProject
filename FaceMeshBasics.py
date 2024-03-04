import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("Videos/8.mp4")
pTime = 0

mpDraw = mp.solutions.drawing_utils #draw in our faces
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2) #create facemesh
drawSpac = mpDraw.DrawingSpec(thickness=1 , circle_radius=2)

while True:
    success , img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    print(results)
    if results.multi_face_landmarks:#display results
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img , faceLms , mpFaceMesh.FACEMESH_CONTOURS , drawSpac , drawSpac)
            #if you want to print id use enumerate
            for id , lm in enumerate(faceLms.landmark):
                #print(lm)
                ih , iw , ic = img.shape
                x , y = int(lm.x * iw) , int(lm.y * ih)
                print(id , x , y)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 2)#scale thickness

    cv2.imshow("Image", img)
    cv2.waitKey(1)


