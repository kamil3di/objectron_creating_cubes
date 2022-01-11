import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

# For static images:
IMAGE_FILES = []
with mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            model_name='Shoe') as objectron:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB and process it with MediaPipe Objectron.
    results = objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw box landmarks.
    if not results.detected_objects:
      print(f'No box landmarks detected on {file}')
      continue
    print(f'Box landmarks of {file}:')
    annotated_image = image.copy()
    for detected_object in results.detected_objects:
      mp_drawing.draw_landmarks(
          annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
      mp_drawing.draw_axis(annotated_image, detected_object.rotation,
                           detected_object.translation)
      cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.99,
                            model_name='Shoe') as objectron:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = objectron.process(image)

    #Here I create contour points to draw boxes
    def draw_boxes(detected_object):
      shape = image.shape 
      pointList = []
      for point in range(1,9):
        X = int(detected_object.landmarks_2d.landmark[point].x*shape[1])
        Y = int(detected_object.landmarks_2d.landmark[point].y*shape[0])
        pointList.append(np.array([X,Y]))

      contourPts1 = np.array([[pointList[0]],[pointList[4]],[pointList[5]],[pointList[1]]])
      contourPts2 = np.array([[pointList[2]],[pointList[3]],[pointList[7]],[pointList[6]]])
      contourPts3 = np.array([[pointList[0]],[pointList[1]],[pointList[3]],[pointList[2]]])
      contourPts4 = np.array([[pointList[4]],[pointList[5]],[pointList[7]],[pointList[6]]])
      contourPts5 = np.array([[pointList[0]],[pointList[4]],[pointList[6]],[pointList[2]]])
      contourPts6 = np.array([[pointList[1]],[pointList[5]],[pointList[7]],[pointList[3]]])

      contourList = [contourPts1,contourPts2,contourPts3,contourPts4,contourPts5,contourPts6]
      return contourList

    # Draw the box landmarks on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detected_objects:
        for detected_object in results.detected_objects:
            #This for loop is designed to draw boxes
            contourList = draw_boxes(detected_object)
            for contour in contourList:
              cv2.drawContours(image, [contour.astype(int)],-1,(0,255,0),-3)
            #cv2.rectangle(image,pointfor2,pointfor5,(0,255,0),3)
            mp_drawing.draw_landmarks(
              image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
            mp_drawing.draw_axis(image, detected_object.rotation,
                                 detected_object.translation)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Objectron', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
