from .social_distancing_config import NMS_THRESH
from .social_distancing_config import MIN_CONF
import numpy as np
import cv2





def detect_people(frame, net, ln, personIdx=0):
    # grab the dimensions of the frame and  initialize the list of
    # results
    (H, W) = frame.shape[:2]
    results = []

    # construct a blob from the input frame and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes
    # and associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    detections = net.forward()

    #detect(frame, detections)

    # initialize our lists of detected bounding boxes, centroids, and
    # confidences, respectively
    boxes = []
    centroids = []
    confidences = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:

            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter detections by (1) ensuring that the object
            # detected was a person and (2) that the minimum
            # confidence is met
            if classID == personIdx and confidence > MIN_CONF:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # centroids, and confidences
                boxes.append([x, y, int(width), int(height)])
                centroids.append((centerX, centerY))
                confidences.append(float(confidence))

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # update our results list to consist of the person
            # prediction probability, bounding box coordinates,
            # and the centroid
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    # return the list of results
    return results

def drawBoundingBox(frame, box, color):
    (startX, startY, endX, endY) = box

    # draw a red rectangle around detected objects
    cv2.rectangle(
        frame, (int(startX), int(startY)), (int(endX), int(endY)), color, thickness=2
    )

def computeCentroid(box):
    (startX, startY, endX, endY) = box
    return np.array([startX + ((endX - startX) / 2), startY + ((endY - startY) / 2)])


def detect(frame, net):
    (H, W) = frame.shape[:2]
    results = []

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    # layerOutputs = net.forward(ln)
    detections = net.forward()

    boxes = []
    centroids = []
    confidences = []



    # loop over the detections
    for detection in detections[0, 0, :, :]:
        confidence = float(detection[2])

        # if the confidence is above a threshold
        if confidence > MIN_CONF:
            classID = detection[1]
            print(classID)

            # proceed only if the object detected is indeed a human
            if classID == 1:
                print('hallo')
                # get coordinates of the bbox
                left = detection[3] * W
                top = detection[4] * H
                right = detection[5] * W
                bottom = detection[6] * H

                #print(right)

                #box = detection[0:4] * np.array([W, H, W, H])
                #(centerX, centerY, width, height) = box.astype("int")

                box = [left, top, right, bottom]

                boxes.append(box)
                centroids.append(computeCentroid(box))
                confidences.append(float(confidence))



    idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # update our results list to consist of the person
            # prediction probability, bounding box coordinates,
            # and the centroid
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)

    # return the list of results
    return results




