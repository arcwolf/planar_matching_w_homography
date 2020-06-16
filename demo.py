import numpy as np
import cv2
# import time

ESC = 27 
camera = cv2.VideoCapture(0)
# orb = cv2.ORB_create()
orb = cv2.BRISK_create() # BRISK
bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
MIN_MATCH_COUNT = 10

"""
cv2.BFMatcher(): Brute-Force Matcher
- normType: NORM_HAMMING should be used with ORB, BRISK and BRIEF
- If crossCheck==true, then the knnMatch() method with k=1 will only return pairs (i,j) such that for i-th query descriptor the j-th descriptor in the matcher's collection is the nearest and vice versa.
"""

imgTrainColor = cv2.imread('./img_source/test6.png') # <- sample data
imgTrainGray = cv2.cvtColor(imgTrainColor, cv2.COLOR_BGR2GRAY)
kpTrain = orb.detect(imgTrainGray, None)
kpTrain, desTrain = orb.compute(imgTrainGray, kpTrain) # keypoints, descriptions

is_firstloop = True

# initiate camera
while True:
    ret, imgCamColor = camera.read() # read image from camera
    imgCamGray = cv2.cvtColor(imgCamColor, cv2.COLOR_BGR2GRAY)
    kpCam = orb.detect(imgCamGray, None)
    kpCam, desCam = orb.compute(imgCamGray, kpCam)
    matches = bf.match(desTrain, desCam)

    # shape初期化
    if is_firstloop:
        h1, w1 = imgCamColor.shape[:2]
        h2, w2 = imgTrainColor.shape[:2]
        nWidth = w1 + w2
        nHeight = max(h1, h2)
        hdif = (h1 - h2) / 2
        is_firstloop = False
    
    # 2つの特徴点の距離で対応関係判定
    list_matched_points = []
    for m in matches:
        if m.distance < 100: # <-- マッチング精度の調整に必要
            list_matched_points.append(m)

    # print(len(matches), len(list_matched_points))

    # find Homography (物体の輪郭, squared)
    if len(list_matched_points) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kpTrain[m.queryIdx].pt for m in list_matched_points ]).reshape(-1,1,2)
        dst_pts = np.float32([ kpCam[m.trainIdx].pt for m in list_matched_points ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
        # cv2.RANSAC: robustly filter matches between keypoints in two images using the Random Sample Consensus (RANSAC) algorithm
        matchesMask = mask.ravel().tolist()
        h, w, d = imgTrainColor.shape
        pts = np.float32([ [0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts, M)
        imgCamColor = cv2.polylines(imgCamColor, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    else:
        print( "Not enough matches are found - {}/{}".format(len(list_matched_points), MIN_MATCH_COUNT) )
        matchesMask = None

    # draw feature matching states
    draw_params = dict(
        matchColor = (0, 255, 0), # draw matches in green color
        singlePointColor = None,
        matchesMask = matchesMask, # draw only inliers
        flags = 2
    )
    
    # img_result = cv2.drawMatches(imgTrainColor, kpTrain, imgCamColor, kpCam, matches, None, **draw_params)
    img_result = cv2.drawMatches(imgTrainColor, kpTrain, imgCamColor, kpCam, list_matched_points, None, **draw_params)
    cv2.imshow('Camara', img_result)
    
    key = cv2.waitKey(20)                                 
    if key == ESC:
        break

cv2.destroyAllWindows()
camera.release()