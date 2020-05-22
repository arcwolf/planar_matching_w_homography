import cv2
import matplotlib.pyplot as plt

# 画像ファイルの読み込み
img = cv2.imread('./img_source/test5.png')

# detectors (keypoint matching)
dict_detector = {
    "ORB (Oriented FAST and Rotated BRIEF)" : cv2.ORB_create(),
    "AgastFeatureDetector" : cv2.AgastFeatureDetector_create(),
    "MSER" : cv2.MSER_create(),
    "Fast" : cv2.FastFeatureDetector_create(),
    "BRIST" : cv2.BRISK_create(),
    "AKAZE" : cv2.AKAZE_create(),
    "KAZE" : cv2.KAZE_create(),
    "SimpleBlobDetector" : cv2.SimpleBlobDetector_create(),
    "SIFT" : cv2.xfeatures2d.SIFT_create(), # patented in the latest version
    "SURF" : cv2.xfeatures2d.SURF_create(), # patented in the latest version
}

for name, detector in dict_detector.items():
    # 特徴検出
    keypoints = detector.detect(img)
    
    # 画像への特徴点の書き込み
    out = cv2.drawKeypoints(img, keypoints, None)

    # 表示
    plt.imshow(out)
    plt.title(name)
    plt.savefig("./img_result/result_KPM_book_{}".format(name))
    plt.show()

    # # 特徴点, 特徴量
    # keypoints, descriptors = detector.detectAndCompute(img, None)
    
    
# print(keypoints)
# print(descriptors)
