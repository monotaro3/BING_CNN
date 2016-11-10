#!coding:utf-8
import os
import cv2 as cv
import numpy as np
from main import readBINGproposals,make_bboxeslist,calcIoU




def main():
    imgpath = "C:/work/vehicle_detection/images/test/ny_mall2.tif"
    n_proposals = 10000
    root, ext = os.path.splitext(imgpath)
    bing_file = root + ".bng"
    gt_file = root + ".txt"
    print(bing_file)

    rproposals = readBINGproposals(bing_file, n_proposals)
    rproposals_detect = [False]*len(rproposals)
    print(len(rproposals))

    vehicle_list = make_bboxeslist(gt_file)
    vehicle_detect = [False] * len(vehicle_list)

    img = cv.imread(imgpath)
    img_ = np.array(img)

    for i in range(len(rproposals)):
        cv.rectangle(img_, (rproposals[i][0], rproposals[i][1]), (rproposals[i][2], rproposals[i][3]), (0, 255, 0))

    for i in range(len(vehicle_list)):
        for j in range(len(rproposals)):
            if calcIoU(vehicle_list[i],rproposals[j]) > 0.3:  #IoU threshold
                vehicle_detect[i] = True
                rproposals_detect[j] = True


    for i in range(len(rproposals)):
        if rproposals_detect[i] == True:
            cv.rectangle(img, (rproposals[i][0], rproposals[i][1]),(rproposals[i][2], rproposals[i][3]), (255, 0, 0))

    w = 0.6
    x,y,c = img.shape
    x = int(x*w)
    y = int(y*w)
    img_ = cv.resize(img_,(y,x))
    cv.imshow("test",img_)
    cv.waitKey(0)
    cv.destroyAllWindows()
    img = cv.resize(img,(y,x))
    cv.imshow("test",img)
    cv.waitKey(0)
    cv.destroyAllWindows()


    print("%d/%d %f %%" % (vehicle_detect.count(True),len(vehicle_list),vehicle_detect.count(True)/len(vehicle_list)*100))





if __name__ == "__main__":
    main()
