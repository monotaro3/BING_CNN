#!coding:utf-8

import cv2 as cv
import numpy as np
import math
import time
from concurrent import futures
import os
import chainer
from chainer import report, training, Chain, datasets, iterators, optimizers,cuda,serializers
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import tuple_dataset
from datetime import datetime

from cnn_structure import vehicle_classify_CNN

class bingwindow():
    def __init__(self,rproposal):
        self.xmin = rproposal[0]
        self.ymin = rproposal[1]
        self.xmax = rproposal[2]
        self.ymax = rproposal[3]
        self.result = None
        self.vehiclecover = False

    def windowimg(self,img): #arg:RGB image
        return cv.resize(img[self.ymin:self.ymax+1,self.xmin:self.xmax+1,:],(48,48)).transpose(2,0,1)/255.

    def cover(self,bbox):
        if calcIoU([self.xmin,self.ymin,self.xmax,self.ymax],bbox) > 0.3:
            self.vehiclecover = True
            return True
        else:
            return False
        # bboxcenter = bbox[0] + int((bbox[2]-bbox[0])/2),bbox[1] + int((bbox[3]-bbox[1])/2)
        # windowcenter = int((self.xmin + self.xmax)/2),int((self.ymin + self.ymax)/2)
        # windowsize = int(((self.xmax -self.xmin + 1) +  (self.ymax -self.ymin + 1))/2)
        # if math.sqrt((bboxcenter[0]-windowcenter[0])**2 + (bboxcenter[1]-windowcenter[1])**2) < windowsize*0.45:
        #     self.vehiclecover = True
        #     return True
        # else:
        #     return False

    def draw(self,img):
        if self.result == 1 and self.vehiclecover == True: #True Positive with red
            cv.rectangle(img, (self.xmin, self.ymin), (self.xmax, self.ymax), (0, 0, 255))
        elif self.result == 0 and self.vehiclecover == True: #False Negative with green
            cv.rectangle(img, (self.xmin, self.ymin), (self.xmax, self.ymax), (0, 255, 0))
        elif self.result == 1 and self.vehiclecover == False: #False Positive with blue
            cv.rectangle(img, (self.xmin, self.ymin), (self.xmax, self.ymax), (255, 0, 0))

    def draw_(self,img):
        cv.rectangle(img, (self.xmin, self.ymin), (self.xmax, self.ymax), (0, 255, 0))

def readBINGproposals(filepath,number):
    bing_txt = open(filepath,"r")
    bboxes = []
    line = bing_txt.readline()
    line = bing_txt.readline()
    i = 1
    while (line and i <= number):
        d, xmin, ymin, xmax, ymax = line.split(",")
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        bboxes.append([xmin, ymin, xmax, ymax])
        line = bing_txt.readline()
        i += 1
    return bboxes

def make_bboxeslist(gt_file):
    gt_txt = open(gt_file,"r")
    bboxes = []
    line = gt_txt.readline()
    while (line):
        category, xmin, ymin, xmax, ymax = line.split(",")
        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)
        bboxes.append([xmin, ymin, xmax, ymax])
        line = gt_txt.readline()
    return bboxes

def predictor(cnn_path,data,batch,gpu = 0):
    model = L.Classifier(vehicle_classify_CNN())
    optimizer = optimizers.SGD()
    serializers.load_npz(os.path.join(cnn_path,"gradient_cnn.npz"), model)
    optimizer.setup(model)
    serializers.load_npz(os.path.join(cnn_path,"gradient_optimizer.npz"), optimizer)

    if gpu == 1:
        model.to_gpu()
    r = list(range(0, len(data), batch))
    r.pop()
    # results = np.empty((0,1),int)
    # result = None
    for i in r:
        if gpu == 1:x = cuda.to_gpu(data[i:i+batch])
        else:x = data[i:i+batch]
        result = F.softmax(model.predictor(x).data).data.argmax(axis=1)
        if gpu == 1:result = cuda.to_cpu(result)
        if i == 0:
            results = result
        else:
            results = np.concatenate((results, result), axis=0)
    if len(r) == 0:j=0
    else:j = i + batch
    if gpu == 1:x = cuda.to_gpu(data[j:])
    else:x = data[j:]
    result = F.softmax(model.predictor(x).data).data.argmax(axis=1)
    if gpu == 1: result = cuda.to_cpu(result)
    if len(r) == 0:
        results = result
    else:
        results = np.concatenate((results, result), axis=0)
    return results

def calcIoU(a, b):  # (xmin,ymin,xmax,ymax)
    if a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3]:
        return 0
    else:
        x = [a[0], a[2], b[0], b[2]]
        y = [a[1], a[3], b[1], b[3]]
        x.sort()
        y.sort()
        x = x[1:3]
        y = y[1:3]
        intersect = (x[1] - x[0]) * (y[1] - y[0])
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersect
        IoU = intersect / union
        return IoU

def main():
    logfile = open("bing_cnn.log", "a")
    date = datetime.now()
    startdate = date.strftime('%Y/%m/%d %H:%M:%S')
    f_startdate = date.strftime('%Y%m%d_%H%M%S')
    print("execution:" + startdate)
    print("execution:" + startdate, file=logfile)
    exec_time = time.time()

    imgpath = "C:/work/vehicle_detection/images/test/ny_mall2.tif"
    cnn_path = "C:/work/PycharmProjects/gradient_slide_cnn/"

    n_proposals = 10000

    print("image:" + imgpath)
    print("image:" + imgpath, file=logfile)

    root, ext = os.path.splitext(imgpath)
    bing_file = root + ".bng"
    gt_file = root + ".txt"

    rproposals = readBINGproposals(bing_file,n_proposals)

    # #大きいウィンドウを排除
    # _rproposals =[]
    # for i in rproposals:
    #     if (i[2] - i[0] + 1)*(i[3]-i[1]+1) < 3000:
    #         _rproposals.append(i)
    # rproposals = _rproposals
    # print(len(rproposals))

    bingwindows = []

    for i in rproposals:
        bingwindows.append(bingwindow(i))

    vehicle_list = make_bboxeslist(gt_file)
    vehicle_detected = [False]*len(vehicle_list)

    img = cv.imread(imgpath)

    # img_ = np.array(img)
    # for i in bingwindows:
    #     i.draw_(img_)
    # w = 0.5
    # x,y,c = img_.shape
    # x = int(x*w)
    # y = int(y*w)
    # img_ = cv.resize(img_,(y,x))
    # cv.imshow("test",img_)
    # cv.waitKey(0)
    # cv.destroyAllWindows()



    windowimgs = []
    for i in bingwindows:
        windowimgs.append(i.windowimg(img))

    npwindows = np.array(windowimgs, np.float32)
    mean_image = np.load(os.path.join(cnn_path,"mean_image.npy"))
    npwindows -= mean_image

    print("predicting windows...")
    print("predicting windows...",file=logfile)
    start = time.time()
    results = predictor(cnn_path,npwindows,50,gpu=1)
    end = time.time()
    time_predicting = end - start
    print("finished.(%.3f seconds)" % time_predicting)
    print("finished.(%.3f seconds)" % time_predicting,file=logfile)

    for i in range(len(bingwindows)):
        bingwindows[i].result = results[i]

    print("analyzing results...")
    print("analyzing results...", file=logfile)
    start = time.time()
    for i in range(len(vehicle_list)):
        for j in range(len(bingwindows)):
            if bingwindows[j].result == 1 and bingwindows[j].cover(vehicle_list[i]):
                vehicle_detected[i] = True
    end = time.time()
    time_analysis = end - start
    print('finished.(%.3f seconds)' % time_analysis)
    print('finished.(%.3f seconds)' % time_analysis, file=logfile)

    TP,TN,FP,FN = 0,0,0,0
    detectobjects = 0

    for i in bingwindows:
        if i.result == 1 and i.vehiclecover == True:TP += 1
        elif i.result == 0 and i.vehiclecover == False:TN += 1
        elif i.result == 1 and i.vehiclecover == False:FP += 1
        else:FN += 1
        if i.result == 1:detectobjects += 1
        i.draw(img)

    exec_time = time.time() - exec_time

    print("---------result--------")
    print("Overall Execution time  :%.3f seconds" % exec_time)
    print("GroundTruth vehicles    :%d" % len(vehicle_detected))
    print("detected objects        :%d" % detectobjects)
    print("PR(d vehicles/d objects):%d/%d %f" %(vehicle_detected.count(True),detectobjects,vehicle_detected.count(True)/detectobjects))
    print("RR(detected vehicles)   :%d/%d %f" % (vehicle_detected.count(True),len(vehicle_detected),vehicle_detected.count(True)/len(vehicle_detected)))
    print("TP,TN,FP,FN             :%d,%d,%d,%d" % (TP, TN, FP, FN))

    print("---------result--------",file=logfile)  #to logfile
    print("Overall Execution time  :%.3f seconds" % exec_time,file=logfile)
    print("GroundTruth vehicles    :%d" % len(vehicle_detected),file=logfile)
    print("detected objects        :%d" % detectobjects,file=logfile)
    print("PR(d vehicles/d objects):%d/%d %f" %(vehicle_detected.count(True),detectobjects,vehicle_detected.count(True)/detectobjects),file=logfile)
    print("RR(detected vehicles)   :%d/%d %f" % (vehicle_detected.count(True),len(vehicle_detected),vehicle_detected.count(True)/len(vehicle_detected)),file=logfile)
    print("TP,TN,FP,FN             :%d,%d,%d,%d" % (TP, TN, FP, FN),file=logfile)
    print("",file=logfile)

    logfile.close()
    result_img = root + "_result" + f_startdate + ".jpg"
    cv.imwrite(result_img, img)

    w = 0.6
    x,y,c = img.shape
    x = int(x*w)
    y = int(y*w)
    img = cv.resize(img,(y,x))
    cv.imshow("test",img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()

