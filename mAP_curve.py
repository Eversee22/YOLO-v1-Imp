import matplotlib.pyplot as plt
import csv
import numpy as np

x = []
y = []
classes = [[] for i in range(20)]
names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
         "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
with open('yolov1.eval.txt') as f:
    reader = csv.reader(f)
    data = list(reader)
    for d in data[1:]:
        x.append(int(d[0]))
        y.append(float(d[-1]))
        for i in range(20):
            classes[i].append(float(d[i+1]))
if True:
    plt.figure()
    plt.plot(x,y)
    plt.xlabel('steps')
    plt.ylabel('mAP')
    #plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.xticks(x,rotation=60)
    plt.locator_params('x', nbins=20)
    # plt.locator_params('y', nbins=10)
    plt.grid()
    plt.show()
else:
    plt.figure()
    for i in range(20):
        ax=plt.subplot(10,2,i+1)
        # plt.text(0.4,0.01,names[i])

        plt.plot(x,classes[i],label=names[i],color='g',marker='o',markersize=2,linewidth=1)
        plt.legend()

        plt.xlabel('steps')
        plt.ylabel('mAP')
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        # if i+1 > 18:
        plt.xticks(x, rotation=60)
        ax.locator_params('x',nbins=20)
        ax.locator_params('y',nbins=5)
        # plt.yticks(y, rotation=30)
        # else:
        #     plt.xticks([])
        # if (i+1) % 2:
        #     plt.yticks(y)
        # else:
        #     plt.yticks([])

        #ax.xaxis.grid(True)
        # plt.locator_params('x', nbins=20)
        # plt.locator_params('y', nbins=5)
        plt.grid()
    plt.show()