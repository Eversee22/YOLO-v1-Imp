import _pickle as cPickle
import os
import numpy as np

path = "results/yolo_org"
class_file = "data/voc.names"
with open(class_file) as f:
    lines = f.readlines()
classes = [l.strip() for l in lines]
steps = [i for i in range(200,1000,200)]
for i in range(1000,40001,1000):
    steps += [i]

def open_evalf(output_dir):
    fname = 'yolov1.eval.txt'
    if not os.path.exists(output_dir + '/' + fname):
        eval_f = open(output_dir + '/' + fname, 'w')
        eval_f.write('{},'.format('steps'))
        for cls in classes:
            eval_f.write('{},'.format(cls))
        eval_f.write('{}\n'.format('mAP'))
    else:
        eval_f = open(output_dir + '/' + fname, 'a')

    return eval_f

evalf = open_evalf('.')
for s in steps:
    evalf.write('{},'.format(s))
    aps = []
    for cls in classes:
        file = os.path.join(path, "yolo_{0}_{1}_pr.pkl".format(s,cls))
        with open(file, 'rb') as f:
            rpa = cPickle.load(f)
            ap = rpa['ap']
            aps += [ap]
            evalf.write('{:.3f},'.format(ap))
    evalf.write('{:.3f}\n'.format(np.mean(aps)))
evalf.close()
print('over')



