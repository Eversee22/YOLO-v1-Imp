import sys
import os
import argparse
import numpy as np
import _pickle as cPickle
import time

from voc_eval_py3 import voc_eval

def parse_args():
    parser = argparse.ArgumentParser(description="model-evaluate results")
    parser.add_argument('--output', dest='output_dir', default='results', help='results directory', type=str)
    parser.add_argument('target', nargs=1, help='[train|test|valid|eval]', type=str)
    parser.add_argument('--weights', dest='weights', default='weights/yolov1.weights', help='weights path', type=str)
    parser.add_argument('--voc_dir', dest='voc_dir', default='VOCdevkit', type=str)
    parser.add_argument('--year', dest='year', default='2007', type=str)
    parser.add_argument('--image_set', dest='image_set', default='test', type=str)
    parser.add_argument('--classes', dest='class_file', default='data/voc.names', type=str)

    if len(sys.argv) ==1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args

def get_voc_results_file_template(image_set, out_dir = 'results'):
    filename = 'comp4_det_' + image_set + '_{:s}.txt'
    path = os.path.join(out_dir, filename)
    return path

def do_python_eval(devkit_path, year, image_set, classes, output_dir = 'results',model='yolov1', eval_f=None):
    annopath = os.path.join(
        devkit_path,
        'VOC' + year,
        'Annotations',
        '{}.xml')
    imagesetfile = os.path.join(
        devkit_path,
        'VOC' + year,
        'ImageSets',
        'Main',
        image_set + '.txt')
    cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = True if int(year) < 2010 else False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    print('devkit_path=',devkit_path,', year = ',year)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # if not os.path.exists(output_dir+'/yolov1.eval.txt'):
    #     eval_f = open(output_dir+'/yolov1.eval.txt','w')
    #     eval_f.write('{},'.format('weights'))
    #     for cls in classes:
    #       eval_f.write('{},'.format(cls))
    #     eval_f.write('{}\n'.format('mAP'))
    # else:
    #     eval_f = open(output_dir+'/yolov1.eval.txt','a')

    for i, cls in enumerate(classes):
        if cls == '__background__':
            continue
        filename = get_voc_results_file_template(image_set).format(cls)
        rec, prec, ap = voc_eval(
            filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
            use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, model+'_'+cls + '_pr.pkl'), 'wb') as f:
            cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    #print('~~~~~~~~')
    #print('Results:')
    eval_f.write('{},'.format(model))
    for ap in aps:
        #print('{:.3f}'.format(ap))
        eval_f.write('{:.3f},'.format(ap))
    #print('{:.3f}'.format(np.mean(aps)))
    eval_f.write('{:.3f}\n'.format(np.mean(aps)))
    #eval_f.close()

def sort_weights(weights_dir):
    ws = []
    wd = {}
    for wf in os.listdir(weights_dir):
        try:
            wd[int(wf.split('_')[-1].split('.')[0])] = wf
        except Exception as e:
            pass
    listk = list(wd.keys())
    listk.sort()
    for k in listk:
        ws.append(wd[k])

    return ws

def open_evalf(output_dir):
    fname = 'yolov1.eval.txt'
    if not os.path.exists(output_dir + '/' + fname):
        eval_f = open(output_dir + '/' + fname, 'w')
        eval_f.write('{},'.format('weights'))
        for cls in classes:
            eval_f.write('{},'.format(cls))
        eval_f.write('{}\n'.format('mAP'))
    else:
        eval_f = open(output_dir + '/' + fname, 'a')

    return eval_f

def eval_all(args, output_dir, weights_dir='backup/train_org'):
    evalf = open_evalf(output_dir)
    time_start = time.time()
    for wf in sort_weights(weights_dir):
        time_i = time.time()
        os.system('./darknet yolo valid cfg/yolov1/yolo.cfg {0}'.format(weights_dir + "/" + wf))
        print('Evaluating detections')
        baseW = wf.split('.')[0]
        do_python_eval(args.voc_dir, args.year, args.image_set, classes, output_dir, baseW, evalf)
        print('Eval time:{:.4f}'.format(time.time()-time_i))
    evalf.close()
    print('Total time: {:.4f}'.format(time.time()-time_start))

if __name__=='__main__':
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    target = args.target[0]
    with open(args.class_file, 'r') as f:
        lines = f.readlines()

    classes = [t.strip('\n') for t in lines]
    #print(target)
    weights = args.weights
    if target == 'test':
        os.system('./darknet yolo test cfg/yolov1/yolo.cfg {0} data/dog.jpg'.format(weights))
    if target == 'train':
        weights = 'weights/extraction.conv.weights'
        os.system('./darknet yolo train cfg/yolov1/yolo.train.cfg {0} data/dog.jpg'.format(weights))
    if target == 'valid':
        os.system('./darknet yolo valid cfg/yolov1/yolo.cfg {0}'.format(weights))
        print('Evaluating detections')
        baseW = weights.split('/')[-1].split('.')[0]
        evalf = open_evalf(output_dir)
        do_python_eval(args.voc_dir, args.year, args.image_set, classes, output_dir, baseW, evalf)
        evalf.close()
    if target == 'eval':
        eval_all(args, output_dir)
