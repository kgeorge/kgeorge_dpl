
from __future__ import division, print_function, absolute_import
from inspect import getsourcefile
import os.path
import sys
import time



import numpy as np
import re
import cPickle
import argparse
import urllib
from urlparse import urlparse
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#load data
dataset_dir='data/vgg-16'
urlrefs_filename='fall11_urls.txt'
r_url = re.compile('^\s*(?P<imgid>\S+)\s*(?P<url>\S+)')
import os
import cPickle
from PIL import Image


LIMIT_ASPECT=3.0/2.0
def is_image_good(filepath):
    b_good_img=False
    try:
        with Image.open(filepath) as im:
            aspect=float(im.size[0])/im.size[1]
            if aspect < 1.0:
                aspect = 1.0/aspect
            assert(aspect >= 1.0)
            if aspect <= LIMIT_ASPECT:
                b_good_img=True
            else:
                print('image {f} aspect {a} not good'.format(f=filepath, a=aspect))
    except IOError:
        pass
    return b_good_img


checkpoint_path = os.path.join('data', 'chkpt',  'test_save.ckpt')

tgt_dir = os.path.join(dataset_dir, 'imagenet', 'src')
cropped_tgt_dir = os.path.join(dataset_dir, 'imagenet', 'cropped')

def main(N=10):
    global tgt_dir, cropped_tgt_dir
    n_downloaded=0
    if not os.path.isdir(tgt_dir):
                os.makedirs(tgt_dir)

    if not os.path.isdir(cropped_tgt_dir):
                os.makedirs(cropped_tgt_dir)
    '''
    for l in file(os.path.join(dataset_dir, urlrefs_filename)):
        if not l:
            continue
        l = l.strip()
        m= r_url.match(l)
        if m:
            p=urlparse(m.group('url'))
            filename=os.path.split(p.path)[1]
            new_filename= m.group('imgid') + os.path.splitext(filename)[1]

            #print('img name: {n}, url={u}, path={p}'.format(n=new_filename, u=m.group('url'), p=filename))

            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            new_filepath=os.path.join(tgt_dir, new_filename)
            if not os.path.isfile(new_filepath):
                try:
                    new_filepath, _ = urllib.urlretrieve(m.group('url'), new_filepath, _progress)
                except IOError:
                    continue

            statinfo = os.stat(new_filepath)
            if is_image_good(new_filepath):
                n_downloaded +=1

            if n_downloaded >= N:
                print('downloaded {n} images to {t}'.format(n=N, t=tgt_dir))
                break
    '''
    baseline_dim=256.0
    cropped_dim=224
    for root, dirs, files in os.walk(tgt_dir):
            for f in files:
                src_img_path=os.path.join(root, f)
                if is_image_good(src_img_path):
                    with Image.open(src_img_path) as im:
                        cols=im.size[0]
                        rows=im.size[1]
                        new_rows = baseline_dim
                        new_cols = baseline_dim * cols/rows
                        if cols < rows:
                            new_cols =  baseline_dim
                            new_rows =  baseline_dim * rows/cols
                        new_cols = int(new_cols)
                        new_rows= int(new_rows)
                        im = im.resize((new_cols, new_rows), Image.ANTIALIAS)
                        center_c=int(new_cols/2)
                        center_r= int(new_rows/2)
                        assert(center_c >= int(baseline_dim/2.0))
                        assert(center_r >= int(baseline_dim/2.0))
                        im = im.crop((center_c-112, center_r-112, center_c+112, center_r+112))
                        print('size of cropped image {f}  is {s}'.format(f=f, s=im.size))
                        im.save(os.path.join(cropped_tgt_dir, f))
    pass




if __name__ == "__main__":
    main()
    pass