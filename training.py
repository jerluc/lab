import sys
import string
import random
import cPickle
import numpy as np
from scipy import misc, ndimage
from PIL import Image, ImageDraw, ImageFont
from os import listdir, makedirs
from os.path import isfile, abspath, join, dirname, exists
from multiprocessing.pool import Pool

font_dir = './data/fonts'
charset = string.ascii_letters + string.digits
sample_size = 5000
height, width = 28, 28

def load_fonts():
    fonts = [f for f in listdir(font_dir) if isfile(join(font_dir, f)) and 'ttf' in f]
    assert len(fonts) > 0, 'No TTF fonts found'
    return fonts

def rand_font():
    return '%s/%s' % (font_dir, random.choice(fonts))

def create_image(input):
    txt, fonts = input

    bgshade = random.randint(128, 256)
    bgcolor = (bgshade, bgshade, bgshade)
    font_face = rand_font()
    fontsize = 28#random.randint(16, 28)
    font = ImageFont.truetype(font_face, size=fontsize)
    txt_width, txt_height = font.getsize(txt)
    x, y = 0, 0 
    fgshade = random.randint(0, bgshade - 75)
    fgcolor = (fgshade, fgshade, fgshade)
    
    image = Image.new('RGBA', (height, width), bgcolor)

    draw = ImageDraw.Draw(image)
    draw.text((x, y), txt, fgcolor, font=font)

    im_arr = misc.fromimage(image)
    im_r = ndimage.interpolation.rotate(im_arr, random.randint(-20, 20), cval=bgshade)

    image = misc.toimage(im_r).crop((0, 0, 28, 28))

    return misc.fromimage(image, flatten=1).flatten()
    

if __name__ == '__main__':
    # Setup directories
    assert exists(font_dir), 'Font directory must exist'

    # Load fonts
    fonts = load_fonts()

    pool = Pool(processes=16)

    f_dim = len(charset) * sample_size

    labels = np.ndarray([f_dim])
    features = np.ndarray([labels.shape[0], width * height])
    sys.stdout.write('Generating dataset:')
    sys.stdout.flush()
    for i in range(len(charset)):
        c = charset[i]
        sys.stdout.write(' %s' % c)
        sys.stdout.flush()
        images = pool.map(create_image, [(c, fonts) for _ in xrange(sample_size)])
        assert len(images) == sample_size

        for j in range(len(images)):
            idx = (i + 1) * j
            labels[idx] = ord(c)
            features[idx, :] = images[j]

    assert len(labels) == len(features) == f_dim

    

    print('\nSplitting out training data from test and validation sets')
    randomized = [i for i in range(f_dim)]
    random.shuffle(randomized)
    
    idx_train = randomized[:int(.8 * f_dim)]
    idx_rest = randomized[int(.8 * f_dim):]
    idx_test, idx_validate = idx_rest[:len(idx_rest) / 2], idx_rest[len(idx_rest) / 2:]

    assert len(idx_train) + len(idx_test) + len(idx_validate) == f_dim

    x_train, y_train = features[idx_train, :], labels[idx_train]
    x_test, y_test = features[idx_test, :], labels[idx_test]
    x_validate, y_validate = features[idx_validate, :], labels[idx_validate]

    datasets = [ 
        [x_train, y_train],
        [x_validate, y_validate],
        [x_test, y_test]
    ]

    print('Saving dataset to file')
    with open('./charset.pkl', 'w+b') as f:
        cPickle.dump(datasets, f)

