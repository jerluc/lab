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
width, height = 28, 28

def load_fonts():
    fonts = [f for f in listdir(font_dir) if isfile(join(font_dir, f)) and 'ttf' in f]
    assert len(fonts) > 0, 'No TTF fonts found'
    return fonts

def rand_font():
    return '%s/%s' % (font_dir, random.choice(fonts))

def create_image(input):
    txt, fonts = input

    bgshade = 0.#random.randint(128, 256)
    font_face = rand_font()
    fontsize = 20 
    font = ImageFont.truetype(font_face, size=fontsize)
    x, y = 0, 0 
    fgshade = 1.#random.randint(0, bgshade - 50)
    
    image = Image.new('F', (width, height), bgshade)

    draw = ImageDraw.Draw(image)
    draw.text((x, y), txt, fgshade, font=font)

    im_r = misc.fromimage(image)
    im_r = ndimage.interpolation.rotate(im_r, random.randint(-10, 10), cval=bgshade)

    txt_image = misc.toimage(im_r)

    bg_image = Image.new('F', (width, height), bgshade)
    txt_image = txt_image.crop(image.getbbox())
    txt_image = txt_image.resize((20, 20), Image.ANTIALIAS)
    txt_width, txt_height = txt_image.size
    bg_image.paste(txt_image, ((width - txt_width) / 2, (height - txt_height) / 2))

    im_r = misc.fromimage(bg_image, flatten=1)
    
    flattened = im_r.flatten()

    return flattened
    

if __name__ == '__main__':
    # Setup directories
    assert exists(font_dir), 'Font directory must exist'

    # Load fonts
    fonts = load_fonts()

    pool = Pool(processes=16)

    f_dim = len(charset) * sample_size

    labels = np.ndarray([f_dim], dtype='int64')
    idx = 0
    features = np.ndarray([labels.shape[0], width * height], dtype='float32')
    sys.stdout.write('Generating dataset:')
    sys.stdout.flush()
    for i in range(len(charset)):
        c = charset[i]
        assert ord(c) > 0, 'WTF'
        sys.stdout.write(' %s' % c)
        sys.stdout.flush()
        images = pool.map(create_image, [(c, fonts) for _ in xrange(sample_size)])
        assert len(images) == sample_size

        for j in range(len(images)):
            labels[idx] = ord(c)
            features[idx, :] = images[j]
            idx += 1

    assert len(labels) == len(features) == f_dim

    assert sum(int(i) == 0 for i in labels) == 0, 'Invalid labels'

    print('\nSplitting out training data from test and validation sets')
    randomized = [i for i in range(f_dim)]
    random.shuffle(randomized)
    
    idx_train = randomized[:int(.7 * f_dim)]
    idx_rest = randomized[int(.7 * f_dim):]
    idx_test, idx_validate = idx_rest[:len(idx_rest) / 2], idx_rest[len(idx_rest) / 2:]

    assert len(idx_train) + len(idx_test) + len(idx_validate) == f_dim

    training = features[idx_train, :], labels[idx_train]
    testing = features[idx_test, :], labels[idx_test]
    validation = features[idx_validate, :], labels[idx_validate]

    datasets = (training, validation, testing)

    print('Saving dataset to file')
    with open('./charset.pkl', 'w+b') as f:
        cPickle.dump(datasets, f)

