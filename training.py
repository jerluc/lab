import sys
import string
import random
from scipy import misc, ndimage
from PIL import Image, ImageDraw, ImageFont
from os import listdir, makedirs
from os.path import isfile, abspath, join, dirname, exists
from multiprocessing.pool import ThreadPool

font_dir = './data/fonts'

charset = string.ascii_letters + string.digits

def load_fonts():
    fonts = [f for f in listdir(font_dir) if isfile(join(font_dir, f)) and 'ttf' in f]
    assert len(fonts) > 0, 'No TTF fonts found'
    return fonts

def rand_font():
    return '%s/%s' % (font_dir, random.choice(fonts))

def create_image(input):
    txt, i, fonts, training_dir = input

    height, width = 28, 28 
    bgshade = random.randint(128, 256)
    bgcolor = (bgshade, bgshade, bgshade)
    font_face = rand_font()
    fontsize = random.randint(16, 28)
    font = ImageFont.truetype(font_face, size=fontsize)
    txt_width, txt_height = font.getsize(txt)
    x, y = 0, 0 
    fgshade = random.randint(0, bgshade - 50)
    fgcolor = (fgshade, fgshade, fgshade)
    
    image = Image.new('RGBA', (height, width), bgcolor)

    draw = ImageDraw.Draw(image)
    draw.text((x, y), txt, fgcolor, font=font)

    im_arr = misc.fromimage(image)
    im_r = ndimage.interpolation.rotate(im_arr, random.randint(-20, 20), cval=bgshade)

    image = misc.toimage(im_r).crop((0, 0, 28, 28))

    image.save('%s/%s/%s.jpg' % (training_dir, ord(txt), i), 'JPEG')
    
def create_dir(dir):
    if not exists(dir):
        makedirs(dir)

if __name__ == '__main__':
    assert len(sys.argv) == 2, 'Usage: python training.py TRAINING_DIR'
    training_dir = sys.argv[1]

    # Setup directories
    create_dir(font_dir)
    create_dir(training_dir)

    # Load fonts
    fonts = load_fonts()

    for c in charset:
        create_dir('%s/%s' % (training_dir, ord(c)))
        # Generate training data
        pool = ThreadPool(processes=16)
        pool.map(create_image, [(c, i, fonts, training_dir) for i in xrange(5000)])

