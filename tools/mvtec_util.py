import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import math
from scipy.misc import toimage
import numpy as np

def count_images(path, img_cat, t='augment'):
    
    # training image count
    print(img_cat + ": train ---------------------------")
    train_imgs = os.listdir(path + '/' + img_cat + '/train/good')
    print("training - {}".format(len(train_imgs)))

    # test image count
    print(img_cat + ": test  ---------------------------")
    test_dirs  = os.listdir(path + '/' + img_cat + '/test/')
    for dir in test_dirs:
        test_imgs = os.listdir(path + '/' + img_cat + '/test/' + dir)
        print("test({}) - {}".format(dir, len(test_imgs)))

    # augment image count (if exists)
    if os.path.exists(path + '/' + img_cat + '/' + t + '/') :
        print(img_cat + ": {} -------------------------".format(t))
        aug_dirs = os.listdir(path + '/' + img_cat + '/' + t + '/')
        for dir in aug_dirs:
            aug_imgs = os.listdir(path + '/' + img_cat + '/' + t + '/' + dir)
            print("{}({}) - {}".format(t,dir, len(aug_imgs))) 

def display_sample(path, img_cat, channel, n_img=5):

    img_dir = path + '/' + img_cat + '/' + channel + '/'
    ad_cats = os.listdir(img_dir)
    print(ad_cats)
        
    for ad_cat in ad_cats:
        print(ad_cat)
        img_path = img_dir + ad_cat
        imgs = os.listdir(img_path)
        n_img = len(imgs) if n_img > len(imgs) else n_img

        plt.figure(figsize=(16, max(6,n_img)))
        for i in range(n_img):
            if imgs[i-1][0] == '.': continue
            img = mpimg.imread(img_path + '/' + imgs[i-1])
            plt.subplot(n_img//5 +1, min(5,n_img), i+1)
            plt.imshow(img)
        plt.show()

def augment_image(path, k, degree, folder='test', bgcolor='gray', resize=1., print_interval = 200, t='augment'):
    image_cnt = 0
    for c in os.listdir('{}/{}/{}'.format(path, k,folder)):
        print('path: ' + c)
        if c[0] == '.' : 
            print('skip ' + c)
            continue
            
        image_cnt_in_c = 0
        for i in os.listdir('{}/{}/{}/{}'.format(path, k,folder,c)):
            if i[0] == '.' : 
                print('skip ' + i)
                continue
            else :
                image  = Image.open("{}/{}/{}/{}/{}".format(path, k,folder,c,i)).convert('L')
         
            # image resize
            image = image.resize((int(image.size[0]*resize), int(image.size[1]*resize)))

            
            # target directory가 없으면 생성
            if not os.path.exists("{}/{}/{}/{}".format(path,k,t,c)):
                os.makedirs("{}/{}/{}/{}".format(path,k,t,c))
            
            # degree 만큼 회전하면서 파일 생성
            for r in range(360//degree):
                rimage  = image.rotate(r*degree, fillcolor=bgcolor)
                rimage  = toimage(np.array([math.log(i)*1.25 + 2 for i in np.array(rimage).flatten()/255]).reshape(256,256))
                rimage.save("{}/{}/{}/{}/{}-{}.png".format(path,k,t,c,i,r))
                image_cnt += 1
                image_cnt_in_c += 1
                if (image_cnt_in_c % print_interval == 0) : print("... {} images were generated".format(image_cnt_in_c))
                    
        print("{} images were generated in {}".format(image_cnt_in_c, c))
         
    print("total : {} images were generated".format(image_cnt))

