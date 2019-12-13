from tqdm import tqdm
import caffe
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys
# from solver_type import GPU_SOLVER

#Program to segment images

def image_histogram_equalization(image, number_bins=256):
    # from http://www.janeriksolem.net/2009/06/histogram-equalization-with-python-and.html

    # get image histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = 255 * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape), cdf


def seg_images(test_dir, out_dir, deploy_path, weights_path, test_size=(600, 400), link_file="data.csv"):
    images = os.listdir(test_dir)
    caffe.set_device(0)
    caffe.set_mode_gpu()
    # print(deploy_path)
    net = caffe.Net(deploy_path, weights_path, caffe.TEST)
    collect_names = list()

    if len(images) == 0:
        raise Exception("No images found in: {}".format(test_dir))
        return 

    for img in tqdm(images):
        img_path = os.path.join(test_dir, img)
        blank_name = img.split(".")[0]
        im = Image.open(img_path)
        org_size = im.size
        im = im.resize(test_size, Image.ANTIALIAS)
        data_in = np.array(im).T[np.newaxis]
        # img, _ = image_histogram_equalization(data_in, number_bins=256)

        net.blobs['data'].data[...] = data_in
        net.forward()

        image_out = (net.blobs["sigmoid-score1"].data)[0, 0].T

        out = Image.fromarray((image_out * 255).astype(np.uint8))
        out = out.resize(org_size, Image.ANTIALIAS)
        pred_name = blank_name + ".png"
        pred_path = os.path.join(out_dir, pred_name)
        out.save(pred_path, "PNG")
        collect_names.append((img_path, pred_path))
    out_data = os.path.join(out_dir, link_file)
    with open(out_data, "w+") as f:
        for (gt, pred) in collect_names:
            f.write("{},{}\n".format(gt, pred))
    
    



def main():
    # Get path to images which should be segmented, alias test dataset
    # IN_DIR = sys.argv[1]
    test_size = (600,400)


    if GPU_SOLVER:
        DEPLOY_PATH = "/media/compute/homes/galberding/FleckDetect/deploy.prototxt"
        MODEL_PATH =  "/media/compute/homes/galberding/FleckDetect/snapshot/ras_iter_23300.caffemodel"
        # MODEL_PATH =  "/media/compute/homes/galberding/FleckDetect/modelZoo/sgd_auth.caffemodel"
        # SOC - Dataset
        # IN_DIR = IMAGE_PATH = "/media/compute/homes/galberding/FleckDetect/SOC6K_Release/ValSet/Imgs/"
        # MSRA-B - Dataset
        # IN_DIR  = "/media/compute/homes/galberding/FleckDetect/MSRA-B/Img/images/"
        IN_DIR  = "/media/compute/homes/galberding/FleckDetect/FleckDataSet/val/original/"
        # OUT_DIR = "/media/compute/homes/galberding/FleckDetect/msra_adam_preds/"
        OUT_DIR = "/media/compute/homes/galberding/FleckDetect/current_fleck_set/"
        caffe.set_device(0)
        caffe.set_mode_gpu()
    else:
        DEPLOY_PATH = "/home/schorschi/Documents/SS19/FleckDetect/CaffeTut/deploy.prototxt"
        MODEL_PATH =  "/home/schorschi/Documents/SS19/FleckDetect/CaffeTut/ras_iter_10000.caffemodel"
        IN_DIR = IMAGE_PATH = "/home/schorschi/Documents/SS19/FleckDetect/CaffeTut/test/"
        caffe.set_mode_cpu()


    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    #


    # load model
    net = caffe.Net(DEPLOY_PATH, MODEL_PATH, caffe.TEST)
    images = None

    for (dirpath, dirnames, filenames) in os.walk(IN_DIR):
        images = (filenames)
        break

    # print(images)
    collect_names = list()

    for img_name in images:
        # print(img_name)
        if img_name[-4:] is "_adam.png":
            continue
        try:


            im = Image.open(IN_DIR + img_name)
            # print("open")
            org_size = im.size
            im = im.resize(test_size)
            print("Segmenting "+ img_name)

            # Show the network
            # print(net.inputs)
            data_in = np.array(im).T[np.newaxis]

            img, _ = image_histogram_equalization(data_in, number_bins=256)

            print(img.shape)

            net.blobs['data'].data[...] = data_in

            net.forward()
            print("writing image: "+ img_name)
            image_out = (net.blobs["sigmoid-score1"].data)[0,0].T
            # print(image_out.shape)
            #
            # plt.imshow(image_out)
            # plt.savefig(IN_DIR +img_name[:-4]+"_mat.png")

            out = Image.fromarray((image_out * 255).astype(np.uint8))
            out = out.resize(org_size , Image.ANTIALIAS)
            pred_name = img_name[:-4]+"_pred.png"
            out.save(OUT_DIR  + pred_name, "PNG")
            collect_names.append((img_name, pred_name))

            # image_out = Image.fromarray((image_out))
            # im.save(IN_DIR +img_name[:-4]+"_mat.png")
        except IOError as e:
            print ("cannot create thumbnail for '%s'" % img_name)
            print(e)

    # Create association between lables and predictions
    with open(OUT_DIR+"data.csv", "w+") as f:
        for (gt, pred) in collect_names:
            f.write("{},{}\n".format(gt, pred))



if __name__ == "__main__":
    main()
