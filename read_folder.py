import csv
import random

import pandas as pd

import numpy as np

import PIL.Image as Image
from PIL import ImageFile
import os
import shutil
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)



def pixel_equal(image, x, y):
    # get image pixel
    piex = image.load()[x, y]
    threshold1 = 250 #close to white, range0-255
    threshold2 = 30#close to black
    # 比较每个像素点的RGB值是否大于阈值
    if piex[0] > threshold1 and piex[1] > threshold1 and piex[2] > threshold1:
        return 1
    elif piex[0] < threshold2 and piex[1] < threshold2 and piex[2] < threshold2:
        return 2
    else:
        return 3

def get_filenames_using_os(folder_path):
    filenames = os.listdir(folder_path)
    for i in range(0,len(filenames)):
        filenames[i] = filenames[i].replace(".jpg","")
    return filenames


def split(img_floderi,clip_img_patho ):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    clip_img_path =clip_img_patho   # save path after cropping
    img_floder = img_floderi  # 原tif图片文件夹

    img_list = os.listdir(img_floder)
    print(img_list)
    img_size = 256  # image cropping size32-1024

    l = 0  # 表示imageX列image几个
    right_num1 = 0  # count white pixels
    right_num2 = 0  # count black pixels
    for img_name in img_list:
        name = img_name[:-4]
        # print(img_floder + '/' + img_name)
        img0 = Image.open(img_floder + '/' + img_name)  # PIL的形式可以显示size
        img = np.array(Image.open(img_floder + '/' + img_name))  # array的形式才能显示shape
        print(img0.size)

        w, h = img0.size[0], img0.size[1]  # original image width and height
        # print(h, w)
        i = 0
        print("image", i, "")
        for i in range(0, h, img_size*5):
            for j in range(0, w, img_size*5):
                end_i, end_j = i + img_size, j + img_size
                cropped = img[i:end_i, j:end_j]  # cropped image values
                # img_orig = Image.fromarray(cv.cvtColor(cropped, cv.COLOR_BGR2RGB)) # bgr to rgb
                img_orig = Image.fromarray(cropped)  # 实现array到image的转换
                to_image = Image.new('RGB', (img_size, img_size))  # create blank image for cropped tile

                to_image.paste(img_orig, (0, 0))
                # print(to_image.size)
                # to_image.save(clip_img_path + name + '_' + str(m) + '_' + str(n) + ".jpg")
                i = i + 1
                for m in range(0, to_image.size[0]):
                    for n in range(0, to_image.size[1]):
                        if pixel_equal(to_image, m, n) == 1:
                            right_num1 += 1
                        elif pixel_equal(to_image, m, n) == 2:
                            right_num2 += 1
                        else:
                            continue
                # print(right_num1, right_num2)

                # discard images that dont meet requirements
                if right_num1 / (img_size * img_size) < 0.4 and right_num2 / (img_size * img_size) < 0.05:
                    to_image.save(clip_img_path + '_' + name + str(l) + '_' + str(img_size) + ".jpg")
                    right_num1 = 0
                    right_num2 = 0
                else:
                    right_num1 = 0
                    right_num2 = 0
                l += 1

        l = 0

def save2csv(input_data,name_pic,class_pic,output):
    data = []
    y = 0
    for i in input_data:
        print(y,i)
        data.append([pic_name[y], 0, 0, 0, 0, 0, 0, class_pic[y], 0, 0, 0])
        if y == len(class_pic)-1:
            continue
        else:
            y = y+1
    print(data)
    df = pd.DataFrame(data, columns=['image_name', 'patient_id', 'sex', 'age_approx', 'anatom_site_general_challenge',
                                     'diagnosis', 'benign_malignant', 'target', 'tfrecord', 'width', 'height'])
    df.to_csv(output+".csv", index=False)
def get_random_elements(lst, n):
    return random.sample(lst, n)







def empty_directory_if_not_empty(directory):
    # check if directory is empty
    if os.listdir(directory):
        # if not empty, clear the directory
        shutil.rmtree(directory)
        os.mkdir(directory)
    else:
        print(f"The directory '{directory}' is already empty.")


def copy_file(src, dst):
    with open(src, 'rb') as fsrc:
        with open(dst, 'wb') as fdst:

            fdst.write(fsrc.read())
        fdst.close()
addr_list_1 = ["random","spilt","split2"]
addr_list_C = ["C\\0","C\\1","C\\2","C\\3","C\\4"]
addr_list_L = ["L\\0","L\\1","L\\3","L\\4"]
addr_list_Y = ["Y\\0","Y\\1","Y\\3","Y\\4"]
suffle_e = []
if __name__ == '__main__':

    base_addr = "D:\\test\\test\\outside\\random\\all40倍\\"
    for i in addr_list_C:
        filename = get_filenames_using_os(base_addr+i)
        # print(filename)
        random.shuffle(filename)
        # elem = get_random_elements(filename, 1)
        elem = filename
        suffle_e.append(elem)
        empty_directory_if_not_empty(base_addr + addr_list_1[0] + "\\"+i+"\\" )
        for j in elem:

            copy_file(base_addr +"\\"+ i+ "\\"+j, base_addr + addr_list_1[0] + "\\"+i + "\\"+j)
        # print(filename)

    for i in addr_list_L:
        filename = get_filenames_using_os(base_addr+i)
        # print(filename)
        random.shuffle(filename)
        # elem = get_random_elements(filename, 2)
        elem = filename
        suffle_e.append(elem)
        empty_directory_if_not_empty(base_addr + addr_list_1[0] + "\\" + i+ "\\")
        for j in elem:
            copy_file(base_addr +"\\"+ i+ "\\"+j, base_addr + addr_list_1[0] + "\\"+i + "\\"+j)
        # print(filename)
    for i in addr_list_Y:
        filename = get_filenames_using_os(base_addr+i)
        # print(filename)
        random.shuffle(filename)
        # print("获取随机3个数字",get_random_elements(filename, 3))
        # elem = get_random_elements(filename, 3)
        elem = filename
        suffle_e.append(elem)
        empty_directory_if_not_empty(base_addr + addr_list_1[0] + "\\" + i+ "\\")
        for j in elem:
            copy_file(base_addr+"\\"+i+ "\\"+j,base_addr+addr_list_1[0]+"\\"+i+"\\"+j)
            print(base_addr+"\\"+i+ "\\"+j,base_addr+addr_list_1[0]+"\\"+i+"\\"+j)
        # print(filename)
    # for i in addr_list_2:
    #     split(base_addr+i)#crop images
    print(suffle_e)
    # print(get_filenames_using_os("D:\\test\\test\\outside\\sp20\\C\\C-BFH"))
    # filename = get_filenames_using_os("D:\\test\\test\\outside\\sp20\\C\\C-BFH")
    # save2csv(filename,'data.csv')
    for i in addr_list_C:

        empty_directory_if_not_empty(base_addr+"\\"+addr_list_1[1]+"\\"+i+ "\\")
        split(base_addr+addr_list_1[0]+"\\"+i+ "\\",base_addr+"\\"+addr_list_1[1]+"\\"+i+ "\\")
    for i in addr_list_L:
        empty_directory_if_not_empty(base_addr + "\\" + addr_list_1[1] + "\\" + i + "\\")
        split(base_addr+addr_list_1[0]+"\\"+i+ "\\",base_addr+"\\"+addr_list_1[1]+"\\"+i+ "\\")
    for i in addr_list_Y:
        empty_directory_if_not_empty(base_addr + "\\" + addr_list_1[1] + "\\" + i + "\\")
        split(base_addr+addr_list_1[0]+"\\"+i+ "\\",base_addr+"\\"+addr_list_1[1]+"\\"+i+ "\\")
    pic_class = []
    pic_name = []
    empty_directory_if_not_empty("data\\jpeg-dfsp-768x768\\test")
    y = 0
    for i in addr_list_C:
        elem = get_filenames_using_os(base_addr+"\\"+addr_list_1[1]+"\\"+i+"\\")
        print(elem)
        for j in elem:
            copy_file(base_addr+"\\"+addr_list_1[1]+"\\"+i+"\\"+j+".jpg", "data\\jpeg-dfsp-768x768\\test"+"\\"+j+".jpg")
            pic_class.append(y)
            pic_name.append(j)
        y =y+1
    y = 0
    for i in addr_list_L:
        elem = get_filenames_using_os(base_addr + "\\" + addr_list_1[1] + "\\" + i + "\\")
        print(elem)
        for j in elem:
            copy_file(base_addr+"\\"+addr_list_1[1]+"\\"+i+"\\"+j+".jpg",
                  "data\\jpeg-dfsp-768x768\\test"+"\\"+j+".jpg")
            pic_class.append(y)
            pic_name.append(j)
        if y ==1:
            y = 3
        else:
            y = y + 1
            print(y)
    y = 0
    for i in addr_list_Y:
        elem = get_filenames_using_os(base_addr + "\\" + addr_list_1[1] + "\\" + i + "\\")
        print(elem)
        for j in elem:
            copy_file(base_addr+"\\"+addr_list_1[1]+"\\"+i+"\\"+j+".jpg",
                  "\\data\\jpeg-dfsp-768x768\\test"+"\\"+j+".jpg")
            pic_class.append(y)
            pic_name.append(j)
        if y == 1:
            y = 3
        else:
            y = y + 1
            print(y)
    print(pic_name,"\n")
    print(pic_class)
    save2csv(get_filenames_using_os("data\\jpeg-dfsp-768x768\\test"),pic_name,pic_class,"data\\jpeg-dfsp-768x768\\test")
    column_number = 7  # 列的索引是从0开始的
    file_path = 'data\\jpeg-dfsp-768x768\\test.csv'

    # list to store column data
    column_data = []

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        row_i = 0
        for row in reader:
            if row_i == 0:
                row_i = 1
                continue


            column_data.append(row[column_number])
    column_data = [int(item) if item.isdigit() else None for item in column_data]
    print(column_data)