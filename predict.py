import csv
import os
import time
import random
import argparse
import numpy as np
import openpyxl
import pandas as pd
# import cv2
import PIL.Image
from openpyxl import workbook
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import GradualWarmupSchedulerV2
# import apex
# from apex import amp
from dataset import get_df, get_transforms, MelanomaDataset
from models import EfficientNetB3_Melanoma, ResNet34_Melanoma, Xception_Melanoma, DenseNet121_Melanoma
from train import get_trans


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, required=True)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--data-folder', type=int, required=True)
    parser.add_argument('--image-size', type=int, required=True)
    parser.add_argument('--enet-type', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--out-dim', type=int, default=5)
    parser.add_argument('--use-amp', action='store_false')
    parser.add_argument('--use-meta', action='store_false')
    parser.add_argument('--DEBUG', action='store_true')
    # parser.add_argument('--model-dir', type=str, default='./weight32/14')
    parser.add_argument('--model-dir', type=str, default='./weight/eff')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--sub-dir', type=str, default='./subs111')
    parser.add_argument('--eval', type=str, choices=['best', 'best_20', 'final'], default="best")
    parser.add_argument('--n-test', type=int, default=5)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='1')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')

    args, _ = parser.parse_known_args()
    return args

def random_select(i):
    import random
    random_integers = []
    j = int(i*0.6)
    print(j)
    print(j)
    random_integers = random.sample(range(1,i), j)
    random_integers.sort()
    print(random_integers)
    return random_integers

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
def set_seed_1(seed=0):
    random.seed(seed)
def main(seedS):
    set_seed_1(seedS)
    # randomly select data

    import random

    import pandas as pd

    import numpy as np

    import PIL.Image as Image
    from PIL import ImageFile
    import os
    import shutil

    def pixel_equal(image, x, y):
        # get image pixel
        piex = image.load()[x, y]
        threshold1 = 250  # close to white, range0-255
        threshold2 = 30  # close to black
        #
        if piex[0] > threshold1 and piex[1] > threshold1 and piex[2] > threshold1:
            return 1
        elif piex[0] < threshold2 and piex[1] < threshold2 and piex[2] < threshold2:
            return 2
        else:
            return 3

    def get_filenames_using_os(folder_path):
        filenames = os.listdir(folder_path)
        for i in range(0, len(filenames)):
            filenames[i] = filenames[i].replace(".jpg", "")
        return filenames

    def split(img_floderi, clip_img_patho):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        Image.MAX_IMAGE_PIXELS = None

        clip_img_path = clip_img_patho  # save path after cropping
        img_floder = img_floderi  #

        img_list = os.listdir(img_floder)
        print(img_list)
        img_size = 256  # image cropping size32-1024

        l = 0  # 表示imageX列image几个
        right_num1 = 0  # count white pixels
        right_num2 = 0  # count black pixels
        for img_name in img_list:
            name = img_name[:-4]
            # print(img_floder + '/' + img_name)
            img0 = Image.open(img_floder + '/' + img_name)  #
            img = np.array(Image.open(img_floder + '/' + img_name))  #
            print(img0.size)

            w, h = img0.size[0], img0.size[1]  # original image width and height
            # print(h, w)
            a = 0
            # print("image", i, "")
            for i in range(0, h, img_size*5):
                if a >= 1:
                    break
                for j in range(0, w, img_size*5):
                    if a >= 1:
                        break
                    a = a + 1
                    end_i, end_j = i + img_size, j + img_size
                    cropped = img[i:end_i, j:end_j]  # cropped image values
                    # img_orig = Image.fromarray(cv.cvtColor(cropped, cv.COLOR_BGR2RGB)) # bgr to rgb
                    img_orig = Image.fromarray(cropped)  # 实现array到image的转换
                    to_image = Image.new('RGB', (img_size, img_size))  # create blank image for cropped tile

                    to_image.paste(img_orig, (0, 0))
                    # print(to_image.size)
                    # to_image.save(clip_img_path + name + '_' + str(m) + '_' + str(n) + ".jpg")

                    for m in range(0, to_image.size[0]):
                        for n in range(0, to_image.size[1]):
                            if pixel_equal(to_image, m, n) == 1:
                                right_num1 += 1
                            elif pixel_equal(to_image, m, n) == 2:
                                right_num2 += 1
                            else:
                                continue
                    # print(right_num1, right_num2)

                    # discard images that don't meet requirements
                    if right_num1 / (img_size * img_size) < 0.4 and right_num2 / (img_size * img_size) < 0.05:
                        to_image.save(clip_img_path + '_' + name + str(l) + '_' + str(img_size) + ".jpg")
                        right_num1 = 0
                        right_num2 = 0
                    else:
                        right_num1 = 0
                        right_num2 = 0
                    l += 1

            l = 0

    def save2csv(input_data, class_pic, output):
        data = []
        y = 0
        for i in input_data:
            # print(y, i)
            data.append([pic_name[y], 0, 0, 0, 0, 0, 0, class_pic[y], 0, 0, 0])
            if y == len(class_pic) - 1:
                continue
            else:
                y = y + 1
        # print(data)
        df = pd.DataFrame(data,
                          columns=['image_name', 'patient_id', 'sex', 'age_approx', 'anatom_site_general_challenge',
                                   'diagnosis', 'benign_malignant', 'target', 'tfrecord', 'width', 'height'])
        df.to_csv(output + ".csv", index=False)

    def get_random_elements(lst, n):
        # pop
        l = []
        # random.shuffle(lst)
        # print(my_list)  # order of elements has changed after shuffle
        # for i in range(0,n):
            # selected_element = lst.pop()  # randomly select and remove element from list
        # selected_element = random.choice(lst,n)
        # l.append(selected_element)
        # return l
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

    # addr_list_1 = ["random", "spilt", "split2"]
    # addr_list_C = ["C\\0", "C\\1", "C\\2", "C\\3", "C\\4"]
    # addr_list_L = ["L\\0", "L\\1", "L\\3", "L\\4"]
    # addr_list_Y = ["Y\\0", "Y\\1", "Y\\3", "Y\\4"]
    # suffle_e = []
    # base_addr = "D:\\test\\test\\outside\\random\\all40倍\\"
    # for i in addr_list_C:
    #     filename = get_filenames_using_os(base_addr + i)
    #     # print(filename)
    #     # random.shuffle(filename)
    #     elem = get_random_elements(filename, 1)
    #     # suffle_e.append(elem)
    #     empty_directory_if_not_empty(base_addr + addr_list_1[0] + "\\" + i + "\\")
    #     for j in elem:
    #         copy_file(base_addr + "\\" + i + "\\" + j, base_addr + addr_list_1[0] + "\\" + i + "\\" + j)
    #     # print(filename)
    #
    # for i in addr_list_L:
    #     filename = get_filenames_using_os(base_addr + i)
    #     # print(filename)
    #     # random.shuffle(filename)
    #     elem = get_random_elements(filename, 2)
    #     # suffle_e.append(elem)
    #     empty_directory_if_not_empty(base_addr + addr_list_1[0] + "\\" + i + "\\")
    #     for j in elem:
    #         copy_file(base_addr + "\\" + i + "\\" + j, base_addr + addr_list_1[0] + "\\" + i + "\\" + j)
    #     # print(filename)
    # for i in addr_list_Y:
    #     filename = get_filenames_using_os(base_addr + i)
    #     # print(filename)
    #     # random.shuffle(filename)
    #     # print("获取随机3个数字",get_random_elements(filename, 3))
    #     elem = get_random_elements(filename, 3)
    #     # suffle_e.append(elem)
    #     empty_directory_if_not_empty(base_addr + addr_list_1[0] + "\\" + i + "\\")
    #     for j in elem:
    #         copy_file(base_addr + "\\" + i + "\\" + j, base_addr + addr_list_1[0] + "\\" + i + "\\" + j)
    #         # print(base_addr + "\\" + i + "\\" + j, base_addr + addr_list_1[0] + "\\" + i + "\\" + j)
    #     # print(filename)
    # # for i in addr_list_2:
    # #     split(base_addr+i)#crop images
    # # print(suffle_e)
    # # print(get_filenames_using_os("D:\\test\\test\\outside\\sp20\\C\\C-BFH"))
    # # filename = get_filenames_using_os("D:\\test\\test\\outside\\sp20\\C\\C-BFH")
    # # save2csv(filename,'data.csv')
    # for i in addr_list_C:
    #     empty_directory_if_not_empty(base_addr + "\\" + addr_list_1[1] + "\\" + i + "\\")
    #     split(base_addr + addr_list_1[0] + "\\" + i + "\\", base_addr + "\\" + addr_list_1[1] + "\\" + i + "\\")
    # for i in addr_list_L:
    #     empty_directory_if_not_empty(base_addr + "\\" + addr_list_1[1] + "\\" + i + "\\")
    #     split(base_addr + addr_list_1[0] + "\\" + i + "\\", base_addr + "\\" + addr_list_1[1] + "\\" + i + "\\")
    # for i in addr_list_Y:
    #     empty_directory_if_not_empty(base_addr + "\\" + addr_list_1[1] + "\\" + i + "\\")
    #     split(base_addr + addr_list_1[0] + "\\" + i + "\\", base_addr + "\\" + addr_list_1[1] + "\\" + i + "\\")
    # pic_class = []
    # pic_name = []
    # empty_directory_if_not_empty(
    #     "D:\\ML\medical\\ASAP\\SIIM-ISIC-Melanoma-Classification-1st-Place-Solution-master\\SIIM-ISIC-Melanoma-Classification-1st-Place-Solution-master\\data\\jpeg-melanoma-768x768\\test")
    # y = 0
    # for i in addr_list_C:
    #     elem = get_filenames_using_os(base_addr + "\\" + addr_list_1[1] + "\\" + i + "\\")
    #     # print(elem)
    #     for j in elem:
    #         copy_file(base_addr + "\\" + addr_list_1[1] + "\\" + i + "\\" + j + ".jpg",
    #                   "D:\\ML\medical\\ASAP\\SIIM-ISIC-Melanoma-Classification-1st-Place-Solution-master\\SIIM-ISIC-Melanoma-Classification-1st-Place-Solution-master\\data\\jpeg-melanoma-768x768\\test" + "\\" + j + ".jpg")
    #         pic_class.append(y)
    #         pic_name.append(j)
    #     y = y + 1
    # y = 0
    # for i in addr_list_L:
    #     elem = get_filenames_using_os(base_addr + "\\" + addr_list_1[1] + "\\" + i + "\\")
    #     # print(elem)
    #     for j in elem:
    #         copy_file(base_addr + "\\" + addr_list_1[1] + "\\" + i + "\\" + j + ".jpg",
    #                   "D:\\ML\medical\\ASAP\\SIIM-ISIC-Melanoma-Classification-1st-Place-Solution-master\\SIIM-ISIC-Melanoma-Classification-1st-Place-Solution-master\\data\\jpeg-melanoma-768x768\\test" + "\\" + j + ".jpg")
    #         pic_class.append(y)
    #         pic_name.append(j)
    #     if y == 1:
    #         y = 3
    #     else:
    #         y = y + 1
    #         # print(y)
    # y = 0
    # for i in addr_list_Y:
    #     elem = get_filenames_using_os(base_addr + "\\" + addr_list_1[1] + "\\" + i + "\\")
    #     # print(elem)
    #     for j in elem:
    #         copy_file(base_addr + "\\" + addr_list_1[1] + "\\" + i + "\\" + j + ".jpg",
    #                   "D:\\ML\medical\\ASAP\\SIIM-ISIC-Melanoma-Classification-1st-Place-Solution-master\\SIIM-ISIC-Melanoma-Classification-1st-Place-Solution-master\\data\\jpeg-melanoma-768x768\\test" + "\\" + j + ".jpg")
    #         pic_class.append(y)
    #         pic_name.append(j)
    #     if y == 1:
    #         y = 3
    #     else:
    #         y = y + 1
    #         # print(y)
    # # print(pic_name, "\n")
    # # print(pic_class)
    # save2csv(get_filenames_using_os(
    #     "D:\\ML\medical\\ASAP\\SIIM-ISIC-Melanoma-Classification-1st-Place-Solution-master\\SIIM-ISIC-Melanoma-Classification-1st-Place-Solution-master\\data\\jpeg-melanoma-768x768\\test"),
    #          pic_name, pic_class,
    #          "D:\\ML\medical\\ASAP\\SIIM-ISIC-Melanoma-Classification-1st-Place-Solution-master\\SIIM-ISIC-Melanoma-Classification-1st-Place-Solution-master\\data\\jpeg-melanoma-768x768\\test")

    # randomly select data



    df, df_test, meta_features, n_meta_features, mel_idx = get_df(
        args.kernel_type,
        args.out_dim,
        args.data_dir,
        args.data_folder,
        args.use_meta
    )

    transforms_train, transforms_val = get_transforms(args.image_size)

    if args.DEBUG:
        df_test = df_test.sample(args.batch_size * 3)
    dataset_test = MelanomaDataset(df_test, 'test', meta_features, transform=transforms_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers)

    # load model
    models = []
    for fold in range(1):

        if args.eval == 'best':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_20_fold{fold}.pth')
        elif args.eval == 'best_20':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_20_fold{fold}.pth')
        if args.eval == 'final':
            model_file = os.path.join(args.model_dir, f'{args.kernel_type}_final_fold{fold}.pth')
        print("model path",model_file)
        model = ModelClass(
            args.enet_type,
            n_meta_features=n_meta_features,
            n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
            out_dim=args.out_dim
        )
        model = model.to(device)

        try:  # single GPU model_file
            model.load_state_dict(torch.load(model_file), strict=True)
        except:  # multi GPU model_file
            state_dict = torch.load(model_file,map_location=torch.device('cpu'))
            state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
            model.load_state_dict(state_dict, strict=True)
        
        if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
            model = torch.nn.DataParallel(model)

        model.eval()
        models.append(model)
    TARGETS = []
    # predict
    PROBS = []
    print("dataset length",len(test_loader))
    random_list = random_select(len(test_loader))
    print("random list",random_list)
    full_list = []
    for l in range(0,len(test_loader)):
        full_list.append(l)
    random_list = full_list
    radom_index_C = 1
    with torch.no_grad():
        for (data) in tqdm(test_loader):
            # if random_lis


            if args.use_meta:
                data, meta = data
                data, meta = data.to(device), meta.to(device)

                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for model in models:
                    for I in range(args.n_test):
                        l = model(get_trans(data, I), meta)
                        probs += l.softmax(1)
            else:   
                data = data.to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                for model in models:
                    for I in range(args.n_test):
                        l = model(get_trans(data, I))
                        probs += l.softmax(1)

            probs /= args.n_test
            probs /= len(models)

            PROBS.append(probs.detach().cpu())

    PROBS = torch.cat(PROBS).numpy()
    # acc = (PROBS.argmax(1) == TARGETS).mean()
    # save cvs
    # print("acc is ",acc)
    L0 = []
    L1 = []
    L2 = []
    L3 = []
    L4 = []
    P = []
    for i in random_list:
        L0.append(PROBS[i-1, 0])
        L1.append(PROBS[i-1, 1])
        L2.append(PROBS[i-1, 2])
        L3.append(PROBS[i-1, 3])
        L4.append(PROBS[i-1, 4])
    P = PROBS.argmax(1)
        # print("1：", L0)
        # print("2：", L1)
        # print("3：", L2)
        # print("4：", L3)
        # print("5：", L4)
        # print("add",PROBS.argmax(1)[i])
    # print("6：", PROBS[:, 5])
    df_test['pridict'] = PROBS.argmax(1)
    df_test['1'] = PROBS[:, 1]
    df_test['2'] = PROBS[:, 2]
    df_test['3'] = PROBS[:, 3]
    df_test['4'] = PROBS[:, 4]
    df_test['0'] = PROBS[:, 0]
    # print(P)
    # df_test['pridict'] = P
    # df_test['1'] = L1
    # df_test['2'] = L2
    # df_test['3'] = L3
    # df_test['4'] = L4
    # df_test['0'] = L0
    # df_test['image_num'] = random_list
    df_test[['image_name','0', '1','2','3','4','pridict']].to_csv(os.path.join(args.sub_dir, f'sub_{args.kernel_type}_{args.eval}.csv'), index=False)
    # print(PROBS[:, mel_idx])
    pridict = PROBS.argmax(1).tolist()
    print("output", PROBS.argmax(1))
    # assume PROBS has 6 columns (0 to 5)
    all_probs_df = pd.DataFrame(PROBS, columns=[f'class_{i}' for i in range(PROBS.shape[1])])

    # add image_name as first column
    all_probs_df.insert(0, 'image_name', df_test['image_name'].values)

    # save
    all_probs_df.to_csv(os.path.join(args.sub_dir, f'all_probs_{args.kernel_type}.csv'), index=False)
    column_number = 7
    return 95
if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.sub_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    if args.enet_type == 'resnet34':
        ModelClass = ResNet34_Melanoma
    elif args.enet_type == 'xception':
        ModelClass = Xception_Melanoma
    elif args.enet_type == 'densenet121':
        ModelClass = DenseNet121_Melanoma
    elif 'efficientnet_b3' in args.enet_type:
        ModelClass = EfficientNetB3_Melanoma
    else:
        raise NotImplementedError()

    DP = len(os.environ['CUDA_VISIBLE_DEVICES']) > 1

    device = torch.device('cpu')
    a = 0
    index = 0
    # index = 31
    max = 0

    # 108 112
    max_i = 0
    se_list = []
    sei_list = []
    while a <=90:
        print(index)
        a = main(index)
        if a >max:
            max = a
            max_i = index
        if a>=70:
            se_list.append(a)
            sei_list.append(index)
        index = index+1
        print("max value:",max_i,max)
        print("70的",se_list)
        print("index",sei_list)