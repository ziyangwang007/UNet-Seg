import os
import shutil
import random

random.seed(0)


# def split_data(file_path,xml_path, new_file_path, train_rate, val_rate, test_rate):
def split_data(file_path,new_file_path, train_rate, val_rate):
    each_class_image = []
    # each_class_label = []
    for image in os.listdir(file_path):
        each_class_image.append(image)
    # for label in os.listdir(xml_path):
    #     each_class_label.append(label)
    data= each_class_image
    total = len(each_class_image)
    random.shuffle(data)
    each_class_image = data
    train_images = each_class_image[0:int(train_rate * total)]
    val_images = each_class_image[int(train_rate * total):int((train_rate + val_rate) * total)]
    # test_images = each_class_image[int((train_rate + val_rate) * total):]
    # train_labels = each_class_label[0:int(train_rate * total)]
    # val_labels = each_class_label[int(train_rate * total):int((train_rate + val_rate) * total)]
    # test_labels = each_class_label[int((train_rate + val_rate) * total):]

    for image in train_images:
        print(image)
        old_path = file_path + '/' + image
        new_path1 = new_file_path + '/' + 'train' +'/' + 'images'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + image
        shutil.copy(old_path, new_path)

    # for label in train_labels:
    #     print(label)
    #     old_path = xml_path + '/' + label
    #     new_path1 = new_file_path + '/' + 'train' + '/' + 'labels'
    #     if not os.path.exists(new_path1):
    #         os.makedirs(new_path1)
    #     new_path = new_path1 + '/' + label
    #     shutil.copy(old_path, new_path)

    for image in val_images:
        old_path = file_path + '/' + image
        new_path1 = new_file_path + '/' + 'val' +'/' + 'images'
        if not os.path.exists(new_path1):
            os.makedirs(new_path1)
        new_path = new_path1 + '/' + image
        shutil.copy(old_path, new_path)

    # for label in val_labels:
    #     old_path = xml_path + '/' + label
    #     new_path1 = new_file_path + '/' + 'val' + '/' + 'labels'
    #     if not os.path.exists(new_path1):
    #         os.makedirs(new_path1)
    #     new_path = new_path1 + '/' + label
    #     shutil.copy(old_path, new_path)

    # for image in test_images:
    #     old_path = file_path + '/' + image
    #     new_path1 = new_file_path + '/' + 'test' + '/' + 'images'
    #     if not os.path.exists(new_path1):
    #         os.makedirs(new_path1)
    #     new_path = new_path1 + '/' + image
    #     shutil.copy(old_path, new_path)

    # for label in test_images:
    #     old_path = file_path + '/' + label
    #     new_path1 = new_file_path + '/' + 'test' + '/' + 'masks'
    #     if not os.path.exists(new_path1):
    #         os.makedirs(new_path1)
    #     new_path = new_path1 + '/' + label
    #     shutil.copy(old_path, new_path)


if __name__ == '__main__':
    file_path = "./ISIC2016/train/images" #你只要改images,labels
    new_file_path = "./my_ISIC"
    # split_data(file_path,xml_path, new_file_path, train_rate=0.7, val_rate=0.2, test_rate=0.1)
    split_data(file_path, new_file_path, train_rate=0.9, val_rate=0.1)