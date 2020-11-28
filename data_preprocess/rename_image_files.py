import os
import random


def rename():
    """
    批量重命名数据文件，前缀名+固定宽度序号+后缀名
    """
    root_folder = r"C:\Users\admin\Documents\data\rename_test"  # data folder
    prefix_name = "sdzw_"
    start_number = 1
    file_type = ".jpg"
    print("正在生成以 prefix_name + start_number + file_type 迭代的文件名")
    count = 0
    file_list = os.listdir(root_folder)
    random.shuffle(file_list)
    for file in file_list:
        old_file_name = os.path.join(root_folder, file)
        if os.path.isdir(old_file_name):
            continue
        new_file_name = os.path.join(root_folder, prefix_name + str(count + start_number).zfill(6) + file_type)
        os.rename(old_file_name, new_file_name)
        count += 1
    print("一共修改了" + str(count) + "个文件")


if __name__ == "__main__":
    rename()
