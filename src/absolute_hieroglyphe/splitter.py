import csv
import os
import shutil

csv_file = 'data/Tableau_imagette_MB.csv'
src_dir = 'data/Image_DAO_Sith/'
dst_dir = 'dataset/'

indicies = {'₁': '_1', '₂': '_2',  '₃': '_3',  '₄': '_4',  '₅': '_5',  '₆': '_6',  '₇': '_7',  '₈': '_8',  '₉': '_9'}

def split(col_name, skip_split=False):
    col_index = -1
    class_names = []

    with open(csv_file) as f:
        reader = csv.reader(f)

        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)
        else:
            if not skip_split:
                for file in os.listdir(dst_dir):
                    path = os.path.join(dst_dir, file)
                    if os.path.isfile(path):
                        os.remove(path)
                    else:
                        shutil.rmtree(path)

        for row in reader:
            if col_index == -1:
                for i in range(len(row)):
                    if row[i] == col_name:
                        col_index = i
                        break
                if col_index == -1:
                    print(f"Error: '{col_name}' couldn't be found in the file.")
                    exit(1)
                else:
                    if skip_split:
                        print("Skipping split")
                        return col_index
                continue

            class_name = row[col_index]
            if class_name == "" or class_name == "NA":
                continue
            if class_name not in class_names:
                class_names.append(class_name)

            dst = dst_dir + class_name + '/'
            if not os.path.isdir(dst):
                os.mkdir(dst)

            name = row[0].split(' (')[0]
            for match in indicies.items():
                name = name.replace(match[0], match[1])

            filename = src_dir + name
            if os.path.isfile(filename + '.jpg'):
                ext = '.jpg'
            else:
                ext = '.png'
            filename += ext

            if os.path.isfile(filename):
                shutil.copyfile(filename, dst + name + ext)
            else:
                print(f"File '{filename}' doesn't exists")

    return col_index
