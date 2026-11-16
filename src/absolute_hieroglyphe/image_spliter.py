import csv
import os
import shutil

csv_file = 'data/Tableau_imagette_MB.csv'
src_dir = 'data/Image_DAO_Sith/'
dst_dir = 'dataset/'

indicies = {'₁': '_1', '₂': '_2',  '₃': '_3',  '₄': '_4',  '₅': '_5',  '₆': '_6',  '₇': '_7',  '₈': '_8',  '₉': '_9'}

with open(csv_file) as f:
    reader = csv.reader(f)

    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    if not os.path.isdir(dst_dir + 'yes/'):
        os.mkdir(dst_dir + 'yes/')
    if not os.path.isdir(dst_dir + 'no/'):
        os.mkdir(dst_dir + 'no/')

    for row in reader:

        if row[9] == "non": # colonne "Pouces_distingues"
            dst = dst_dir + 'no/'
        else:
            dst = dst_dir + 'yes/'

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
