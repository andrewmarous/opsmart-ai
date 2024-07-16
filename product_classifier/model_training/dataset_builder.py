import random
import csv
import glob
import argparse

parser = argparse.ArgumentParser(description='Process training data path.')
parser.add_argument('--data', '-d', type=str, nargs='?')
args = parser.parse_args()

data_path = args.data

# get subfolder names
subfolders = glob.glob(data_path + '*')
for i in range(len(subfolders)):
    subfolders[i] = subfolders[i][5:]
class_dict = {subfolders[i]: i+1 for i in range(len(subfolders))}

# create label set
label_set = []
for k, v in class_dict.items():
    image_filenames = glob.glob(f"data/{k}/*.jpg")
    for filename in image_filenames:
        label_set.append([filename, class_dict[k]])

# shuffle pairings and split them 80/10/10 for training, validation, test
random.shuffle(label_set)
train_cutoff = int(len(label_set) * 0.8)
test_cutoff = int(len(label_set) * 0.9)
train_labels = label_set[:train_cutoff]
validation_labels = label_set[test_cutoff:]

# make training labels csv
with open('train_labels.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(train_labels)

# make validation labels csv
with open('validation_labels.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(validation_labels)


# create label dictionary for server
flipped_dict = {value : key for key, value in class_dict.items()}
lst = []
for k, v in flipped_dict.items():
    lst.append([k, v])
with open('label_dict.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(lst)
