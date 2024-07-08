import os
import csv
import numpy as np


def make_pair_intraclass(csv_folder, output_csv, datatype):
    os.makedirs(output_csv, exist_ok=True)
    combined_pairs = []

    for root, dirs, files in os.walk(csv_folder):
        for file in files:
            full_path = os.path.join(root, file)

            with open(full_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                rows = list(reader)

                for i in range(len(rows)):
                    for j in range(i + 1, len(rows)):
                        img1, label1 = rows[i]
                        img2, label2 = rows[j]
                        combined_pairs.append([img1, img2, label1, label2])

    with open(output_csv + f'/intraclass_pair_{datatype}.csv', 'w', newline='') as save_f:
        writer = csv.writer(save_f)
        writer.writerow(['img1', 'img2', 'label1', 'label2'])
        writer.writerows(combined_pairs)

    print(f'combinded pair saved at {output_csv}' + f'/intraclass_pair_{datatype}.csv')


datatype = 'train'
csv_folder = f'/mnt/c/OxygenAi/resources/human_with_bag/ctw_re_uid_2024-07-01-2024-07-01.bag-images/image-samples-by-class/csv_for_training/{datatype}'
output_csv = 'manifest/intraclass_pair_with_label'
make_pair_intraclass(csv_folder, output_csv, datatype)

datatype = 'test'
csv_folder = f'/mnt/c/OxygenAi/resources/human_with_bag/ctw_re_uid_2024-07-01-2024-07-01.bag-images/image-samples-by-class/csv_for_training/{datatype}'
make_pair_intraclass(csv_folder, output_csv, datatype)
