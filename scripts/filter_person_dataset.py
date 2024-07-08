import os
import cv2
import csv
import shutil

# Set the path to the folder containing the images
root_dir = '/mnt/c/OxygenAi/resources/human_with_bag/ctw_re_uid_2024-07-01-2024-07-01.bag-images/image-samples-by-class/ctw_re_uid_2024-07-01-2024-07-01.bag-images/re_uid'

# Set the path to the CSV file to save the labels
# csv_file = '/mnt/c/OxygenAi/resources/human_with_bag/ctw_re_uid_2024-07-01-2024-07-01.bag-images/image-samples-by-class/ctw_re_uid_2024-07-01-2024-07-01.bag-images/label.csv'

# Set the path to the folder to copy the person folder
copy_dir = '/mnt/c/OxygenAi/resources/human_with_bag/ctw_re_uid_2024-07-01-2024-07-01.bag-images/image-samples-by-class/ctw_re_uid_2024-07-01-2024-07-01.bag-images/filtered'

# Set the path to the CSV file to save the folder status
status_file = '/mnt/c/OxygenAi/resources/human_with_bag/ctw_re_uid_2024-07-01-2024-07-01.bag-images/image-samples-by-class/ctw_re_uid_2024-07-01-2024-07-01.bag-images/status.csv'


def display_images(images):
    max_images_per_row = 7
    rows = []
    max_height = max(image.shape[0] for image in images)

    # Resize images to the same height
    resized_images = [cv2.resize(image, (int(image.shape[1] * max_height / image.shape[0]), max_height)) for image in images]

    for i in range(0, len(resized_images), max_images_per_row):
        row_images = resized_images[i : i + max_images_per_row]

        # Make sure all images in the row have the same width
        max_width = max(img.shape[1] for img in row_images)
        row_images = [
            cv2.copyMakeBorder(img, 0, 0, 0, max_width - img.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0]) for img in row_images
        ]

        row = cv2.hconcat(row_images)
        rows.append(row)

    # Make sure all rows have the same width
    max_width = max(row.shape[1] for row in rows)
    rows = [cv2.copyMakeBorder(row, 0, 0, 0, max_width - row.shape[1], cv2.BORDER_CONSTANT, value=[0, 0, 0]) for row in rows]

    # Concatenate all rows vertically
    concatenated_image = cv2.vconcat(rows)

    # Display the concatenated image in fullscreen
    window_name = 'Images'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, concatenated_image)

    while True:
        key = cv2.waitKey(0)
        if key == ord('x'):
            cv2.destroyAllWindows()
            break
        elif key == ord('d'):
            cv2.destroyAllWindows()
            return 'delete'
        elif key == ord('c'):
            cv2.destroyAllWindows()
            return 'copy'


def copy_folder(src, dst):
    if not os.path.exists(dst):
        os.makedirs(dst)
    dst_folder = os.path.join(dst, os.path.basename(src))
    if os.path.exists(dst_folder):
        print(f"Folder {dst_folder} already exists. Skipping copy.")
    else:
        shutil.copytree(src, dst_folder)
        print(f"Copied folder to {dst_folder}")


def read_status_file(status_file):
    if os.path.exists(status_file):
        with open(status_file, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            status_dict = {rows[0]: rows[1] for rows in reader}
    else:
        status_dict = {}
    return status_dict


def write_status_file(status_file, status_dict):
    with open(status_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in status_dict.items():
            writer.writerow([key, value])


# Read the existing folder status from the CSV file
status_dict = read_status_file(status_file)


for person_folder in os.listdir(root_dir):
    # Skip folders that have already been processed
    if person_folder in status_dict:
        print(f"Skipping {person_folder}, already processed as {status_dict[person_folder]}")
        continue

    # Loop through the images in the folder
    print(person_folder)
    current_folder = os.path.join(root_dir, person_folder)
    images = []
    image_paths = []
    for filename in os.listdir(current_folder):
        print("filename:", filename)
        # Check if the file is an image
        if filename.endswith('.jpg') or filename.endswith('.png'):
            full_path = os.path.join(current_folder, filename)
            rel_path = os.path.relpath(full_path, root_dir)

            # Open the image using OpenCV
            img = cv2.imread(full_path)
            if img is not None:
                images.append(img)
                image_paths.append(rel_path)

    if images:
        result = display_images(images)
        if result == 'delete':
            print(f"Deleting folder: {current_folder}")
            shutil.rmtree(current_folder)
            status_dict[person_folder] = 'deleted'
        elif result == 'copy':
            print(f"Copying folder: {current_folder}")
            copy_folder(current_folder, copy_dir)
            status_dict[person_folder] = 'copied'
        else:
            status_dict[person_folder] = 'skipped'

    # Write the updated folder status to the CSV file
    write_status_file(status_file, status_dict)
