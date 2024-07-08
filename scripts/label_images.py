import os
import cv2
import csv
import numpy as np
import shutil

# Set the path to the folder containing the images
root_dir = '/mnt/c/OxygenAi/resources/ctw_re_uid_2024-07-01-2024-07-01.bag-images/image-samples-by-class/ctw_re_uid_2024-07-01-2024-07-01.bag-images/re_uid'

# Set the path to the CSV file to save the labels
csv_file = '/mnt/c/OxygenAi/resources/ctw_re_uid_2024-07-01-2024-07-01.bag-images/image-samples-by-class/ctw_re_uid_2024-07-01-2024-07-01.bag-images/label.csv'

# Create a list to store the labels
labels = []

# Check if the CSV file exists
if os.path.exists(csv_file):
    # Read the existing labels from the CSV file
    with open(csv_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header row
        existing_labels = [row[0] for row in reader]
else:
    existing_labels = []

# Set the number of classes
num_classes = 11


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

    # Display the concatenated image
    cv2.imshow('Images', concatenated_image)
    while True:
        key = cv2.waitKey(0)
        if key == ord('x'):
            cv2.destroyAllWindows()
            break
        elif key == ord('d'):
            cv2.destroyAllWindows()
            return 'delete'


for person_folder in os.listdir(root_dir):
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
            # Check if the image has already been labeled
            if rel_path in existing_labels:
                print(f"Skipping already labeled image: {rel_path}")
                continue

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
            continue

        # Get the class label from the user for each image
        for rel_path in image_paths:
            label = input(f"Enter the class label for {rel_path}: ")

            # Validate the input
            while not label.isdigit() or int(label) < 0 or int(label) >= num_classes:
                label = input(f"Invalid input. Enter the class label for {rel_path} (0-{num_classes-1}): ")

            # Add the label to the list
            labels.append((rel_path, int(label)))

    # Write the new labels to the CSV file
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
            writer.writerow(['Image', 'Label'])  # header row
        writer.writerows(labels)
