import os
import csv
import random


def get_image_paths(image_folder):
    image_paths = []
    labels = []

    # Iterate through each chunk folder
    for chunk_folder in os.listdir(image_folder):
        chunk_folder_path = os.path.join(image_folder, chunk_folder)

        # Skip if not a directory
        if not os.path.isdir(chunk_folder_path):
            continue

        # Iterate through each subfolder within the chunk folder
        subfolders = [
            f
            for f in os.listdir(chunk_folder_path)
            if os.path.isdir(os.path.join(chunk_folder_path, f))
        ]
        subfolders.sort(key=int)  # Ensure numeric order

        for subfolder in subfolders:
            subfolder_path = os.path.join(chunk_folder_path, subfolder)
            images = [
                os.path.join(chunk_folder, subfolder, img)
                for img in os.listdir(subfolder_path)
                if img.endswith(".jpg")
            ]
            image_paths.extend(images)
            labels.extend([int(subfolder)] * len(images))

    return list(zip(image_paths, labels))


def create_varied_pairs(image_paths, num_pairs):
    all_possible_pairs = set()
    num_images = len(image_paths)

    # Generate all possible unique pairs
    for i in range(num_images):
        for j in range(i + 1, num_images):
            img1, label1 = image_paths[i]
            img2, label2 = image_paths[j]
            all_possible_pairs.add((img1, img2, label1, label2))

    # Randomly sample the desired number of pairs
    pairs = random.sample(
        list(all_possible_pairs), min(num_pairs, len(all_possible_pairs))
    )

    return pairs


def save_to_csv(pairs, output_file):
    # Write the image pairs and their labels to a CSV file
    with open(output_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Image1", "Image2", "Label1", "Label2"])
        for pair in pairs:
            csvwriter.writerow(pair)


if __name__ == "__main__":
    image_folder = "/home/jeans/internship/resources/datasets/mon/train"
    output_file = "image_pairs_train.csv"
    num_pairs = 70000  # Set the desired number of varied pairs
    image_paths = get_image_paths(image_folder)
    varied_pairs = create_varied_pairs(image_paths, num_pairs)
    save_to_csv(varied_pairs, output_file)

    print(
        f"CSV file '{output_file}' created successfully with {len(varied_pairs)} pairs."
    )

    image_folder = "/home/jeans/internship/resources/datasets/mon/val"
    output_file = "image_pairs_val.csv"
    num_pairs = 15000  # Set the desired number of varied pairs
    image_paths = get_image_paths(image_folder)
    varied_pairs = create_varied_pairs(image_paths, num_pairs)
    save_to_csv(varied_pairs, output_file)

    print(
        f"CSV file '{output_file}' created successfully with {len(varied_pairs)} pairs."
    )

    image_folder = "/home/jeans/internship/resources/datasets/mon/test"
    output_file = "image_pairs_test.csv"
    num_pairs = 15000  # Set the desired number of varied pairs
    image_paths = get_image_paths(image_folder)
    varied_pairs = create_varied_pairs(image_paths, num_pairs)
    save_to_csv(varied_pairs, output_file)

    print(
        f"CSV file '{output_file}' created successfully with {len(varied_pairs)} pairs."
    )
