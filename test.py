import os
import shutil

root_dir = '/mnt/c/OxygenAi/resources/human_with_bag/ctw_re_uid_2024-07-01-2024-07-01.bag-images/image-samples-by-class/ctw_re_uid_2024-07-01-2024-07-01.bag-images/re_uid'

for person_id in os.listdir(root_dir):
    person_id_path = os.path.join(root_dir, person_id)
    if os.path.isdir(person_id_path):
        file_count = len([f for f in os.listdir(person_id_path) if os.path.isfile(os.path.join(person_id_path, f))])
        if file_count < 3:
            shutil.rmtree(person_id_path)
            print(f"Removed subfolder {person_id_path} with {file_count} files")
