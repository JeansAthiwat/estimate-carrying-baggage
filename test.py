import os
import json
from tkinter import Tk, Label, Button, Entry, StringVar, filedialog
from PIL import Image, ImageTk


class ImageLabeler:
    def __init__(self, root, base_folder):
        self.root = root
        self.root.title("Image Labeler")
        self.base_folder = base_folder
        self.person_ids = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
        self.current_person_id_index = 0
        self.current_image_index = 0
        self.object_counts = {}

        self.label = Label(root)
        self.label.pack()

        self.entry_label = Label(root, text="Enter object count:")
        self.entry_label.pack()

        self.entry_text = StringVar()
        self.entry = Entry(root, textvariable=self.entry_text)
        self.entry.pack()

        self.save_button = Button(root, text="Save and Next", command=self.save_and_next)
        self.save_button.pack()

        self.load_image()

    def load_image(self):
        if self.current_person_id_index < len(self.person_ids):
            person_id = self.person_ids[self.current_person_id_index]
            person_folder = os.path.join(self.base_folder, person_id)
            image_files = [f for f in os.listdir(person_folder) if f.endswith('.jpg')]

            if self.current_image_index < len(image_files):
                image_file = image_files[self.current_image_index]
                image_path = os.path.join(person_folder, image_file)
                image = Image.open(image_path)
                image = image.resize((500, 500), Image.ANTIALIAS)
                photo = ImageTk.PhotoImage(image)

                self.label.config(image=photo)
                self.label.image = photo

                self.root.title(f"Labeling: {person_id}/{image_file}")
            else:
                self.current_person_id_index += 1
                self.current_image_index = 0
                self.load_image()
        else:
            self.save_to_json()
            self.root.quit()

    def save_and_next(self):
        person_id = self.person_ids[self.current_person_id_index]
        person_folder = os.path.join(self.base_folder, person_id)
        image_files = [f for f in os.listdir(person_folder) if f.endswith('.jpg')]

        if person_id not in self.object_counts:
            self.object_counts[person_id] = {}

        if self.current_image_index < len(image_files):
            image_file = image_files[self.current_image_index]
            object_count = self.entry_text.get()
            if object_count.isdigit():
                self.object_counts[person_id][image_file] = int(object_count)
                self.entry_text.set('')
                self.current_image_index += 1
                self.load_image()
            else:
                print("Please enter a valid number.")

    def save_to_json(self):
        output_file = os.path.join(self.base_folder, 'object_counts.json')
        with open(output_file, 'w') as json_file:
            json.dump(self.object_counts, json_file, indent=4)
        print(f"Data saved to {output_file}")


def main():
    root = Tk()
    base_folder = filedialog.askdirectory(title="Select Folder Containing Images")
    if base_folder:
        app = ImageLabeler(root, base_folder)
        root.mainloop()


if __name__ == "__main__":
    main()
