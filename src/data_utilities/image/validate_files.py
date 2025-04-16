import os

import PIL.Image as Image





def main():
    path = "data"

    # Traverse all files in folder and subfolders
    for root, dirs, files in os.walk(path):
        for file in files:
            # Check if file is an image
            if file.endswith(('.jpg', '.jpeg', '.png')):
                # Open image file
                img = Image.open((os.path.join(root, file)))
                try:
                    img.load()
                except Exception as e:
                    print(f"Error loading {file}: {e}")


if __name__ == "__main__":
    main()