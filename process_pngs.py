# Import required libraries
import os
import re
from PIL import Image

# Directory to process PNG files
dir_path = os.path.dirname(os.path.abspath(__file__)) + '/proc_images'
print(dir_path)
dir_path = 'C:\\train_dataset/test_images/image_1000'
target_size = 128

# Go through each file in the directory
for filename in os.listdir(dir_path):
    # Check if the file is a PNG file
    if filename.endswith('.jpg'):
        # Get the full path of the file
        file_path = os.path.join(dir_path, filename)

        # Open the image file
        with Image.open(file_path) as img:
            print('processing ... ' + img.filename)

            # Scale down the image to target_size
            img_resized = img.resize((target_size, target_size))

            # Save the scaled image
            img_resized.save(file_path)
            continue

            # Extract the PNG info
            png_info = img.info
            
            # Join the info into a single string
            info_string = ', '.join([f"{key}: {value}" for key, value in png_info.items()])

            # Define the regular expression to find the text between 'parameters:' and 'Negative prompt:'
            pattern = r"parameters:(.*?)Negative prompt:"
            
            # Find matches
            matches = re.search(pattern, info_string, re.DOTALL)
            if matches:
                # Extract the matched text
                extracted_text = matches.group(1).strip()

                # Write the extracted text to a text file
                with open(f"{file_path[:-4]}_filtered_info.txt", "w") as text_file:
                    text_file.write(extracted_text)