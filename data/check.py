import os
import shutil

def check_and_replace_images(folder_a, folder_b):
    # List all files in folder A
    files_in_a = os.listdir(folder_a)
    
    for filename in files_in_a:
        file_path_a = os.path.join(folder_a, filename)
        
        # Check if the file size is 0 bytes
        if os.path.getsize(file_path_a) == 0:
            file_path_b = os.path.join(folder_b, filename)
            
            # Check if the corresponding file exists in folder B
            if os.path.exists(file_path_b):
                # Copy the file from folder B to folder A
                shutil.copy(file_path_b, file_path_a)
                print(f"Replaced corrupted file: {file_path_a} with {file_path_b}")
            else:
                print(f"File {file_path_b} does not exist in folder B.")
        else:
            print(f"File {file_path_a} is not corrupted.")
    
    print("Done checking and replacing corrupted images.")

# Specify the paths to folder A and folder B
folder_a = '/home/ziyang/Downloads/aaaKANUNet/data/my_Kvasir-SEG/train/masks'
folder_b = '/home/ziyang/Downloads/Kvasir-SEG/masks'

# Run the check and replace function
check_and_replace_images(folder_a, folder_b)
