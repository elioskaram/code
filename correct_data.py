import os
import shutil

def move_files_to_patient_folders(root_folder):
    # Get a list of all files in the root folder
    files_in_root = os.listdir(root_folder)

    # Process each file in the root folder
    for file_name in files_in_root:
        # Check if the file is a txt file (patient ID file)
        if file_name.endswith('.txt'):
            # Extract the patient ID from the file name
            patient_id = os.path.splitext(file_name)[0]

            # Create a folder with the patient ID if it doesn't exist
            patient_folder = os.path.join(root_folder, patient_id)
            if not os.path.exists(patient_folder):
                os.makedirs(patient_folder)

            # Move all files starting with the patient ID to the patient folder
            for file_to_move in files_in_root:
                if file_to_move.startswith(patient_id):
                    source_path = os.path.join(root_folder, file_to_move)
                    destination_path = os.path.join(patient_folder, file_to_move)
                    shutil.move(source_path, destination_path)

if __name__ == "__main__":
    # Replace 'your_root_folder' with the actual path of the folder containing the files.
    root_folder = 'data/training_data'
    move_files_to_patient_folders(root_folder)