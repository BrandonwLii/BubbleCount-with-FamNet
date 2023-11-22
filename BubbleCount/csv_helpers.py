# csv related helpers
import os
import shutil
import csv

def backup_and_clear_csv(file_path):
    backup_path = file_path.replace('.csv', '_ss.csv')
    while os.path.exists(backup_path):
        backup_path = backup_path.replace('.csv', '_ss.csv')

    # Check if the original file exists
    if os.path.exists(file_path):
        # Create a backup by copying the original file
        shutil.copyfile(file_path, backup_path)

        # Clear the contents of the original file
        open(file_path, 'w').close()
    return

def save_to_csv(data_list, file_path, header = ['Exemplar', 'Target', 'Count']):
    # First, backup and clear the original file
    backup_and_clear_csv(file_path)

    # Then, write new data to the original file
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data_list)

    print(f"The counts are saved to {file_path}.")
    return