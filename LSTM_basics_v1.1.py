import pandas as pd
import os

# Get fitting data
subject = 14  # participant id

subject_data = pd.DataFrame()
main_folder_name = 'DeviceMotion_data'
folders = os.listdir('./%s' % (main_folder_name))
activities_to_model = ['jog', 'wlk']

# Iterate all folders to gather user data
for activity in activities_to_model:
    activity_folders = [foldername for foldername in folders if activity in foldername]
    for folder in activity_folders:
        data_files = os.listdir('./%s/%s' % (main_folder_name, folder))
        for file in data_files:
            if '_' + str(subject) + '.csv' in file:
                subject_data_i = pd.read_csv('./%s/%s/%s' % (main_folder_name, folder,file))
            subject_data_i['activity'] = activity
            subject_data = subject_data.append(subject_data_i)

# User data ready
subject_data.describe()


# User data ready
subject_data.describe()


