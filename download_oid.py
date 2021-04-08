from csv import reader
import os

# https://github.com/EscVM/OIDv4_ToolKit'
# https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy_visualizer/circle.html
# https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv


def download_oid(root_dir, per_class_limit=80):
    with open('OID/csv_folder_nl/class-descriptions.csv', 'r', encoding="utf8") as read_obj:
        csv_reader = reader(read_obj, delimiter=',')
        for row in csv_reader:
            os.system(
                f'python "OID/OIDv4_ToolKit/main.py" downloader_ill --sub h --classes "{row[1]}" --type_csv train --limit {per_class_limit} --Dataset "{root_dir}"')
