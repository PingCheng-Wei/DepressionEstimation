'''
DAIC-WOZ Depression Database

The package includes 189 folders of sessions 300-492 and
other documents as well as matlab scripts in util.zip
Excluded sessions: 342,394,398,460
-------------------------------------------------------------------------
run from the commend line as such:
    in Linux: python3 download_DAIC-WOZ.py --out_dir=<where/to/store/absolute_path> --username=<the_give_username> --password=<the_given_password>
'''

import os
import requests
from zipfile import ZipFile
from io import BytesIO
import argparse


def download_dataset(OUTPUT_DIR, USERNAME, PASSWORD):
    '''
    Automatically download the "DAIC-WOZ dataset" for depression estimation from the website
    :param OUTPUT_DIR: either folder in current local path or absolute path to other folder
    :return: void
    '''

    all_sessions = list(range(308, 492 + 1))
    excluded = [342, 394, 398, 460]

    # for website download authentication
    username = USERNAME
    password = PASSWORD

    dataset_destination = OUTPUT_DIR

    # download all sessions
    for session in all_sessions:
        if session not in excluded:
            print("Start downloading session {} ...".format(session))
            # get the full url
            zipurl = 'https://dcapswoz.ict.usc.edu/wwwdaicwoz/{}_P.zip'.format(session)
            #  get the link with given username & password
            r = requests.get(zipurl, auth=(username, password))
            # open the ZIP file with binary stream
            zfile = ZipFile(BytesIO(r.content))
            # extract all contents and store in a given folder
            zfile.extractall(os.path.join(dataset_destination, "{}_P".format(session)))

        else:
            print("{}_P is excluded session".format(session))
            continue

    # download documents.zip
    print("Start downloading documents ...")
    zipurl = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/documents.zip"
    #  get the link with given username & password
    r = requests.get(zipurl, auth=(username, password))
    # open the ZIP file with binary stream
    zfile = ZipFile(BytesIO(r.content))
    # extract all contents and store in the "documents" folder
    file = zfile.extractall(os.path.join(dataset_destination, "documents"))

    # download util.zip
    print("Start downloading util ...")
    zipurl = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/util.zip"
    #  get the link with given username & password
    r = requests.get(zipurl, auth=(username, password))
    # open the ZIP file with binary stream
    zfile = ZipFile(BytesIO(r.content))
    # extract all contents and store in the "util" folder
    file = zfile.extractall(os.path.join(dataset_destination, "util"))

    # function for download the file
    def download_file(url, destination, name):
        downloaded_obj = requests.get(url, auth=(username, password))
        with open(os.path.join(destination, name), "wb") as file:
            file.write(downloaded_obj.content)

    # download all csv and pdf files
    print("Start downloading pdf and csv ...")
    pdf = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/DAICWOZDepression_Documentation_AVEC2017.pdf"
    pdf_name = pdf.split('/')[-1]
    download_file(pdf, dataset_destination, pdf_name)

    csv1 = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/dev_split_Depression_AVEC2017.csv"
    csv1_name = csv1.split('/')[-1]
    download_file(csv1, dataset_destination, csv1_name)

    csv2 = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/full_test_split.csv"
    csv2_name = csv2.split('/')[-1]
    download_file(csv2, dataset_destination, csv2_name)

    csv3 = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/test_split_Depression_AVEC2017.csv"
    csv3_name = csv3.split('/')[-1]
    download_file(csv3, dataset_destination, csv3_name)

    csv4 = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/train_split_Depression_AVEC2017.csv"
    csv4_name = csv4.split('/')[-1]
    download_file(csv4, dataset_destination, csv4_name)



if __name__ == '__main__':
    # Parse comment line arguments
    parser = argparse.ArgumentParser(description='Download DAIC-WOZ dataset')
    parser.add_argument('--out_dir', required=True,
                        metavar="output path to store the datasets",
                        help='output path to store the datasets')
    parser.add_argument('--username', required=True,
                        metavar="username you received after submitting the request",
                        help='username you received after submitting the request')
    parser.add_argument('--password', required=True,
                        metavar="password you received after submitting the request",
                        help='password you received after submitting the request')
    args = parser.parse_args()

    # download the whole DAIC-WOZ dataset
    download_dataset(args.out_dir, args.username, args.password)
    print('done!')



