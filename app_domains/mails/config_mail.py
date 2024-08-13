"""Configuration global parameters for the mail exchange server detection"""

import os
from os.path import join

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# running setup
CONFIG_MAIL = {
    "MAIN_DIR": MAIN_DIR,

    # GENERAL
    # files,
    "input_folder": join(MAIN_DIR, "input", "com_mini_mini"),
    "output_final_file_path": join(MAIN_DIR, "output","mail_check.csv"),

    # REQUESTS ENGINE
    "TIMEOUT_BETWEEN_VISIT" : 1, # seconds between two MX requests
    # "TIMEOUT_RESOLVER": 5,

    # request engine progress notifications
    "LOG_EVERY_N": 50,

    # DATABASE STORAGE
    # if True, the program will convert any csvs it finds in the output folder into the defined database and move the csvs to the storage folder
    # set these in the .env folder for security purposes, as this file is stored on the git.
    "USE_DB": False,
    "DBNAME": "crawlerdb",  # database where you're going to store all the of data
    "DBHOST": "localhost",  # hostname or ip address of the db server
    "DBPORT": 5432,  # db access port
    "DBUSER": "dbuser",  # db username
    "DBPASS": "dbpass",  # db password

}


# load environment file if available

def ProcessEnvSetting(setting):
    key, value = setting[0], setting[1]
    # convert text values to python recognized values
    if value.lower() == 'true':
        value = True
    elif value.lower() == 'false':
        value = False
    else:
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
    # Overwrite existing key. If a non-existant key is added, it will just be ignored.
    CONFIG_MAIL[key] = value


try:
    with open(join(os.path.dirname(os.path.dirname(__file__)), ".env"), 'r') as f:
        for line in f:
            setting = line.replace('\n', '').split('=')
            if len(setting) == 2:
                ProcessEnvSetting(setting)
except IOError:
    with open(join(os.path.dirname(os.path.dirname(__file__)), ".env"), 'w') as f:
        f.write('')
