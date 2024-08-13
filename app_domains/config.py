"""Configuration global parameters of the overall crawler
In particular, input mail selection, HTTP requests orchestration and scenario selection"""

import os
from os.path import join

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# running setup
RUN_CONFIG = {
    "MAIN_DIR": MAIN_DIR,

    # GENERAL
    # files
    # "input_folder": join(MAIN_DIR, "input", "com_extramini"),
    "input_folder": join(MAIN_DIR, "input", "com_mini"),
    # "input_folder": join(MAIN_DIR, "input", "com1k"),
    # "input_folder": join(MAIN_DIR, "input", "com10k"),

    # Max number of domains the controller can handle in one go.
    "CONTROLLER_LIMIT": 10000,  # Files will be split into these chunks before being fed to the controller

    "output_final_file_path": join(MAIN_DIR, "output", "perf2.csv"),
    "CSV_OUTPUT_DELIMITER": ',',
    # Set this to the character you want as the delimiter, will apply to the output_file_final_path

    # Scenario
    "DO_REQUESTS": True,
    # "DO_REQUESTS": False,
    "DO_PAGE_PROCESSING": True,
    # "DO_PAGE_PROCESSING": False,
    "DO_CLASSIFICATION": True,
    # "DO_CLASSIFICATION": False,
    "DO_CONTENT_CLASSIFICATION": True,
    # "DO_CONTENT_CLASSIFICATION": False,
    "DO_SOCIAL_MEDIA": True,
    # "DO_SOCIAL_MEDIA": False,
    "DO_MAIL_EXCHANGE": True,
    # "DO_MAIL_EXCHANGE": False,
    "DO_HCJ_EXTRACTION": True,
    # "DO_HCJ_EXTRACTION": False,
    "DO_JS_INTERPRETATION": True,
    # "DO_JS_INTERPRETATION": False,
    "DO_CONCAT_FORMAT": True,
    "DO_SAMPLING": False,

    # REQUESTS ENGINE
    # "USE_UVLOOP": True,
    "USE_UVLOOP": False,
    # timeout for to request and extract the response of a single url
    "MINUTES_TO_TIMEOUT": 2,
    # maximum number of connection by TCP connector
    "LIMIT_REQUEST": 50,
    # maximum size of the queue = batch of urls automatically processed when a Python interpreter is
    # available (1 queue per process)
    # "MAX_WORKERS": 100,
    "MAX_WORKERS": 50,
    # number of folds to split input file (example: a 24,000 urls input file is processed in three batches of
    # 8,000, each batch is entirely done by one process)
    "BATCH_SPLIT": 3,
    # request engine/classification on multiple processes
    "MULTI_PROCESSING": True,
    # "MULTI_PROCESSING": False,
    # number of processes to use for request engine: 1 process = 1 Python interpreter = one queue = one batch at a time
    "MAX_PROCESSES": 3,
    "MAX_MB_SINGLE_URL": 200,

    # Whether to use Threads or Processes
    "PARALLEL_PREFER": "processes",
    # how many requests per second we should make at the most
    "REQUEST_RATE_LIMIT": 10000,

    # Force new visit if a website have already been visited in a former run
    "force_new_visit": False,
    # Allow multiple loops of requests for Timout error/Server disconnection errors
    "ENABLE_REVISITING": True,
    "MAX_REVISIT": 1,

    ## Chrome options
    "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36",

    # path where url pickle are saved (if debug-mode=True)
    "PATH_URL_SAVE": join(MAIN_DIR, "inter", "urls"),

    # path where file pickle are saved(if debug-mode=True)
    "PATH_DOC_SAVE": join(MAIN_DIR, "inter"),

    # request engine progress notifications
    "LOG_EVERY_N": 500,

    # PAGE PROCESSING
    # Number of parallel workers (<= number of CPUs)
    "WORKERS_POST_PROCESSING": 5,

    # PARKING CLASSIFICATION
    # if True, save the original sentence in the column kw_park_notice, else just the attribute keyword in english
    "LANGUAGE_DEVELOPMENT_MODE": False,
    "PATH_URL_REVISIT_SAVE": join(MAIN_DIR, "inter", "revisit"),
    "name_model": "xgb_v1",
    "taxonomy_path": join(MAIN_DIR, "input", "taxonomy.csv"),
    "unique_words": join(MAIN_DIR, "input", "vocab_unique_words.csv"),
    "feature": "root",

    # SAMPLING 
    # The sampling_rate will determine how many domains per 100 will be randomly sampled. Each sample will get a unique browser visit 
    # to have a screenshot taken, which will be saved along with its raw-html. This data will then be prepped ready to be used in
    # LabTools for ML training.
    # Set the sampling_rate to a fractional percentage of the desired rate. e.g. 0.1 = 10% of pages will be sampled, or 1.0 = 100% of pages will be sampled.
    "SAMPLING_RATE": 0.001,
    "SAMPLING_MAX_SCREENSHOT_HEIGHT_PX": 1200,
    "SAMPLING_LOCAL_FOLDER": join(MAIN_DIR, "output", "sampling"),
    # "SAMPLING_DBNAME": "samplingdb", # direct db insert not (yet) implemented, use the rest API below.
    # "SAMPLING_DBHOST": "localhost",
    # "SAMPLING_DBPORT": 5432,
    # "SAMPLING_DBUSER": "dbuser",
    # "SAMPLING_DBPASS": "dbpass",

    "SAMPLING_REST_API_IMAGE_URL": "https://labtools.centr.org/image/",
    # web address if using a rest framework to accept samples, e.g. "https://labtools.centr.org/image/"
    "SAMPLING_REST_API_SAMPLE_URL": "https://labtools.centr.org/samples",
    # web address if using a rest framework to accept samples, e.g. "https://labtools.centr.org/samples"
    "SAMPLING_REST_API_USERNAME": "labtools",  # username used to upload samples using the rest API
    "SAMPLING_REST_API_PASSWORD": "8L%zDaC4E98#rX",  # password used to upload samples using the rest API
    "SAMPLING_REST_API_ALLOW_UPDATES": False,  # CRUD feature:
    # output_processing will try to Create a new sample from the crawler's final_ file using LabTool's REST API.
    # If a sample with the same "categorisation" data already exists (dict for dict match), LabTools will respond
    # with a HTTP Redirect code (302) to an url of the form "update-sample/<primary key>/".
    # The reponse.content will contain the database details of this sample (including the pk), so if
    # SAMPLING_REST_API_ALLOW_UPDATES is set to True, output_processing will re-upload (patch) the sample to the given URL.

    # DATABASE STORAGE
    # if True, the program will convert any csvs it finds in the output folder into the defined database and move the csvs to the storage folder
    # set these in the .env folder for security purposes, as this file is stored on the git.
    "USE_DB": False,
    "DBNAME": "signs_of_life",  # database where you're going to store all the of data
    "DBHOST": "dbpostgres",  # hostname or ip address of the db server
    "DBPORT": '5432',  # db access port
    "DBUSER": "postgres",  # db username
    "DBPASS": "password",  # db password

    # DEBUGGING
    "VERBOSE_DEBUGGING": False,  # allows printing of extra debugging information while developing.
    # RUNTIME PERFORMANCE LOGGING
    "PERFORMANCE_LOGGING": False,  # runs the custom performance logging module to profile and analyse code performance
    # save intermediary files
    "DEBUG_MODE": False,
    "DEBUG_PRINT": False,
}

# version
fp_vers = join(MAIN_DIR, "app_domains", "version.txt")
if not os.path.isfile(fp_vers):
    raise FileNotFoundError("Missing version.txt in app_domains")
with open(fp_vers) as f:
    RUN_CONFIG["version"] = f.read()


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
    # Overwrite existing key. If a non-existent key is added, it will just be ignored.
    RUN_CONFIG[key] = value


env_file_path = join(os.path.dirname(os.path.dirname(__file__)), ".env")

try:
    with open(env_file_path, 'r') as f:
        for line in f:
            setting = line.replace('\n', '').split('=')
            if len(setting) == 2:
                ProcessEnvSetting(setting)
except IOError:
    print(f"Error: Could not read {env_file_path}")
    # Handle the error or raise an exception as needed

from utils import PerformanceLogger

PLOG = PerformanceLogger(filename="main_perf.log", enable_logging=RUN_CONFIG['PERFORMANCE_LOGGING'])
