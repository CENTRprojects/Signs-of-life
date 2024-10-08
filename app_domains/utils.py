"""Various utilitary functions for the crawler"""
from os.path import join
import os
import pandas as pd
import numpy as np
import re
from bs4.element import Comment
import json
from openpyxl.utils.dataframe import dataframe_to_rows
import inspect 
from datetime import datetime
import time
import idna

# shutting logs
import logging
loggers = [logging.getLogger()]  # get the root logger
loggers = loggers + [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for log in loggers:
    log.setLevel(logging.WARNING)


RE_FULL_DIG = re.compile("^\d+$")

def save_obj(obj, name, path):
    """save json"""
    with open(join(path, 'obj_' + name + '.json'), "w") as f:
        json.dump(obj, f)


def load_obj(name, path):
    """load json"""
    with open(join(path, 'obj_' + name + '.json'), "r") as f:
        # return msgpack.unpackb(f.read())
        return json.load(f)


def gather_url_saved(path_url_save):
    """Read all URL level pickles"""
    list_urls = os.listdir(path_url_save)
    list_existing_url = []
    for url_json in list_urls:
        try:
            with open(join(path_url_save, url_json), 'r') as f:
                try:
                    resu = json.load(f)
                except EOFError:
                    continue
                except ImportError:
                    continue
                except Exception as e:
                    if "doorstopper" not in join(path_url_save, url_json):
                        print("Error with JSON file : {}\t{}--\t{}".format(type(e), str(e), join(path_url_save, url_json)))
                    continue
            list_existing_url.append(resu)

        except Exception as e:
            print("Error while reading JSON file : {}\t{}".format(type(e), str(e)))
            continue

    return list_existing_url


def remove_url_saved(path_url_save):
    """Delete temporary saved URL request responses"""
    list_urls = os.listdir(path_url_save)
    for f in list_urls:
        if "doorstopper" not in f:
            try:
                os.remove(join(path_url_save, f))
            except FileNotFoundError as e:
                print(f"remove_url_saved: File not found {f}")


def clean_link(lk):
    """Normalize URL link down to 'domain name + . + TLD'"""
    lk = lk.replace("\\", "/")
    if lk.startswith("https://www."):
        lk = lk[12::]
    if lk.startswith("http://www."):
        lk = lk[11::]
    if lk.startswith("https://"):
        lk = lk[8::]
    if lk.startswith("http://"):
        lk = lk[7::]
    if lk.startswith("//"):
        lk = lk[2::]
    if lk.startswith("www."):
        lk = lk[4::]
    if lk.endswith("/"):
        lk = lk[:-1]
    return lk


def link_to_domain(lk):
    """extract domain name of a link"""
    lk = clean_link(lk)
    lk = lk.split("/")[0]
    lk = lk.split("?")[0]
    return lk


def tag_visible(element):
    """Check the HTML tag 'element' is displayed to user"""
    if element.parent.name in ['style', 'script', 'meta']:
        return False
    if isinstance(element, Comment):
        return False
    return True


def split_big_file():
    """Split large domain file into files with n_sample rows"""

    fp = r"/media/totoyoutou/ssd/freelance/sign_of_life_crawler/input/speed10k_500/spd10k500.csv"
    pout = r"/media/totoyoutou/ssd/freelance/sign_of_life_crawler/input/speed10k_500"
    n_sample = 500
    sep = ","

    df = pd.read_csv(fp, delimiter=sep)

    lg = len(df)
    last_name = fp.split("\\")[-1]

    for i in range(0, int(lg / n_sample) + 1):
        df_tempo = df[i * n_sample: (i + 1) * n_sample]

        df_tempo.to_csv(join(pout, last_name[0:-4] + "_part_" + str(i) + ".csv"), index=False)

    print("done")

def get_db_sample_for_testing(datestr, tld, qty = 100):
    """Load samples from database"""
    import output_processing as op
    import random
    if op.db is None:
        op.ConnectDB()
    cur = op.db.cursor()
    try:
        #cur.execute(f"select date, tld, input_url, final_url, category, subcategory, pred_is_parked, is_error, comment from signs_of_life_crawler where date='{datestr}'")
        cur.execute("select * from signs_of_life_crawler where date={}".format(datestr))
    except:
        op.db.rollback()
        raise
    col_names = [desc[0] for desc in cur.description]
    all_data = cur.fetchall()
    all_data_len = len(all_data)
    print('Found {} rows in database. {} transferred to all_data'.format(cur.rowcount, all_data_len))
    if all_data_len == 0:
        print("No rows in database!")
        return None, None
    random_indices = set()
    iterations = 0
    while len(random_indices) < qty:
        iterations += 1
        random_indices.add(random.randrange(0,all_data_len))
    print(random_indices)
    print('Found {} samples in {} iterations'.format(len(random_indices),iterations))
    sample = [all_data[idx] for idx in random_indices]
    cur.close()
    return col_names, sample

def save_db_sample_to_csv_for_testing(datestr, tld, qty = 100):
    """Save samples into database"""
    import csv
    headers, sample = get_db_sample_for_testing(datestr, tld, qty)
    if headers is None:
        return
    filename = join(os.path.pardir, 'output', '{}-db_sample.{}.{}.csv'.format(datestr,qty,tld))

    print('Writing {} samples to csv.'.format(len(sample)))
    with open(filename,'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(headers)
        csvwriter.writerows(sample)
    print('Sample data saved to: {}'.format(filename))

def write_in_excel_df(ws, df, t_col=0, t_row=0):
    """write the DataFrame df into the Excel worksheet ws"""
    rows = dataframe_to_rows(df)
    for r_idx, row in enumerate(rows, 1):
        for c_idx, value in enumerate(row, 1):
            ws.cell(row=r_idx + t_row, column=c_idx + t_col, value=value)


def csv_name_to_json_name(x):
    """Remove .JSON"""
    x = x[0:-4]
    return "obj_documents_" + x + ".json"


# performance logging

'''
    Runtime Performance (Logging)
        Takes a string and logs it to the runtime performance logfile along with running times
    Usage:
        from utils import PerformanceLogger 
        plog = PerformanceLogger()
          Options:
            filepath        str     the folder you want to store the log in, defaults to ./logging
            filename        str     the log's filename, defaults to performance_runtime.log
            to_screen       bool    output to screen as well, defaults True
            enable_logging  bool    to allow disabling the log for squeezing out performance

        plog = PerformanceLogger()
        plog.perf_go("My Speedy App")                       # Start your engines, inits the timer
        plog.perf_lap(f"Finished Loop {loop_counter}")      # mark points in code you wish to count as a lap. Considers the code before it, not after, so put at the end of code you want to profile.
        plog.perf_end("All Done")                           # end the performance counter, reports all of the laptimes

        # You can also log some intersting events along with runtime module information
        plog.it("I've just started someting interesting")
        plog.it("I've just finished something interesting")

        # You can surpress or focus on particular modules by just adding their names to the filter
        plog.surpress_modules(["spammy_function_1", "spammy_module_2"])             # stop these log messages being sent to screen or logfile
        plog.focus_modules(["interesting_function_1", "interesting_function_2"])    # stop all output *except these*

'''

class PerformanceLogger(object):
    def __init__(self, filepath="logging/", filename="performance_runtime.log", to_screen=True, screen_log_surpression=False, enable_logging=True):
        self.to_screen = to_screen
        self.logpath = os.path.join(filepath,filename)
        self.errorpath = os.path.join(filepath,"error_perf.log")
        self.perf_running = False
        self.perf_start = None
        self.perf_stop = None
        self.laps = []
        self.tick_last = 0.0
        self.tick = 0
        self.lap = 0.0
        self.screen_log_surpression = screen_log_surpression
        self.surpressed = set()
        self.focused = set()
        self.enabled = enable_logging
        #self.it(f"Printing to screen: {self.to_screen}\nSaving to file: {self.logpath}")

    def it(self, msg, perf_go=False, perf_end=False, is_error=False):
        if self.enabled is not True:
            return
        frame = inspect.stack()[1]
        filename = os.path.basename(frame[0].f_code.co_filename).replace('.py','')
        modname = frame[0].f_code.co_name
        args = frame[0].f_code.co_argcount
        perf_msg = ""
        if perf_end:
            self.perf_end()
        if self.perf_start is not None:
            self.perf_tick()
            perf_msg += f"( tick = {self.get_perf_str(self.tick)} )"
        if perf_go is True:
            self.perf_go()
        try:
            log_msg = datetime.now().strftime(f"%Y-%m-%d %H:%M:%S => ")
            log_msg += perf_msg+" : "
            log_msg += filename+":"
            log_msg += f"{modname}({args}) => "
            log_msg += f"{msg}\n"
        except ValueError:
            print(filename, modname, args, msg)
            raise
        if len(self.focused) > 0:
            #if modname not in self.focused and filename not in self.focused and not self.is_phrase_in_msg(self.focused, log_msg) and modname is not 'Log':
            if 'surpress_modules' in log_msg or self.is_phrase_in_msg(self.focused, log_msg):
                if self.screen_log_surpression:
                    print(f'ALLOWING {log_msg}')
            else:
                if self.screen_log_surpression:
                    print(f'DENYING {log_msg}\n\n')
                return    
        #if modname is not 'Log' or modname in self.surpressed or filename in self.surpressed or self.is_phrase_in_msg(self.surpressed, msg):
        if 'surpress_modules' in log_msg or not self.is_phrase_in_msg(self.surpressed, log_msg):
            if self.screen_log_surpression:
                print(f'ENCOURAGING {log_msg}')
        else:
            if self.screen_log_surpression:
                print(f'SURPRESSING {log_msg}\n\n')
            return
        with open(self.logpath, mode = 'a') as f:    
            try:
                f.write(log_msg)
            except:
                pass
        if is_error is True: 
            with open(self.errorpath, mode='a') as f:
                try:
                    f.write(log_msg)
                except:
                    pass
        if self.to_screen == True:
            if self.screen_log_surpression:
                print(f'LOG_MSG:\t{log_msg}\n\n')
            else:
                print(f'{log_msg}')
    def is_phrase_in_msg(self, phrases, msg):
        for phrase in phrases:
            if phrase in msg:
                if self.screen_log_surpression:
                    print(f'CHECK ==> {phrases} in: {msg}')
                return True
        if self.screen_log_surpression:
            print(f'CHECK ==> {phrases} not in: {msg}')
        return False
    def perf_go(self, msg=""):
            self.perf_start = time.perf_counter()
            self.laps = []
            self.tick_last = 0.0
            self.tick = 0
            self.lap = time.perf_counter()
            self.it(f"{msg}\t=> Performance Clock Started.")
    def perf_tick(self):
        self.tick = time.perf_counter() - self.tick_last
        self.tick_last = time.perf_counter()
    def perf_lap(self, msg):
        lap = time.perf_counter()
        laptime = lap - self.lap
        self.laps.append((msg, laptime))
        self.lap = lap
        self.it(f"LAP: {self.get_perf_str(laptime)} : {msg}")
    def get_perf_str(self,tick):
        if tick < 60.0:
            return f"{tick: .2f}s"
        else:
            return time.strftime('%Hh %Mm %Ss', time.gmtime(tick))
    def perf_end(self, msg=""):
        if self.perf_start is not None:
            self.perf_stop = time.perf_counter() - self.perf_start
            self.perf_start = None
            for lap in self.laps:
                self.it(f"{lap[0]} => {self.get_perf_str(lap[1])}")
            self.it(f"{msg}\t=> => {self.get_perf_str(self.perf_stop)}")
        else:
            self.it(f"There is no timer running! use perf_go() to begin one")
    def surpress_modules(self, modules):
        """Add a list of functions you want to globally surpress in the logs, for dev purposes to help with quietening it. \n
            If any modules have been added to focusmodules, this list will still be considered.\n        
        Arguments:\n
            modules {list} -- a list of function names or modules, both will be checked against this surpression list. \n
        """        
        self.it(f'Log surpressing: {modules}')
        self.surpressed = self.surpressed.union(modules)
    def focus_modules(self,modules):
        """Grant log privileges to the given list of modules. Only these will be able to output into the log. Module functions can be further filtered by surpression. \n
        Arguments:\n
            modules {list} -- list of function names or modules, all of which will be granted exclusive logging permission\n
        """                        
        self.it(f'Log isolating: {modules}')
        self.focused = modules

class CustomJSONizer(json.JSONEncoder):
    """
    Used to allow np.booleans to be properly exported by json.dumps.
    Usage: 
        from .utils import CustomJSONizer
        json.dumps(dict, cls=CustomJSONizer)
    """
    def default(self, obj):
        return super().encode(bool(obj)) \
            if isinstance(obj, np.bool_) \
                else super().default(obj)

def convert_idna(url):
    """Convert url into IDNA format (with only ascii letters)"""
    try:
        return idna.encode(url).decode("utf-8")
    except:
        return url

def is_str_full_digit(x):
    """Check if the string x is only made of digits"""
    if isinstance(x, str):
        return RE_FULL_DIG.search(x) is not None
    else:
        return False

def normalize_hcj_value(val):
    """Normalize the string/number val"""
    return str(val).lower()


def normalize_hcj_list(lst):
    """Normalize list lst"""
    return [str(e).lower() for e in lst]


def normalize_hcj_set(st):
    """Normalize set st"""
    nst = set()
    for k in st:
        nst.add(str(k).lower())
    return nst


def normalize_hcj_keys(dico):
    """Normalize dictionnary keys of dico"""
    ndico = {}
    for k, v in dico.items():
        ndico[str(k).lower()] = v
    return ndico

