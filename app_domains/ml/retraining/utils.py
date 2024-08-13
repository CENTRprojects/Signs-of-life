"""Utilitary functions for the retraining pipeline"""
import inspect
from openpyxl.utils.dataframe import dataframe_to_rows

from ml.retraining.imports import *
# from utils import PerformanceLogger

DT_FORMAT_NAMING_WITH_SEC = "%Y%m%d-%H%M%S"


def gtm():
    """Get date time up to seconds"""
    return datetime.datetime.now().strftime(DT_FORMAT_NAMING_WITH_SEC)


def print_all_done():
    add_to_log("-" * 64 + "\n" + "-" * 64 + "\n" + "-" * 24 + "ALL DONE" + "-" * 32 + "\n" + "-" * 64 + "\n" + "-" * 64)


def write_in_excel_df(ws, df, t_col=0, t_row=0):
    rows = dataframe_to_rows(df)
    for r_idx, row in enumerate(rows, 1):
        for c_idx, value in enumerate(row, 1):
            ws.cell(row=r_idx + t_row, column=c_idx + t_col, value=value)


def clean_link(lk):
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
    lk = clean_link(lk)
    lk = lk.split("/")[0]
    lk = lk.split("?")[0]
    return lk

def is_not_trivial_init2k_sample(text):
    return text not in ["", " ", ".", ".."]


def replace_dico(x, dico):
    if x in dico:
        return dico[x]
    else:
        return x


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

