import json
import io
import os
import time


class LogFile():
    def __init__(self, logfile_dir, logfile_name):
        self.logfile_dir = logfile_dir
        self.logfile_name = logfile_name

    def write_log(self, origin, log_data):
        logfile_path = os.path.join(self.logfile_dir,self.logfile_name)
        with io.open(logfile_path, "a", encoding='utf-8') as flog:
            flog.write(f'{origin} : Starting a new log :{time.strftime("%Y.%m.%d-%H:%M:%S")}\n')
            if (type(log_data) is str):
                flog.write(log_data)
            else:
                flog.write(json.dumps(log_data, ensure_ascii=False))
            flog.write(f'\n{origin} : Ending log.\n')


if __name__ == "__main__":
    logfile_dir = ''
    logfile_name = 'results.txt'
    logfile = LogFile(logfile_dir, logfile_name)