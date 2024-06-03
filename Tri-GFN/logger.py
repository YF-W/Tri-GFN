import sys
import os
import logging

def write2txt(file, info):
    with open(file, 'a') as f:
        f.write(str(info[0]) + '\t' + str(info[1])
                + '\t' + str(info[2]) + '\t' + str(info[3]) + '\n')
        f.close()

def metrics_info(metrics):
    metric_show = ['\t'.join(metrics)]
    metric = '\t'.join(metric_show)
    return "metrics:\t%s" % metric

def record_info(record):
    rec = [format(x, '.4f') for x in record]
    info_show = ['\t'.join(rec)]
    rec = '\t'.join(info_show)
    return rec

class Logger(object):

    def __init__(self, filename):
        dir_name = os.path.dirname('./log/')
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        self.logger = logging.getLogger(filename)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d: %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
        logger_name = os.path.join(dir_name, filename + ".log")

        fh = logging.FileHandler(logger_name)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.removeHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.removeHandler(ch)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def _flush(self):
        for handler in self.logger.handlers:
            handler.flush()

    def debug(self, message):
        self.logger.debug(message)
        self._flush()

    def info(self, message):
        self.logger.info(message)
        self._flush()

    def warning(self, message):
        self.logger.warning(message)
        self._flush()

    def error(self, message):
        self.logger.error(message)
        self._flush()

    def critical(self, message):
        self.logger.critical(message)
        self._flush()
