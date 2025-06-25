import sys
import os

class Tee(object):
    """터미널과 파일 동시 출력을 위한 클래스"""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

def setup_logging(results_dir):
    """로그 파일 설정"""
    log_file = open(os.path.join(results_dir, "experiment_log.txt"), "w")
    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)
    return log_file