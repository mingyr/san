import os
import re
import subprocess
import signal
import string
import shlex
from monitor import Parser, Cmd

def test():
    results = []

    num_iters = 5

    folds = [0, 1, 2, 3, 4]
    test_samples = [104, 96, 96, 116, 116]

    for i, d in enumerate(folds):
        for j in range(num_iters):

            cmdline_str = 'python test.py --type=classification --test_batch_size={} --data_dir=/data/yuming/eeg-processed-data/vep/san/selected-channels/1d-leave-out/{} --output_dir=output/adp6/{}/{} --gpus=0'.format(test_samples[i], d, d, j)
            print(cmdline_str)

            cmdline = shlex.split(cmdline_str)

            # parser = Parser().get_custom_parser("\\#trials\\:\\d+")
            # parser = Parser().get_custom_parser(".+")

            parser = Parser().get_custom_parser("(metric \\-> (\\d|\\.)+)")
            cmd = Cmd(parser, results)
            cmd.start(cmdline, "tmp")
            cmd.wait()

    print("")
    for r in results:
        print(r)



if __name__ == '__main__':
    test()

    

