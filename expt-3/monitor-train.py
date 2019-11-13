import os
import re
import subprocess
import signal
import string
from monitor import Parser, Cmd

def train(parse = False, pseudo = False):
    import shlex

    num_iters = 5

    folds = [0, 1, 2, 3, 4]
    train_samples = [1210, 1217, 1217, 1200, 1200]
    xval_samples = [214, 215, 215, 212, 212]

    for i, d in enumerate(folds):
        for j in range(num_iters):

            cmdline_str = 'python train.py --type=classification --train_batch_size=256 --xval_batch_size={} --data_dir=/data/yuming/eeg-processed-data/vep/san/selected-channels/1d-leave-out/{} --output_dir=output/adp6/{}/{} --checkpoint_dir=adaptor5 --gpus=0'.format(xval_samples[i], d, d, j)
            print(cmdline_str)

            cmdline = shlex.split(cmdline_str)

            if parse:
                parser = Parser().get_custom_parser("\\#trials\\:\\d+")
                parser = Parser().get_custom_parser(".+")
                cmd = Cmd(parser)
                cmd.start(cmdline, "tmp")
                cmd.wait()
            else:
                cmd = Cmd()
                cmd.start(cmdline)
                cmd.wait()
 
if __name__ == '__main__':
    train(pseudo = True)

