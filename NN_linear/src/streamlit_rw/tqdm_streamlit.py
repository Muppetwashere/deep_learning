import sys
import time

import streamlit as st
import tqdm
from keras_tqdm import TQDMCallback


class TQDMStreamlit:
    def __init__(self, desc: str, total: int, leave: bool, file=None, initial: int = 0,
                 mininterval=0.1, smoothing=0.3, remaining_est: bool = True):
        """

        :param desc: Initial description
        :param total: Maximum value of the progress bar
        :param leave: Boolean: leave or not the progress bar once finished
        :param file: Used for compatibility only
        """
        self.desc: str = desc
        self.total: int = total
        self.leave: bool = leave
        self.smoothing = smoothing
        self.remaining_est = remaining_est
        self.file: str = file  # For compatibility
        self.original_mininterval = mininterval
        self.initial = initial

        self.n: int = self.initial  # Current progress
        self.avg_time = None
        self.last_print_n = self.initial
        self.last_print_t = time.time()
        self.start_t = time.time()
        self.miniters: float = self.original_mininterval
        self.mininterval: float = self.original_mininterval

        self.st_text = st.text(self.desc)
        self.progress = st.progress(0)
        self.refresh()

    def update(self, n: int = 1, desc: str = None):
        """
        Manually update the progress bar, useful for streams
        such as reading files.

        :param n: int, optional
            Increment to add to the internal counter of iterations
            [default: 1].
        :param desc: str, optional
            Modifies the description of the progress bar
        """

        if n < 0:
            self.last_print_n += n  # for auto-refresh logic to work
        self.n += n
        if desc is not None:
            self.desc = desc

        # check counter first to reduce calls to time()
        if self.n - self.last_print_n >= self.miniters:
            delta_t = time.time() - self.last_print_t
            if delta_t >= self.mininterval:
                cur_t = time.time()
                delta_it = self.n - self.last_print_n  # >= n
                # elapsed = cur_t - self.start_t
                # EMA (not just overall average)
                if self.smoothing and delta_t and delta_it:
                    rate = delta_t / delta_it
                    self.avg_time = tqdm.tqdm.ema(
                        rate, self.avg_time, self.smoothing)

                self.refresh()

                # Store old values for next call
                self.last_print_n = self.n
                self.last_print_t = cur_t

    # noinspection PyUnusedLocal
    def refresh(self, nolock=False, lock_args=None):
        percentage = max(0, min(100, int(100 * self.n / self.total)))  # Integer value of %age, clipped btwn 0 and 100

        elapsed = time.time() - self.start_t
        bar_format = '{n_fmt}/{total_fmt}, {percentage:3.0f}% - {desc}'
        if self.remaining_est:
            bar_format += ' - [{elapsed}<{remaining}, {rate_fmt}]'

        tqdm_str = tqdm.tqdm.format_meter(self.n, self.total, elapsed, prefix=self.desc,
                                          rate=1 / self.avg_time if self.avg_time else None,
                                          bar_format=bar_format)

        # self.st_text.text('%d/%d, %d%% - %s' % (self.n, self.total, percentage, self.desc))
        self.st_text.text(tqdm_str)
        self.progress.progress(percentage)

    def close(self):
        # todo: see if it would be possible to hide the progress bar
        self.progress.progress(100)

    def reset(self, desc: str, total: int):
        self.total = total

        self.n: int = self.initial  # Current progress
        self.avg_time = None
        self.last_print_n = self.initial
        self.last_print_t = time.time()
        self.start_t = time.time()
        self.miniters: float = self.original_mininterval
        self.mininterval: float = self.original_mininterval

        self.update(n=0, desc=desc)


class TQDMStreamlitCallback(TQDMCallback):
    def __init__(self,
                 outer_description="Training",
                 inner_description_initial="Epoch {epoch}",
                 inner_description_update="Epoch: {epoch} - {metrics}",
                 metric_format="{name}: {value:0.3f}",
                 separator=", ",
                 leave_inner=False,
                 leave_outer=True,
                 output_file=sys.stderr, **kwargs):
        super(TQDMStreamlitCallback, self).__init__(outer_description=outer_description,
                                                    inner_description_initial=inner_description_initial,
                                                    inner_description_update=inner_description_update,
                                                    metric_format=metric_format,
                                                    separator=separator,
                                                    leave_inner=leave_inner,
                                                    leave_outer=leave_outer,
                                                    output_file=output_file, **kwargs)

    def tqdm(self, desc, total, leave, initial=0):
        """
        Extension point. Override to provide custom options to tqdm_notebook initializer.
        :param desc: Description string
        :param total: Total number of updates
        :param leave: Leave progress bar when done
        :return: new progress bar
        :param initial: Initial counter state
        """
        return TQDMStreamlit(desc=desc, total=total, leave=leave, initial=initial)

    def build_tqdm_inner(self, desc, total):
        if self.tqdm_inner is None:
            return super().build_tqdm_inner(desc, total)
        else:
            self.tqdm_inner.reset(desc, total)
            return self.tqdm_inner
