import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import List, Callable, Tuple, Any


class InteractiveDataPlotter:
    """
    Plot and interact with large number of 1d data points.

    Parameters
    ----------
    data : The data to plot. If 'datas' is provided, this is ignored.
    datas : List of data to plot. Specify nrows and ncols and plot all the given data in the same figure.
    titles : List of titles for each data.
    startidx : The index of the data to start with.
    nrows : Number of rows in the plot.
    ncols : Number of columns in the plot.
    subplot_kw : Keyword arguments to pass to plt.subplots.
    section_infos : List of functions that take the data index, section index and the section data and return a tuple of (metric_name, metric_value).
    y_range_offset : The range of the y-axis is calculated as the mean of the data +/- y_range_offset * (max - min). Default is 1.1.
    data_split : If only 'data' is provided, we can split 'data' into this many sections and plot them in the same figure, as if 'datas' was provided.
    """
    def __init__(self, data=None, datas: list = None, titles: list = None, startidx=0, nrows=1, ncols=1, subplot_kw={}, section_infos: List[Callable[[int, int, Any], Tuple[str, Any]]] = None, y_range_offset=None, data_split=1):
        assert data is not None or datas is not None, "Either data or datas must be provided."

        if datas is None:
            if data_split > 1:
                if isinstance(data, torch.Tensor):
                    datas = data.chunk(data_split, dim=0)
                else:
                    datas = np.array_split(data, data_split, axis=0)
            else:
                datas = [data]
            
        self.datas = datas
        self.titles = [''] * len(self.datas) if titles is None else [f'{title}\n' for title in titles]
        self.section_infos = section_infos

        if not all([len(data) == len(self.datas[0]) for data in self.datas]):
            print("Data have different length, using the shortest.")
        self.minN = min([len(data) for data in self.datas])

        self.y_range_offset = 1.1 if y_range_offset is None else y_range_offset
        self.nrows = nrows
        self.ncols = ncols
        self.prev_ylims = [None] * len(self.datas)
        
        self.index = startidx
        self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, **subplot_kw)
        self.plot_data()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.show()

    def plot_data(self):
        for i, data in enumerate(self.datas):
            r, c = i // self.ncols, i % self.ncols
            section = data[self.index]
            self.axes[r][c].clear()
            self.axes[r][c].plot(section)
            self.axes[r][c].grid()

            metrics_str = self.calculate_section_metrics(i)
                
            self.axes[r][c].set_title(f'{self.titles[i]}{metrics_str}Data {self.index + 1}/{len(self.datas[i])}')
            self.axes[r][c].set_ylim(self.calculate_ylim(i))
            self.axes[r][c].set_xlim(0, len(data[self.index]) - 1)
        self.fig.tight_layout()
        self.fig.canvas.draw()

    def calculate_section_metrics(self, dataidx):
        section = self.datas[dataidx][self.index]
        metrics_str = ''
        if self.section_infos is not None:
            metrics = []
            for _, metric in enumerate(self.section_infos):
                metric_val = metric(dataidx, self.index, section)
                if metric_val is not None:
                    metrics.append(metric_val)
            metrics_str = " - ".join([f"{metric}: {value}" for metric, value in metrics])
            metrics_str = f"{metrics_str}\n"
        return metrics_str

    def calculate_ylim(self, dataidx):
        section = self.datas[dataidx][self.index]
        if (isinstance(section, torch.Tensor)):
            min_val = section.min().item()
            max_val = section.max().item()
        else:
            min_val = np.min(section)
            max_val = np.max(section)
        mean_val = (min_val + max_val) / 2
        range_offset = (max_val - min_val) * self.y_range_offset
        
        lower_limit = mean_val - range_offset / 2
        upper_limit = mean_val + range_offset / 2

        if self.prev_ylims[dataidx] is not None:
            prev_lower_limit, prev_upper_limit = self.prev_ylims[dataidx]
            if min_val >= prev_lower_limit:
                lower_limit = prev_lower_limit
            if max_val <= prev_upper_limit:
                upper_limit = prev_upper_limit

        self.prev_ylims[dataidx] = (lower_limit, upper_limit)
        return lower_limit, upper_limit

    def on_key_press(self, event):
        if event.key == 'right':
            self.index = (self.index + 1) % self.minN
            self.plot_data()
        elif event.key == 'left':
            self.index = (self.index - 1) % self.minN
            self.plot_data()


class LongDataPlotter:
    def __init__(self, data, window_size=128, stride=128 // 3, y_range_size=2, start_section=0, datas: list = None, titles: list = None, nrows=1, ncols=1, subplot_kw={}):
        self.datas = [data] if datas is None else datas
        self.titles = ['\n'] * len(self.datas) if titles is None else [f'{title}\n' for title in titles]

        if not all([len(data) == len(self.datas[0]) for data in self.datas]):
            print("Data have different length, using the shortest.")
        self.minN = min([len(data) for data in self.datas])

        self.window_size = window_size
        self.stride = stride
        self.y_range_size = y_range_size
        self.current_section = start_section

        self.nrows = nrows
        self.ncols = ncols
        self.prev_ylims = [None] * len(self.datas)

        self.start_index = min(self.stride * start_section, self.minN - self.window_size)
        self.end_index = min(self.start_index + self.window_size, len(data))
        self.fig, self.axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, subplot_kw=subplot_kw)
        self.plot_data()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        plt.show()

    def plot_data(self):
        for i, data in enumerate(self.datas):
            r, c = i // self.ncols, i % self.ncols
            section = data[self.start_index:self.end_index]
            self.axes[r][c].clear()
            self.axes[r][c].plot(section)
            self.axes[r][c].grid()
            self.axes[r][c].set_title(f'{self.titles[i]}Section {self.current_section}: {self.start_index + 1}-{self.end_index} / {self.minN}')
            self.axes[r][c].set_ylim(self.calculate_ylim(i))
            self.axes[r][c].set_xlim(0, self.window_size - 1)
        self.fig.tight_layout()
        self.fig.canvas.draw()

    def calculate_ylim(self, dataidx):
        section = self.datas[dataidx][self.start_index:self.end_index]
        min_val = section.min().item()
        max_val = section.max().item()
        mean_val = (min_val + max_val) / 2
        lower_limit = mean_val - self.y_range_size / 2
        upper_limit = mean_val + self.y_range_size / 2

        if self.prev_ylims[dataidx] is not None:
            prev_lower_limit, prev_upper_limit = self.prev_ylims[dataidx]
            if min_val >= prev_lower_limit:
                lower_limit = prev_lower_limit
            if max_val <= prev_upper_limit:
                upper_limit = prev_upper_limit

        self.prev_ylims[dataidx] = (lower_limit, upper_limit)
        return lower_limit, upper_limit

    def on_key_press(self, event):
        if event.key == 'right':
            self.current_section += 1
            self.start_index = min(self.start_index + self.stride, self.minN - self.window_size)
            self.end_index = min(self.start_index + self.window_size, self.minN)
            self.plot_data()
        elif event.key == 'left' and self.current_section > 0:
            self.current_section -= 1
            self.start_index = max(self.start_index - self.stride, 0)
            self.end_index = min(self.start_index + self.window_size, self.minN)
            self.plot_data()