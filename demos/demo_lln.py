from BaseApp import BaseApp
import tkinter as tk
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import scipy.stats as stats

def sample_data(dist,mean,stddev,num_samples,num_lines):

    xdata = np.arange(1,num_samples+1)

    if dist=='Bernoulli':
        X = stats.bernoulli(mean)
    elif dist=='Gaussian':
        X = stats.norm(loc=mean, scale=stddev)
    elif dist=='Exponential':
        X = stats.expon(loc=0,scale=mean)
        stddev = mean
    elif dist=='Uniform':
        width = np.sqrt(12)*stddev
        a = mean-width/2
        X = stats.uniform(loc=a, scale=width)
    else:
        print("This should not happen")

    samples = X.rvs(size=(num_samples, num_lines))
    zz = np.cumsum(samples.T, axis=1) / np.tile(np.arange(1, num_samples + 1), (num_lines, 1))
    ydata = zz.T
    
    stddev = X.std() / np.sqrt(range(1, num_samples+1))
    upper = X.mean() + stddev
    lower = X.mean() - stddev

    # isclose, pcnt_close = None, None
    # isclose = np.zeros((num_samples, num_lines), dtype=bool)
    # for i in range(num_lines):
    #     isclose[1:, i] = np.logical_and(ydata[1:, i] > lower, ydata[1:, i] < upper)
    # pcnt_close = np.mean(isclose, axis=1)

    return {'xdata':xdata, 'ydata':ydata, 'upper':upper, 'lower':lower}

class TkContainer(BaseApp):

    distribution_names = ['Gaussian','Uniform','Exponential','Bernoulli']

    def __init__(self):
        super(TkContainer, self).__init__(
            title="Law of large numbers",
            geometry="1600x600",
            figsize=(16, 3),
            subplots=(1,2))

    def initialize_parameters(self):

        self.mean = tk.DoubleVar(master=self.root, value=0.0)
        self.meanstr = tk.StringVar(master=self.root, value='0.0')

        self.stddev = tk.DoubleVar(master=self.root, value=0.1)
        self.stddevstr = tk.StringVar(master=self.root, value='0.1')

        self.num_lines = tk.IntVar(master=self.root, value=10)
        self.num_linesstr = tk.StringVar(master=self.root, value='10')

        self.num_samples = tk.IntVar(master=self.root, value=20)
        self.num_samplesstr = tk.StringVar(master=self.root, value='20')
        
        self.selected_dist = tk.StringVar(master=self.root, value=self.distribution_names[0])

    def initialize_data(self):
        # Generate the data (xdata, ydata, upper, lower, isclose, pcnt_close)
        self.data = sample_data(self.selected_dist.get(),
                                self.mean.get(),
                                self.stddev.get(),
                                self.num_samples.get(),
                                self.num_lines.get())

    def initialize_fig(self):

        dist = self.selected_dist.get()
        mean = self.mean.get()
        stddev = self.stddev.get()
        num_lines = self.num_lines.get()
        num_samples = self.num_samples.get()
        
        ax = self.ax[0]
        ax.clear()
    
        if dist=='Gaussian' or dist=='Uniform' or dist=='Exponential':

            if dist=='Gaussian':
                X = stats.norm(loc=mean, scale=stddev)
                xx = np.linspace(mean-3*stddev,mean+3*stddev,200)
            elif dist=='Uniform':
                scale = np.sqrt(12)*stddev
                loc = mean - scale/2
                X = stats.uniform(loc=loc, scale=scale)
                xx = np.linspace(mean-3*stddev,mean+3*stddev,200)
            elif dist=='Exponential':
                X = stats.expon(loc=0,scale=mean)
                stddev = mean
                xx = np.linspace(0,mean+3*stddev,200)
            else:
                print("THIS SHOULD NOT HAPPEN")

            yy = X.pdf(xx)
            minx = min(xx)
            maxx = max(xx)
            maxy = max(yy)

            ax.plot(xx,yy,linewidth=4)

            xx = np.linspace(mean-stddev,mean+stddev,100)
            yy = X.pdf(xx)
            ax.fill_between(xx, 0*yy, yy, color='blue',alpha=0.2)

            ax.set_ylim(0,maxy*2)
            ax.set_xlim(minx,maxx)
        
        elif dist=='Bernoulli':
            p, loc = math.modf(mean)
            X = stats.bernoulli(p)
            maxy = 1

            markerline, stemlines, baseline = ax.stem([0,1],[1-p,p])
            markerline.set_markerfacecolor('r')
            markerline.set_markersize(18)
            markerline.set_markeredgewidth(2)
            stemlines.set_lw(4)
            ax.set_ylim(0,1)
            ax.set_xticks([loc,loc+0.5,loc+1])
            ax.set_xlim(loc-0.5,loc+1.5)

        else:
            print("This shouldn't happen")

        ax.vlines(mean,0,maxy*1.1,color='red',linewidth=2,linestyles='dashed')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        xticks = ax.get_xticks()
        ax.set_xticklabels(['{:.2f}'.format(x) for x in xticks], fontsize=26)
        ax.set_yticks([])

        # Paint convergence plot -------------------------------------
        ax = self.ax[1]
        ax.clear()
        for i in range(num_lines):
            ax.plot(self.data['xdata'], self.data['ydata'][:, i], linewidth=2)

        # ax.plot(range(1,num_samples+1),self.data['lower'],'-',linewidth=2,color='k',alpha=0.5)
        # ax.plot(range(1,num_samples+1),self.data['upper'],'-',linewidth=2,color='k',alpha=0.5)
        # ax.fill_between(range(1, num_samples+1), self.data['lower'], self.data['upper'], color=[0.9, 0.9, 0.9])
        
        if stddev>0:
            ax.set_ylim(mean-2*stddev,mean+2*stddev)
        else:
            ax.set_ylim(mean-1,mean+1)
        ax.set_xlim(0,num_samples*1.1)
        ax.set_xlabel('N', fontsize=16)
        ax.hlines(mean, 0, num_samples, 'r',linestyle='--', linewidth=4)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        xticks = ax.get_xticks()
        ax.set_xticklabels(['{:.0f}'.format(x) for x in xticks], fontsize=26)
        yticks = ax.get_yticks()
        # ax.set_yticklabels(['{:.2f}'.format(x) for x in yticks], fontsize=26)
        ax.set_xlabel('N',fontsize=30)

        plt.draw()

    def add_widgets(self):

        header_width = 40

		# Header ------------------------------------------
        self.get_header(self.root,text='Distribution',char='.',width=header_width)\
			.pack(side=tk.TOP, fill=tk.X)

        # select model combo box ........................................
        self.get_combobox(self.root,
                          text='Family',
                          textvariable = self.selected_dist,
                          values = self.distribution_names,
                          command = self.select_dist)\
			.pack(side=tk.TOP, fill=tk.X)

        # Mean input box ......................................
        self.get_entry_label(self.root,
                     text="Mean",
                     textvariable=self.meanstr,
                     validatecommand=self.set_mean) \
            .pack(side=tk.TOP, fill=tk.X)

        # Stddev input box ......................................
        self.get_entry_label(self.root,
                     text="Stddev",
                     textvariable=self.stddevstr,
                     validatecommand=self.set_stddev) \
        .pack(side=tk.TOP, fill=tk.X)

		# Header ------------------------------------------
        self.get_header(self.root,text='Samples',char='.',width=header_width)\
			.pack(side=tk.TOP, fill=tk.X)

        # num lines input box ......................................
        self.get_entry_label(self.root,
                     text="Number lines",
                     textvariable=self.num_linesstr,
                     validatecommand=self.set_numlines) \
            .pack(side=tk.TOP, fill=tk.X)

        # num samples input box ......................................
        self.get_entry_label(self.root,
                     text="N max",
                     textvariable=self.num_samplesstr,
                     validatecommand=self.set_num_samples) \
            .pack(side=tk.TOP, fill=tk.X)

        # resample button ......................................
        self.get_button(self.root, text="Resample", command=self.click_resample) \
            .pack(side=tk.TOP, fill=tk.X)

    def update_figure(self):
        self.initialize_fig()

    def select_dist(self,event):

        # set stddev for distribution
        if self.selected_dist.get()=="Bernoulli":
            mean = float(self.meanstr.get())
            sigma = np.sqrt(mean*(1-mean))
            self.stddev.set(sigma)
            self.stddevstr.set("{:.1f}".format(sigma))

        if self.selected_dist.get()=="Exponential":
            mean = float(self.meanstr.get())
            sigma = mean
            self.stddev.set(sigma)
            self.stddevstr.set("{:.1f}".format(sigma))

        self.click_resample()
                
    def set_mean(self):
        try:
            oldval = self.mean.get()
            newval = float(self.meanstr.get())
            if not np.isclose(oldval,newval):
                self.mean.set(newval)

                # set stddev for distribution
                if self.selected_dist.get()=="Bernoulli":
                    sigma = np.sqrt(newval*(1-newval))
                    self.stddev.set(sigma)
                    self.stddevstr.set("{:.1f}".format(sigma))

                if self.selected_dist.get()=="Exponential":
                    sigma = newval
                    self.stddev.set(sigma)
                    self.stddevstr.set("{:.1f}".format(sigma))

                self.click_resample()
            return True
        except ValueError:
            return False

    def set_stddev(self):
        try:
            oldval = self.stddev.get()
            newval = float(self.stddevstr.get())
            if not np.isclose(oldval,newval):
                self.stddev.set(newval)

                # set stddev for distribution
                if self.selected_dist.get()=="Bernoulli":
                    p = 0.5*(1 - np.sqrt(1-newval**2))
                    self.mean.set(p)
                    self.meanstr.set("{:.1f}".format(p))

                if self.selected_dist.get()=="Exponential":
                    mean = newval
                    self.mean.set(mean)
                    self.meanstr.set("{:.1f}".format(mean))

                self.click_resample()
            return True
        except ValueError:
            return False

    def set_numlines(self):
        try:
            oldval = self.num_lines.get()
            newval = int(self.num_linesstr.get())
            if not np.isclose(oldval,newval):
                self.num_lines.set(newval)
                self.click_resample()
            return True
        except ValueError:
            return False

    def set_num_samples(self):
        try:
            oldval = self.num_samples.get()
            newval = int(self.num_samplesstr.get())
            if not np.isclose(oldval,newval):
                self.num_samples.set(newval)
                self.click_resample()
            return True
        except ValueError:
            return False

    def click_resample(self):
        self.data = sample_data(self.selected_dist.get(),
                                self.mean.get(),
                                self.stddev.get(),
                                self.num_samples.get(),
                                self.num_lines.get())
        self.update_figure()
        return True

####################################################
if __name__ == "__main__":
    app = TkContainer()
    tk.mainloop()

