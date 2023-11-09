import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from BaseApp import BaseApp
import tkinter as tk
import matplotlib.animation as animation


class TkContainer(BaseApp):

    signal_types = ['manual','sine wave','random']

    def __init__(self):
        super(TkContainer, self).__init__(
            title="Time Series demo",
            geometry="1400x500",
            figsize=(12, 4))

    def initialize_parameters(self):

        self.muY = tk.DoubleVar(master=self.root, value=0)
        self.varY = tk.DoubleVar(master=self.root, value='1.0')
        self.window_size = tk.IntVar(master=self.root, value='10')
        self.signal_type = tk.StringVar(master=self.root, value='manual')

    def add_widgets(self):

        header_width = 40

        # Header ------------------------------------------
        self.get_header(self.root,text='Deterministic signal',char='.',width=header_width)\
            .pack(side=tk.TOP, fill=tk.X)

		# select model combo box ........................................
        self.get_combobox(self.root,
						  	text='Type',
							textvariable = self.signal_type,
							values = self.signal_types,
							command = self.setit)\
			.pack(side=tk.TOP, fill=tk.X)


        # Header ------------------------------------------
        self.get_header(self.root,text='Y',char='.',width=header_width)\
            .pack(side=tk.TOP, fill=tk.X)
        
        # muY input box ......................................
        self.get_scale(self.root,
                       variable=self.muY,
                       command=self.setit,
                       from_= -2,
                       to=2,
                       resolution = 0.1,
                       length=200,
                       text='muY')\
                .pack(side=tk.TOP, fill=tk.X)
        
        # muY input box ......................................
        self.get_scale(self.root,
                       variable=self.varY,
                       command=self.setit,
                       from_=0,
                       to=2,
                       resolution = 0.1,
                       length=200,
                       text='varY')\
                .pack(side=tk.TOP, fill=tk.X)

        # Header ------------------------------------------
        self.get_header(self.root,text='Moving average',char='.',width=header_width)\
            .pack(side=tk.TOP, fill=tk.X)
        
        self.get_scale(self.root,
                       variable=self.window_size,
                       command=self.setit,
                       from_=1,
                       to=100,
                       resolution = 1,
                       length=200,
                       text='window_size')\
                .pack(side=tk.TOP, fill=tk.X)

    def initialize_fig(self):

        ax = self.ax

        num_points = 100
        self.R2 = np.full(num_points,np.NaN)

        self.points = np.full(num_points,np.NaN)
        self.line_points = ax.plot(range(num_points),self.points,'k.-',linewidth=2,markersize=20)

        self.marker_mu = ax.plot(num_points,0,'r+',markersize=40,markeredgewidth=6)

        self.muhat = np.full(num_points,np.NaN)
        self.muhat[-1] = 1
        self.line_muhat = ax.plot(range(num_points),self.muhat,'m-',linewidth=6)

        # self.text = ax.text(3,2.5,"t={}\nforecast MSE={:.2f}".format(0,0),fontsize=32)

        ax.set_ylim(-3,3)
        ax.set_xlim(0,num_points)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_yticklabels(ax.get_yticklabels(),fontsize=26)

        ax.set_xticklabels(["now-100","now-80","now-60","now-40","now-20","now"],fontsize=26)

        self.ani = animation.FuncAnimation(fig=self.fig, func=self.update, frames=10000, interval=30)

    def update(self,frame):

        sigmaY = np.sqrt(self.varY.get())
        window = self.window_size.get()
        signal_type = self.signal_type.get()

        if signal_type=='manual':
            muY = self.muY.get()
        elif signal_type=='sine wave':
            muY = 2*np.sin(frame/10)
        elif signal_type=='random':
            muY = 4*np.random.rand()-2

        # update moving average
        self.muhat[:-1] = self.muhat[1:]
        self.muhat[-1] = np.mean(self.points[-window:])
        self.line_muhat[0].set_ydata(self.muhat)

        # receive measurement
        self.points[:-1] = self.points[1:]
        self.points[-1] = stats.norm(loc=muY,scale=sigmaY).rvs()
        self.marker_mu[0].set_ydata(muY)
        self.line_points[0].set_ydata(self.points)

        # compute R2
        # self.R2[:-1] = self.R2[1:]
        # yhat = self.muhat[-1]
        # yhatbaseline = self.points[-2]
        # ytrue = self.points[-1]
        # self.R2[-1] = 1 - ((ytrue-yhat)**2)/((ytrue-yhatbaseline)**2)

        # R2smoothed = np.mean(self.R2[-30:])
        # self.text.set_text("t={}\nforecast R2={:6.2f}".format(frame,R2smoothed))

        return self.line_points, self.line_muhat

    def setit(self,event):
         pass

####################################################
if __name__ == "__main__":
	app = TkContainer()
	tk.mainloop()
