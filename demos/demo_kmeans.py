import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from BaseApp import BaseApp
import tkinter as tk

class TkContainer(BaseApp):

    rs = 3242
    num_iterations = 0

    def __init__(self,rs=23):
        super(TkContainer, self).__init__(title="K-means demo",
            geometry="800x800",figsize=(12, 4))

        self.num_iterations = 0

    def initialize_parameters(self):
        pass

    def initialize_data(self):
        self.data = np.vstack(( stats.multivariate_normal(mean=[30,30], 
                                      cov=[[40, 0], [0, 100]],seed=self.rs).rvs(
                                          size=15),
                            stats.multivariate_normal(mean=[70,70], 
                                      cov=[[45, 0], [0, 50]],seed=self.rs+1).rvs(
                                          size=15),
                            stats.multivariate_normal(mean=[30,70], 
                                      cov=[[45, 0], [0, 50]],seed=self.rs+2).rvs(
                                          size=15) ) )

        self.centroids = np.array([(18, 46),(45, 76),(76, 41)])
        self.assignment = np.array([np.argmin(np.sum((self.centroids-d)**2,axis=1)) for d in self.data])
        self.prev_assignment = -np.ones(self.data.shape)

    def add_widgets(self):

        self.get_button(self.root, text="Assign", command=self.assign) \
            .pack(side=tk.TOP, fill=tk.X)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def initialize_fig(self):

        self.datadots = list()
        for i in range(self.data.shape[0]):
            z, = self.ax.plot(self.data[i,0],self.data[i,1],
                    'ro',
                    markersize=8,
                    label='data',
                    zorder=10)
            self.datadots.append(z)
                    
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.grid(which="both")

        self.drag_centroid = None
        self.dots, = self.ax.plot(self.centroids[:,0], self.centroids[:,1], 
            "ks", 
            markersize=18,
            label='centroids',
            zorder=20)

        self.lines = []

        self.txt = plt.text(.05, .9, self.format_string(),
                       horizontalalignment='left',
                       verticalalignment='center',
                       transform=self.ax.transAxes,
                       fontsize=30)

    def assign(self):
        self.assignment = np.array([np.argmin(np.sum((self.centroids-d)**2,axis=1)) for d in self.data])
        if np.array_equal(self.assignment,self.prev_assignment):
            self.conclude()
            return

        self.prev_assignment = self.assignment
        self.num_iterations += 1
        self.update_plot()

    def update_plot(self,fromInit=False):

        dotcolors = [[1,0,0],[0,1,0],[0,0,1]]
        self.dots.set_data(self.centroids[:,0], self.centroids[:,1])
        
        if not fromInit:
            if len(self.lines)==0:
                for p, pt in enumerate(self.data):
                    c = self.centroids[self.assignment[p]]
                    line = self.ax.plot([c[0],pt[0]],[c[1],pt[1]],'k') 
                    self.lines.append(line[0])
            else:
                for p, pt in enumerate(self.data):
                    c = self.centroids[self.assignment[p]]
                    self.lines[p].set_data([c[0],pt[0]],[c[1],pt[1]])
                    
            for i in range(len(self.assignment)):
                self.datadots[i].set_color(dotcolors[self.assignment[i]])


        self.txt.set_text(self.format_string())
        self.fig.canvas.draw()

    def on_click(self, event):
        if event.button == 1 and event.inaxes in [self.ax]:
            e = np.array([event.xdata,event.ydata])
            self.drag_centroid = np.argmin( np.sum((self.centroids-e)**2, axis=1) ) 

    def on_release(self, event):
        if event.button == 1 and event.inaxes in [self.ax] and self.drag_centroid!=None:
            self.drag_centroid = None
            self.update_plot()

    def on_motion(self, event):
        if self.drag_centroid==None:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.centroids[self.drag_centroid] = (event.xdata,event.ydata)
        self.update_plot()

    def compute_cost(self):
        J = 0
        for i, centroid in enumerate(self.centroids):
            J +=  sum(sum((self.data[self.assignment==i,:] - centroid)**2))
        return J

    def format_string(self):
        J = self.compute_cost()
        return "score: {:.0f}\niterations: {:d}".format(J,self.num_iterations)
    
    def conclude(self):   
        # self.txt.set_text(self.format_string())
        J = self.compute_cost()

        t = plt.text(.5, .6, "** Final score **\n {:.0f} in {:d} iterations".format(J,self.num_iterations),
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=self.ax.transAxes,
                       fontsize=60,
                       backgroundcolor = 'red')
        t.set_zorder(100)

        self.button_assign['state'] = "disabled"
        self.fig.canvas.draw()

####################################################
if __name__ == "__main__":
    app = TkContainer()
    tk.mainloop()

