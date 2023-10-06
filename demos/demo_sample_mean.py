from BaseApp import BaseApp
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

class TkContainer(BaseApp):

    xx = np.linspace(-4,4,400)

    def __init__(self):
        super(TkContainer, self).__init__(
            title="Distribution of the sample mean when Y is Gaussian",
            geometry="1600x600",
            figsize=(8, 3))

    def initialize_parameters(self):
        self.N = tk.IntVar(master=self.root, value=1)
        self.sigmaY = tk.DoubleVar(master=self.root, value=1)
        str1, str2 = self.get_string(self.N.get(),self.sigmaY.get())
        self.Nstr = tk.StringVar(master=self.root, value=str1)
        self.sigmaYstr = tk.StringVar(master=self.root, value=str2)

    def initialize_fig(self):

        ax = self.ax
        ax.clear()

        N = self.N.get()
        sigmaY = self.sigmaY.get()
        X = stats.norm(loc=0, scale=sigmaY/np.sqrt(N))
        self.line_pdf = ax.plot(self.xx,X.pdf(self.xx),linewidth=4)[0]
        ax.set_xlim(-3,3)
        ax.set_ylim(0,5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticklabels(ax.get_xticks(),fontsize=25)
        ax.set_yticks([])

        plt.draw()

    def add_widgets(self):

        # sigmaY input box ......................................        
        self.scale2 = self.get_scale(self.root,
                       variable=self.sigmaY,
                       command=self.update_figure,
                       from_=0.1,
                       to=3,
                       length=200,
                       textvariable=self.sigmaYstr).pack(side=tk.TOP, fill=tk.X)

        # param1 input box ......................................
        self.scale1 = self.get_scale(self.root,
                       variable=self.N,
                       command=self.update_figure,
                       from_=1,
                       to=100,
                       length=200,
                       textvariable=self.Nstr).pack(side=tk.TOP, fill=tk.X)
        

    def update_figure(self,event):
        ax = self.ax
        N = self.N.get()
        sigmaY = self.sigmaY.get()

        strN, strSigma = self.get_string(N,sigmaY)
        self.Nstr.set(strN)
        self.sigmaYstr.set(strSigma)
        
        X = stats.norm(loc=0, scale=sigmaY/np.sqrt(N))
        self.line_pdf.set(ydata=X.pdf(self.xx))
        
        plt.draw()

    def get_string(self,N,sigmaY):
        strN = "N ({:})".format(N)
        strSigma = "sigmaY ({:.2f})".format(sigmaY)
        return strN, strSigma

####################################################
if __name__ == "__main__":
    app = TkContainer()
    tk.mainloop()

