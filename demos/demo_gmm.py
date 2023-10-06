import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from BaseApp import BaseApp
import scipy.stats as stats
from sklearn.datasets import make_blobs
from matplotlib.patches import Ellipse

colors = ['red','green','blue']
ellipsevals = [1,2,3]

np.random.seed(seed=234)    # GOOD
# np.random.seed(seed=5646)   # 2 clusters
# np.random.seed(seed=4343)   # SINGULAR
# np.random.seed(seed=12)   # SLOW CONVERGENCE

class TkContainer(BaseApp):

    def __init__(self):
        super(TkContainer, self).__init__(title="Gaussian Mixture demo",
            geometry="900x600",figsize=(12, 4))

    def initialize_parameters(self):
        self.stdev = tk.DoubleVar(master=self.root,value=1.3)
        self.stdevstr = tk.StringVar(master=self.root, value='1.3')

    def initialize_data(self):
        pass

    def add_widgets(self):

        self.get_button(self.root,text="EM step",command=self.EMstep_pressed)\
            .pack(side=tk.TOP, fill=tk.X)

        self.get_button(self.root,text="M step",command=self.Mstep_pressed)\
            .pack(side=tk.TOP, fill=tk.X)

        self.get_button(self.root,text="E step",command=self.Estep_pressed)\
            .pack(side=tk.TOP, fill=tk.X)

        self.get_button(self.root,text="Reset",command=self.Reset_pressed)\
            .pack(side=tk.TOP, fill=tk.X)

        self.get_entry_label(self.root,
                        text="stdev",
                        textvariable=self.stdevstr,
                        validatecommand=self.set_stdev)\
            .pack(side=tk.TOP, fill=tk.X)


    def initialize_data(self):
        N = 100
        K = 3

        self.N = N
        self.K = K
        self.Y, _ = make_blobs(n_samples=self.N,
                          centers=K,
                          n_features=2,
                          cluster_std=self.stdev.get())

        self.Pi = np.repeat(1/K,K)
        self.Mu = np.empty((K,2))        
        xmin, xmax = self.Y[:,0].min(), self.Y[:,0].max()
        ymin, ymax = self.Y[:,1].min(), self.Y[:,1].max()
        self.Mu[:,0] =  stats.uniform(loc=xmin,scale=xmax-xmin).rvs(K)
        self.Mu[:,1] =  stats.uniform(loc=ymin,scale=ymax-ymin).rvs(K)
        self.Sigma2 = np.empty((K,2,2))
        self.Sigma2[:,0,0] = 1
        self.Sigma2[:,0,1] = 0
        self.Sigma2[:,1,0] = 0
        self.Sigma2[:,1,1] = 1


        self.loglike = -np.inf
        self.counter = 0

    def initialize_fig(self):

        self.ax.clear()
        self.ellipses = list()
        self.centers = list()
        self.dataplt = None

        K = self.K
        Y = self.Y

        plt.ion()

        xmin, xmax = Y[:,0].min(), Y[:,0].max()
        ymin, ymax = Y[:,1].min(), Y[:,1].max()
        dx = xmax-xmin
        dy = ymax-ymin
        d = max(dx, dy)
        a = 1.1
        xmin = xmin - a*(d-dx)/2
        xmax = xmax + a*(d-dx)/2
        ymin = ymin - a*(d-dy)/2
        ymax = ymax + a*(d-dy)/2
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)

        for k in range(K):
            ell = list()
            color = colors[k]
            for nstd in ellipsevals:
                ell.append(self.plot_cov_ellipse(self.Sigma2[k, :, :], self.Mu[k, :], nstd=nstd,color=color))
            self.ellipses.append(ell)
            c,  = plt.plot(self.Mu[k, 0], self.Mu[k, 1], '*', markersize=40, color=color)
            self.centers.append( c )
        self.dataplt = plt.scatter(self.Y[:, 0], self.Y[:, 1], color='k', s=150)
        plt.xticks([])
        plt.yticks([])
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)

        self.txt = plt.text(0, 0, self.format_string(),
                       horizontalalignment='left',
                       verticalalignment='center',
                       transform=self.ax.transAxes,
                       fontsize=30)

    def format_string(self):
        return "steps: {}\nloglike: {:.2f}".format(self.counter, self.loglike)

    def update_fig(self):

        K = self.K
        Mu = self.Mu 

        for k in range(K):

            ell = self.ellipses[k]
            for i, nstd in enumerate(ellipsevals):
                width, height, theta = self.get_ellipse_data(self.Sigma2[k, :, :], nstd=nstd)
                ell[i].set_center(Mu[k, :])
                ell[i].set_width(width)
                ell[i].set_height(height)
                ell[i].set_angle(theta)
                ell[i].set_color(colors[k])

            self.centers[k].set_xdata(Mu[k, 0])
            self.centers[k].set_ydata(Mu[k, 1])

        self.dataplt.set_color(self.Gamma )

        self.txt.set_text(self.format_string())
        plt.draw()

    def eval_gauss(self):

        Y = self.Y 
        Mu = self.Mu 
        Sigma2 = self.Sigma2
        N = self.N
        K = self.K
        D = Mu.shape[1]

        values = np.empty((N, K))

        for k in range(K):
            sigma2 = Sigma2[k, :, :]
            mu = Mu[k, :]
            den = (2 * np.pi) ** (D / 2) * np.sqrt(abs(np.linalg.det(sigma2)))
            sigma2inv = np.linalg.inv(sigma2)
            for i in range(N):
                y = Y[i, :]
                num = np.exp(-0.5 * (y - mu).T @ sigma2inv @ (y - mu))
                values[i, k] = num / den

        return values

    def E(self):
        self.gauss = self.eval_gauss()
        pigauss = self.Pi * self.gauss
        den = pigauss.sum(axis=1).reshape((self.N, 1))
        self.Gamma = pigauss / den

    def M(self):
        N = self.N
        K = self.K
        Gamma = self.Gamma
        Y = self.Y

        Pi = np.empty(K)
        Mu = np.empty((K, 2))
        Sigma2 = np.empty((K, 2, 2))

        for k in range(K):

            gamma = Gamma[:, k]
            Nk = sum(gamma)

            Pi[k] = Nk / N
            Mu[k, :] = sum(Y * gamma.reshape((N, 1))) / Nk

            sigma2 = np.empty((2, 2))
            for i in range(N):
                v = Y[i, :] - Mu[k, :]
                sigma2 += gamma[i] * v.reshape((2, 1)) * v
            sigma2 /= Nk

            Sigma2[k, :, :] = sigma2

        self.Pi = Pi
        self.Mu = Mu 
        self.Sigma2 = Sigma2

    def EMstep_pressed(self):
        if self.counter%2==0:
            self.Estep_pressed()
            self.Mstep_pressed()

    def Estep_pressed(self):
        if self.counter%2==0:
            self.E()
            self.update_fig()
            self.counter += 1

    def Mstep_pressed(self):
        if self.counter%2==1:
            self.M()
            self.loglike = np.sum(np.log(np.sum(self.Pi * self.gauss, axis=1)))
            self.update_fig()
            self.counter += 1

    def Reset_pressed(self):
        self.initialize_data()
        self.initialize_fig()
    
    def set_stdev(self):
        try:
            self.stddev.set(float(self.stddevstr.get()))
            return True
        except ValueError:
            return
    # ------------------------------------------------------

    def eigsorted(self,cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    def get_ellipse_data(self, cov, nstd=2):
        vals, vecs = self.eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        width, height = 2 * nstd * np.sqrt(vals)
        return width, height, theta

    def plot_cov_ellipse(self, cov, pos, nstd=2, color='black'):
        width, height, theta = self.get_ellipse_data(cov, nstd)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta,
                        fill=False, linewidth=2, color=color)
        self.ax.add_artist(ellip)
        return ellip


####################################################
if __name__ == "__main__":
    app = TkContainer()
    tk.mainloop()
