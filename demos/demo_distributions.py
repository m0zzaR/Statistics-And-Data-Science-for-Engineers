from BaseApp import BaseApp
import tkinter as tk
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def get_distribution(dist,param1,param2):
    if dist=='Gaussian':
        mean = param1
        stddev = param2
        return stats.norm(loc=mean, scale=stddev)
    elif dist=='Uniform':
        return stats.uniform(loc=param1, scale=param2-param1)
    elif dist=='Exponential':
        mean = param1
        return stats.expon(loc=0, scale=mean)
    elif dist=='Binomial':
        N = int(param1)
        p = param2
        return stats.binom(N,p)
    elif dist=='Bernoulli':
        p, loc = math.modf(param1)
        return stats.bernoulli(p,loc=loc)
    else:
        return None

class TkContainer(BaseApp):

    distribution_names = ['Bernoulli','Binomial','Uniform','Gaussian','Exponential']
    distribution_info = dict.fromkeys(distribution_names)

    distribution_info['Bernoulli'] = {
        'type': 'discrete',
        'numparam' : 1,
        'param1' : 'p',
        'param1_range': [0,1],
        'param1_nom' : 0.2,
        'param2' : 'N/A',
        'param2_range': None,
        'param2_nom' : np.NaN,
        'xx': [0,1],
    }

    distribution_info['Binomial'] = {
        'type': 'discrete',
        'numparam' : 2,
        'param1' : 'N',
        'param1_range': [1,10],
        'param1_nom' : 5,
        'param2' : 'p',
        'param2_range': [0,1],
        'param2_nom' : 0.2,
        'xx': np.arange(0,11),
        'currN' : 5
    }

    distribution_info['Uniform'] = {
        'type': 'continuous',
        'numparam' : 2,
        'param1' : 'lower',
        'param1_range': [-6,-0.01],
        'param1_nom' : -1.0,
        'param2' : 'upper',
        'param2_range': [0.01,6],
        'param2_nom' : 1.0,
        'xx': np.linspace(-6,6,200),
        'minx': -6.0,
        'maxx': 6.0
    }

    distribution_info['Gaussian'] = {
        'type': 'continuous',
        'numparam' : 2,
        'param1' : 'mean',
        'param1_range': [-5,5],
        'param1_nom' : 0,
        'param2' : 'stddev',
        'param2_range': [0.01,3],
        'param2_nom' : 1,
        'xx': np.linspace(-6,6,200),
        'minx': -6,
        'maxx': 6
    }

    distribution_info['Exponential'] = {
        'type': 'continuous',
        'numparam' : 1,
        'param1' : 'lambda',
        'param1_range': [0.01,4],
        'param1_nom' : 1.0,
        'param2' : 'N/A',
        'param2_range': None,
        'param2_nom' : np.NaN,
        'xx': np.linspace(0.01,4,200),
        'minx': 0,
        'maxx': 4
    }

    def __init__(self):
        super(TkContainer, self).__init__(
            title="Probability density functions",
            geometry="1600x600",
            figsize=(8, 3))

    def initialize_parameters(self):

        dist_init = 'Exponential'
        self.selected_dist = tk.StringVar(master=self.root, value=dist_init)

        dist_info = self.distribution_info[dist_init]

        p1 = dist_info['param1_nom']
        p2 = dist_info['param2_nom']

        p1_nom_norm, p2_nom_norm = self.normalize_params(dist_info,p1,p2)
        self.param1 = tk.DoubleVar(master=self.root, value=p1_nom_norm)
        self.param2 = tk.DoubleVar(master=self.root, value=p2_nom_norm)

        str1, str2 = self.get_string(p1,p2)
        self.param1str = tk.StringVar(master=self.root, value=str1)
        self.param2str = tk.StringVar(master=self.root, value=str2)

    def initialize_fig(self):

        dist = self.selected_dist.get()
        dist_info = self.distribution_info[dist]
        ax = self.ax
        ax.clear()

        p1_norm = self.param1.get()
        if dist_info['numparam']==2:
            p2_norm = self.param2.get()
        else:
            p2_norm = None
        p1, p2 = self.unnormalize_params(dist,dist_info,p1_norm,p2_norm)

        X = get_distribution(dist,p1, p2)
        xx = dist_info['xx']

        if dist_info['type']=='continuous':
            yy = X.pdf(xx)
            maxy = max(yy)
            self.line_pdf = ax.plot(xx,yy,linewidth=4)[0]
            ax.set_ylim(0,maxy*1.1)
            ax.set_xlim(dist_info['minx'],dist_info['maxx'])

        elif dist=='Bernoulli':
            maxy = 1
            self.markerline, self.stemlines, _ = ax.stem(xx,X.pmf(xx))
            self.markerline.set_markerfacecolor('r')
            self.markerline.set_markersize(18)
            self.markerline.set_markeredgewidth(2)
            self.stemlines.set_lw(4)
            ax.set_ylim(0,1)
            ax.set_xticks([0,0.5,1])
            ax.set_xlim(-0.5,1.5)

        elif dist=='Binomial':
            maxy = 1
            self.markerline, self.stemlines, _ = ax.stem(xx,X.pmf(xx))
            self.markerline.set_markerfacecolor('r')
            self.markerline.set_markersize(18)
            self.markerline.set_markeredgewidth(2)
            self.stemlines.set_lw(4)
            ax.set_ylim(0,1)
            ax.set_xticks(xx)
            ax.set_xlim(xx[0]-0.5,xx[-1]+0.5)
        
        else:
            print("This should not happen")

        self.line_mean = ax.vlines(X.mean(),0,maxy*1.1,color='red',linewidth=2,linestyles='dashed')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_xticklabels(ax.get_xticks(),fontsize=25)
        ax.set_yticks([])

        plt.draw()

    def add_widgets(self):

		# # Header ------------------------------------------
        # header_width = 40
        # self.get_header(self.root,text='Distribution',char='.',width=header_width)\
		# 	.pack(side=tk.TOP, fill=tk.X)

        # select model combo box ........................................
        self.get_combobox(self.root,
                          text='Family',
                          textvariable = self.selected_dist,
                          values = self.distribution_names,
                          command = self.select_dist)\
			.pack(side=tk.TOP, fill=tk.X)
        

        # param1 input box ......................................
        self.scale1 = self.get_scale(self.root,
                       variable=self.param1,
                       command=self.set_params,
                       from_=0,
                       to=1,
                       length=200,
                       textvariable=self.param1str).pack(side=tk.TOP, fill=tk.X)
        

        # param2 input box ......................................        
        self.scale2 = self.get_scale(self.root,
                       variable=self.param2,
                       command=self.set_params,
                       from_=0,
                       to=1,
                       length=200,
                       textvariable=self.param2str).pack(side=tk.TOP, fill=tk.X)
    
    def update_figure(self,p1,p2):

        dist = self.selected_dist.get()
        dist_info = self.distribution_info[dist]

        X = get_distribution(dist,p1,p2)
        xx = dist_info['xx']

        if dist_info['type']=='continuous':
            yy = X.pdf(xx)
            self.line_pdf.set(xdata=xx,ydata=yy)

        elif dist=='Bernoulli':
            self.markerline.set_ydata(X.pmf(xx))
            seg = self.stemlines.get_segments()
            seg[0][1,1] = X.pmf(0)
            seg[1][1,1] = X.pmf(1)
            self.stemlines.set_segments(seg)

        elif dist=='Binomial':
            self.markerline.set_ydata(X.pmf(xx))
            seg = self.stemlines.get_segments()
            for i in range(len(seg)):
                seg[i][1,1] = X.pmf(i)
            self.stemlines.set_segments(seg)
        
        else:
            print("This should not happen")

        seg = self.line_mean.get_segments()
        seg[0][:,0] = X.mean()
        self.line_mean.set_segments(seg)

        plt.draw()

    def select_dist(self,event):
        dist = self.selected_dist.get()
        dist_info = self.distribution_info[dist]
        p1 = dist_info['param1_nom']
        p2 = dist_info['param2_nom']
        p1_nom_norm, p2_nom_norm = self.normalize_params(dist_info,p1,p2)
        self.param1.set(p1_nom_norm)
        self.param2.set(p2_nom_norm)
        str1, str2 = self.get_string(p1,p2)
        self.param1str.set(str1)
        self.param2str.set(str2)
        self.initialize_fig()

    def set_params(self,event):
        dist = self.selected_dist.get()
        dist_info = self.distribution_info[dist]

        p1_norm = self.param1.get()
        if dist_info['numparam']==2:
            p2_norm = self.param2.get()
        else:
            p2_norm = None
        p1, p2 = self.unnormalize_params(dist,dist_info,p1_norm,p2_norm)
        str1, str2 = self.get_string(p1,p2)

        self.param1str.set(str1)    
        if dist_info['numparam']==2:
            self.param2str.set(str2)

        if dist=='Binomial' and dist_info['currN']!=p1:
            dist_info['currN']==p1
            self.initialize_fig()
        else:
            self.update_figure(p1,p2)

    def get_string(self,p1,p2):
        dist = self.selected_dist.get()
        dist_info = self.distribution_info[dist]
        if isinstance(p1,int):
            str1 = "{} ({})".format(dist_info['param1'],p1)
        else:
            str1 = "{} ({:.2f})".format(dist_info['param1'],p1)

        if dist_info['numparam']>=2:
            str2 = "{} ({:.2f})".format(dist_info['param2'],p2)
        else:
            str2 = '-'
        return str1, str2

    def normalize_params(self,dist_info,p1,p2):
        p1min = dist_info['param1_range'][0]
        p1max = dist_info['param1_range'][1]
        p1_norm = (p1-p1min)/(p1max-p1min) 

        if dist_info['numparam']>=2:
            p2min = dist_info['param2_range'][0]
            p2max = dist_info['param2_range'][1]
            p2_norm = (p2-p2min)/(p2max-p2min) 
        else:
            p2_norm = None

        return p1_norm, p2_norm

    def unnormalize_params(self,dist,dist_info,p1_norm,p2_norm):
        p1min = dist_info['param1_range'][0]
        p1max = dist_info['param1_range'][1]
        p1 = p1_norm*(p1max-p1min) + p1min

        if dist=='Binomial':
            p1 = round(p1)

        if dist_info['numparam']>=2:
            p2min = dist_info['param2_range'][0]
            p2max = dist_info['param2_range'][1]
            p2= p2_norm*(p2max-p2min) + p2min
        else:
            p2 = None
        return p1, p2

####################################################
if __name__ == "__main__":
    app = TkContainer()
    tk.mainloop()

