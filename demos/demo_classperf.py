import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
rnd = np.random.RandomState(42)
from BaseApp import BaseApp
import tkinter as tk

colors = {'tp':'g','tn':'r','fp':'c','fn':'m','acc':'b'}

class TkContainer(BaseApp):

    def __init__(self):
        super(TkContainer, self).__init__(
            title="Classifier performance",
            geometry="1200x800",
            figsize=(16, 6),
            subplots=(3,1) )

    def initialize_parameters(self):
        self.thresh = tk.DoubleVar(master=self.root, value=0)

        self.N0=40
        self.mu0 = -1
        self.sigma0 = 1
        self.N1=40
        self.mu1 = 1
        self.sigma1 = 1

        self.N0str = tk.StringVar(master=self.root,value=str(self.N0))
        self.mu0str = tk.StringVar(master=self.root,value=str(self.mu0))
        self.sigma0str = tk.StringVar(master=self.root,value=str(self.sigma0))
        self.N1str = tk.StringVar(master=self.root,value=str(self.N1))
        self.mu1str = tk.StringVar(master=self.root,value=str(self.mu1))
        self.sigma1str = tk.StringVar(master=self.root,value=str(self.sigma1))

    def sample_data(self):
        self.N = self.N0+self.N1
        self.X0 = stats.norm(loc=self.mu0,scale=self.sigma0).rvs(self.N0)
        self.X1 = stats.norm(loc=self.mu1,scale=self.sigma1).rvs(self.N1)

        xmin = min(np.concatenate((self.X0,self.X1)))
        xmax = max(np.concatenate((self.X0,self.X1)))
        width = xmax-xmin
        self.xmin = xmin-0.1*width
        self.xmax = xmax+0.1*width

    def initialize_data(self):
        self.sample_data()
        self.update_metrics()

    def add_widgets(self):

        header_width = 30

        # Header ------------------------------------------
        self.get_header(self.root,text='Data',char='.',width=header_width)\
            .pack(side=tk.TOP, fill=tk.X)

        # resample button ......................................
        self.get_button(self.root,text="Resample",command=self.press_resample)\
            .pack(side=tk.TOP, fill=tk.X)
        
        
        # N0 ......................................
        self.get_entry_label(self.root,
                        text="N0",
                        textvariable=self.N0str,
                        validatecommand=self.set_params)\
            .pack(side=tk.TOP, fill=tk.X)
        
        # mu0 ......................................
        self.get_entry_label(self.root,
                        text="mu0",
                        textvariable=self.mu0str,
                        validatecommand=self.set_params)\
            .pack(side=tk.TOP, fill=tk.X)
        
        # sigma0 ......................................
        self.get_entry_label(self.root,
                        text="sigma0",
                        textvariable=self.sigma0str,
                        validatecommand=self.set_params)\
            .pack(side=tk.TOP, fill=tk.X)

        # N1 ......................................
        self.get_entry_label(self.root,
                        text="N1",
                        textvariable=self.N1str,
                        validatecommand=self.set_params)\
            .pack(side=tk.TOP, fill=tk.X)


        # mu1 ......................................
        self.get_entry_label(self.root,
                        text="mu1",
                        textvariable=self.mu1str,
                        validatecommand=self.set_params)\
            .pack(side=tk.TOP, fill=tk.X)
        
        # sigma1 ......................................
        self.get_entry_label(self.root,
                        text="sigma1",
                        textvariable=self.sigma1str,
                        validatecommand=self.set_params)\
            .pack(side=tk.TOP, fill=tk.X)
        
        # Header ------------------------------------------
        self.get_header(self.root,text='Threshold',char='.',width=header_width)\
            .pack(side=tk.TOP, fill=tk.X)
        
                
        # stddev X input box ......................................
        self.slider_thresh = self.get_scale(self.root,
                       variable=self.thresh,
                       command=self.update_fig,
                        from_=self.xmin,
                        to=self.xmax,
                        resolution=0.01,
                        length=300,
                       text='')\
                .pack(side=tk.TOP, fill=tk.X)

    def initialize_fig(self):

        self.ax[0].clear()
        self.ax[1].clear()
        self.ax[2].clear()

        self.ax[0].axhline(0,color='k',linewidth=1)
        self.ax[0].axhline(1,color='k',linewidth=1)
        self.ax[0].axhline(0.5,color='k',linestyle=':',linewidth=2)
        self.ax[0].plot(self.X0,np.zeros(self.N0),'o',color=colors['tn'],markersize=10,markeredgewidth=4,markerfacecolor='none')
        self.ax[0].plot(self.X1,np.ones(self.N1),'+',color=colors['tp'],markersize=14,markeredgewidth=4)
        self.threshline  =self.ax[0].axvline(self.thresh.get(),linestyle='--',color='k',linewidth=2)
        self.ax[0].set_xticks([])
        self.ax[0].set_yticks([])
        self.ax[0].spines['top'].set_visible(False)
        self.ax[0].spines['right'].set_visible(False)
        self.ax[0].spines['bottom'].set_visible(False)
        self.ax[0].spines['left'].set_visible(False)
        self.ax[0].set_xlim(self.xmin,self.xmax)

        thresh = np.linspace(self.xmin,self.xmax)
        TP = np.empty(len(thresh))
        TN = np.empty(len(thresh))
        FP = np.empty(len(thresh))
        FN = np.empty(len(thresh))
        for i, t in enumerate(thresh):
            TP[i] = sum(self.X1>t)
            TN[i] = sum(self.X0<t)
            FP[i] = sum(self.X0>t)
            FN[i] = sum(self.X1<t)
        acc = (TP+TN)/self.N


        self.ax[1].plot(thresh,acc,label='Acc={:.2f}'.format(self.acc),linewidth=4,color=colors['acc'])
        self.dot_acc = self.ax[1].plot(self.thresh.get(),self.acc,'o',markersize=16,color=colors['acc'])
        self.ax[1].set_yticks([0,1])
        self.ax[1].set_yticklabels([0,1],fontsize=26)
        self.ax[1].legend(fontsize=26,loc='upper right')
        self.ax[1].set_xlim(self.xmin,self.xmax)
        self.ax[1].tick_params(labelsize=20)

        self.ax[2].plot(thresh,TP,label=f'TP={self.TP}',linewidth=4,color='r')
        self.ax[2].plot(thresh,TN,label=f'TN={self.TN}',linewidth=4,color='g')
        self.ax[2].plot(thresh,FP,label=f'FP={self.FP}',linewidth=4,color='m')
        self.ax[2].plot(thresh,FN,label=f'FN={self.FN}',linewidth=4,color='c')

        self.dot_tp = self.ax[2].plot(self.thresh.get(),self.TP,'o',markersize=16,color=colors['tp'])
        self.dot_tn = self.ax[2].plot(self.thresh.get(),self.TN,'o',markersize=16,color=colors['tn'])
        self.dot_fp = self.ax[2].plot(self.thresh.get(),self.FP,'o',markersize=16,color=colors['fp'])
        self.dot_fn = self.ax[2].plot(self.thresh.get(),self.FN,'o',markersize=16,color=colors['fn'])

        self.ax[2].set_yticks([0,1])
        self.ax[2].set_yticklabels([0,1],fontsize=26)
        self.ax[2].legend(fontsize=26,loc='upper right')
        self.ax[2].set_xlim(self.xmin,self.xmax)
        self.ax[2].tick_params(labelsize=20)

    def update_metrics(self):

        thresh = self.thresh.get()

        self.TP = sum(self.X1>thresh)
        self.TN = sum(self.X0<thresh)
        self.FP = sum(self.X0>thresh)
        self.FN = sum(self.X1<thresh)
        self.acc = (self.TP+self.TN)/self.N

    # def format_string(self):
    #     return '{}: {:.2f}\n{}: {}\n{}: {}\n{}: {}\n{}: {}'.format('Acc'.rjust(10),self.acc,
    #                                                                    'TP'.rjust(10),self.TP,
    #                                                                    'TN'.rjust(10),self.TN,
    #                                                                    'FP'.rjust(10),self.FP,
    #                                                                    'FN'.rjust(10),self.FN)

    def press_resample(self):
        self.set_params()

    def update_fig(self,notused=None):
        self.update_metrics()
        self.threshline.set_data([self.thresh.get(),self.thresh.get()],[0,1])
        self.dot_acc[0].set_data(self.thresh.get(),self.acc)
        self.ax[1].legend(['Acc={:.2f}'.format(self.acc)],fontsize=26,loc='upper right')
        self.dot_tp[0].set_data(self.thresh.get(),self.TP)
        self.dot_tn[0].set_data(self.thresh.get(),self.TN)
        self.dot_fp[0].set_data(self.thresh.get(),self.FP)
        self.dot_fn[0].set_data(self.thresh.get(),self.FN)
        self.ax[2].legend([f'TP={self.TP}',f'TN={self.TN}',f'FP={self.FP}',f'FN={self.FN}'],fontsize=26,loc='upper right')
        plt.draw()

    def set_params(self):
        try:
            self.N0 = int(self.N0str.get())
            self.mu0 = float(self.mu0str.get())
            self.sigma0 = float(self.sigma0str.get())
            self.N1 = int(self.N1str.get())
            self.mu1 = float(self.mu1str.get())
            self.sigma1 = float(self.sigma1str.get())
            self.initialize_data()
            self.initialize_fig()
            self.update_fig()
            plt.draw()
            return True

        except ValueError:
            return False
    
####################################################
if __name__ == "__main__":
    app = TkContainer()
    tk.mainloop()

