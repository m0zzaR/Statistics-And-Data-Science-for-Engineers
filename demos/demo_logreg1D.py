import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from BaseApp import BaseApp
import tkinter as tk

class TkContainer(BaseApp):

    def __init__(self):
        super(TkContainer, self).__init__(
            title="Simple logistic regression",
            geometry="1300x800",
            figsize=(3, 3),
            subplots=None )

    def initialize_parameters(self):
        self.N = 200
        self.xplot = np.linspace(0, 10, self.N)
        self.mu0 = tk.DoubleVar(master=self.root, value=2.6)
        self.sigma = tk.DoubleVar(master=self.root, value=0.3)
        self.mu1 = tk.DoubleVar(master=self.root, value=4.4)
        self.probY1 = tk.DoubleVar(master=self.root, value=0.5)
        self.pY1gx = 0.0  # tk.DoubleVar(master=self.root, value=0.0)
        self.x = tk.DoubleVar(master=self.root, value=6.0)\
        
        self.show_x = tk.BooleanVar(master=self.root, value=False)
        self.show_pxgY = tk.BooleanVar(master=self.root, value=False)
        self.show_pYgx = tk.BooleanVar(master=self.root, value=False)


    def add_widgets(self):
        
        header_width = 40

        # Header ------------------------------------------
        self.get_header(self.root,text='P(Y)',char='.',width=header_width)\
            .pack(side=tk.TOP, fill=tk.X)
        
        self.get_scale(self.root,
                        variable=self.probY1,
                        command=self.update_fig,
                        from_=0.01,
                        to= 0.99,
                        resolution=.01,
                        length=300,
                        text='P(Y=1)') \
            .pack(side=tk.TOP)
        
        # Header ------------------------------------------
        self.get_header(self.root,text='P(X|Y)',char='.',width=header_width)\
            .pack(side=tk.TOP, fill=tk.X)
        
        self.get_scale(self.root,
                        variable=self.mu0,
                        command=self.update_fig,
                        from_=-10,
                        to=10,
                        length=300,
                        resolution=.1,
                        text='mu0') \
            .pack(side=tk.TOP)

        self.get_scale(self.root,
                        variable=self.mu1,
                        command=self.update_fig,
                        from_=-10,
                        to=10,
                        length=300,
                        resolution=.1,
                        text='mu1') \
            .pack(side=tk.TOP)

        self.get_scale(self.root,
                        variable=self.sigma,
                        command=self.update_fig,
                        from_=0.1,
                        to=5,
                        resolution=.1,
                        length=300,
                        text='sigma') \
            .pack(side=tk.TOP)

        # Header ------------------------------------------
        self.get_header(self.root,text='Input',char='.',width=header_width)\
            .pack(side=tk.TOP, fill=tk.X)

        self.get_scale(self.root,
                        variable=self.x,
                        command=self.update_fig,
                        from_=self.xplot.min(),
                        to= self.xplot.max(),
                        resolution=0.1,
                        length=300,
                        text='x') \
            .pack(side=tk.TOP)

        # Header ------------------------------------------
        self.get_header(self.root,text='Show',char='.',width=header_width)\
            .pack(side=tk.TOP, fill=tk.X)

        # function checkbox ........................................
        self.get_checkbox(self.root, text='Show p(x|Y)',variable=self.show_pxgY, command=self.click_show_pxgY)\
            .pack(side=tk.TOP, fill=tk.X)
        
        self.get_checkbox(self.root, text='Show x',variable=self.show_x, command=self.click_show_x)\
            .pack(side=tk.TOP, fill=tk.X)
        
        self.get_checkbox(self.root, text='Show p(Y|x)',variable=self.show_pYgx, command=self.click_show_pYgx)\
            .pack(side=tk.TOP, fill=tk.X)

    def update_distributions(self):

        mu0 = self.mu0.get()
        mu1 = self.mu1.get()
        sigma = self.sigma.get()
        probY1 = self.probY1.get()

        self.norm0 = norm(loc=mu0, scale=sigma)
        self.norm1 = norm(loc=mu1, scale=sigma)

        self.theta0 = np.log(probY1/(1-probY1)) + (mu0**2-mu1**2)/2/sigma**2
        self.theta1 = (mu1-mu0)/sigma**2
        self.sigmoid = 1/(1+np.exp(-(self.theta0+self.theta1*self.xplot)))

        self.pY1gx = 1/(1+np.exp(-(self.theta0+self.theta1*self.x.get())))

        if not np.isclose(self.theta1,0.0):
            self.threshold = -self.theta0/self.theta1
        else:
            self.threshold = np.inf;

    def initialize_fig(self):

        # remove 2D spines
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        self.ax = self.fig.add_subplot(projection='3d')
        self.ax.set_box_aspect(aspect=(4, 1, 1))

        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.yaxis.set_ticklabels([0,1],fontsize=40,va='bottom',ha='left')

        self.ax.xaxis.set_ticklabels([])
        self.ax.yaxis.set_ticks([0,1])
        self.ax.yaxis.set_ticklabels(['Y=0','Y=1'])
        self.ax.zaxis.set_ticklabels([])

        self.update_distributions()

        probY1 = self.probY1.get()
        xmin = self.xplot.min()-1

        self.threshold_line = self.ax.plot3D(self.threshold*np.ones(4),[0,1,1,1],[0,0,0,1],'--',linewidth=3)

        self.x_line = self.ax.plot3D(self.x.get()*np.ones(2),[0,1],[0,0],'k:',linewidth=3)
        self.x_text = self.ax.text(self.x.get(),-0.7,0,'x={:.2f}'.format(self.x.get()),'x',fontsize=24,alpha=0.7)

        self.pxgY1 = self.ax.plot3D(self.xplot, np.ones(self.N),probY1*self.norm1.pdf(self.xplot), linewidth=4)
        self.PY1gx_sigma = self.ax.plot3D(self.xplot,np.ones(self.N), self.sigmoid,linewidth=4)
        self.pxgY0 = self.ax.plot3D(self.xplot, np.zeros(self.N),(1-probY1)*self.norm0.pdf(self.xplot), linewidth=4)

        self.PY0gx_stem = self.ax.plot3D([self.x.get(),self.x.get()],[0,0],[0,1-self.pY1gx],'k-',linewidth=4)
        self.PY0gx_mark = self.ax.plot3D([self.x.get()],[0],[1-self.pY1gx],'ro',markersize=12)
        self.PY1gx_stem = self.ax.plot3D([self.x.get(),self.x.get()],[1,1],[0,self.pY1gx],'k-',linewidth=4)
        self.PY1gx_mark = self.ax.plot3D([self.x.get()],[1],[self.pY1gx],'ro',markersize=12)

        self.PY1_stem = self.ax.plot3D([xmin,xmin],[1,1],[0,probY1],'b-',linewidth=4)
        self.PY1_mark = self.ax.plot3D([xmin],[1],[probY1],'mo',markersize=12)
        self.pY0_stem = self.ax.plot3D([xmin,xmin],[0,0],[0,1-probY1],'b-',linewidth=4)
        self.PY0_mark = self.ax.plot3D([xmin],[0],[1-probY1],'mo',markersize=12)
        
        self.pY1_text = self.ax.text(-1,1,probY1,'P(Y=1)\n{:.2f}'.format(probY1),'x',fontsize=24)
        self.pY0_text = self.ax.text(-1,0,1-probY1,'P(Y=0)\n{:.2f}'.format(1-probY1),'x',fontsize=24)

        self.pY1gx_text = self.ax.text(self.x.get(),1,self.pY1gx,'P(Y=1|x)\n{:.2f}'.format(self.pY1gx),'x',fontsize=24)
        self.pY0gx_text = self.ax.text(self.x.get(),0,1-self.pY1gx,'P(Y=0|x)\n{:.2f}'.format(1-self.pY1gx),'x',fontsize=24)

        self.click_show_pxgY()
        self.click_show_x()
        self.click_show_pYgx()

    def update_fig(self,notused=None):
        
        self.update_distributions()

        probY1 = self.probY1.get()
        xmin = self.xplot.min()-1


        self.x_line[0].set_data_3d(self.x.get()*np.ones(2),[0,1],[0,0])
        self.x_text.set_x(self.x.get())
        self.x_text.set_text('x={:.2f}'.format(self.x.get()))

        self.PY1_stem[0].set_data_3d([xmin,xmin],[1,1],[0,probY1])
        self.PY1_mark[0].set_data_3d([xmin],[1],[probY1])
        self.pY0_stem[0].set_data_3d([xmin,xmin],[0,0],[0,1-probY1])
        self.PY0_mark[0].set_data_3d([xmin],[0],[1-probY1])

        self.pxgY0[0].set_data_3d(self.xplot,np.zeros(self.N),(1-probY1)*self.norm0.pdf(self.xplot))
        self.pxgY1[0].set_data_3d(self.xplot,np.ones(self.N),probY1*self.norm1.pdf(self.xplot))
        self.PY1gx_sigma[0].set_data_3d(self.xplot,np.ones(self.N), self.sigmoid)
        self.threshold_line[0].set_data_3d(self.threshold*np.ones(4),[0,1,1,1],[0,0,0,1])

        self.PY0gx_stem[0].set_data_3d([self.x.get(),self.x.get()],[0,0],[0,1-self.pY1gx])
        self.PY1gx_stem[0].set_data_3d([self.x.get(),self.x.get()],[1,1],[0,self.pY1gx])
        self.PY0gx_mark[0].set_data_3d([self.x.get()],[0],[1-self.pY1gx])
        self.PY1gx_mark[0].set_data_3d([self.x.get()],[1],[self.pY1gx])

        self.pY1_text.set_z(probY1)
        self.pY1_text.set_text('P(Y=1)\n{:.2f}'.format(probY1))
        self.pY0_text.set_z(1-probY1)
        self.pY0_text.set_text('P(Y=0)\n{:.2f}'.format(1-probY1))

        self.pY1gx_text.set_x(self.x.get())
        self.pY1gx_text.set_z(self.pY1gx)
        self.pY1gx_text.set_text('P(Y=1|x)\n{:.2f}'.format(self.pY1gx))

        self.pY0gx_text.set_x(self.x.get())
        self.pY0gx_text.set_z(1-self.pY1gx)
        self.pY0gx_text.set_text('P(Y=0|x)\n{:.2f}'.format(1-self.pY1gx))

        plt.draw()

    def click_show_pxgY(self):
        show = self.show_pxgY.get()
        self.pxgY1[0].set_visible(show)
        self.pxgY0[0].set_visible(show)
        plt.draw()

    def click_show_x(self):
        show = self.show_x.get()
        self.x_line[0].set_visible(show)
        self.x_text.set_visible(show)
        plt.draw()

    def click_show_pYgx(self):
        show =self.show_pYgx.get()
        self.PY1gx_stem[0].set_visible(show)
        self.PY1gx_mark[0].set_visible(show)
        self.PY0gx_stem[0].set_visible(show)
        self.PY0gx_mark[0].set_visible(show)
        self.threshold_line[0].set_visible(show)
        self.PY1gx_sigma[0].set_visible(show)
        self.pY1gx_text.set_visible(show)
        self.pY0gx_text.set_visible(show)
        plt.draw()

####################################################
if __name__ == "__main__":
    app = TkContainer()
    tk.mainloop()

