import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from BaseApp import BaseApp
import tkinter as tk

xmin, xmax = -6,6
ymin, ymax = -4,4
muX = 2

def true_funct(x):
    if True:  
        theta0 = -2
        theta1 = 1                       # LINEAR GENERATING FUNCTION\
        return theta0 + theta1*x
    else:
        return 1.5*np.sin(x)          # SINUSOIDAL GENERATING FUNCTION

def sample_XY(N,stddevX,stddevY):
    x = stats.norm(loc=muX,scale=stddevX).rvs(N)
    y = true_funct(x) + stats.norm(loc=0,scale=stddevY).rvs(N)
    return x, y

def sample_Y(N,x,stddevY):
    y = true_funct(x) + stats.norm(loc=0,scale=stddevY).rvs(N)
    return x, y

def get_contour_data(stddevX,stddevY):
	nx, ny = (100, 100)
	x = np.linspace(xmin,xmax, nx)
	y = np.linspace(ymin,ymax, ny)
	Xg, Yg = np.meshgrid(x, y)
	Xg = Xg.T
	Yg = Yg.T
	Z = np.empty((nx, ny))
	for i, xx in enumerate(x):
		Z[i, :] = stats.norm.pdf(y, loc=true_funct(xx), scale=stddevY)
	for i, yy in enumerate(y):
		Z[:, i] *= stats.norm.pdf(x, loc=muX, scale=stddevX)
	Z /= sum(sum(Z))
	return Xg, Yg, Z

class TkContainer(BaseApp):

    train_plt = None

    def __init__(self):
        super(TkContainer, self).__init__(
            title="Linear Regression demo",
            geometry="1400x700",
            figsize=(12, 4))

    def initialize_parameters(self):

        self.stddevX = tk.DoubleVar(master=self.root, value=2)
        self.stddevXstr = tk.StringVar(master=self.root, value='2.0')

        self.stddevY = tk.DoubleVar(master=self.root, value=1.0)
        self.stddevYstr = tk.StringVar(master=self.root, value='1.0')

        self.theta0 = tk.DoubleVar(master=self.root, value=0.0)
        self.theta1 = tk.DoubleVar(master=self.root, value=0.0)

        self.Ntrain = tk.IntVar(master=self.root, value=20)
        self.Ntrainstr = tk.StringVar(master=self.root, value='20')

        self.param = tk.IntVar(master=self.root, value=2)
        self.paramstr = tk.StringVar(master=self.root, value='2')

        self.show_model = tk.BooleanVar(master=self.root, value=False)
        self.show_ls = tk.BooleanVar(master=self.root, value=False)
        self.show_pXY = tk.BooleanVar(master=self.root, value=False)
        self.show_function = tk.BooleanVar(master=self.root, value=False)

    def initialize_data(self):
        self.Dtrain= sample_XY(self.Ntrain.get(),self.stddevX.get(),self.stddevY.get())

    def add_widgets(self):

        header_width = 40

        # Header ------------------------------------------
        self.get_header(self.root,text='Data generator',char='.',width=header_width)\
            .pack(side=tk.TOP, fill=tk.X)
        
        # contour checkbox ........................................
        self.get_checkbox(self.root, text='Show XY distribution',variable=self.show_pXY, command=self.click_pXY_checkbox)\
            .pack(side=tk.TOP, fill=tk.X)

        # function checkbox ........................................
        self.get_checkbox(self.root, text='Show true function',variable=self.show_function, command=self.click_function_checkbox)\
            .pack(side=tk.TOP, fill=tk.X)
        
        # stddev X input box ......................................
        self.get_scale(self.root,
                       variable=self.stddevX,
                       command=self.set_stddevX,
                       from_= 0,
                       to=3,
                       resolution = 0.1,
                       length=200,
                       text='stddev X')\
                .pack(side=tk.TOP, fill=tk.X)

        # stddev Y input box ......................................
        self.get_scale(self.root,
                       variable=self.stddevY,
                       command=self.set_stddevY,
                       from_= 0,
                       to=3,
                       resolution = 0.1,
                       length=200,
                       text='stddev Y')\
                .pack(side=tk.TOP, fill=tk.X)
        
        # Header ------------------------------------------
        self.get_header(self.root,text='Training data',char='.',width=header_width)\
            .pack(side=tk.TOP, fill=tk.X)

        # N train input box ......................................
        self.get_entry_label(self.root,
                        text="N train",
                        textvariable=self.Ntrainstr,
                        validatecommand=self.set_Ntrain)\
            .pack(side=tk.TOP, fill=tk.X)
        
        # resample buttons ......................................
        self.get_button(self.root,text="Resample x from X",command=self.press_resample_X)\
            .pack(side=tk.TOP, fill=tk.X)
        
        self.get_button(self.root,text="Resample y from Y|X=x",command=self.press_resample_Y)\
            .pack(side=tk.TOP, fill=tk.X)

        # Header ------------------------------------------
        self.get_header(self.root,text='Candidate model',char='.',width=header_width)\
            .pack(side=tk.TOP, fill=tk.X)

        # model checkbox ........................................
        self.get_checkbox(self.root, text='Show candidate model',variable=self.show_model, command=self.click_model_checkbox)\
            .pack(side=tk.TOP, fill=tk.X)
        
        self.get_scale(self.root,
                       variable=self.theta0,
                       command=self.change_theta0,
                       from_=-6,
                       to=3,
                       resolution=0.1,
                       length=200,
                       text='theta0').pack(side=tk.TOP, fill=tk.X)
        
        self.get_scale(self.root,
                       variable=self.theta1,
                       command=self.change_theta1,
                       from_=-3,
                       to=3,
                       resolution=0.1,
                       length=200,
                       text='theta1').pack(side=tk.TOP, fill=tk.X)
        
        # Header ------------------------------------------
        self.get_header(self.root,text='Leasts squares model',char='.',width=header_width)\
            .pack(side=tk.TOP, fill=tk.X)
        
        # LS checkbox ........................................
        self.get_checkbox(self.root, text='Show least squares model',variable=self.show_ls, command=self.click_ls_checkbox)\
            .pack(side=tk.TOP, fill=tk.X)
        
    def initialize_fig(self):

        ax = self.ax

        # pXY contour ....................................
        Xg, Yg, Z, = get_contour_data(self.stddevX.get(), self.stddevY.get())
        self.plt_pXY = ax.contour(Xg, Yg, Z)
        self.plt_pXY.set_alpha(0.0)

        # True function ....................................
        self.xx = np.linspace(xmin,xmax,200)
        self.plt_truefunc, = ax.plot(self.xx, true_funct(self.xx), c='b', linewidth=6)
        self.plt_truefunc.set_visible(self.show_function.get())

        # Data ....................................
        self.plt_train, = ax.plot(self.Dtrain[0],self.Dtrain[1],'o',c='k',markersize=12)

        # candidate model ....................................
        self.plt_model, = ax.plot(self.xx, self.eval_model(self.xx), c='m', linewidth=6)
        self.plt_model.set_visible(self.show_model.get())

        # leaast squares model ....................................
        yhat, yhatstddev = self.eval_least_squares(self.Dtrain[0],self.Dtrain[1],self.xx)
        self.plt_ls, = ax.plot(self.xx, yhat, c='cyan', linewidth=6)
        self.plt_ls.set_visible(self.show_ls.get())

        # prediction bounds
        self.plt_ls_bounds = ax.fill_between(self.xx, yhat-2*yhatstddev,yhat+2*yhatstddev,alpha=0.2)
        self.plt_ls_bounds.set_visible(self.show_ls.get())

        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xticks([0])
        ax.set_yticks([0])
        ax.grid(linestyle='-',linewidth=2)

        self.txt = ax.text(1, .1, self.format_string(),
            horizontalalignment='right',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=30)
        self.txt.set_visible(self.show_model.get())

        ax.text(1, 0.5, 'x',
            horizontalalignment='right',
            verticalalignment='bottom',
            transform=ax.transAxes,
            fontsize=40)

        ax.text(0.48, 1, 'y',
            horizontalalignment='right',
            verticalalignment='top',
            transform=ax.transAxes,
            fontsize=40)

    def update_figure(self,replotpXY=False,replotLS=False):

        if (self.plt_pXY is not None) and replotpXY:
            for coll in self.plt_pXY.collections:
                coll.remove()
            Xg, Yg, Z, = get_contour_data(self.stddevX.get(), self.stddevY.get())
            self.plt_pXY = self.ax.contour(Xg, Yg, Z)
            if self.show_pXY.get():
                self.plt_pXY.set_alpha(1.0)
            else:
                self.plt_pXY.set_alpha(0.0)

        if self.plt_model is not None:
            self.plt_model.set_ydata(self.eval_model(self.xx))

        if (self.plt_ls is not None) and replotLS:
            yhat, yhatstddev = self.eval_least_squares(self.Dtrain[0],self.Dtrain[1],self.xx)
            self.plt_ls.set_ydata(yhat)

            # reset the fill between
            path = self.plt_ls_bounds.get_paths()[0]
            xnew = self.xx
            y0new = yhat - 2*yhatstddev
            y1new = yhat + 2*yhatstddev
            v_x = np.hstack([xnew[0],xnew,xnew[-1],xnew[::-1],xnew[0]])
            v_y = np.hstack([y1new[0],y0new,y0new[-1],y1new[::-1],y1new[0]])
            path.vertices = np.vstack([v_x,v_y]).T
            path.codes = np.array([1]+(2*len(xnew)+1)*[2]+[79]).astype('uint8')

        if self.plt_train is not None:
            self.plt_train.set_xdata(self.Dtrain[0])
            self.plt_train.set_ydata(self.Dtrain[1])

        if self.txt is not None:
            self.txt.set_text(self.format_string())

        if self.train_plt is not None:
            self.train_plt.remove()

        plt.draw()

    def eval_model(self,X):
        if len(X)==0:
            return np.nan
        return self.theta0.get() + self.theta1.get()*X
    
    def eval_least_squares(self,X,Y,xx):
        if len(X)==0:
            return np.nan
        muhatX = X.mean()
        muhatY = Y.mean()
        N = Y.shape[0]
        sigmahatXY = np.sum((X-muhatX)*(Y-muhatY))/(N-1)
        sigmahatX2 = np.sum((X-muhatX)**2)/(N-1)
        theta1 =  sigmahatXY/sigmahatX2
        theta0 = muhatY - theta1*muhatX
        yhat = theta0 + theta1*xx
        sigmahatY2 = np.sum((Y-theta0-theta1*X)**2)/(N-1)
        stddevYhat = np.sqrt(sigmahatY2*(1/N + ((self.xx-muhatX)**2)/sigmahatX2/(N-1)))
        return yhat, stddevYhat
    
    def set_stddevX(self,event):
        try:
            self.press_resample_X(replotpXY=True)
            return True
        except ValueError:
            return

    def set_stddevY(self,event):
        try:
            self.press_resample_Y(replotpXY=True)
            return True
        except ValueError:
            return

    def set_Ntrain(self):
        try:
            self.Ntrain.set(int(self.Ntrainstr.get()))
            self.press_resample_X()
            return True
        except ValueError:
            return False
        
    def change_theta0(self,event):
         self.update_figure()

    def change_theta1(self,event):
         self.update_figure()
         
    def press_resample_X(self,replotpXY=True):
        self.Dtrain = sample_XY(self.Ntrain.get(),self.stddevX.get(), self.stddevY.get())
        self.update_figure(replotpXY,replotLS=True)

    def press_resample_Y(self,replotpXY=True):
        self.Dtrain = sample_Y(self.Ntrain.get(),self.Dtrain[0],self.stddevY.get())
        self.update_figure(replotpXY,replotLS=True)

    def click_model_checkbox(self):
        a = self.show_model.get()
        self.plt_model.set_visible(a)
        self.txt.set_visible(a)
        plt.draw()

    def click_ls_checkbox(self):
        self.plt_ls.set_visible(self.show_ls.get())
        self.plt_ls_bounds.set_visible(self.show_ls.get())
        plt.draw()

    def click_pXY_checkbox(self):
        if self.show_pXY.get():
            self.plt_pXY.set_alpha(1.0)
        else:
            self.plt_pXY.set_alpha(0.0)
        plt.draw()

    def click_function_checkbox(self):
        self.plt_truefunc.set_visible(self.show_function.get())
        plt.draw()

    def format_string(self):

        X, Y = self.Dtrain

        Yhat = self.eval_model(X)
        MSE = np.mean((Y-Yhat)**2)
        MAE = np.mean(np.abs(Y-Yhat))
        MAPE = np.mean(np.abs(Y-Yhat)/np.abs(Y))
        R2 = 1 - MSE / np.mean((Y-np.mean(Y))**2)

        return "MSE = {:.1f}\nMAE = {:.1f}\nMAPE = {:.1f}\nR2 = {:.1f}".format(MSE,MAE, MAPE, R2)

####################################################
if __name__ == "__main__":
	app = TkContainer()
	tk.mainloop()
