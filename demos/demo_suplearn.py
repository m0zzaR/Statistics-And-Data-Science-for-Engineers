from BaseApp import BaseApp
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.neighbors import KNeighborsRegressor

def true_funct(x):
	return np.sin(x)

def sample_data(N,stddevX,stddevY):
	x = stats.norm(loc=0,scale=stddevX).rvs(N)
	y = true_funct(x) + stats.norm(scale=stddevY).rvs(N)
	return x, y

def get_contour_data(stddevX,stddevY):
	nx, ny = (100, 100)
	x = np.linspace(-6, 6, nx)
	y = np.linspace(-1.8, 1.8, ny)
	Xg, Yg = np.meshgrid(x, y)
	Xg = Xg.T
	Yg = Yg.T
	Z = np.empty((nx, ny))
	for i, xx in enumerate(x):
		Z[i, :] = stats.norm.pdf(y, loc=true_funct(xx), scale=stddevY)
	for i, yy in enumerate(y):
		Z[:, i] *= stats.norm.pdf(x, loc=0, scale=stddevX)
	Z /= sum(sum(Z))
	return Xg, Yg, Z

class MetaModel:

	def eval_perf(self,param, D):
		if D is None:
			return np.nan, np.nan, np.nan, np.nan
		X, Y = D
		if len(Y)==0:
			return np.nan, np.nan, np.nan, np.nan
		else:
			Yhat = self.predict(param, X)
			ybar = np.mean(Y)
			mse = np.mean((Y-Yhat)**2)
			mae = np.mean(np.abs(Y-Yhat))
			mape = np.mean(np.abs(Y-Yhat)/np.abs(Y))
			R2 = 1 - mse / np.mean((Y-ybar)**2)
			return mse, mae, mape, R2

		
	def __init__(self,param_space,Dtrain,Dtest):

		# parameter range
		self.param_space = param_space

		# sweep parameter space
		Xtrain,Ytrain = Dtrain
		Xtrain = Xtrain.reshape(-1, 1)
		self.modeldict = dict()
		for param in self.param_space:
			self.modeldict[param] = self.train_one(param,Xtrain,Ytrain)

		# compute test and train performance
		self.train_perf = np.array([self.eval_perf(param,Dtrain) for param in self.param_space])
		self.test_perf  = np.array([self.eval_perf(param,Dtest) for param in self.param_space])

	def predict(self, param, X): pass
	def train_one(self,param,Xtrain,Ytrain): pass

	def update_test_perf(self,Dtest):
		self.test_perf = np.array([self.eval_perf(param,Dtest) for param in self.param_space])

class MetaLinearRegression(MetaModel):

	def train_one(self,param,Xtrain,Ytrain):
		q = param
		poly = PolynomialFeatures(q, include_bias=False).fit(Xtrain)
		phi_train = poly.transform(Xtrain)
		model = LinearRegression().fit(phi_train, Ytrain)
		return {'poly': poly, 'model': model}

	def predict(self,param,X):
		model = self.modeldict[param]
		poly = model['poly']
		lr = model['model']
		return lr.predict(poly.fit_transform(X.reshape(-1, 1)))

class MetaKBins(MetaModel):

	def train_one(self,param,Xtrain,Ytrain):
		n_bins = param
		enc = KBinsDiscretizer(n_bins=n_bins,encode='ordinal',subsample=200_000,strategy='quantile')
		X_binned = enc.fit_transform(Xtrain.reshape(-1, 1))
		ybin = np.empty(n_bins)
		for i in range(n_bins):
			ind = i==X_binned[:, 0]
			if not np.any(ind):
				ybin[i] = np.nan
			else:
				ybin[i] = np.mean(Ytrain[ind])
		return {'enc':enc,'ybin':ybin}

	def predict(self,param,X):
		model = self.modeldict[param]
		enc = model['enc']
		ybin = model['ybin']
		return np.array([ybin[int(i)] for i in enc.transform(X.reshape(-1, 1))[:, 0]])

class MetaKNN(MetaModel):

	def train_one(self,param,Xtrain,Ytrain):
		n_neighbors = param
		model = KNeighborsRegressor(n_neighbors=n_neighbors) \
			.fit(Xtrain.reshape(-1, 1), Ytrain)
		return {'model': model}

	def predict(self,param,X):
		model = self.modeldict[param]
		knn = model['model']
		return knn.predict(X.reshape(-1, 1))

class TkContainer(BaseApp):

	metamodel_names = ['K-bins','KNN','Linear regression']
	metamodel_info = {
		'K-bins':{
			'param':'# of bins K',
			'param_space':np.arange(2, 10)
		},
		'KNN':{
			'param':'# of neighbors K',
			'param_space':np.arange(2, 10)
		},
		'Linear regression':{
			'param':'polynomial order',
			'param_space': np.arange(1, 8)
		}
	}
	train_plt = None
	test_plt = None

	def __init__(self):
		super(TkContainer, self).__init__(
			title="Supervised learning demo",
			geometry="1400x800",
			figsize=(12, 4),
			subplots=(2,1))

	def initialize_parameters(self):

		self.stddevX = tk.DoubleVar(master=self.root, value=2)
		self.stddevXstr = tk.StringVar(master=self.root, value='2.0')

		self.stddevY = tk.DoubleVar(master=self.root, value=0.3)
		self.stddevYstr = tk.StringVar(master=self.root, value='0.3')

		self.Ntrain = tk.IntVar(master=self.root, value=0)
		self.Ntrainstr = tk.StringVar(master=self.root, value='0')

		self.Ntest = tk.IntVar(master=self.root, value=0)
		self.Nteststr = tk.StringVar(master=self.root, value='0')

		self.param = tk.IntVar(master=self.root, value=2)
		self.paramstr = tk.StringVar(master=self.root, value='2')

		self.show_pXY = tk.BooleanVar(master=self.root, value=False)
		self.show_function = tk.BooleanVar(master=self.root, value=False)
		self.show_model = tk.BooleanVar(master=self.root, value=False)
		self.show_sweep = tk.BooleanVar(master=self.root, value=False)

		self.selected_model = tk.StringVar(master=self.root, value='K-bins')

	def initialize_data(self):
		self.Dtrain= sample_data(self.Ntrain.get(),self.stddevX.get(),self.stddevY.get())
		self.Dtest = sample_data(self.Ntest.get(),self.stddevX.get(),self.stddevY.get())
		self.build_model()

	def add_widgets(self):

		header_width = 40

		# xy checkbox ........................................
		self.get_checkbox(self.root, text='Show XY distribution',variable=self.show_pXY, command=self.click_pXY_checkbox)\
			.pack(side=tk.TOP, fill=tk.X)

		# function checkbox ........................................
		self.get_checkbox(self.root, text='Show true function',variable=self.show_function, command=self.click_function_checkbox)\
			.pack(side=tk.TOP, fill=tk.X)

		# stddev X input box ......................................
		self.get_entry_label(self.root,
						text="stddev X",
						textvariable=self.stddevXstr,
						validatecommand=self.set_stddevX)\
			.pack(side=tk.TOP, fill=tk.X)

		# stddev Y input box ......................................
		self.get_entry_label(self.root,
						text="stddev Y",
						textvariable=self.stddevYstr,
						validatecommand=self.set_stddevY)\
			.pack(side=tk.TOP, fill=tk.X)

		# Header data ------------------------------------------
		self.get_header(self.root,text='Training data',char='.',width=header_width)\
			.pack(side=tk.TOP, fill=tk.X)

		# N train input box ......................................
		self.get_entry_label(self.root,
						text="N train",
						textvariable=self.Ntrainstr,
						validatecommand=self.set_Ntrain)\
			.pack(side=tk.TOP, fill=tk.X)

		# resample train button ......................................
		self.get_button(self.root,text="Resample train",command=self.press_resample_train)\
			.pack(side=tk.TOP, fill=tk.X)

		# Header Model ----------------------------------------
		self.get_header(self.root,'Model',char='.',width=header_width)\
			.pack(side=tk.TOP, fill=tk.X)

		# model checkbox ........................................
		self.get_checkbox(self.root, text='Show model',variable=self.show_model, command=self.click_model_checkbox)\
			.pack(side=tk.TOP, fill=tk.X)

		# select model combo box ........................................
		self.get_combobox(self.root,
						  	text='Model type',
							textvariable = self.selected_model,
							values = self.metamodel_names,
							command = self.select_model_type)\
			.pack(side=tk.TOP, fill=tk.X)

		# parmaeter input box ......................................
		selected_model = self.selected_model.get()
		metamodel_info = self.metamodel_info[selected_model]

		self.param_input = self.get_combobox(self.root,
						  text= metamodel_info['param'],
						  textvariable=self.paramstr,
						  values=list(metamodel_info['param_space']),
						  command=self.select_model_param)
		self.param_input.pack(side=tk.TOP, fill=tk.X)

		# Header test data ----------------------------------------
		self.get_header(self.root,'Test data',char='.',width=header_width)\
			.pack(side=tk.TOP, fill=tk.X)

		# N test input box ......................................
		self.get_entry_label(self.root,
						text="N test",
						textvariable=self.Nteststr,
						validatecommand=self.set_Ntest)\
			.pack(side=tk.TOP, fill=tk.X)

		# resample test button ......................................
		self.get_button(self.root,text="Resample test",command=self.press_resample_test)\
			.pack(side=tk.TOP, fill=tk.X)

		# Header sweep ------------------------------------------
		self.get_header(self.root,text='Parameter sweep',char='.',width=header_width)\
			.pack(side=tk.TOP, fill=tk.X)

		# sweep plot checkbox ........................................
		self.get_checkbox(self.root, text='Show parameter sweep',variable=self.show_sweep,
						  command=self.click_sweep_checkbox)\
			.pack(side=tk.TOP, fill=tk.X)

	def initialize_fig(self):

		ax0 = self.ax[0]

		# pXY contour ....................................
		Xg, Yg, Z, = get_contour_data(self.stddevX.get(), self.stddevY.get())
		self.plt_pXY = ax0.contour(Xg, Yg, Z)
		self.plt_pXY.set_alpha(0.0)

		# Data ....................................
		self.plt_train, = ax0.plot(self.Dtrain[0],self.Dtrain[1],'o',c='k',markersize=12)
		self.plt_test, = ax0.plot(self.Dtest[0],self.Dtest[1],'+',c='r',markeredgewidth=4,markersize=20)

		# True function ....................................
		self.xx = np.linspace(-6, 6,200)
		self.plt_truefunc, = ax0.plot(self.xx, true_funct(self.xx), c='r', linewidth=6)
		self.plt_truefunc.set_visible(self.show_function.get())

		# Model ....................................
		self.plt_model, = ax0.plot(self.xx, self.eval_model(self.xx), c='m', linewidth=6)
		self.plt_model.set_visible(self.show_model.get())

		ax0.set_ylim(-2,2)
		ax0.spines['right'].set_visible(False)
		ax0.spines['left'].set_visible(False)
		ax0.spines['bottom'].set_visible(False)
		ax0.spines['top'].set_visible(False)
		ax0.set_xticks([0])
		ax0.set_yticks([0])
		ax0.grid(linestyle='-',linewidth=2)

		self.txt = ax0.text(1, .1, self.format_string(),
			horizontalalignment='right',
			verticalalignment='center',
			transform=ax0.transAxes,
			fontsize=30)
		self.txt.set_visible(self.show_model.get())

		ax0.text(1, 0.5, 'x',
			horizontalalignment='right',
			verticalalignment='bottom',
			transform=ax0.transAxes,
			fontsize=40)

		ax0.text(0.48, 1, 'y',
			horizontalalignment='right',
			verticalalignment='top',
			transform=ax0.transAxes,
			fontsize=40)

		# Sweep plot .........................................
		self.make_sweep_plot(self.ax[1])
		self.ax[1].set_visible(self.show_sweep.get())

	def update_figure(self,replotpXY=False):

		if (self.plt_pXY is not None) and replotpXY:
			for coll in self.plt_pXY.collections:
				coll.remove()
			Xg, Yg, Z, = get_contour_data(self.stddevX.get(), self.stddevY.get())
			self.plt_pXY = self.ax[0].contour(Xg, Yg, Z)
			if self.show_pXY.get():
				self.plt_pXY.set_alpha(1.0)
			else:
				self.plt_pXY.set_alpha(0.0)

		if self.plt_model is not None:
			self.plt_model.set_ydata(self.eval_model(self.xx))

		if self.plt_train is not None:
			self.plt_train.set_xdata(self.Dtrain[0])
			self.plt_train.set_ydata(self.Dtrain[1])

		if self.plt_test is not None:
			self.plt_test.set_xdata(self.Dtest[0])
			self.plt_test.set_ydata(self.Dtest[1])

		if self.txt is not None:
			self.txt.set_text(self.format_string())

		if self.train_plt is not None:
			self.train_plt.remove()
			self.test_plt.remove()
			self.make_sweep_plot(self.ax[1])
			self.ax[1].set_visible(self.show_sweep.get())

		plt.draw()

	def set_stddevX(self):
		try:
			self.stddevX.set(float(self.stddevXstr.get()))
			self.press_both_resample(replotpXY=True)
			return True
		except ValueError:
			return

	def set_stddevY(self):
		try:
			self.stddevY.set(float(self.stddevYstr.get()))
			self.press_both_resample(replotpXY=True)
			return True
		except ValueError:
			return

	def set_Ntrain(self):
		try:
			self.Ntrain.set(int(self.Ntrainstr.get()))
			self.press_resample_train()
			return True
		except ValueError:
			return False

	def set_Ntest(self):
		try:
			self.Ntest.set(int(self.Nteststr.get()))
			self.press_resample_test()
			return True
		except ValueError:
			return False

	def press_resample_train(self):
		self.Dtrain = sample_data(self.Ntrain.get(),self.stddevX.get(), self.stddevY.get())
		self.build_model()
		self.update_figure()

	def press_resample_test(self):
		self.Dtest = sample_data(self.Ntest.get(),self.stddevX.get(), self.stddevY.get())
		self.metamodel.update_test_perf(self.Dtest)
		self.update_figure()

	def press_both_resample(self,replotpXY=False):
		self.Dtrain = sample_data(self.Ntrain.get(), self.stddevX.get(), self.stddevY.get())
		self.Dtest = sample_data(self.Ntest.get(), self.stddevX.get(), self.stddevY.get())
		self.build_model()
		self.update_figure(replotpXY)

	def click_pXY_checkbox(self):
		if self.show_pXY.get():
			self.plt_pXY.set_alpha(1.0)
		else:
			self.plt_pXY.set_alpha(0.0)
		plt.draw()

	def click_function_checkbox(self):
		self.plt_truefunc.set_visible(self.show_function.get())
		plt.draw()

	def click_model_checkbox(self):
		self.plt_model.set_visible(self.show_model.get())
		self.txt.set_visible(self.show_model.get())
		plt.draw()

	def select_model_type(self,event):
		selected_model = self.selected_model.get()
		metamodel_info = self.metamodel_info[selected_model]
		param_space = list(metamodel_info['param_space'])

		# update the param combo
		label = self.param_input.children['!label']
		label['text'] = metamodel_info['param']

		combo = self.param_input.children['!combobox']
		combo['values'] = param_space

		self.paramstr.set(param_space[0])
		self.param.set(param_space[0])

		self.build_model()
		self.update_figure()

	def select_model_param(self,event):
		try:
			self.param.set(int(self.paramstr.get()))
			self.build_model()
			self.update_figure()
			return True
		except ValueError:
			return False

	def click_sweep_checkbox(self):
		self.ax[1].set_visible(self.show_sweep.get())
		plt.draw()

	def build_model(self):

		Xtrain, Ytrain = self.Dtrain

		if Xtrain.shape[0]==0:
			self.metamodel = None
			return

		selected_model = self.selected_model.get()
		param_space = self.metamodel_info[selected_model]['param_space']

		if selected_model=='K-bins':
			self.metamodel = MetaKBins(param_space, self.Dtrain, self.Dtest)

		if selected_model=='KNN':
			self.metamodel = MetaKNN(param_space, self.Dtrain, self.Dtest)

		if selected_model == 'Linear regression':
			self.metamodel = MetaLinearRegression(param_space, self.Dtrain, self.Dtest)

	def eval_model(self,X):
		if len(X)==0:
			return np.nan
		if self.metamodel is None:
			return np.empty(X.shape[0])

		return self.metamodel.predict(self.param.get(), X)

	def format_string(self):

		if self.metamodel is None:
			return ''

		param = self.param.get()
		ind, = np.where(self.metamodel.param_space == param)
		selected_model = self.selected_model.get()
		paramname = self.metamodel_info[selected_model]['param']
		train_MSE, train_MAE, train_MAPE, train_R2 = self.metamodel.train_perf[ind[0],:]
		test_MSE, test_MAE, test_MAPE, test_R2 = self.metamodel.test_perf[ind[0],:]

		if self.Ntest.get()==0:
			return "{} = {}\ntrain MSE = {:.3f}\ntrain MAE = {:.3f}\ntrain MAPE = {:.3f}\ntrain R2 = {:.3f}"\
				.format(paramname,param,train_MSE,train_MAE, train_MAPE, train_R2)

		else:
			return "{} = {}\ntrain MSE = {:.3f}\ntrain MAE = {:.3f}\ntrain MAPE = {:.3f}\ntrain R2 = {:.3f}\ntest MSE = {:.3f}\ntest MAE = {:.3f}\ntest MAPE = {:.3f}\ntest R2 = {:.3f}"\
				.format(paramname,param,train_MSE,train_MAE, train_MAPE, train_R2,test_MSE, test_MAE, test_MAPE, test_R2 )

	def make_sweep_plot(self,ax):
		ax.clear()

		if self.metamodel is None:
			param_space = np.arange(10)
			loss_train = np.full(len(param_space), np.nan)
			loss_test = np.full(len(param_space), np.nan)
		else:
			param_space = self.metamodel.param_space
			loss_train = self.metamodel.train_perf[:,0]
			loss_test = self.metamodel.test_perf[:,0]

		self.train_plt, = ax.plot(param_space, loss_train, 'o-', c='b',
								   linewidth=3, markersize=14, label='training MSE')
		self.test_plt, = ax.plot(param_space, loss_test, 'o-', c='r',
								  linewidth=3, markersize=14, label='test MSE')
		ax.legend(fontsize=28, loc='upper left')
		ax.spines['right'].set_visible(False)
		ax.spines['left'].set_visible(False)
		ax.spines['bottom'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.set_xticks(param_space)
		ax.tick_params(axis='x', labelsize=24)
		ax.tick_params(axis='y', labelsize=24)
		selected_model = self.selected_model.get()
		paramname = self.metamodel_info[selected_model]['param']
		ax.set_xlabel(paramname, fontsize=30)
		ax.grid(linestyle=':', linewidth=2)

####################################################
if __name__ == "__main__":
	app = TkContainer()
	tk.mainloop()
