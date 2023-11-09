import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from BaseApp import BaseApp
import tkinter as tk
from sklearn.linear_model import LogisticRegression

def sample_data(num_samples,stddev):
    return make_blobs(n_samples=num_samples,
                        centers=np.array([[-1, -1], [1, 1]]),
                        cluster_std=stddev,
                        n_features=2)

class TkContainer(BaseApp):
    num_samples = 100

    scatter = []
    logreg_surf = None
    decision_bndry = None

    def __init__(self):
        super(TkContainer, self).__init__(
            title="2D logistic regression",
            geometry="1300x800",
            figsize=(10, 7),
            subplots=None )

    def initialize_parameters(self):
        self.show_decision_bndry = tk.BooleanVar(master=self.root,value=False)
        self.show_ce = tk.BooleanVar(master=self.root,value=False)
        self.show_logreg = tk.BooleanVar(master=self.root,value=False)
        self.stddev = tk.DoubleVar(master=self.root,value=0.5)
        self.stddevstr = tk.StringVar(master=self.root,value='0.5')

    def initialize_data(self):

        self.X, self.y = sample_data(self.num_samples,self.stddev.get())

        N = 30
        x1 = np.linspace(-3, 3, N)
        x2 = np.linspace(-3, 3, N)
        self.mesh1, self.mesh2 = np.meshgrid(x1, x2)
        mesh1l = np.reshape(self.mesh1, (N ** 2, 1))
        mesh2l = np.reshape(self.mesh2, (N ** 2, 1))

        logreg = LogisticRegression().fit(self.X, self.y)
        yhatl = logreg.predict_proba(np.hstack((mesh1l, mesh2l)))
        yhatl = yhatl[:, 1]
        self.yhat = np.reshape(yhatl, (N, N))

        theta0 = logreg.intercept_[0]
        theta1 = logreg.coef_[0, 0]
        theta2 = logreg.coef_[0, 1]

        slope = -theta1 / theta2
        intercept = -theta0 / theta2

        self.x1 = np.linspace(-2, 2)
        self.x2 = slope * self.x1 + intercept

    def add_widgets(self):

        # std dev input box ......................................
        self.get_entry_label(self.root,
                             text="stddev",
                             textvariable=self.stddevstr,
                             validatecommand=self.set_stddev) \
            .pack(side=tk.TOP, fill=tk.X)

        # reset view button ......................................
        self.get_button(self.root, text="View from above", command=self.reset_view) \
            .pack(side=tk.TOP, fill=tk.X)

        # resample button ......................................
        self.get_button(self.root, text="Resample", command=self.resample) \
            .pack(side=tk.TOP, fill=tk.X)

        # CE checkbox ........................................
        self.get_checkbox(self.root, text='Show cross-entropy', variable=self.show_ce,
                          command=self.click_ce_checkbox) \
            .pack(side=tk.TOP, fill=tk.X)

        # logreg checkbox ........................................
        self.get_checkbox(self.root, text='Show optimal sigmoid',
                          variable=self.show_logreg,
                          command=self.click_logreg_checkbox) \
            .pack(side=tk.TOP, fill=tk.X)

        # decision boundary checkbox ............................
        self.get_checkbox(self.root, text='Show decision boundary',
                          variable=self.show_decision_bndry,
                          command=self.click_decision_bndry_checkbox) \
            .pack(side=tk.TOP, fill=tk.X)

    def initialize_fig(self):

        self.ax.remove()
        self.ax = plt.subplot(projection='3d')
        self.ax.set_xlabel('x1', fontsize=30)
        self.ax.set_ylabel('x2', fontsize=30)

        # cross entropy surface plot ...................................
        self.ce1_surf = self.ax.plot_surface(self.mesh1, self.mesh2, -np.log(self.yhat)/10,
                        cmap='Greens',
                        alpha=0.5,
                        linewidth=0.1,
                        edgecolors='g',
                        antialiased=False)
        self.ce1_surf.set_visible(self.show_ce.get())

        self.ce2_surf = self.ax.plot_surface(self.mesh1, self.mesh2, -np.log(1 - self.yhat)/10,
                        cmap='Reds',
                        alpha=0.5,
                        linewidth=0.1,
                        edgecolors='r',
                        antialiased=False)
        self.ce2_surf.set_visible(self.show_ce.get())

        # logistic regression sigmoid function .........................
        self.logreg_surf = self.ax.plot_surface(self.mesh1, self.mesh2, self.yhat,
                          cmap='RdYlGn',
                          alpha=0.5,
                          linewidth=1,
                          antialiased=False)
        self.logreg_surf.set_visible(self.show_logreg.get())

        # scatter plot ...............................................
        colors = ['r','g']
        markers=['o','o']
        s = [70,70]
        for i in range(2):
            ind = np.where(self.y==i)[0]
            self.scatter.append( self.ax.scatter3D(self.X[ind,0], self.X[ind,1],i,
                       c=colors[i],
                       marker=markers[i],
                       s = s[i]) )

        # decision boundary ...............................................
        self.decision_bndry = self.ax.plot3D(self.x1, self.x2, 0.5, color='black', linewidth=6)[0]
        self.decision_bndry.set_visible(self.show_decision_bndry.get())

        self.ax.set_xlim(-4,4)
        self.ax.set_ylim(-4,4)
        self.ax.view_init(elev=90., azim=-90)
    
    def click_ce_checkbox(self):
        self.ce1_surf.set_visible(self.show_ce.get())
        self.ce2_surf.set_visible(self.show_ce.get())
        plt.draw()

    def click_logreg_checkbox(self):
        self.logreg_surf.set_visible(self.show_logreg.get())
        plt.draw()

    def click_decision_bndry_checkbox(self):
        self.decision_bndry.set_visible(self.show_decision_bndry.get())
        plt.draw()

    def reset_view(self):
        self.ax.view_init(elev=90., azim=-90)
        plt.draw()

    def resample(self):
        self.initialize_data()
        self.initialize_fig()
        plt.draw()
        return True


    def set_stddev(self):
        try:
            self.stddev.set(float(self.stddevstr.get()))
            self.resample()
            return True

        except ValueError:
            return False


####################################################
if __name__ == "__main__":
    app = TkContainer()
    tk.mainloop()

