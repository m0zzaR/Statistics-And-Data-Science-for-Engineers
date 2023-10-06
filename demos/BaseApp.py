import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import tkinter.ttk as ttk
from tkinter import font as tkFont  # for convenience

class BaseApp:

    style = None
    font = ('Ubuntu Mono',18)

    def __init__(self,title="Demo title",geometry="600x600",figsize=(12, 4),subplots=None):

        # Root window
        self.root = tk.Tk()
        self.font = tkFont.Font(family=self.font[0], size=self.font[1])
        self.root.protocol('WM_DELETE_WINDOW', self.kill)
        self.root.wm_title(title)
        self.root.geometry(geometry)

        self.set_style()
        self.initialize_parameters()
        self.initialize_data()

        # Create figure canvas
        if subplots==None:
            self.fig, self.ax = plt.subplots(figsize=figsize, dpi=50)
        else:
            numrows = subplots[0]
            numcols = subplots[1]
            if numrows==1 and numcols>1:
                self.fig, self.ax = plt.subplots(numrows,numcols, figsize=figsize, dpi=50)
            elif numcols==1 and numrows>1:
                self.fig, self.ax = plt.subplots(numrows,numcols, figsize=figsize, dpi=50)
            else:
                print("ERROR")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        NavigationToolbar2Tk(self.canvas, self.root).update()
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.draw()

        self.add_widgets()
        self.initialize_fig()

    def initialize_parameters(self):
        pass

    def initialize_data(self):
        pass

    def add_widgets(self):
        pass

    def initialize_fig(self):
        pass

    def set_style(self):
        padding = 5
        self.style = ttk.Style(self.root)
        self.style.configure("SDSE.TCheckbutton",padding=padding,font=self.font)
        self.style.configure("SDSE.TEntry",padding=padding,font=self.font)
        self.style.configure("SDSE.TButton",padding=padding,font=self.font)
        self.style.configure("SDSE.TLabel",padding=padding,font=self.font)
        self.style.configure("SDSE.TScale",padding=padding,font=self.font)
        self.style.configure("SDSE.TCombobox",padding=padding,font=self.font)

    def kill(self):
        self.root.quit()
        self.root.destroy()

    def get_button(self,root,text,command):
        return ttk.Button(master=root,
                  style='SDSE.TButton',
                  text=text,
                  command=command)

    def get_checkbox(self,root, text,variable, command):
        return ttk.Checkbutton(root,
                       text=text,
                       variable=variable,
                       style="SDSE.TCheckbutton",
                       onvalue=True,
                       offvalue=False,
                       command=command)

    def get_entry_label(self,root,text,textvariable,validatecommand):

        f = ttk.Frame(root)

        ttk.Entry(f,
                 width=5,
                 textvariable=textvariable,
                #  style="SDSE.TEntry",
                 font=self.style.lookup("SDSE.TEntry", "font"),
                 justify='right',
                 validate="focusout",
                 validatecommand=validatecommand) \
            .pack(side=tk.LEFT,
                  padx=8,
                  pady=8,
                  fill=tk.X)

        ttk.Label(f,
                  text=text,
                  style="SDSE.TLabel")\
            .pack(side=tk.LEFT,
                  padx=8,
                  pady=8,
                  fill=tk.X)
        return f

    def get_scale(self,root,variable,command,from_,to,length,text=None,textvariable=None):

        f = ttk.Frame(root)

        ttk.Scale(master=f,
                 variable=variable,
                 command=command,
                 from_=from_,
                 to=to,
                 length=length,
                 orient=tk.HORIZONTAL)\
            .pack(side=tk.LEFT,
                  padx=8,
                  pady=8,
                  fill=tk.X)

        if textvariable is None:
            ttk.Label(f,
                    text=text,
                    style="SDSE.TLabel")\
                .pack(side=tk.LEFT,
                    padx=8,
                    pady=8,
                    fill=tk.X)
        else:
            ttk.Label(f,
                    textvariable=textvariable,
                    style="SDSE.TLabel")\
                .pack(side=tk.LEFT,
                    padx=8,
                    pady=8,
                    fill=tk.X)
            
        return f


    def get_combobox(self,root,text,textvariable,values,command):

        f = ttk.Frame(root)
        cb = ttk.Combobox(f,
                          textvariable=textvariable,
                          font=self.style.lookup("SDSE.TCombobox", "font"))
                        #   style="SDSE.TCombobox")
        cb['values'] = values
        cb.bind('<<ComboboxSelected>>', command)
        cb.pack(side=tk.LEFT,
                  padx=8,
                  pady=8,
                  fill=tk.X)

        ttk.Label(f,
                  text=text,
                  style="SDSE.TLabel")\
            .pack(side=tk.LEFT,
                  padx=8,
                  pady=8,
                  fill=tk.X)
        return f

    def get_header(self,root,text,char,width):
        ndots = int((width - len(text)) / 2)
        sstr = ndots * char + text + ndots * char
        return ttk.Label(root,text=sstr,style="SDSE.TLabel")
