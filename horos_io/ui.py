import tkinter as tk
from tkinter import ttk

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def shitty_confirmation_ui(ax: plt.Axes) -> tk.Tk:
    window = tk.Tk()
    window.geometry("+%d+%d" % (50, 50))
    image_window = FigureCanvasTkAgg(ax.figure, window)
    image_window.get_tk_widget().grid(column=0, row=0, columnspan=2, rowspan=2)
    confirm = ttk.Button(window, text="CONFIRM",
                         command=lambda: (setattr(window, "truth", True),
                                          setattr(window, "remark", remark.get("1.0", "end")),
                                          window.quit(),
                                          window.destroy()))
    confirm.grid(column=0, row=2)
    deny = ttk.Button(window, text="DENY",
                      command=lambda: (setattr(window, "truth", False),
                                       setattr(window, "remark", remark.get("1.0", "end")),
                                       window.quit(),
                                       window.destroy()))
    deny.grid(column=1, row=2)
    remark_label = tk.Label(window, text="Remarks: ")
    remark_label.grid(column=2, row=0)
    remark = tk.Text(window)
    remark.grid(column=2, row=1, rowspan=2)
    return window


def shitty_manual_confirm(sth: plt.Axes) -> bool:
    """
    plots an image to a window and waits for you to confirm or deny... yep; creates a new fucking window every time
    """
    window = shitty_confirmation_ui(sth)
    window.mainloop()
    return window.truth, window.remark
