import sys
import argparse
import tkinter as tk
import tkinter.ttk
sys.path.append('./pyuiutils/')
import pyuiutils.uiutils as uiutils

import argparse
import numpy as np
import filtering

class LaplacianUIFrame(tk.Frame):
    """ Frame to contain the laplacian image editing GUI.
    Contains two sub-frames - an ImageFrame and a SliderFrame. """

    def __init__(self, parent, root, num_levels):
        tk.Frame.__init__(self, parent)
        self.num_levels = num_levels

        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.image_frame = ImageFrame(self, root)
        self.image_frame.grid(column=0, row=0, sticky=tk.NSEW)

        self.slider_frame = SliderFrame(self, root)
        self.slider_frame.grid(row=0, column=1, sticky=tk.NSEW)

    def make_pyr(self, img):
        self.pyr = filtering.construct_laplacian(img, self.num_levels)

    def update_img(self, *args):
        weights = [s.get() for s in self.slider_frame.sliders]

        img_float = filtering.reconstruct_laplacian(self.pyr, weights)
        img = (img_float*255).clip(0, 255).astype(np.uint8)
        self.image_frame.image_widget.draw_cv_image(img)


class ImageFrame(uiutils.BaseFrame):
    """ Frame to load, display, and save out a reconstructed image. """
    def __init__(self, parent, root):
        uiutils.BaseFrame.__init__(self, parent, root, 3, 1)
        self.config(highlightthickness=1, highlightbackground="black")


        tk.Button(self, text='Load Image', command=self.load_img).grid(
                row=0, column=0, sticky=tk.N)


        self.image_widget = uiutils.ImageWidget(self)
        self.image_widget.grid(row=1, column=0, sticky=tk.NSEW)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)

        tk.Button(self, text='Save Image', 
                command=self.save_image).grid(row=2, column=0, sticky=tk.W + tk.E)

    def load_img(self, img_name=None):
        img_name, img = self.ask_for_image(img_name)
        if img is not None:
            self.image_widget.draw_cv_image(img)
            self.parent.make_pyr(img.astype(np.float32)/255)
            self.image_name = img_name

    def save_image(self):
        f = uiutils.ask_for_image_path_to_save(self)
        if f is not None:
            self.image_widget.write_to_file(f, False)

class SliderFrame(uiutils.BaseFrame):
    """ Frame to provide sliders that set the weight of each pyramid level
    in the reconstruction. """
    def __init__(self, parent, root):
        uiutils.BaseFrame.__init__(self, parent, root, 5, 2)
        self.config(highlightthickness=1, highlightbackground="black")

        self.sliders = []
        for i in range(self.parent.num_levels):
            tk.Label(self, text='Scale ' + str(i)).grid(row=i, column=0, sticky=tk.E)
            slider = tk.Scale(self, from_=0.0, to=2.0, resolution=0.01, orient=tk.HORIZONTAL)
            self.sliders.append(slider)
            slider.set(1.0)
            slider.grid(row=i,column=1)
            slider.bind('<ButtonRelease-1>', self.parent.update_img)

if __name__ == "__main__":
    root = tk.Tk()

    parser = argparse.ArgumentParser('Run the Laplacian Image Editing GUI.')

    parser.add_argument('--image', '-i', help='An image to load.', default=None)
    parser.add_argument('--levels', '-l', help='Levels of laplacian pyramid', type=int, default=5)
    args = parser.parse_args()

    root.title('CSCI 497P P1 - Laplacian Image Editing')
    w, h = root.winfo_screenwidth(), root.winfo_screenheight() - 50
    root.geometry('{}x{}+0+0'.format(w, h))
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    app = LaplacianUIFrame(root, root, args.levels)
    app.grid(row=0, sticky=tk.NSEW)

    if args.image:
        app.image_frame.load_img(args.image)
    root.mainloop()


