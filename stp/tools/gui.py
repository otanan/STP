#!/usr/bin/env python3
"""GUI element handler.

**Author: Jonathan Delgado**

Handles creating GUI elements such a graphical loading bars for long processes.

"""
from tkinter import ttk
import tkinter as tk
import time

######################## GUI ########################

def _center_window(master, width, height):
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        # Calculate (x, y) coordinates for the Tk window
        x = (screen_width/2) - (width/2)
        y = (screen_height/2) - (height/2)
        # Set the dimensions of the screen and where it is placed
        master.geometry('%dx%d+%d+%d' % (width, height, x, y))


class ProgressBar:
    """ Tkinter progress bar.
    
        Object which will handle creating a Tkinter progress window for the purpose of showing as a graphical loading bar.
        
        Attributes:
            MAX_VALUE: the max value of steps before completing the bar.

            width: the width of the GUI bar.

            height: the height of the GUI bar.
            
            title: the window title.
    """

    def __init__(self, MAX_VALUE, width=200, height=40, title='Loading...'):
        """ Initialize object to hold important quantities and create/style the bar. """


        # Must know maximum to calculate percentages
        self._MAX_VALUE = MAX_VALUE
        # Initialize value property
        self.value = 0
        # self.master = tk.Toplevel()
        self.master = tk.Tk()
        self.bar = ttk.Progressbar(self.master, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.width = width; self.height = height
        self.title = title

        self._style()
        _center_window(self.master, self.width, self.height)


    def _style(self):
        """ Handles designing the bar. """

        self.bar.pack(fill='both', expand=True)
        self.set_title(self.title)
        self.master.style = ttk.Style()
        self.master.style.theme_use('classic')


    def set_title(self, title):
        """ Updates the bar's title.
            
            Args:
                title (str): the new title.

            Returns:
                (None): none
        
        """
        self.title = title
        self.master.wm_title(self.title)


    def update(self, amount=1):
        """ Main function for interacting with the bar. Handles updating progress.
            
            Kwargs:
                amount (int): the amount to update the progress by.
        
        """
        self.value += amount

        # print( f'Progress update counter: \n{self.value}' )
        status = self.value / self._MAX_VALUE * 100
        self.bar['value'] = status
        # time.sleep(1) # Uncomment to see the progress bar update slowly

        # Updates the window itself
        self.master.update()


    def next(self):
        """ Mask for update to use same syntax in the case of using instead of ShadyBar for example. """
        self.update()


    def finish(self):
        """ Finishes the bar. Briefly shows it has been completed before safely destroying the Tkinter window. """
        self.bar['value'] = 100
        self.master.update()
        # Briefly show a completed bar
        time.sleep(0.2)
        # Close the window
        self.master.destroy()




#------------- Entry code -------------#

def main():
    print('gui.py')
    

if __name__ == '__main__':
    main()