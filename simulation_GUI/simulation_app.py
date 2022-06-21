import tkinter as tk
from tkinter import filedialog as fd

window = tk.Tk()

window.rowconfigure(0, minsize=100, weight=1)
window.columnconfigure(0, minsize=100, weight = 3)
window.columnconfigure(1, minsize=100, weight = 1)

#### set up two main frames which split the screen into animation part and simulation input ###
frame_animation = tk.Frame(master=window, relief=tk.FLAT, borderwidth=0)
frame_animation.grid(column=0, row=0, sticky=tk.NSEW)
frame_simulation = tk.Frame(master=window, relief=tk.GROOVE, borderwidth=5,  bg="blue")
frame_simulation.grid(column=1, row=0, sticky=tk.NSEW)


#### fill the animation side with the appropriate widgets ####
frame_animation.columnconfigure(0, minsize=10, weight=1)
frame_animation.rowconfigure(0, minsize=10, weight=6)
for i in range(1,4):
    frame_animation.rowconfigure(i, minsize=10, weight=1)


### animation canvas ###
canvas_animation = tk.Canvas(master=frame_animation, bg="white")
canvas_animation.grid(column=0, row=0, sticky=tk.NSEW)
x_axis = canvas_animation.create_line(0, 100 , 100, 100)
# draw axes

### controls tab ###
frame_animationControls = tk.Frame(master=frame_animation, relief=tk.FLAT, borderwidth=5, bg="green")

frame_animationControls.rowconfigure(0, minsize=10, weight=1)
frame_animationControls.columnconfigure(0, minsize=10, weight=1)
frame_animationControls.columnconfigure(1, minsize=10, weight=1)

frame_animationControls.grid(column=0, row=1, sticky=tk.NSEW)

# control buttons
play_image = tk.PhotoImage(file="./play_image.png")

button_startPause = tk.Button(master=frame_animationControls, text="PLAY")
button_stop = tk.Button(master=frame_animationControls, text="STOP")

button_startPause.grid(row=0, column=0, sticky=tk.NSEW)
button_stop.grid(row=0, column=1, sticky=tk.NSEW)

### slider tab ###
slider_frameControl = tk.Scale(master = frame_animation, from_ = 0, to=1, orient="horizontal")
slider_frameControl.grid(row=2, column=0, sticky=tk.NSEW)

### open folder tab ###
frame_fileDialog = tk.Frame(master=frame_animation, relief=tk.FLAT, borderwidth=5)
frame_fileDialog.grid(column=0, row=3, sticky=tk.NSEW)

frame_fileDialog.rowconfigure(0, minsize=10, weight=1)
frame_fileDialog.columnconfigure([0,1,2], minsize=10, weight=1)

# input box
entry_fileName =tk.Entry(master=frame_fileDialog)
entry_fileName.grid(column=0, row=0, sticky=tk.E)

# open button
button_loadFile = tk.Button(master=frame_fileDialog, text="Load")
button_loadFile.grid(row=0, column=1, sticky=tk.NSEW)

# text field to confirm file loaded
label_fileLoadIndicator = tk.Label(master=frame_fileDialog, text="No file loaded")
label_fileLoadIndicator.grid(row=0, column=2, sticky = tk.W)




# =============================================================================
# def handle_keypress(event):
#     """Print the character associated to the key pressed"""
#     print(event.char)
# 
# # Bind keypress event to handle_keypress()
# window.bind("<Key>", handle_keypress)
# 
# def handle_click(event):
#     print("The button was clicked!")
# 
# button = tk.Button(text="Click me!")
# button.pack()
# button.bind("<Button-1>", handle_click)
# =============================================================================
window.mainloop()