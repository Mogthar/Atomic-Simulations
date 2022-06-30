import tkinter as tk
from tkinter import filedialog as fd
import simulation as sim
import animation as anim

window = tk.Tk()

window.rowconfigure(0, minsize=100, weight=1)
window.columnconfigure(0, minsize=100, weight = 3) # add padding
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
    

### animation canvas and the animation object tied to it###
canvas_animation = tk.Canvas(master=frame_animation, bg="white") ## set a size for the canvas
canvas_animation.grid(column=0, row=0, sticky=tk.NSEW)
animation = anim.Animation(canvas_animation)

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
def render_frame_at_index(index):
    animation.render_frame(animation.get_frame(int(index)))
    
slider_frameControl = tk.Scale(master = frame_animation, from_ = 0, to=0, orient="horizontal", command = render_frame_at_index)
slider_frameControl.grid(row=2, column=0, sticky=tk.NSEW)

### open folder tab ###
frame_fileDialog = tk.Frame(master=frame_animation, relief=tk.FLAT, borderwidth=5)
frame_fileDialog.grid(column=0, row=3, sticky=tk.NSEW)

frame_fileDialog.rowconfigure(0, minsize=10, weight=1)
frame_fileDialog.columnconfigure([0,1], minsize=10, weight=1)
    
def set_label_text(label, text):
    label['text'] = text
    return

# load file buutton
def load_trajectory_file():
    filetypes = (('trajectory files', '*.traject'),('xyz file', '*.xyz'))
    file = fd.askopenfile(mode ='r', title = 'Load trajectory file', initialdir ='.', filetypes = filetypes)
    # can use try except here
    if file != None:
        set_label_text(label_fileLoadIndicator, file.name)
        # reset the animation
        animation.reset()
        animation.load_frames_from_file(file)
        slider_frameControl.configure(to = animation.get_number_of_frames() - 1)
        slider_frameControl.set(0)
        render_frame_at_index(0)
    else:
        set_label_text(label_fileLoadIndicator, 'No file selected')
    return

button_loadFile = tk.Button(master=frame_fileDialog, text="Load", command = load_trajectory_file)
button_loadFile.grid(row=0, column=1, sticky=tk.NSEW)

# text field to confirm file loaded
label_fileLoadIndicator = tk.Label(master=frame_fileDialog, text="No file loaded")
label_fileLoadIndicator.grid(row=0, column=2, sticky = tk.W)

window.mainloop()