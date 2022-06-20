import tkinter as tk

window = tk.Tk()

### split the screen into animation side and the simulation side ###
animation_frame_height = 1080
animation_frame_width = 1280
simulation_frame_height = 1080
simulation_frame_width = 640


## instead of pack maybe use columns because then one can specify the relative scaling of each using columnconfigure
frame_animation = tk.Frame(master=window, width=animation_frame_width, height=animation_frame_height, relief=tk.FLAT, borderwidth=0,  bg="red")
frame_animation.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)
frame_simulation = tk.Frame(master=window, width=simulation_frame_width, height=simulation_frame_height, relief=tk.GROOVE, borderwidth=5,  bg="blue")
frame_simulation.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)


### fill the animatin side with the appropriate widgets ###

### animation canvas ###
canvas_height = (2/3) * animation_frame_height
canvas_animation = tk.Canvas(master=frame_animation, bg="white", height=canvas_height, width=animation_frame_width)

canvas_animation.pack(fill=tk.X, side=tk.TOP)

### controls tab ###
controls_height = 50
controls_width = animation_frame_width
frame_animationControls = tk.Frame(master=frame_animation, width=controls_width, height=controls_height, relief=tk.FLAT, borderwidth=5, bg="green")

frame_animationControls.rowconfigure(0, minsize=10, weight=1)
frame_animationControls.columnconfigure([0, 1], minsize=10, weight=1)

frame_animationControls.pack(side=tk.TOP, fill=tk.X)

# control buttons
button_width = 50
button_height = controls_height

play_image = tk.PhotoImage(file="./play_image.png")

button_startPause = tk.Button(master=frame_animationControls, width=button_width, height = button_height, image=play_image)
button_stop = tk.Button(master=frame_animationControls, width=button_width, height = button_height, text="STOP")

button_startPause.grid(row=0, column=0)
button_stop.grid(row=0, column=1)

### 


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