from moviepy.editor import *

clip = VideoFileClip("project_video_output.mp4").resize(0.2)
clip.write_gif("project_video_output.gif")
