""" Script for counting through the entire set of experiment images"""

from BubbleCount.counting_model import CountingPipe

target_dir = "F:\\MEng Project\\cropped_images"
args = {"target_path": target_dir}
pipe = CountingPipe(args, num_boxes=6)

pipe.counting_all_unadjusted()