""" Script for counting through the entire set of experiment images

    Make sure the desired exemplars are saved in the exemplar folder, which is then
    specified in the args. And same with the target folder. 
"""

from BubbleCount.counting_model import CountingPipe

target_dir = "F:\\MEng Project\\cropped_images"
args = {"target_path": target_dir}
pipe = CountingPipe(args, num_boxes=6)

pipe.counting_all_unadjusted()