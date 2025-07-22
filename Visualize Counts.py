import BubbleCount.image_preprocess as image_preprocess
import BubbleCount.csv_helpers as csv_helpers

import csv
import os

from BubbleCount.counting_model import CountingPipe

model = CountingPipe()  # Only really need the count_hybrid() function in this notebook
args = {
    "sample_path": "./Exemplars/",
    "target_path": "./2025/Targets",
    "result_path": "./2025/Outputs/out.csv",
    "output_dir": "./2025/Outputs/",
    "model_path": "./data/pretrainedModels/FamNet_Save1.pth",
    "raw_img_dir": "./2025/Images/SEN10_1.8_6_300-400",
}
# Crop images to AOI
image_preprocess.crop_to_interest(
    image_dir=args["raw_img_dir"],
    region=[65, 770, 1090, 970],
    output_dir=args["target_path"],
)


def simplify_target_name(target):
    temp = target.split("_")[1:-1]
    temp = "_".join(temp)
    serial_number = temp.split(".")[-1]
    batch_name = ".".join(temp.split(".")[0:-2])

    return f"{batch_name}_{serial_number}"


result_to_csv = []

# Memory leak tracker

# Load exemplars and targets
Exemplars = image_preprocess.load_exemplars_from_directory(
    args["sample_path"], reverse_bbox=False
)
Targets = image_preprocess.load_target_images_from_directory(args["target_path"])

sample_image = Exemplars[1]

csv_helpers.backup_and_clear_csv(args["result_path"])

process = psutil.Process(os.getpid())


with open(args["result_path"], "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Exemplar", "num_exemps", "rev_bbox", "Target", "Count"])

    base_memory_usage = process.memory_info().rss
    last_mem = base_memory_usage

    for target_image in Targets:
        num_exemps = 4
        # Creating the hybrid
        hybrid, hybrid_boxes = image_preprocess.insert_cropped(
            sample_image["image"],
            target_image["image"],
            sample_image["box"],
            num_exemps,
        )
        target_name = (
            f"{simplify_target_name(target_image['file_name'])}_{num_exemps}exemps"
        )

        # counting
        hybrid_count = model.count_hybrid_and_visualize(
            hybrid,
            hybrid_boxes,
            sample_image["file_name"],
            target_name,
            output_directory=args["output_dir"],
        )

        result_to_csv.append(
            [sample_image["file_name"], num_exemps, False, target_name, hybrid_count]
        )

        # append to csv
        writer.writerows(result_to_csv)

        # track memory leaks
        memory_usage = process.memory_info().rss
        loop_memory_usage = memory_usage - base_memory_usage

        mem_diff = loop_memory_usage - last_mem

        last_mem = loop_memory_usage

        print(
            f"Memory Gain Since Starting:{loop_memory_usage} (Difference: {mem_diff})"
        )

print(f"The counts are saved to {args['result_path']}.")
