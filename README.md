# Bubble Counting with FamNet

This repo introduces a method for implementing the FamNet on a large 
image set where it is inconvenient to manually select exemplars for every 
images. It also contains the experiemnts conducted for investigating the
quality of the counting result. FamNet is based from this paper: 
```
Learning To Count Everything
Viresh Ranjan, Udbhav Sharma, Thu Nguyen and Minh Hoai
Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.
```
Link to arxiv preprint: https://arxiv.org/pdf/2104.08391.pdf

## Introduction
The notebooks, "EXP_9_Batches" and "EXP_1_Batch", shows the experiments 
of the counting pipeline. The pipeline modules are saved in the package 
`BubbleCount`. 
The directories `Exemplars`, `Targets`, and `Outputs` are used in the pipeline

## Visualization
Set up by creating a virtual environment and installing requirements.
```bash
python -m venv .venv
source .venv/Scripts/Activate
pip install -r requirements.txt
```

Run Visualize Counts.ipynb and modify the following in `args` to your liking:

`"result_path": "./2025/Outputs/out.csv"`
Directory to store prediction CSV.

`"output_dir": "./2025/Outputs/"`
Directory for predicted images.

`"raw_img_dir": "./2025/Images/SEN10_1.8_6_300-400"`
Raw images to be cropped.

`"target_path": "./2025/Targets"`
Cropped images to be predicted.

Uncomment the following line in the 4th cell to crop raw images (only needs to be run once):
`image_preprocess.crop_to_interest(image_dir=args["raw_img_dir"],region=[65, 770, 1090, 970],output_dir=args["target_path"])`
