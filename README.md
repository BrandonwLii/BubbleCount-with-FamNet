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

