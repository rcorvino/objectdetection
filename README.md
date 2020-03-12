# objectdetection
An alternative solution to Tech Xperience – Case competition - Philips case

## Basic idea
This solution uses object detection to identify 4 classes of objects:

1. shaver
2. smart-baby-bottle
3. toothbrush
4. wake-up-light

The limited number of images prevents efficient training for accurate image classification. Hence, we adopt a solution that comes from face recognition: solving a one-shot learning problem. This technique learns from one example to recognize again the same person (in our case the same object).

We use keras and [Mask R-CNN](https://github.com/matterport/Mask_RCNN).
Ideas on how to implement this solution are mostly taken from [here](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/).


## Get the code
```
git clone https://github.com/rcorvino/objectdetection.git
```

## Run the docker image
The docker image is on dockerHub

```
docker pull rosildecorvino/objdetect
docker run --name imgdet -it rosildecorvino/objdetect python objectDetection.py
docker cp <your-path-to-validation-folder> imgdet:/root/validation
docker start -ai imgdet
```
