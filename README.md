# objectdetection
alternative solution to Tech Xperience – Case competition - Philips case

## Basic idea
Use object detection to identify 4 classes of objects:

1. shaver
2. smart-baby-bottle
3. toothbrush
4. wake-up-light

The limited number of images prevent efficient trainig for image classification. Hence we adopt a solution that comes from face recognition system: one-shot learning problem, which means learning from one example to recognize a person (in our case an object) again.

We use keras and [Mask R-CNN](https://github.com/matterport/Mask_RCNN).
Ideas on how to implement this solution are mostely taken from [here](https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/).
