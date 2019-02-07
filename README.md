# Face-Swap-OpenCV
![face swap sample](https://raw.githubusercontent.com/BruceMacD/Face-Swap-OpenCV/master/images/face_swapped.png)

This is a basic face-swap implementation using OpenCV. Check out the code for a step-by-step explanation.

## Usage
./face_swap.py -i <data/input1.jpg> -i <data/input2.jpg>

```
./face_swap.py -i data/headshot1.jpg -i data/headshot2.jpg
```

## Requirements
* OpenCV v3.0+
* numpy
* dlib (for the landmark detection)
* Python 3

## Sources
Based on the work of Satya Mallick

https://www.learnopencv.com/face-swap-using-opencv-c-python/

Facial landmarks

https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

Head shot 1

https://www.pexels.com/photo/adult-attractive-beautiful-beauty-415829/

Head shot 2

https://www.pexels.com/photo/man-wearing-black-zip-up-jacket-near-beach-smiling-at-the-photo-736716/

![delauney triangulation](https://github.com/BruceMacD/Face-Swap-OpenCV/blob/master/images/delauney_landmarks.png)
