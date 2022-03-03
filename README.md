# Machine-Vision-KIT-VL

I tried out some of the concepts introduced in the Machine Vision lecture in winter term 2021/2022. The results are brought together in this repos.

## Structure

The project is structured the following way (relevant files are in the src directory):

* corner_detection: Implementation of the Harris Corner Detector discussed at the end of Chapter 3
* data: Warehouse for all the constant and variable data
* edge_detection: Relevant files associated with the Edge Detection discussed in Chapter 3
* resources: Different images used in the project including results
* utils: Utility functions

## Results

Here are some of the results obtained from the different methods.

### Edge Detection with the Canny Operator

![alt text](https://github.com/JanIsHacking/Machine-Vision-KIT-VL/blob/master/src/resources/results/comparison_scrambled_2.png?raw=true)

![alt text](https://github.com/JanIsHacking/Machine-Vision-KIT-VL/blob/master/src/resources/results/comparison_smart_building.jpg?raw=true)

### Corner (and Edge) Detection with the Harris Corner Detector

![alt text](https://github.com/JanIsHacking/Machine-Vision-KIT-VL/blob/master/src/resources/results/comparison_solved_2_HCD.png?raw=true)

![alt text](https://github.com/JanIsHacking/Machine-Vision-KIT-VL/blob/master/src/resources/results/comparison_tiger_bird_HCD.png?raw=true)
