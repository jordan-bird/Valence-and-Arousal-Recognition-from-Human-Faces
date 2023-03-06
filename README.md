# Automating Russell's Circumplex of Affect

This machine learning model automates Russell's 1980 work, a circumplex model of affect. The valence and arousal of the face are predicted. 

# Model overview

The model is a mirrored, branched CNN. The CNN contains 4 layers of 64 filters followed by a dense layer of 64 neurons. Each layer is surrounded by a dropout at a rate of 20%. The model is trained on 100,000 images from the [AffectNet dataset](http://mohammadmahoor.com/affectnet/)

# Interpretation

According to Russell (1980), _"emotional valence describes the extent to which an emotion is positive or negative, whereas arousal refers to its intensity, i.e., the strength of the associated emotional state"_. 

![diagram](https://i.imgur.com/L8XLsuW.png)

As shown in the example code, the first value output corresponds to the valence and the second corresponds to the arousal. 

# Usage

Run `predict_image.py` and specify the flags for an image file containing at least one human face `--image`, as well as the choice of model `--model`. The model provided in this repo (`model-100k-linear`) is the final model from the paper.

`python predict_image.py --image=test_image_8.png --model=model-100k-linear`

# Reference

This model is the output of a published study by Jordan J. Bird, Azhar Aulia Saputra, Naoyuki Kubota, and Ahmad Lotfi. 

Bird, J. J., Saputra, A. A., Kubota, N., Lotfi, A. (2022). Affective Computing in Computer Vision: A Study on Facial Expression Recognition. 13th International Congress on Advanced Applied Informatics Winter (IIAI-AAI-Winter) (pp. 84-88).

[Download PDF](http://jordanjamesbird.com/publications/Affective-Computing-in-Computer-Vision-A-Study-on-Facial-Expression-Recognition.pdf)
