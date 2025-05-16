# X-Ray_Pneumonia

This repository holds codes for a hobby / portfolio project.
Currently worked on by Patrick and Andrei.

Main functionalities:
  - Training (backend):
    - model training: own CNN model, transfer learning (MobileNet), fine tuning (MobileNet)
    - tracking of training perfomance with `mlflow`
  - Model administration & tracking (backend):
    - model registration with `mlflow`
    - csv logging of model performance on new, unseen data
    - supervision of and automated switch between two competing models ("champion" vs "challenger")
  - Frontend:
    - image upload and inferrence (pneumonia indication)
    - visualization of ML models' performance
    - embedded API documentation and model administration panel


Readme under construction. For a short presentation of the project check out [this video](https://www.youtube.com/watch?v=aaeOJk1loig).
