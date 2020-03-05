Download the zip file called submission from the portal.
The imgs/train contains all of the 17K training images.
The train.csv file contains the uids of each image along with the type and color attributes.

For evaluation, the run.py file is invoked with a dataframe containing only test image uids. These test images will be put into submission/imgs/test directory before invoking predict.
Ensure that your model is created and read from submission/model directory.
Please generate a requirements.txt, by running a pip freeze >> requirements.txt and submit that along with all your code and models. This is important from the POV of creating your environment before execution.

zip up submission directory AFTER removing all the training and test images and then submit on the portal as submission.zip
