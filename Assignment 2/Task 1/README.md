# Tensorboard integration to monitor training process
---
For execution: Please go to the file location where main.py has been saved, and from the command line execute the below command (tested on Python 3.7.4).

```
python main.py
```

This will create a log directory in the location provided in the "current_dir". To initiate tensorboard please go to the log directory (the root) and from 
command line execute the below command

``
tensorboard --logdir LOG_DIRECTORY_NAME
``

and then follow the instruction on the terminal.

Also we can monitor how our model is performing if the experiment continues to run and tensorboard is also enabled at the same time.

Important Point:

Tensorboard will read data from a full directory therefore always point --logdir to a directory.
