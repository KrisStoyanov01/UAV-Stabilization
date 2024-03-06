Robust Reinforcement Learning for Drones Control under Gusting Conditions

The code can be used to train a new model, but several limitations should be noted:
1) Speed - the training process can take up to 48h, depending on the hardware. Significant results may not appear even after 20h of training.
2) CPU intensivity - the model trains solely on the CPU so please note that this can lead to significant performance decrease and unstable system.
3) RAM requirements - during the training process a lot of RAM is being used and its consumptions is steadily increasing. Because of this, I do not suggest training on a machine with less than 32 GB of RAM. 
Sometimes the process can require too much RAM and if it is left usupervised, this can lead to sudden stopping of the training process and loss of progress.
4) The project requires quite a large pool of Python Libraries, several of which have additional requirems about their versions. I have not uploaded any of them, due to their massive sizes (the total size on my machine is more than 20GB and there are a lot of dependencies that should be cared of.
5) There is no training data uploaded. The first reason for this is it's massive size. The second one is the constantly changing environment - a model trained for specific conditions is not guaranteed to perform well if there are changes in the conditions. Therefore it is best if after the new conditions are taken care of, to change the model parameters and retrain.

The used Hardware for the project is an Optitrack Motioncapture system, Crazyflie 2.1 drones with Single marker, Crazyflie Radio and the Crazyswarm library.
Note: During the development of the project the library Crazyswarm was declared as deprecated and work began on Crazyswarm 2. Unfortunately, Crazyswarm 2 was declared stable and open for the public in January 2023, which was too late for such a signifacant change. The team behind Crazyswarm 2 is promising backwards compatability, but due to the project being still in early stages of development, this is not guaranteed. If running the trained model experimentaly on drones in the real world, please take care of this note. 
