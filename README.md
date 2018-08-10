# Actor Critic Algorithm for Pycrysfml

Algorithm was adapted from this tutorial: https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69

## Setting up Dependancies

These instructions are for a docker container containing all of the dependancies. If you are using ubuntu you could follow these instructions outside of a container.

### Set up a container with openai gym, keras, and tensorflow

You need nvidia-docker installed to utilize the gpu from a docker container.

Start a container

    $ nvidia-docker run -it -v /path/to/storage/:/path/in/container/ tensorflow/tensorflow:latest-gpu /bin/bash

Within the container, set up gym and keras

    $ apt update
    $ pip install gym
    $ pip install keras

### Set up the code

Clone crystal-ac-ml and pycrysfml. For ease of transition, either put them in the home directory or change all filepaths in hkl.py.

    $ apt install git
    $ git clone https://github.com/scattering/crystal-ac-ml.git

Follow instructions in pycrysfml/doc/Ubuntu_deps.txt to set up pycrysfml

Replace pycrysfml/hklgen/sxtal_model.py with the sxtal_model.py file in this repository. This updates the model class to accept empty lists for the observed data, so you can create a model without giving it all of the initial data. 

Replace the prnio.cfl file in pycrysfml/hklgen/examples/sxtal/prnio.cfl with the prnio_optimized.cfl file. This file contains optimized atomic position values from fitting the data with FullProf. It will allow you to get better fits on the praseodymium nickolate crsytal model when only fitting a single parameter.

Edit the filepath in the top of hkl.py to match your directory structure (i.e., enter the path to the prnio.int file in pycrysfml).

Copy __init__.py and hkl.py into /usr/local/lib/python2.7/dist-packages/gym/envs/ to register our environment (hkl.py) with openai gym.

Now you are all set up to run actor_critic.py from anywhere in your filesystem.

### Run with SLURM

If you are going to run the code remotely, using slurm rather than directly from inside the container, you need to first save this container as an image.

Exit the container and use 'docker ps -a' to get the container id. Then commit it.

    $ docker commit cont-id image-name

Now that the image is commited, it can be run from slurm

cd into your storage folder. Then run:

    /storage/you$  srun --mem=0 --nodelist=ncnrgpu1 bash -c '/storage/etc/sudocker run runtime=nvidia -d -v /storage/you/:/path/in/container/ img-name python /path/to/actor_critic.py'

The parts of the srun command are:

    --nodelist: the node to run on
    bash -c: run a bash command
    --runtime=nvida: use nvidia docker, remove this to run with regular docker


### Additional Notes for SLURM

To move containers around nodes, save the container to a tar file using 

    $ docker save container-id
