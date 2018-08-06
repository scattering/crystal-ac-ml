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

Clone crystal-ac-ml and pycrysfml

    $ apt install git
    $ git clone https://github.com/scattering/crystal-ac-ml.git
    $ git clone https://github.com/scattering/pycrysfml.git

Follow instructions in pycrysfml/doc/Ubuntu_deps.txt to set up pycrysfml
#### todo add prnio.clf optimized
#### todo fix sxtal model

Edit the filepath in the top of hkl.py to match your directory structure (i.e., enter the path to the prnio.int file in pycrysfml).

Copy __init__.py and hkl.py into /usr/local/lib/python2.7/dist-packages/gym/envs/ to register our environment (hkl.py) with openai gym.

Now you are all set up to run actor_critic.py from anywhere in your filesystem.
