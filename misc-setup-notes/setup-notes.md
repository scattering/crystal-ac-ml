# Miscellanious Setup Notes

## Locally install Bumps

    $ git clone https://github.com/bumps/bumps.git
    $ pip install --user bumps

Then access bumps form ~/.local/bin/bumps

## Bumps from the command line

Run a dream fit

    $ bumps file/path/ --fit=dream --steps=1000 --burn=500 --store=M2 --parallel=8 --batch

 * parallel: set equal to the number of gpus to use
 * batch: don't display the graphs at the end

## Graphics on Sparkle/GPU

 * Download xming on your machine with graphics and run it - this needs to be active to forward graphics to your machine
 * Go into Putty settings: connection/SSH/X11 and enable X11 forwarding
 * Run a docker container with this tag:
    -v /storage/yourDir/:/path/in/container/
   This mounts the storage directory to your docker container
 * Write images to a png in /path/in/container/
 * Exit the docker container and access the png files in /storage/yourDir/
 * To view: use Eye of Gnome
    $ apt install eog
    $ eog filename.png

## Docker

For Ubuntu, follow https://docs.docker.com/install/linux/docker-ce/ubuntu/#supported-storage-drivers for installation instructions.
 
Create a docker container: https://www.jamescoyle.net/how-to/1503-create-your-first-docker-container

Run a Ubuntu container

    $  docker run -ti ubuntu:18.04 /bin/bash
---

Run a docker container with tensorflow, GPU: https://hub.docker.com/r/tensorflow/tensorflow/

    $  docker pull tensorflow/tensorflow:latest-gpu
    $  nvidia-docker run -it -p 8888:8888 tensorflow/tensorflow:latest-gpu

Go to localhost:8888 to access the jupyter notebook server

To run without starting a server (i.e, get straight into the command line), run:

    $  nvidia-docker run -it tensorflow/tensorflow:latest-gpu /bin/bash
---

To save progress, exit a container, but dont stop it. Then use the commit command to save an image: https://docs.docker.com/engine/reference/commandline/commit/

    root~$ exit
    $ docker ps -a  #list all containers to see what your containers name/id is
    $ docker commit container-name images-name
---

DNS Errors

If apt update/related cmds fail with warning that it could not resolve ubuntu.com, failed to download updates: 
https://stackoverflow.com/questions/24991136/docker-build-could-not-resolve-archive-ubuntu-com-apt-get-fails-to-install-a

Fix:

Check what DNS servers the host machine uses:

    $ nmcli dev show | grep 'IP4.DNS'

Edit /etc/docker/daemon.json with:

    { "dns": ["your ip adresses"] }

Restart docker

    $ sudo service docker restart

