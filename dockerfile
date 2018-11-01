FROM tensorflow/tensorflow:latest-gpu

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR "/home"

RUN apt update

RUN apt upgrade -y

RUN pip install gym

RUN pip install keras

RUN apt install -y build-essential \
        gfortran \
        python \
        python-dev \
        python-numpy \
        python-scipy \
        python-matplotlib \
        python-pip \
        subversion \
        git \
        swig

RUN git clone https://github.com/bumps/bumps.git

RUN pip install ./bumps

RUN git clone https://github.com/scattering/crystal-ac-ml.git

RUN git clone https://github.com/scattering/pycrysfml.git

WORKDIR "/home/pycrysfml"

RUN ./build.sh

WORKDIR "/home/crystal-ac-ml"

RUN cp /home/crystal-ac-ml/sxtal_model.py /home/pycrysfml/hklgen/

RUN cp /home/crystal-ac-ml/prnio-optimized.cfl /home/pycrysfml/hklgen/examples/sxtal/prnio.cfl

RUN cp /home/crystal-ac-ml/__init__.py /usr/local/lib/python2.7/dist-packages/gym/envs/

RUN cp /home/crystal-ac-ml/hkl.py /usr/local/lib/python2.7/dist-packages/gym/envs/

CMD /bin/bash
