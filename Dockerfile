FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV MYPATH /home/ID-disentanglement
WORKDIR ${MYPATH}

VOLUME /home/ID-disentanglement