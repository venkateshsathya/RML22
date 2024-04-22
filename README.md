### RML22 Dataset direct download [link](https://drive.google.com/drive/folders/1dEv6gPwPahUfFFRYYxvp3i5M34D7KI9J?usp=sharing).

This code is inspired by and reuses part of the code from [RADIOML](https://github.com/radioML/dataset), [DeepSig](https://www.deepsig.ai/datasets)
![alt text](https://github.com/venkateshsathya/RML22/blob/main/GitHubREADME_1.png?raw=true)


### RML22 Dataset generation code details

##### **Instructions to run the code.**

1. Copy the files under the folder "code" to a local folder "RML22_code".
2. Copy the Podcast.wav file from this [link](https://drive.google.com/drive/folders/1dEv6gPwPahUfFFRYYxvp3i5M34D7KI9J?usp=sharing) to "RML22_code". This file is the information source for analog modulation types.
4. This notebook contains cells with code to generate dataset, train on CNN architecure, test and plot results: accuracy versus SNR and confusion matrix. By default it generates RML22 dataset in the first of the notebook. The second cell trains a CNN model with an architecture and training parameters as shown below

![alt text](https://github.com/venkateshsathya/RML22/blob/main/DL_Architecture_TrainingParameters.png?raw=true)
5. The third cell tests and plots accuracy versus SNR and a confusion matrix.
6. The five types of datasets that could be generated are clean(no artifacts), with thermal noise over clean dataset (AWGN), with clock effects(SRO/CFO/phase offset) on clean dataset, with fading over clean dataset and the final one RML22 with all artifacts. The default code generates RML22.

### To generate your own dataset, you need the following GNU radio dependencies. You can also skip installing the dependencies if you do not wish to generate your own dataset and directly download [RML22](https://drive.google.com/drive/folders/1dEv6gPwPahUfFFRYYxvp3i5M34D7KI9J?usp=sharing) and train your model on this.

##### **Instructions to install GNURadio module and out-of-tree GR-MAPPER module.**  :cowboy_hat_face:
```
1. conda create --name gnuradio
2. conda activate gnuradio
3. conda install -c conda-forge gnuradio=3.8.3
4. conda install -c conda-forge scipy
5. conda install -c conda-forge matplotlib
6. git clone https://github.com/myersw12/gr-mapper.git
7. cd gr-mapper && mkdir build && cd build
8. chmod -R 777 ../../
9. conda install -c conda-forge gnuradio-build-deps
10. conda activate $CONDA_DEFAULT_ENV
11. conda install -c conda-forge cppunit
12. cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DLIB_SUFFIX="" ..
13. cmake --build .
14. cmake --build . --target install

```

##### **List of potential issues.** :confused:

1. Gr-mapper may or may not work with GNURadio 3.9
   See details in [link](https://github.com/conda-forge/gnuradio-feedstock/issues/42)
2. The cmake instructions in this [link](https://github.com/myersw12/gr-mapper) does not work. Please do not use them.
   Please follow instructions given above which were taken from [CondaInstall](https://wiki.gnuradio.org/index.php/CondaInstall).
3. Any issues with cmake: good link to check out is this [link](https://github.com/conda-forge/gnuradio-feedstock/issues/49)
   "ryanvolz" user on github seems to respond to related queries. 
4. If you are getting solving environment error, then do the following "conda config --set channel_priority false". This can help resolve the issue. [link](https://stackoverflow.com/questions/57518050/conda-install-and-update-do-not-work-also-solving-environment-get-errors)
5. If you encounter errors such as *gr::vmcircbuf_sysv_shm: shmget (2): No space left on device*, you might want to run the program only from command line and not an IDE. We have noticed issues such as low memory allocation when using an IDE that causes this issue. You can also potentially fix the issue by following instrucitons in this [link](https://stackoverflow.com/questions/24486153/gnu-radio-python-script-shmget-2-no-space-left-on-device)
