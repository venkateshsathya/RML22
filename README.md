# AMCDataset
### Dataset generation code for modulation classification

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
