# Spatio_Temporal_MCPM
Code used in my MSC thesis to implement MCPM model with spatio-temporal data.

# Dependencies
The code was tested on Python 3.7, TensorFlow 1.14, and TensorFlow-Probability 0.7.

# Installation
The code can be downloaded and installed as follows:
```
git clone git@github.com:BenMorris1/Spatio_Temporal_MCPM.git
cd Spatio_Temporal_MCPM
python3 setup.py
```

# Use
The main code is contained in ```Spatio_Temporal_MCPM/mcpm.py```.
The folder ```methods/``` contains functions to run the MCPM and LGCP models.
The experiments discussed in the thesis are contained in the folder ```Experiments/```.
The folder ```Results_visualisation/``` contains jupyter notebooks the produce the plots and performance statistics for each experiment.
The folder ```graphics/``` contains jupyter notebooks used to produce plots used elsewhere in my thesis.


## Acknowledgements
The MCPM model code was built by [Virginia Aglietti](https://warwick.ac.uk/fac/sci/statistics/staff/research_students/aglietti/) to implement the model described in [Efficient Inference in Multi-task Cox Process Models](https://arxiv.org/abs/1805.09781) and can also be found in her repository [here](https://github.com/VirgiAgl/MCPM).
