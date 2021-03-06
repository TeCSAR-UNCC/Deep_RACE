# Deep Learning Reliability Awareness of Converters at the Edge (Deep RACE)
![POWERED BY TeCSAR](https://raw.githubusercontent.com/TeCSAR-UNCC/Deep_RACE/master/logo/tecsarPowerBy.png)

Deep RACE is a real-time reliability modeling and assessment of power semiconductor devices embedded into a wide range of smart power electronics systems. Deep RACE departures from classical learning and statistical modeling to deep learning based data analytics, combined with full system integration for scalable real-time reliability modeling and assessment. In this regard, it leverages the Long Short-Term Memory (LSTM) networks as a branch of Recurrent Neural Networks (RNN) to aggregate reliability across many power converters with similar underlying physic. Also, It offers real-time online assessment by selectively combining the aggregated training model with device-specific behaviors in the field.
## Prerequisites
First make sure you have already installed pip3, Tkinter, and git tools:
``` bash
sudo apt install git python3-pip python3-tk
```
You should also install follwoing python packages:
```bash
sudo -H pip3 install tensorflow scipy matplotlib seaborn
```
## Installation
You only need to clone the Deep RACE repository:
```bash
git clone https://github.com/TeCSAR-UNCC/Deep_RACE
```
## Training the network models
Change the path to the `Deep_RACE` directory and run the `train.py`:
```bash
cd Deep_RACE
python3 ./train.py
```

All the training models will be saved automatically in `./inference_models/` folder. You can load them by running `inference.py` file.

### Prediction output
The `./train.py` will generate and save the predition out put in a text file. The file name is based on the selected MOSFET device number. As an instance, a text file with the name of `./prediction_output/res_dev2.txt` will be generated for `dev#2`.

### Testing different MOSFET devices
You can test different devices from `RoIFor5Devs.mat` by altering [this line](https://github.com/TeCSAR-UNCC/Deep_RACE/blob/68688f2b89a651f0985364c74c2ae949a696338b/train.py#L69) in the `./train.py`.


## Citing Deep RACE
Please cite the Deep RACE if it helps your research work.
```
@ARTICLE{8629973,
author={M. {Baharani} and M. {Biglarbegian} and B. {Parkhideh} and H. {Tabkhi}},
journal={IEEE Internet of Things Journal},
title={Real-Time Deep Learning at the Edge for Scalable Reliability Modeling of Si-MOSFET Power Electronics Converters},
year={2019},
volume={6},
number={5},
pages={7375-7385},
keywords={electronic engineering computing;Internet of Things;Kalman filters;MOSFET;neural nets;power aware computing;power convertors;power electronics;real-time systems;reliability;scalable decentralized devices-specific reliability monitoring;MOSFET convertors;Internet-of-Things devices;real-time deep learning processing capabilities;Deep RACE solution;MOSFET data;1.87-W computing power;edge IoT device;scalable reliability modeling;advanced high-frequency power converters;active reliability assessment;power electronic devices;real-time reliability modeling;high-frequency MOSFET power electronic converters;edge node;real-time reliability awareness;deep learning algorithmic solution;collective reliability training;collective MOSFET converters;device resistance changes;edge-to-cloud solution;Si-MOSFET power electronics converters;deep learning reliability awareness;Kalman filter;particle filter;Reliability;Real-time systems;Deep learning;MOSFET;Power electronics;Predictive models;Degradation;Deep learning;high-frequency power converter;long short-term memory (LSTM);MOSFET;reliability modeling},
doi={10.1109/JIOT.2019.2896174},
ISSN={2372-2541},
month={Oct},}
```

## Author
* Reza Baharani:  *Python code* - [My personal webpage](https://rbaharani.com/)
## License
Copyright (c) 2018, University of North Carolina at Charlotte All rights reserved. - see the [LICENSE](https://raw.githubusercontent.com/TeCSAR-UNCC/Deep_RACE/master/LICENSE) file for details.
## Acknowledgments

The five Si-MOSFET ΔR<sub>ds(on)</sub> are extracted from NASA MOSFET Thermal Overstress Aging Data Set which is available [here](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/). Please cite their paper if you are going to use their data samples. Here is its BibTeX:
```
@article{celaya2011prognostics,
title={{Prognostics of power {MOSFET}s under thermal stress accelerated aging using data-driven and model-based methodologies}},
author={Celaya, Jose and Saxena, Abhinav and Saha, Sankalita and Goebel, Kai F},
year={2011}
}
```
