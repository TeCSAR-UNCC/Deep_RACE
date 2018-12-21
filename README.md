# Deep Learning Reliability Awareness of Converters at the Edge (Deep RACE)
![POWERED BY TeCSAR](https://raw.githubusercontent.com/TeCSAR-UNCC/Deep_RACE/master/logo/tecsarPowerBy.png)

Deep RACE is a real-time reliability modeling and assessment of power semiconductor devices embedded into a wide range of smart power electronics systems. Deep RACE departures from classical learning and statistical modeling to deep learning based data analytics, combined with full system integration for scalable real-time reliability modeling and assessment. In this regard, it leverages the Long Short-Term Memory (LSTM) networks as a branch of Recurrent Neural Networks (RNN) to aggregate reliability across many power converters with similar underlying physic. Also, It offers real-time online assessment by selectively combining the aggregated training model with device-specific behaviors in the field.
## Prerequisites
First make sure you have already install pip3, Tkinter, and git tools:
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
## Running the test
Change the path to the `Deep_Race` directory and run the application:
```bash
cd Deep_RACE
./dR.py
```
### Testing different MOSFET devices
You can test different devices from `RoIFor5Devs.mat` by altering [this line](https://github.com/TeCSAR-UNCC/Deep_RACE/blob/faa2f1aed804ba607b24fe0e2e6b9eb724fb0982/dR.py#L69) in the `./dR.py`.

## Author
* Reza Baharani:  *Python code* - [My personal webpage](https://rbaharani.com/)
## License
This project is licensed under the MIT License - see the [LICENSE](https://raw.githubusercontent.com/TeCSAR-UNCC/Deep_RACE/master/LICENSE) file for details.
## Acknowledgments

The five Si-MOSFET ΔR<sub>ds(on)</sub> are extracted from NASA MOSFET Thermal Overstress Aging Data Set which is available [here](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/). Please cite their paper if you are going to use their data samples. Here is its BiBTeX:
```
@article{celaya2011prognostics,
title={{Prognostics of power {MOSFET}s under thermal stress accelerated aging using data-driven and model-based methodologies}},
author={Celaya, Jose and Saxena, Abhinav and Saha, Sankalita and Goebel, Kai F},
year={2011}
}
```
