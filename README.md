<h1 align="center"> CS_FD_TRAINING</h1> 

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![YAML](https://img.shields.io/badge/yaml-%23ffffff.svg?style=for-the-badge&logo=yaml&logoColor=151515)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

---

### Table of Contents

- [Description](#description)
- [How to run](#how-to-run)
- [Training](#training)
- [Quality Code](#quality-code)
- [Logging](#logging)

---

### Description

This repo is part of a project to predict the probability of fraud in a credit card transaction. The solution separates training and prediction/transformation in two different repositories. This repository is the training repository. The goal is to **provide an isolated training environment for data scientists to experiment with different models and parameters**.

The environment provided here is a **Docker** container, so that the training can be done in a **reproducible** way, independently of the services used by the team. The predict/transform endpoints are defined in the [cs_fd_mleng](https://github.com/AlexandreH13/cs_fd_mleng) repository.

---

### How to run

The sugestion is to build the docker image and run the container.

```bash
docker build -t project_tag .
```

After the building, you can run the container with the following command:

```bash
docker run -it image_name
```
---

### Training

This solution suggests using a **multi repo** structure. Before training, the data used needs to be transformed. This can be done using the `/transform` endpoint located in the [cs_fd_mleng](https://github.com/AlexandreH13/cs_fd_mleng) repository. NOTE: We are not using any storage service, neither we are versioning the csv files. The data in this repository is persisted in the `src/data/processed` folder.

The model training is **parameterized** by the `train_params.yaml` file in order to facilitate the experimenting process for data scientists. For our purpose, the training is divided for each **state**. In that sense, the main parameters are:

- Model: The model to be used. Currently, the model is **Logistic Regression**.
- State_Name: The name of the state.

Please, check the `train_params.yaml` file for more details.

To run an experiment, you can use the `train.sh` script:

```bash
./train.sh
```

NOTE: The model is currently persisted in the `artifacts` folder. Ideally, the model should be persisted in a storage location.

---

### Quality Code

For this project, we used the following tools:

- [isort](https://github.com/PyCQA/isort): sort imports
- [black](https://github.com/psf/black): format code

---

### Logging

The logging module is used to log the information of the application. Error information and the metrics are logged in the `app.log` file. The log file is located in the `src/logs` folder. NOTE: The log file is not versioned.

---