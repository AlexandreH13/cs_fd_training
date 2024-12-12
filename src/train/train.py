import logging
import os
import pickle

logging.basicConfig(filename="src/logs/app.log", level=logging.DEBUG)

import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


class Train:
    """
    Class responsible for training the model.

    Returns:
        _type_: _description_
    """

    def __init__(self):
        self.CONFIG_PATH = "src/train/"
        self.config = self.load_config("train_params.yaml")
        self.state = self.config["state"]
        self.model = self.config["model"]

    def load_config(self, config_name):
        """Load the config file.

        Args:
            config_name (str): Name of the config file

        Returns:
            dict: Config file
        """
        logging.info(f"Carregando o arquivo de configuração: {config_name}")
        with open(os.path.join(self.CONFIG_PATH, config_name)) as file:
            config = yaml.safe_load(file)

        return config

    def get_model(self):
        """Select the model to train and load the parameters located in the train_params.yaml file."""
        logging.info(f"Selecionando o modelo: {self.model}")
        models_dct = {"lr": LogisticRegression, "rf": RandomForestClassifier}

        model_selec = models_dct[self.model]
        model_params = self.config[self.model]
        return model_selec(**model_params)

    def get_training_data(self):
        """Load the training data.

        Returns:
            pd.DataFrame: Training data
        """
        logging.info(f"Carregando os dados de treinamento para o estado: {self.state}")
        try:
            df = pd.read_csv(
                f'{self.config["processed_data_dir"]}/train_{self.config["state"]}.csv'
            )
        except FileNotFoundError as e:
            logging.error(
                f"Dataset de treinamento não encontrado. O mesmo pode não ter sido processado. {e}"
            )
        return df

    def sample_data(self):
        """Split the data into training and testing sets.

        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        logging.info(f"Dividindo os dados para o estado: {self.state}")
        df = self.get_training_data()
        X = df.drop(["is_fraud", "state"], axis=1)
        y = df["is_fraud"]
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config["sample"]["test_size"],
            random_state=self.config["sample"]["random_state"],
        )
        return X_train, X_test, y_train, y_test

    def train_model(self):
        """Train the model. Save the model in the artifacts folder."""
        logging.info(f"Iniciando o treinamento para o estado: {self.state}")
        X_train, X_test, y_train, y_test = self.sample_data()
        model = self.get_model()
        model.fit(X_train, y_train)

        self.evaluate_model(model, X_test, y_test)

        self.persist_model(model)

    def evaluate_model(self, model, X_test, y_test):
        """Evaluate the model.

        Args:
            model (object): Model to be evaluated
            X_test (pd.DataFrame): Testing data
            y_test (pd.DataFrame): Testing data
        """
        logging.info(f"Avaliando o modelo para o estado: {self.state}")
        metrics_dic = {"f1": f1_score, "auc": roc_auc_score}

        y_pred = model.predict(X_test)
        for metric in self.config["evaluation"]:
            logging.info(
                f"Métrica: {metric}, Valor: {metrics_dic[metric](y_test, y_pred)}"
            )

    def persist_model(self, model):
        """Save the model in the artifacts folder.

        Args:
            model (object): Model to be saved
        """
        logging.info(f"Salvando o modelo para o estado: {self.state}")
        try:
            pickle.dump(
                model, open(f'{self.config["models_dir"]}/model_{self.state}.pkl', "wb")
            )
        except Exception as e:
            logging.error(f"Erro ao persistir o modelo: {e}")

    def run(self):
        """Execute the training process."""
        self.train_model()


if __name__ == "__main__":
    train = Train()
    train.run()
