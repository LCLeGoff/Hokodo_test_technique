import numpy as np
import pandas as pd
from sklearn import linear_model, metrics, ensemble


class ModelBase:
    """
    Class giving basic methods to Model class (inheritance)
    """

    def __init__(
            self, df_data: pd.DataFrame, df_validation: pd.DataFrame, name_to_predict: str = 'Winner',
            feature_names: str = None, train_pc: float = 0.9):
        """

        :param df_data: dataframe containing the features and the Y. Train and test dataset  would be extract from it
        :param df_validation: dataframe used for model validation
        :param name_to_predict: name of the Ys
        :param feature_names: names of the features used to train
        :param train_pc: percentage of df_data used for training
        """
        self.df_data = df_data.copy().reset_index(drop=True)
        self.df_validation = df_validation.copy().reset_index(drop=True)
        self.name_to_predict = name_to_predict

        if feature_names is None:
            self.feature_names = list(df_data.columns)
            self.feature_names.remove(name_to_predict)
        else:
            self.feature_names = feature_names

        self.train_size = int(len(self.df_data) * train_pc)
        self.test_size = int(len(self.df_data) * (1-train_pc))

        shuffle_index = list(df_data.index)
        np.random.shuffle(shuffle_index)

        self.df_data = self.df_data.loc[shuffle_index].reset_index(drop=True)

        self.x_train = self.df_data.loc[:self.train_size, self.feature_names]
        self.x_test = self.df_data.loc[self.train_size:self.train_size + self.test_size, self.feature_names]

        self.y_train = self.df_data.loc[:self.train_size, self.name_to_predict]
        self.y_test = self.df_data.loc[self.train_size:self.train_size + self.test_size, self.name_to_predict]

        self.x_validate = self.df_validation.loc[:, self.feature_names]
        self.y_validate = self.df_validation.loc[:, self.name_to_predict]

        # filled by the inheritor model class
        self.y_train_predict = None
        self.y_test_predict = None
        self.y_validate_predict = None
        self.model = None

    def accuracy(self, validation: bool = False):
        """
        compute the train, test and validation accuracy
        :param validation: if True compute validation accuracy instead of train and test accuracy
        :return: train and test accuracy or validation accuracy
        """
        if validation is True:
            validation_accuracy = np.round(metrics.accuracy_score(self.y_validate, self.y_validate_predict), 3)
            return validation_accuracy
        else:
            train_accuracy = np.round(metrics.accuracy_score(self.y_train, self.y_train_predict), 3)
            test_accuracy = np.round(metrics.accuracy_score(self.y_test, self.y_test_predict), 3)
            return train_accuracy, test_accuracy

    def confusion_matrix(self, validation: bool = False):
        """
        Compute the train, test and validation confusion matrix
        :param validation: if True compute validation confusion matrix instead of train and test confusion matrix
        :return: train and test confusion matrix or validation confusion matrix
        """

        if validation is True:

            validation_confusion_matrix = metrics.confusion_matrix(
                self.y_validate, self.y_validate_predict, normalize='true')

            validation_confusion_matrix = pd.DataFrame(
                validation_confusion_matrix, columns=['Negative', 'Positive'], index=['False', 'True']).round(4)

            return validation_confusion_matrix
        else:
            train_confusion_matrix = metrics.confusion_matrix(self.y_train, self.y_train_predict, normalize='true')
            train_confusion_matrix = pd.DataFrame(
                train_confusion_matrix, columns=['Negative', 'Positive'], index=['False', 'True']).round(4)

            test_confusion_matrix = metrics.confusion_matrix(self.y_test, self.y_test_predict, normalize='true')
            test_confusion_matrix = pd.DataFrame(
                test_confusion_matrix, columns=['Negative', 'Positive'], index=['False', 'True']).round(4)

            return train_confusion_matrix, test_confusion_matrix

    def predict(self):
        """
        Compute the prediction of the train, test and validation Y.
        """

        self.y_train_predict = self.model.predict(self.x_train)
        self.y_test_predict = self.model.predict(self.x_test)
        self.y_validate_predict = self.model.predict(self.x_validate)


class LogisticRegressionClass(ModelBase):
    """
    Logistic regression model class
    """

    def __init__(
            self, df_data: pd.DataFrame, df_validation: pd.DataFrame, name_to_predict: str = 'Winner',
            feature_names: str = None, train_pc: float = 0.9):
        """
        :param df_data: inherited from ModelBase
        :param df_validation: inherited from ModelBase
        :param name_to_predict: inherited from ModelBase
        :param feature_names: inherited from ModelBase
        :param train_pc:  inherited from ModelBase
        """

        self.name = 'LogisticRegression'

        ModelBase.__init__(
            self, df_data=df_data, df_validation=df_validation, name_to_predict=name_to_predict,
            feature_names=feature_names, train_pc=train_pc)
        self.model = linear_model.LogisticRegression(fit_intercept=False)
        self.model.fit(self.x_train, self.y_train)
        self.predict()


class RandomForestClass(ModelBase):

    def __init__(
            self, max_depth: float, n_estimators: float,
            df_data: pd.DataFrame, df_validation: pd.DataFrame, name_to_predict: str = 'Winner',
            feature_names: str = None, train_pc: float = 0.9):
        """
        :param max_depth: maximum depth of the trees
        :param n_estimators: number of trees
        :param df_data: inherited from ModelBase
        :param df_validation: inherited from ModelBase
        :param name_to_predict: inherited from ModelBase
        :param feature_names: inherited from ModelBase
        :param train_pc:  inherited from ModelBase
        """

        self.name = 'RandomForest'

        self.max_depth = max_depth
        self.n_estimators = n_estimators
        ModelBase.__init__(
            self, df_data=df_data, df_validation=df_validation, name_to_predict=name_to_predict,
            feature_names=feature_names, train_pc=train_pc)

        self.model = ensemble.RandomForestClassifier(max_depth=self.max_depth, n_estimators=self.n_estimators)
        self.model.fit(self.x_train, self.y_train)
        self.predict()


class HyperparameterExplorationClass:
    """
    Class exploring hyperparameters of a model
    """
    def __init__(
            self, model_class, feature_names_list: list[list], para_tuple_list: list[tuple], para_names: list[str],
            df_data: pd.DataFrame, df_validation: pd.DataFrame, name_to_predict: str = 'Winner',
            train_pc: float = 0.8):
        """

        :param model_class: model class trained on each hyperparameter
        :param feature_names_list: list of feature name list used fro training
        :param para_tuple_list: list of the hyperparameter tuple
        :param para_names: names of the hyperparameters
        :param df_data: dataframe containing the features and the Y. Train and test dataset  would be extract from it
        :param df_validation: dataframe used for model validation
        :param name_to_predict: name of the Ys
        :param train_pc: percentage of df_data used for training
        """

        self.model_class = model_class
        self.feature_names_list = feature_names_list
        self.para_tuple_list = para_tuple_list
        self.para_names = para_names
        self.n_para = len(self.para_names)

        self.name_to_predict = name_to_predict
        self.train_pc = train_pc

        self.df_data = df_data
        self.df_validation = df_validation

        self.exploration_dict = None

        self.df_accuracies = pd.DataFrame(columns=['train_accuracy', 'test_accuracy'])

    def exploration(self):
        """
        Will train model instance on each parameter tuple
        """
        self.exploration_dict = dict()

        k = 0
        if self.n_para == 0:

            for feature_names in self.feature_names_list:
                model = self.model_class(
                    df_data=self.df_data, df_validation=self.df_validation, name_to_predict=self.name_to_predict,
                    feature_names=feature_names, train_pc=self.train_pc)

                model_dict = model.__dict__.copy()
                model_dict['train_accuracy'], model_dict['test_accuracy'] = model.accuracy()
                model_dict['train_confusion'], model_dict['test_confusion'] = model.confusion_matrix()

                self.df_accuracies.loc[k] = (model_dict['train_accuracy'], model_dict['test_accuracy'])
                model_dict['model'] = model
                model_dict['id'] = k
                self.exploration_dict[k] = model_dict
                k += 1
        else:
            for feature_names in self.feature_names_list:
                for paras in self.para_tuple_list:
                    para_dict = {self.para_names[i]: paras[i] for i in range(self.n_para)}

                    model = self.model_class(
                        df_data=self.df_data, df_validation=self.df_validation, name_to_predict=self.name_to_predict,
                        feature_names=feature_names, train_pc=self.train_pc, **para_dict)

                    model_dict = model.__dict__.copy()
                    model_dict['train_accuracy'], model_dict['test_accuracy'] = model.accuracy()
                    model_dict['train_confusion'], model_dict['test_confusion'] = model.confusion_matrix()

                    model_dict['model'] = model
                    model_dict['id'] = k
                    model_dict['para_dict'] = para_dict

                    self.df_accuracies.loc[k] = (model_dict['train_accuracy'], model_dict['test_accuracy'])

                    self.exploration_dict[k] = model_dict
                    k += 1

    def get_best_model_id(self):
        """
        :return: id of the model object with the highest test accuracy
        """
        if self.exploration_dict is None:
            self.exploration()
        best_model_id = self.df_accuracies.sort_values(['test_accuracy']).index[-1]
        return best_model_id

    def get_best_model(self):
        """
        :return: model object with the highest test accuracy
        """
        best_model_id = self.get_best_model_id()
        return self.exploration_dict[best_model_id]

    def print_model(self, i, details=False):
        """
        print model characteristics
        :param i: id of the model to print
        :param details: print confusion matrices if True
        """
        model_dict = self.exploration_dict[i]
        print(model_dict['name'])
        print('features: ', model_dict['feature_names'])
        if 'para_dict' in model_dict:
            print('parameters: ', model_dict['para_dict'])
        print()

        print('## Train accuracy:', model_dict['train_accuracy'])
        if details is True:
            display(model_dict['train_confusion'])

        print('## Test accuracy:', model_dict['test_accuracy'])
        if details is True:
            display(model_dict['test_confusion'])
        print()

    def print_best_model(self, details=False):
        """
        print characteristics of the model with the highest test accuracy
        :param details: print confusion matrices if True
        """

        print('#### Best model ####')

        best_model_id = self.get_best_model_id()
        self.print_model(best_model_id, details=details)

    def print_all_models_equal_or_better_than_speed_model(self, details=False):
        """
        print characteristics of the all model with a test accuracy higher or equal to 0.93 the speed model accuracy
        :param details: print confusion matrices if True
        """
        print('#### All models ####')

        for i in range(len(self.exploration_dict)):
            test_accuracy = self.df_accuracies.loc[i, 'test_accuracy']
            if test_accuracy >= 0.93:
                print('##')
                self.print_model(i, details=details)

    def print_all_models(self, details=False):
        """
        print characteristics of all models
        :param details: print confusion matrices if True
        """
        print('#### All models ####')

        for i in range(len(self.exploration_dict)):
            print('##')
            self.print_model(i, details=details)

    def print_best_model_validation(self):
        """
        print validation accuracy of the model with the highest test accuracy
        """
        print('#### Best model validation####')
        best_model_id = self.get_best_model_id()
        model_dict = self.exploration_dict[best_model_id]
        print('##', model_dict['name'])

        print('Validation accuracy:', model_dict['model'].accuracy(validation=True))
        display(model_dict['model'].confusion_matrix(validation=True))
        print()
        print()

