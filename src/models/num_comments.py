#!/usr/bin/python3
# num_comments.py

import os
import pickle
import optuna
import xgboost
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection, metrics


iterative_params_and_accuracies = []


def xgb_objective(trial, X, y, estimator):
    n_estimators = trial.suggest_int('n_estimators', 40, 280, 40)
    max_depth = trial.suggest_int('max_depth', 2, 12, 2)
    learning_rate = trial.suggest_loguniform('learning_rate', 0.001, 0.1)
    params = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'learning_rate': learning_rate
    }
    estimator.set_params(**params)
    accuracy = model_selection.cross_val_score(estimator, X, y).mean()
    iterative_params_and_accuracies.append([params, accuracy])

    # Try without this and use if accuracy is negative
    '''
    if accuracy < 0: 
        accuracy = model_selection.cross_val_score(estimator, X, y, scoring='neg_mean_squared_error').mean()
        return accuracy
    '''
    return accuracy


def optimize(estimator, X_train, X_val, y_train, y_val, timeout):
    estimator.fit(X_train, y_train)
    study = optuna.create_study(direction='maximize')  # maximize or minimize?
    study.optimize(lambda trial: xgb_objective(trial, X_val, y_val, estimator), timeout=timeout)
    estimator.set_params(**study.best_params)


def generate_and_evaluate_model(dataset, hyperparameterization=False, timeout=300):
    data = pd.read_csv(dataset)
    X = data.drop(labels='n_comments', axis=1)  # change this with actual col name
    y = data['n_comments']  # change this with actual col name

    # 70% train, 15% eval, 15% test
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.15)
    X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train, y_train, test_size=(0.15/0.85))

    if hyperparameterization:
        model = xgboost.XGBRegressor()
        optimize(model, X_train, X_val, y_train, y_val, timeout)
    else:
        model = xgboost.XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.01)
        model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = metrics.mean_squared_error(y_test, predictions)
    return model, accuracy


def graph_feature_model(model):
    feature_importance = model.get_booster().get_score(importance_type='weight')
    print(len(feature_importance))
    sort_features = sorted(feature_importance, key=lambda x: x[1])
    print(len(sort_features))
    fexists = False
    items = []
    for feature in sort_features:
        if len(items) < 8:
            if 'emb_feature' in feature:
                if not fexists:
                    items.append(feature)
                    fexists = True
            else:
                items.append(feature)

    values = [feature_importance[i] for i in items]
    data = pd.DataFrame(data=values, index=items, columns=[
        "score"]).sort_values(by="score", ascending=False)
    data.plot(kind='barh')
    plt.show()


def graph_params_and_accuracies():
    pass


if __name__ == '__main__':
    model, accuracy = generate_and_evaluate_model(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'nyt-articles-2020-final-dataset.csv'), hyperparameterization=True, timeout=600)
    print(accuracy)
    print(model.get_params())
    print(iterative_params_and_accuracies)
    graph_feature_model(model)
    if len(iterative_params_and_accuracies) > 0:
        # with open('params_accuracies_comments.pkl', 'wb') as f:
        #     pickle.dump(iterative_params_and_accuracies, f)
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'num_comments_model.pkl'), 'wb') as file:
            pickle.dump(model, file)
