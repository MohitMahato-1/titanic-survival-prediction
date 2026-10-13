from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from titanic import ROOT, add_features, make_pipeline, read_data, split_data


@pytest.fixture
def passengers():
    return read_data(ROOT / "data/train.csv")


def test_split_is_disjoint_reproducible_and_keeps_both_classes(passengers):
    development, holdout = split_data(passengers)
    repeated, _ = split_data(passengers)
    assert len(development) == 712 and len(holdout) == 179
    assert not set(development.PassengerId) & set(holdout.PassengerId)
    assert set(development.PassengerId) | set(holdout.PassengerId) == set(passengers.PassengerId)
    assert development.PassengerId.tolist() == repeated.PassengerId.tolist()
    assert set(holdout.Survived) == {0, 1}


def test_imputation_is_fitted_on_training_only_and_inputs_are_not_mutated(passengers):
    training = passengers.iloc[:8].copy()
    training["Age"] = [10, 20, 30, np.nan, 10, 20, 30, np.nan]
    before = training.copy(deep=True)
    model = make_pipeline(LogisticRegression(max_iter=1000)).fit(training, training.Survived)
    imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
    assert imputer.statistics_[0] == 20
    unseen = passengers.iloc[8:10].copy()
    unseen["Age"] = [1000, np.nan]
    unseen["Embarked"] = "UNSEEN_PORT"
    model.predict_proba(unseen)
    assert imputer.statistics_[0] == 20
    pd.testing.assert_frame_equal(before, training)


def test_features_cannot_read_target_or_identifier(passengers):
    features = add_features(passengers)
    without_target = add_features(passengers.drop(columns="Survived"))
    pd.testing.assert_frame_equal(features, without_target)
    assert "PassengerId" not in features and "Survived" not in features
    assert (features.FamilySize == passengers.SibSp + passengers.Parch + 1).all()


def test_pipeline_predicts_unlabeled_kaggle_data(passengers):
    development, _ = split_data(passengers)
    model = make_pipeline(LogisticRegression(max_iter=2000)).fit(development, development.Survived)
    test = read_data(ROOT / "data/test.csv", labeled=False)
    probabilities = model.predict_proba(test)
    assert probabilities.shape == (418, 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_duplicate_ids_and_invalid_labels_are_rejected(passengers, tmp_path):
    bad = pd.concat([passengers.iloc[:2], passengers.iloc[:1]])
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        read_data(path)
    bad = passengers.iloc[:2].copy()
    bad.loc[bad.index[0], "Survived"] = 9
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="only 0 or 1"):
        read_data(path)
