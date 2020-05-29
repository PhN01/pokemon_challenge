import logging
from typing import Any, Dict
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm


def preprocess_battles(
        battles: pd.DataFrame, weakness: pd.DataFrame, all_pokemon: pd.DataFrame
):

    log = logging.getLogger(__name__)

    # remove pokemon 32 due to duplicate name
    all_pokemon = all_pokemon.loc[all_pokemon.ID != 32, :]

    log.info("Left joining pokemon types.")
    # get types of pokemon 1 from all_pokemon
    battles = (
        pd.merge(
            battles,
            all_pokemon,
            left_on="Name_1",
            right_on="Name",
            how="left"
        )
            .drop(["ID", "Name"], axis=1)
            .rename(columns={"Type_1": "Type_1_1", "Type_2": "Type_2_1"})
    )
    # get types of pokemon 2 from all_pokemon
    battles = (
        pd.merge(
            battles,
            all_pokemon,
            left_on="Name_2",
            right_on="Name",
            how="left",
            suffixes=("", "_2"),
        )
            .drop(["ID", "Name"], axis=1)
            .rename(columns={"Type_1": "Type_1_2", "Type_2": "Type_2_2"})
    )
    # create column with the pair of primary types of the opposing pokemons
    # this will be used for stratification when splitting the data into a
    # train and test set
    battles['Battle_MainType'] = battles.apply(lambda row: f"{row['Type_1_1']}_{row['Type_1_2']}", axis=1)

    log.info("Left joining strength factors according to pokemon types.")
    # remove 'Types' column from weakness matrix and transform to a dataframe
    weakness = weakness.drop("Types", axis=1)
    types = weakness.columns

    weakness_df = pd.DataFrame()
    for i, type_1 in enumerate(types):
        for j, type_2 in enumerate(types):
            weakness_df = weakness_df.append(
                pd.DataFrame(
                    {
                        "type_1": type_1,
                        "type_2": type_2,
                        "strength": weakness.iloc[i, j],
                    },
                    index=[0],
                ),
                ignore_index=True,
            )

    # get strength from weakness_df
    #
    # column names:
    # strength_21_2 refers to the strength factor induced by the type 2 vs.
    # opponent type 1 of pokemon 2.
    for i in range(1, 3):
        for j in range(1, 3):
            battles[f"strength_{i}{j}_1"] = pd.merge(
                battles,
                weakness_df,
                left_on=[f"Type_{i}_1", f"Type_{j}_2"],
                right_on=["type_1", "type_2"],
                how="left",
            ).loc[:, "strength"]
            battles[f"strength_{i}{j}_2"] = pd.merge(
                battles,
                weakness_df,
                left_on=[f"Type_{i}_2", f"Type_{j}_1"],
                right_on=["type_1", "type_2"],
                how="left",
            ).loc[:, "strength"]

    # column names of the newly created strength factor variables
    strength_cols = [
        "strength_11_1",
        "strength_12_1",
        "strength_21_1",
        "strength_22_1",
        "strength_11_2",
        "strength_12_2",
        "strength_21_2",
        "strength_22_2",
    ]
    # since the strength factor work as multiplicative factors, filling NAs with 1
    # should be consistent (1 indicating that no change of the original stats occurs)
    battles.loc[:, strength_cols] = battles.loc[:, strength_cols].fillna(1)

    log.info("Identifying duplicate battles.")
    # this will be used to make sure duplicate battles are all put into the training
    # set to avoid information leakage when training the model

    # create helper variables to identify duplicate battles
    battles['NL_1'] = battles.loc[:, ['Name_1', 'Level_1']].apply(
        lambda row: f"{row['Name_1']}_{row['Level_1']}", axis=1
    )
    battles['NL_2'] = battles.loc[:, ['Name_2', 'Level_2']].apply(
        lambda row: f"{row['Name_2']}_{row['Level_2']}", axis=1
    )
    battles['NL_pair_clean'] = battles.loc[:, ['NL_1', 'NL_2']].apply(
        lambda row: "_".join(sorted(list(row))), axis=1
    )

    # identify duplicates
    battles['duplicate'] = battles.duplicated(subset='NL_pair_clean', keep=False)

    # now we can remove the helper columns, because they are not needed anymore
    battles = battles.drop(['NL_1', 'NL_2', 'NL_pair_clean'], axis=1)


    log.info("Create one-hot encodings of categorical variables")

    # types
    type_cols = ["Type_1_1", "Type_2_1", "Type_1_2", "Type_2_2"]
    battles.loc[:, type_cols] = battles.loc[:, type_cols].fillna("NoType")
    types = battles.Type_1_1.unique().tolist() + ["NoType"]

    type_oh_1 = (
            pd.get_dummies(battles.loc[:, "Type_1_1"].values.tolist() + types).loc[
            : battles.shape[0], types[:-1]
            ]
            + pd.get_dummies(battles.loc[:, "Type_2_1"].values.tolist() + types).loc[
              : battles.shape[0], types[:-1]
              ]
    )
    type_oh_2 = (
            pd.get_dummies(battles.loc[:, "Type_1_2"].values.tolist() + types).loc[
            : battles.shape[0], types[:-1]
            ]
            + pd.get_dummies(battles.loc[:, "Type_2_2"].values.tolist() + types).loc[
              : battles.shape[0], types[:-1]
              ]
    )

    # weather and time
    weather_oh = pd.get_dummies(battles.WeatherAndTime)

    # concatenate one-hot features to battles
    for df in [type_oh_1, type_oh_2, weather_oh]:
        battles = pd.merge(
            battles, df, left_index=True, right_index=True, suffixes=("_1", "_2")
        )
    # drop original categorical variables
    battles = battles.drop(
        ["Type_1_1", "Type_2_1", "Type_1_2", "Type_2_2", "WeatherAndTime"], axis=1
    )

    # convert boolean variables to integers
    battles.Legendary_1 = battles.Legendary_1.astype(int)
    battles.Legendary_2 = battles.Legendary_2.astype(int)

    return battles


def preprocess_available(
        available_pokemon: pd.DataFrame, grandmaster: pd.DataFrame, weakness: pd.DataFrame, all_pokemon: pd.DataFrame
):
    log = logging.getLogger(__name__)

    # remove pokemon 32 due to duplicate name
    all_pokemon = all_pokemon.loc[all_pokemon.ID != 32, :]
    n_available = available_pokemon.shape[0]
    n_opponents = grandmaster.shape[0]

    log.info("Left joining pokemon types.")
    # get types of pokemon 1 from all_pokemon
    available_pokemon = (
        pd.merge(available_pokemon, all_pokemon, left_on="Name_1", right_on="Name", how="left")
            .drop(["ID", "Name"], axis=1)
            .rename(columns={"Type_1": "Type_1_1", "Type_2": "Type_2_1"})
    )
    # get types of grandmaster pokemons from all_pokemon
    grandmaster = (
        pd.merge(grandmaster, all_pokemon, left_on="Name_2", right_on="Name", how="left")
            .drop(["ID", "Name"], axis=1)
            .rename(columns={"Type_1": "Type_1_2", "Type_2": "Type_2_2"})
    )

    log.info("Prepare battles dataframe.")
    # repeat available pokemons 6 times to prepare a dataframe for battles
    # against the grandmaster pokemons (6)
    battles_p1 = pd.DataFrame()
    for i, row in available_pokemon.iterrows():
        battles_p1 = battles_p1.append([row] * n_opponents, ignore_index=True)

    battles_p2 = pd.DataFrame()
    battles_p2 = battles_p2.append([grandmaster] * n_available, ignore_index=True)

    battles = pd.concat((battles_p1, battles_p2), axis=1)
    del battles_p1
    del battles_p2

    log.info("Left joining strength factors according to pokemon types.")
    # remove 'Types' column from weakness matrix and transform to a dataframe
    weakness = weakness.drop("Types", axis=1)
    types = weakness.columns

    weakness_df = pd.DataFrame()
    for i, type_1 in enumerate(types):
        for j, type_2 in enumerate(types):
            weakness_df = weakness_df.append(
                pd.DataFrame(
                    {
                        "type_1": type_1,
                        "type_2": type_2,
                        "strength": weakness.iloc[i, j],
                    },
                    index=[0],
                ),
                ignore_index=True,
            )

    # get strength from weakness_df
    #
    # column names:
    # strength_21_2 refers to the strength factor induced by the type 2 vs.
    # opponent type 1 of pokemon 2.
    for i in range(1, 3):
        for j in range(1, 3):
            battles[f"strength_{i}{j}_1"] = pd.merge(
                battles,
                weakness_df,
                left_on=[f"Type_{i}_1", f"Type_{j}_2"],
                right_on=["type_1", "type_2"],
                how="left",
            ).loc[:, "strength"]
            battles[f"strength_{i}{j}_2"] = pd.merge(
                battles,
                weakness_df,
                left_on=[f"Type_{i}_2", f"Type_{j}_1"],
                right_on=["type_1", "type_2"],
                how="left",
            ).loc[:, "strength"]
    # column names of the newly created strength factor variables
    strength_cols = [
        "strength_11_1",
        "strength_12_1",
        "strength_21_1",
        "strength_22_1",
        "strength_11_2",
        "strength_12_2",
        "strength_21_2",
        "strength_22_2",
    ]
    # since the strength factor work as multiplicative factors, filling NAs with 1
    # should be consistent (1 indicating that no change of the original stats occurs)
    battles.loc[:, strength_cols] = battles.loc[:, strength_cols].fillna(1)

    log.info("Create one-hot encodings of categorical variables")

    # types
    type_cols = ["Type_1_1", "Type_2_1", "Type_1_2", "Type_2_2"]
    battles.loc[:, type_cols] = battles.loc[:, type_cols].fillna("NoType")
    types = battles.Type_1_1.unique().tolist() + ["NoType"]

    type_oh_1 = (
            pd.get_dummies(battles.loc[:, "Type_1_1"].values.tolist() + types).loc[
            : battles.shape[0], types[:-1]
            ]
            + pd.get_dummies(battles.loc[:, "Type_2_1"].values.tolist() + types).loc[
              : battles.shape[0], types[:-1]
              ]
    )
    type_oh_2 = (
            pd.get_dummies(battles.loc[:, "Type_1_2"].values.tolist() + types).loc[
            : battles.shape[0], types[:-1]
            ]
            + pd.get_dummies(battles.loc[:, "Type_2_2"].values.tolist() + types).loc[
              : battles.shape[0], types[:-1]
              ]
    )

    # weather and time
    weather_oh = pd.get_dummies(battles.WeatherAndTime)

    # concatenate one-hot features to battles
    for df in [type_oh_1, type_oh_2, weather_oh]:
        battles = pd.merge(
            battles, df, left_index=True, right_index=True, suffixes=("_1", "_2")
        )

    # drop original categorical variables
    battles = battles.drop(
        ["Type_1_1", "Type_2_1", "Type_1_2", "Type_2_2", "WeatherAndTime"], axis=1
    )

    # convert boolean variables to integers
    battles.Legendary_1 = battles.Legendary_1.astype(int)
    battles.Legendary_2 = battles.Legendary_2.astype(int)

    # ensure correct selection and ordering of columns according to battles training data
    battles = battles.loc[:, [
            'Level_1', 'Price_1', 'HP_1', 'Attack_1', 'Defense_1', 'Sp_Atk_1', 'Sp_Def_1', 'Speed_1', 'Legendary_1',
            'Level_2', 'Price_2', 'HP_2', 'Attack_2', 'Defense_2', 'Sp_Atk_2', 'Sp_Def_2', 'Speed_2', 'Legendary_2',
            'strength_11_1', 'strength_11_2', 'strength_12_1', 'strength_12_2', 'strength_21_1',
            'strength_21_2', 'strength_22_1', 'strength_22_2',
            'Bug_1', 'Dragon_1', 'Electric_1', 'Fairy_1', 'Fighting_1', 'Fire_1', 'Ghost_1',
            'Grass_1', 'Ground_1', 'Ice_1', 'Normal_1', 'Poison_1', 'Psychic_1', 'Rock_1', 'Water_1',
            'Bug_2', 'Dragon_2', 'Electric_2', 'Fairy_2', 'Fighting_2', 'Fire_2', 'Ghost_2',
            'Grass_2', 'Ground_2', 'Ice_2', 'Normal_2', 'Poison_2', 'Psychic_2', 'Rock_2', 'Water_2',
            'Night', 'Rain', 'Sunshine', 'Unknown', 'Windy'
        ]
    ]

    # sunshine is not present in grandmaster data, thus column is NaN -> replace by 0
    battles = battles.fillna(0)

    cont_cols = ['Level_1', 'Price_1', 'Attack_1', 'Defense_1', 'Sp_Atk_1', 'Sp_Def_1', 'Speed_1', 'Level_2', 'Price_2',
                 'Attack_2', 'Defense_2', 'Sp_Atk_2', 'Sp_Def_2', 'Speed_2']

    with open("./data/99_non_catalogued/scaler_model.pkl", 'rb') as f:
        fitted_scaler = pickle.load(f)

    battles_cont_sc = fitted_scaler.transform(battles[cont_cols])
    battles[cont_cols] = battles_cont_sc

    return battles


def prepare_battles_for_training(
        battles: pd.DataFrame
):
    log = logging.getLogger(__name__)

    log.info('Separating dulpicate battles')
    aside = battles.loc[battles.duplicate, :].copy().drop('duplicate', axis=1)
    battles = battles.loc[[not dupl for dupl in battles.duplicate], :].reset_index(drop=True).drop('duplicate', axis=1)
    n_unique_battles = battles.shape[0]

    np.random.seed(1234)
    idc = np.arange(n_unique_battles)
    shuffled_idc = np.random.permutation(idc)

    train_val_test_split = [0.7, 0.2, 0.1]

    train_split_idx = int(n_unique_battles * train_val_test_split[0])
    val_split_idx = int(n_unique_battles * (train_val_test_split[0] + train_val_test_split[1]))

    log.info("Applying 0.7/0.2/0.1 split into train/val/test sets.")
    train_battles = battles.iloc[shuffled_idc[:train_split_idx], :]
    val_battles = battles.iloc[shuffled_idc[train_split_idx:val_split_idx], :]
    test_battles = battles.iloc[shuffled_idc[val_split_idx:], :]

    assert n_unique_battles == len(train_battles) + len(val_battles) + len(test_battles)

    train_battles = train_battles.append(aside, ignore_index=True)

    y_train = train_battles.BattleResult.copy()
    y_val = val_battles.BattleResult.copy()
    y_test = test_battles.BattleResult.copy()

    # drop response from train/test input
    X_train = train_battles.drop(['BattleResult', 'Name_1', 'Name_2', 'Battle_MainType'], axis=1)
    X_val = val_battles.drop(['BattleResult', 'Name_1', 'Name_2', 'Battle_MainType'], axis=1)
    X_test = test_battles.drop(['BattleResult', 'Name_1', 'Name_2', 'Battle_MainType'], axis=1)

    log.info("Standardizing continuous variables.")
    scaler = StandardScaler()
    cont_cols = ['Level_1', 'Price_1', 'Attack_1', 'Defense_1', 'Sp_Atk_1', 'Sp_Def_1', 'Speed_1', 'Level_2', 'Price_2',
                 'Attack_2', 'Defense_2', 'Sp_Atk_2', 'Sp_Def_2', 'Speed_2']
    X_train_cont = X_train[cont_cols]
    fitted_scaler = scaler.fit(X_train_cont)

    with open("./data/99_non_catalogued/scaler_model.pkl", 'wb') as f:
        pickle.dump(fitted_scaler, f)

    X_train_cont_sc = fitted_scaler.transform(X_train_cont)
    X_train[cont_cols] = X_train_cont_sc

    X_val_cont_sc = fitted_scaler.transform(X_val[cont_cols])
    X_val[cont_cols] = X_val_cont_sc

    X_test_cont_sc = fitted_scaler.transform(X_test[cont_cols])
    X_test[cont_cols] = X_test_cont_sc

    return X_train, X_val, X_test, y_train, y_val, y_test