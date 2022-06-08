import pandas as pd
import numpy as np


class Helpers:

    @classmethod
    def get_feature_values_for_categorical_pairs(cls, feature_value_list: list):
        """
        Assign a normalized number to each pokemon pairs of a categorical features.
        Normalisation is such that numbers are between -1 and 1
        :param feature_value_list: list of the categorical values
        :return: dataframe with in column 1 and 2 the pair values and in column 3 the normalized value assigned

        >>> Helpers.get_feature_values_for_categorical_pairs([0, 1])
           feature1  feature2  feature_pairs
        0         0         0            0.0
        1         0         1            1.0
        2         1         0           -1.0
        3         1         1            0.0


        """

        k = 1.
        df_type_category = pd.DataFrame(
            columns=['feature_pairs'], index=pd.MultiIndex.from_tuples([], names=['feature1', 'feature2']), dtype=float)

        for i1 in range(len(feature_value_list)):
            type1 = feature_value_list[i1]
            df_type_category.loc[(type1, type1), 'feature_pairs'] = 0.
            for i2 in range(i1 + 1, len(feature_value_list)):
                type2 = feature_value_list[i2]
                df_type_category.loc[(type1, type2), 'feature_pairs'] = k
                df_type_category.loc[(type2, type1), 'feature_pairs'] = -k
                k += 1.

        df_type_category = df_type_category / max(np.max(df_type_category.values), -np.min(df_type_category.values))
        df_type_category = df_type_category.reset_index()

        return df_type_category

    @classmethod
    def mean_columns_per_bins_of_another_columns(
            cls, df: pd.DataFrame, column_to_bin: str, column_to_mean: str,
            bins: [str, list] = 10) -> (pd.DataFrame, pd.DataFrame):
        """
        Compute the mean value of column_to_mean in bins of column_to_bin.
        Gives also the number of data use to compute the mean for each bin
        :param df: dataframe of the data
        :param column_to_bin: column to bin
        :param column_to_mean: column to mean
        :param bins: if is int, gives the number of bins to use, if is list, gives the limit of the bins to use
        :return: dataframe with bin centers as index and mean of column_to_mean as column,
        dataframe with bin centers as index and number of data used to compute the mean column_to_mean as column
        """

        df_res = df[[column_to_bin, column_to_mean]].copy()

        if isinstance(bins, int) or isinstance(bins, float):
            bin_min = np.min(df[column_to_bin].values)
            bin_max = np.max(df[column_to_bin].values)
            dx = (bin_max-bin_min)/bins
            bins = np.arange(bin_min-dx/2., bin_max+dx, dx)

        x = (bins[1:]+bins[:-1])/2.

        bins_columns = pd.cut(df_res[column_to_bin], bins, labels=x)
        df_res = df_res.drop(columns=column_to_bin)
        df_res = df_res.merge(bins_columns, right_index=True, left_index=True)
        df_mean = df_res.groupby(column_to_bin).mean()
        df_count = df_res.groupby(column_to_bin).count()

        return df_mean.dropna(), df_count.dropna()
