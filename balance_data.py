from preparing_train_and_validation_data import TrainPreparing

train_data = TrainPreparing()


class BalanceData:
    def __init__(self):
        self.train_df = self.group_ship_count()
        self.balanced_train_df = self.balanced_train_df_method()

    @staticmethod
    def group_ship_count():
        train_data.train_df["grouped_ship_count"] = train_data.train_df.ships.map(
            lambda x: (x + 1) // 2
        ).clip(0, 7)
        return train_data.train_df

    @staticmethod
    def sample_ships(in_df, base_rep_val=1500):
        if in_df["ships"].values[0] == 0:
            return in_df.sample(base_rep_val // 3)
        else:
            return in_df.sample(base_rep_val)

    def balanced_train_df_method(self):
        balanced_train_df = train_data.train_df.groupby("grouped_ship_count").apply(
            self.sample_ships
        )
        return balanced_train_df
