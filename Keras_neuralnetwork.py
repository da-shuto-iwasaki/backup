# coding: utf-8
"""
Kerasを用いて、ニューラルネットワークで学習する。
"""
import pandas as pd
import datetime
from keras.layers import Input, Dense, concatenate, Dropout
from keras.models import Model

# これより前のデータはモデルを学習させるのに利用する。（2016.1〜2017.8 なので、３割を予測に利用する。）
Threshold = datetime.datetime(2017, 3, 1)

if __name__ == "__main__":
    df = pd.read_csv("./A_time/keras/JOAX.csv", index_col=0).sort_values(by=['year','month','day','hour','minute'])
    df = df.dropna().reset_index(drop=True)
    # datetime型は、csvファイルに保存するとstr型に変換されてしまうため、毎回変更が必要。
    df.cm_start_time  = pd.to_datetime(df.cm_start_time,  format='%Y-%m-%dT%H:%M:%S')
    df.start_datetime = pd.to_datetime(df.start_datetime, format='%Y-%m-%dT%H:%M:%S')
    df.end_datetime   = pd.to_datetime(df.end_datetime,   format='%Y-%m-%dT%H:%M:%S')

    #=== 閾値で、学習データと予測データに分ける ===
    df_data    = df[df.cm_start_time < Threshold].reset_index(drop=True)    # 閾値より前の値を利用して、モデルを学習させる。
    df_predict = df[df.cm_start_time >= Threshold].reset_index(drop=True)  # 閾値より後の値でテストをする。

    #=== ニューラルネットワークに入れるそれぞれのカラム ===
    targets = ["setai","kozin","C","T","M1","M2","M3","F1","F2","F3","timing","housou_minute"]
    col_11  = ["JOCX_{}".format(target) for target in targets]
    col_12  = ["JOEX_{}".format(target) for target in targets]
    col_13  = ["JORX_{}".format(target) for target in targets]
    col_14  = ["JOTX_{}".format(target) for target in targets]
    col_15  = ['setai_rate','kozin_rate','C_rate','T_rate','M1_rate','M2_rate','M3_rate','F1_rate','F2_rate','F3_rate',
                'hour','minute','cm_timing','housou_minutes','onair_sec','total_onair_sec','cm_count',
                'is_holiday','housou_kaisuu','last_flag','sai_housou_flag','土','日','月','木','水','火','金',]
    target_cols = ['CM_decline_setai','CM_decline_after5_setai'] # 正解ラベル

    ### モデルの構築 #############################################################
    #=== 入力層 ===
    d1  = len(targets)  # 各局の特徴量がいくつあるか
    d1_ = len(col_15)
    inputs_1 = Input(shape=(d1, ))
    inputs_2 = Input(shape=(d1, ))
    inputs_3 = Input(shape=(d1, ))
    inputs_4 = Input(shape=(d1, ))
    inputs_5 = Input(shape=(d1_,)) # 時間などの特徴量

    #=== 第１層 ===
    d2  = 24    # 暫定値
    d2_ = 32    # 暫定値
    x_11 = Dense(d2,  activation="relu")(inputs_1)
    x_12 = Dense(d2,  activation="relu")(inputs_2)
    x_13 = Dense(d2,  activation="relu")(inputs_3)
    x_14 = Dense(d2,  activation="relu")(inputs_4)
    x_15 = Dense(d2_, activation="relu")(inputs_5)

    #=== 第２層（マージする） ===
    x_2 = concatenate([x_11,x_12,x_13,x_14,x_15])

    #=== 第３層 ===
    d3 = 64     # 暫定値
    x_3 = Dense(d3, activation="relu")(x_2)
    
    #=== 第４層===
    d4 = 32     # 暫定値
    x_4 = Dense(d3, activation="relu")(x_3)

    #=== 出力層 ===
    prediction = Dense(2)(x_4)

    #=== コンパイル ====
    model = Model(inputs=[inputs_1,inputs_2,inputs_3,inputs_4,inputs_5], outputs=prediction)
    model.compile(optimizer="Adagrad",
                  loss="logcosh")      # log(cosh(x))はxが小さければ(x ** 2) / 2とほぼ等しくなり，xが大きければabs(x) - log(2)とほぼ等しくなる。
    ###########################################################################

    #=== モデルの学習 ===
    model.fit([df_data[col_11].values,
               df_data[col_12].values,
               df_data[col_13].values,
               df_data[col_14].values,
               df_data[col_15].values,],
             [df_data[target_cols].values],
             epochs=1000, batch_size=32)

    model.save('my_model6.h5')
    print(model.summary())

