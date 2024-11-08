import os
import pandas as pd
import json

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from preprocess import (
    train_test_set,
    normalize
)

from window_generator import WindowGenerator

from models import (
    Baseline,
    linear_,
    dense_,
    cnn_,
    lstm_,
    compile_and_fit
    )

def read_file():
    csv_path = r"C:\Users\fcommo\.keras\datasets"
    # file = "jena_climate_2009_2016.csv"
    file = "jena_climate_2009_2016_prep.csv"
    df = pd.read_csv(os.path.join(csv_path, file))
    return df

def plot_performance(performance):
    plt.figure(figsize=(16, 8))

    x = range(len(performance))
    labels = performance.keys()
    
    plt.subplot(2, 1, 1)
    y = [performance[k]['mse'] for k in labels]
    plt.bar(x, y, label='MSE')
    plt.ylabel('MSE')
    plt.xticks(x, labels, rotation=45)
    plt.legend()

    plt.subplot(2, 1, 2)
    y = [performance[k]['mae'] for k in labels]
    plt.bar(x, y, label='MAE')
    plt.ylabel('MAE')
    plt.xticks(x, labels, rotation=45)
    plt.legend()

    plt.show()


def main():
    df = read_file()
    train_df, val_df, test_df = train_test_set(df)
    norm_train_df, norm_val_df, norm_test_df, df_std = normalize(df, train_df, val_df, test_df)

    # df_std = df_std.melt(var_name='Column', value_name='Normalized')
    # plt.figure(figsize=(12, 6))
    # ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    # _ = ax.set_xticklabels(df.keys(), rotation=90)
    # plt.show()

    # w1 = WindowGenerator(input_width=24, label_width=1, shift=24,
    #                      train_df=norm_train_df, val_df=norm_val_df, test_df=norm_test_df,
    #                      label_columns=['T (degC)'])
    # print(w1)

    column_indices = {name: i for i, name in enumerate(df.columns)}

    # single_step_window = WindowGenerator(
    #     input_width=1, label_width=1, shift=1,
    #     train_df=norm_train_df, val_df=norm_val_df, test_df=norm_test_df,
    #     label_columns=['T (degC)'])

    # print(single_step_window)

    print("\nBaseline")
    baseline = Baseline(label_index=column_indices['T (degC)'])
    baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                    metrics=[tf.keras.metrics.MeanAbsoluteError()])

    wide_window = WindowGenerator(
        input_width=24, label_width=24, shift=1,
        train_df=norm_train_df, val_df=norm_val_df, test_df=norm_test_df,
        label_columns=['T (degC)'])

    models_list = ['Baseline', 'Linear', 'Dense', 'CNN', 'LSTM']
    val_performance = {m: dict() for m in models_list}
    test_performance = {m: dict() for m in models_list}

    val_performance['Baseline']['mse'], val_performance['Baseline']['mae'] = baseline.evaluate(wide_window.val)
    test_performance['Baseline']['mse'], test_performance['Baseline']['mae'] = baseline.evaluate(wide_window.test, verbose=0)

    print("\nLinear")
    linear = linear_()
    history = compile_and_fit(linear, wide_window)
    val_performance['Linear']['mse'], val_performance['Linear']['mae'] = linear.evaluate(wide_window.val)
    test_performance['Linear']['mse'], test_performance['Linear']['mae'] = linear.evaluate(wide_window.test, verbose=0)

    print("Dense")
    dense = dense_()
    history = compile_and_fit(dense, wide_window)
    val_performance['Dense']['mse'], val_performance['Dense']['mae'] = dense.evaluate(wide_window.val)
    test_performance['Dense']['mse'], test_performance['Dense']['mae'] = dense.evaluate(wide_window.test, verbose=0)

    CONV_WIDTH = 3
    LABEL_WIDTH = 24
    INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
    wide_conv_window = WindowGenerator(
        input_width=INPUT_WIDTH,
        label_width=LABEL_WIDTH,
        train_df=norm_train_df, val_df=norm_val_df, test_df=norm_test_df,
        shift=1,
        label_columns=['T (degC)']
        )

    print("\nConvolutional n-nets")
    cnn = cnn_()
    history = compile_and_fit(cnn, wide_conv_window)
    val_performance['CNN']['mse'], val_performance['CNN']['mae'] = cnn.evaluate(wide_conv_window.val)
    test_performance['CNN']['mse'], test_performance['CNN']['mae'] = cnn.evaluate(wide_conv_window.test, verbose=0)

    print("\nReccurent n-nets")
    lstm = lstm_()
    history = compile_and_fit(lstm, wide_window)
    val_performance['LSTM']['mse'], val_performance['LSTM']['mae'] = lstm.evaluate(wide_window.val)
    test_performance['LSTM']['mse'], test_performance['LSTM']['mae'] = lstm.evaluate(wide_window.test, verbose=0)

    # print(history)
    print("Test performance:")
    print(json.dumps(test_performance, indent=4))
    print("Validation performance:")
    print(json.dumps(val_performance, indent=4))

    wide_window.plot(lstm)

    plot_performance(val_performance)


if __name__ == "__main__":
    main()