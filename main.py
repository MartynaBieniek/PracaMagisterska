from getting_data import get_data
from model import create_cnn
import numpy as np
import time
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold

step_size = [1, 2, 3, 4, 5, 6,7 ,8,9,10]

filtrer_size = [3, 5, 7,9, 11,13,15,17,19,21]

start_initializers = {
    "VarianceScaling_FanAvg_Normal": tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="truncated_normal"),
    "Zeros": tf.keras.initializers.Zeros(),
    "Ones": tf.keras.initializers.Ones(),
    "RandomNormal": tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05),
    "RandomUniform": tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05),
    "TruncatedNormal": tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05),
    "HeNormal": tf.keras.initializers.HeNormal(),
    "HeUniform": tf.keras.initializers.HeUniform(),
    "GlorotNormal": tf.keras.initializers.GlorotNormal(),
    "GlorotUniform": tf.keras.initializers.GlorotUniform(),
    "LecunNormal": tf.keras.initializers.LecunNormal(),
    "LecunUniform": tf.keras.initializers.LecunUniform(),
    "VarianceScaling_FanIn_Normal": tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_in", distribution="truncated_normal"),
    "VarianceScaling_FanOut_Uniform": tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_out", distribution="uniform"),
    "VarianceScaling_FanAvg_Uniform": tf.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
}

pooling_type = ['max', 'avg']

pool_sizes = [2, 3, 4]

strides_list = [1, 2, 3]

activation_functions = [
    "relu",
    "leaky_relu",
    "elu",
    "selu",
    "sigmoid",
    "tanh",
    "swish",
    "softplus",
    "hard_sigmoid",
    "gelu"
]

hidden_units= [8, 16, 32, 64, 96, 128, 192, 256, 384, 512]


def train_model(x, parm):
    X_all, y_all = get_data("dane_widma.xlsx", x)

    if x== 'c': color = "czarny"
    else: color = "niebieski"

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    results = []

    for z in parm:
            fold = 1
            for train_index, val_index in kf.split(X_all):
                print(f"\n=== Pooling method {z} | Fold {fold} ===")

                X_train, X_val = X_all[train_index], X_all[val_index]
                y_train, y_val = y_all[train_index], y_all[val_index]

                input_shape = X_train.shape[1:]
                num_classes = len(np.unique(y_all))

                model = create_cnn(input_shape, num_classes, z)

                start_time = time.time()

                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=16,
                    verbose=0
                )

                end_time = time.time()
                training_time = end_time - start_time

                best_val_acc = max(history.history['val_accuracy'])
                train_acc = history.history['accuracy'][-1]
                best_epoch = np.argmax(history.history['val_accuracy']) + 1
                best_train_acc = max(history.history['accuracy'])
                overfitting = best_train_acc - best_val_acc

                results.append({
                    'Liczba wyjść': z,
                    'Fold': fold,
                    'Dokładność walidacji': best_val_acc,
                    'Dokładność treningowa': train_acc,
                    'Czas treningu (s)': training_time,
                    'Liczba epok': best_epoch,
                    'Overfitting': overfitting
                })

                fold += 1

    df = pd.DataFrame(results)
    df.to_excel(f"{color}_wyniki_liczba_wyjsc_warstwy_ukrytej_polaczonej_1.xlsx", index=False)

def main():
    df = train_model(data_key="all", units_list=hidden_units)
    print("\n=== Training completed ===")


if __name__ == "__main__":
    main()

