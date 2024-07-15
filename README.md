# Tiny-IDS: Tiny-IDS: A Tiny Deep Neural Network-based Intrusion Detection System

## Ebrahim Fard . Mahdi Soltani . Amir Hossein Jahangir . Seokbum Ko

In this repo, we provide the scripts of the paper "Tiny-IDS: A Tiny Deep Neural Network-based Intrusion Detection System", a DNN-based Intrusion Detection System (IDS) that can be utilized in Embedded Systems due to its small number of weights. We used the CIC-IDS2017 dataset to validate the proposed method.

We ran our code on Ubuntu 24.04, and the required packages are in the requirement.txt file. it is recommended to run the code in a virtual environment. 

By running the code, you should see something like this at first:

Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ main_input (InputLayer)         │ (None, 100, 200)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ LSTM1 (LSTM)                    │ (None, 100, 50)        │        50,200 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 5000)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ classifier (Dense)              │ (None, 2)              │        10,002 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 60,202 (235.16 KB)
 Trainable params: 60,202 (235.16 KB)
 Non-trainable params: 0 (0.00 B)

...

The average Delay per flow is: 0.055161 sec +/- 0.010924 sec
              precision    recall  f1-score   support

           0       0.99      0.99      0.99      1007
           1       0.96      0.97      0.97       394

    accuracy                           0.98      1401
   macro avg       0.98      0.98      0.98      1401
weighted avg       0.98      0.98      0.98      1401

[[992  15]
 [ 10 384]]


Notably, the aforementioned information is only a sample, and you may see different results by using the whole dataset.
