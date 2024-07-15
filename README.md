# Tiny-IDS: Tiny-IDS: A Tiny Deep Neural Network-based Intrusion Detection System

## Ebrahim Fard . Mahdi Soltani . Amir Hossein Jahangir . Seokbum Ko

In this repo, we provide the scripts of the paper "Tiny-IDS: A Tiny Deep Neural Network-based Intrusion Detection System", a DNN-based Intrusion Detection System (IDS) that can be utilized in Embedded Systems due to its small number of weights. We used the CIC-IDS2017 dataset to validate the proposed method.

We ran our code on Ubuntu 24.04, and the required packages are in the requirement.txt file. it is recommended to run the code in a virtual environment. 

By running the code, you should see something like this at first:

![Screenshot from 2024-07-14 19-18-25](https://github.com/user-attachments/assets/b534ba55-0b83-49d4-8e03-04cbf2601598)

And this at the end:
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
