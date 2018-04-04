from tkinter import *
from tkinter.filedialog import askopenfilename

# from pandas.tests.indexes.datetimes.test_tools import epochs

import experiments
import _thread
import sys

root = Tk()

root.geometry('550x320')
v = IntVar()

Radiobutton(root, text='LSTM', variable=v, value=1).place(x=20, y=20, anchor=W)
Radiobutton(root, text='Bi-LSTM', variable=v, value=2).place(x=20, y=40, anchor=W)
Radiobutton(root, text='A-LSTM', variable=v, value=3).place(x=20, y=60, anchor=W)


out_file_str = StringVar()  # 输出文件路径
acc_str = StringVar()
auc_str = StringVar()
precision_str = StringVar()
recall_str = StringVar()
f_score_str = StringVar()
epochs_str = StringVar()
output_n_epoch_str = StringVar()
lstm_size_str = StringVar()
input_file_str = StringVar()
out_file_str.set("select output file")
input_file_str.set("select input file")


def callback1():
    input_file = askopenfilename()
    input_file.set(input_file)


def callback2():
    filename = askopenfilename()
    out_file_str.set(filename)

textLabel_out_file = Label(textvariable=out_file_str, justify=LEFT)
textLabel_out_file.place(x=20, y=250, anchor=W)

theButton_select_file = Button(text="选择输出文件", command=callback2)
theButton_select_file.place(x=20, y=280, anchor=W)

textLabel_

theButton_input_file = Button(text='选择输入文件', command=callback1)
theButton_input_file.place(x=20, y=2)

acc_str.set("accuracy")
auc_str.set("auc")
precision_str.set("precision")
recall_str.set("recall")
f_score_str.set("f_score")
epochs_str.set("epochs")
output_n_epoch_str.set("output_n_epochs")
lstm_size_str.set("lstm_size")


loss_out_text = Text(height=4, width=50, state=DISABLED)
loss_out_text.place(x=150, y=30, anchor=W)


class WriteToTextArea(object):
    def __init__(self):
        pass

    def write(self, *arg):
        loss_out_text.config(state=NORMAL)
        loss_out_text.insert(END, arg[0])
        loss_out_text.see(END)
        loss_out_text.config(state=DISABLED)


sys.stdout = WriteToTextArea()

textLabel_epoch = Label(textvariable=epochs_str, justify=LEFT)
textLabel_epoch.place(x=20, y=90, anchor=W)

epochs_text = Text(height=1, width=10)
epochs_text.insert(END, 1000)
epochs_text.place(x=20, y=110, anchor=W)

textLabel_output_n_epoch = Label(textvariable=output_n_epoch_str, justify=LEFT)
textLabel_output_n_epoch.place(x=20, y=120)

output_n_epoch_text = Text(height=1, width=10)
output_n_epoch_text.insert(END, 20)
output_n_epoch_text.place(x=20, y=150, anchor=W)

textLabel_lstm_size = Label(textvariable=lstm_size_str, justify=LEFT)
textLabel_lstm_size.place(x=20, y=170, anchor=W)

lstm_size_text = Text(height=1, width=10)
lstm_size_text.insert(END, 200)
lstm_size_text.place(x=20, y=190, anchor=W)

textLabel_acc = Label(textvariable=acc_str, justify=LEFT)
textLabel_acc.place(x=150, y=100, anchor=W)

acc_text = Text(height=2, width=40)
acc_text.place(x=220, y=100, anchor=W)

textLabel_auc = Label(textvariable=auc_str, justify=LEFT)
textLabel_auc.place(x=150, y=130, anchor=W)

auc_text = Text(height=2, width=40)
auc_text.place(x=220, y=130, anchor=W)

textLabel_precision = Label(textvariable=precision_str, justify=LEFT)
textLabel_precision.place(x=150, y=160, anchor=W)

precision_text = Text(height=2, width=40)
precision_text.place(x=220, y=160, anchor=W)

textLabel_recall = Label(textvariable=recall_str, justify=LEFT)
textLabel_recall.place(x=150, y=190, anchor=W)

recall_text = Text(height=2, width=40)
recall_text.place(x=220, y=190, anchor=W)

textLabel_f = Label(textvariable=f_score_str, justify=LEFT)
textLabel_f.place(x=150, y=220, anchor=W)

f_score_text = Text(height=2, width=40)
f_score_text.place(x=220, y=220, anchor=W)


def call_basic_lstm_experiment(result_file):
    acc, auc, precision, recall, f_score = experiments.basic_lstm_model_experiments(
        result_file)
    acc_text.insert(END, acc)
    auc_text.insert(END, auc)
    precision_text.insert(END, precision)
    recall_text.insert(END, recall)
    f_score_text.insert(END, f_score)


def call_bidirectional_lstm_model_experiments(result_file):
    acc, auc, precision, recall, f_score = experiments.bidirectional_lstm_model_experiments(
        result_file)
    acc_text.insert(END, acc)
    auc_text.insert(END, auc)
    precision_text.insert(END, precision)
    recall_text.insert(END, recall)
    f_score_text.insert(END, f_score)


def rad_call():
    value = v.get()
    result_file = out_file_str.get()
    epochs_value = epochs_text.get('1.0', END)
    output_n_epoch_value = output_n_epoch_text.get('1.0', END)
    lstm_size_value = lstm_size_text.get('1.0', END)
    epochs = int(epochs_value)
    output_n_epoch = int(output_n_epoch_value)
    lstm_size = int(lstm_size_value)
    experiments.ExperimentSetup.epochs = epochs
    experiments.ExperimentSetup.output_n_epochs = output_n_epoch
    experiments.ExperimentSetup.lstm_size = lstm_size

    if result_file is None:
        return
    if value == 0:
        print("select a model")
    elif value == 1:
        print("waiting....")
        _thread.start_new_thread(call_basic_lstm_experiment, (result_file, ))
    elif value == 2:
        print('waiting....')
        _thread.start_new_thread(call_bidirectional_lstm_model_experiments, (result_file, ))
    else:
        pass


theButton_execute = Button(text="确定", command=rad_call)
theButton_execute.place(x=120, y=280, anchor=W)

root.mainloop()
