from tkinter import *
from tkinter.filedialog import askopenfilename


import experiments
import _thread
import sys


class UIInterface(object):
    def __init__(self):
        self.root = Tk()
        self.root.geometry('500x320')

        # 创建模型选择的Radio button
        self._model_selected = IntVar()  # 代表被选择的模型
        self._input_file_select = IntVar()
        self._model_select_radio_button()

        self.out_file_str = StringVar(value="select output__file")  # 输出文件路径
        self._place_some_text()
        self._place_some_label()
        self._place_some_button()
        
        self._change_stdout()

    def _model_select_radio_button(self):
        Radiobutton(self.root, text="LSTM", variable=self._model_selected, value=1).place(x=20, y=20, anchor=W)
        Radiobutton(self.root, text="Bi-LSTM", variable=self._model_selected, value=2).place(x=20, y=40, anchor=W)
        Radiobutton(self.root, text="A-LSTM", variable=self._model_selected, value=3).place(x=20, y=60, anchor=W)
        Radiobutton(self.root, text='Lu', variable=self._input_file_select, value=1).place(x=20, y=90, anchor=W)
        Radiobutton(self.root, text='Sun', variable=self._input_file_select, value=2).place(x=20, y=110, anchor=W)

    def _place_some_label(self):
        Label(self.root, textvariable=self.out_file_str, justify=LEFT).place(x=20, y=270, anchor=W)
        Label(textvariable=StringVar(value="epochs"), justify=LEFT).place(x=20, y=140, anchor=W)
        Label(textvariable=StringVar(value="output_n_epochs"), justify=LEFT).place(x=20, y=180, anchor=W)
        Label(textvariable=StringVar(value="lstm_size"), justify=LEFT).place(x=20, y=220, anchor=W)

        Label(textvariable=StringVar(value="acc"), justify=LEFT).place(x=150, y=100, anchor=W)
        Label(textvariable=StringVar(value="auc"), justify=LEFT).place(x=150, y=130, anchor=W)
        Label(textvariable=StringVar(value="precision"), justify=LEFT).place(x=150, y=160, anchor=W)
        Label(textvariable=StringVar(value="recall"), justify=LEFT).place(x=150, y=190, anchor=W)
        Label(textvariable=StringVar(value="f_score"), justify=LEFT).place(x=150, y=220, anchor=W)

    def _place_some_text(self):
        self.loss_out_text = Text(self.root, height=4, width=50, state=DISABLED)
        self.loss_out_text.place(x=150, y=30, anchor=W)

        self.epochs_text = Text(self.root, height=1, width=10)
        self.epochs_text.insert(END, 1000)
        self.epochs_text.place(x=20, y=160, anchor=W)

        self.output_n_epoch_text = Text(self.root, height=1, width=10)
        self.output_n_epoch_text.insert(END, 20)
        self.output_n_epoch_text.place(x=20, y=200, anchor=W)

        self.lstm_size_text = Text(self.root, height=1, width=10)
        self.lstm_size_text.insert(END, 200)
        self.lstm_size_text.place(x=20, y=240, anchor=W)

        self.acc_text = Text(height=2, width=40)
        self.acc_text.place(x=220, y=100, anchor=W)

        self.auc_text = Text(height=2, width=40)
        self.auc_text.place(x=220, y=130, anchor=W)

        self.precision_text = Text(height=2, width=40)
        self.precision_text.place(x=220, y=160, anchor=W)

        self.recall_text = Text(height=2, width=40)
        self.recall_text.place(x=220, y=190, anchor=W)

        self.f_score_text = Text(height=2, width=40)
        self.f_score_text.place(x=220, y=220, anchor=W)

    def _place_some_button(self):
        Button(self.root,
               text='选择输出文件',
               command=lambda: self.out_file_str.set(askopenfilename())).place(x=20, y=300, anchor=W)
        Button(self.root,
               text="确定",
               command=self._confirm_click).place(x=120, y=300, anchor=W)

    def _confirm_click(self):
        self.auc_text.delete('1.0', END)
        self.precision_text.delete('1.0', END)
        self.recall_text.delete('1.0', END)
        self.f_score_text.delete('1.0', END)
        self.loss_out_text.delete('1.0', END)

        value = self._model_selected.get()
        v = self._input_file_select.get()
        if v == 1:
            experiments.ExperimentSetup.data_source = "lu"
        elif v == 2:
            experiments.ExperimentSetup.data_source = "sun"
        else:
            print("select data source")
            return
        result_file = self.out_file_str.get()
        epochs_value = self.epochs_text.get('1.0', END)
        output_n_epoch_value = self.output_n_epoch_text.get('1.0', END)
        lstm_size_value = self.lstm_size_text.get('1.0', END)
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
            _thread.start_new_thread(self._call_basic_lstm_experiment, (result_file,))
        elif value == 2:
            print('waiting....')
            _thread.start_new_thread(self._call_bi_lstm_experiment, (result_file,))
        else:
            pass

    def _call_basic_lstm_experiment(self, result_file):
        acc, auc, precision, recall, f_score = experiments.basic_lstm_model_experiments(
            result_file)
        self.acc_text.insert(END, acc)
        self.auc_text.insert(END, auc)
        self.precision_text.insert(END, precision)
        self.recall_text.insert(END, recall)
        self.f_score_text.insert(END, f_score)

    def _call_bi_lstm_experiment(self, result_file):
        acc, auc, precision, recall, f_score = experiments.bidirectional_lstm_model_experiments(
            result_file)
        self.acc_text.insert(END, acc)
        self.auc_text.insert(END, auc)
        self.precision_text.insert(END, precision)
        self.recall_text.insert(END, recall)
        self.f_score_text.insert(END, f_score)

    def show(self):
        self.root.mainloop()

    def _change_stdout(self):
        sys.stdout.write = self.__write_to_loss_text

    def __write_to_loss_text(self, *args):
        self.loss_out_text.config(state=NORMAL)
        self.loss_out_text.insert(END, args[0])
        self.loss_out_text.see(END)
        self.loss_out_text.config(state=DISABLED)


if __name__ == "__main__":
    ui = UIInterface()
    ui.show()
