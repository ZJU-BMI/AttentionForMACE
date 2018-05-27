import argparse

import experiments


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str,
                        choices=['LSTM',
                                 "Bi-LSTM",
                                 "ResNet",
                                 "A-LSTM",
                                 "A-LSTM-ResNet",
                                 "LSTM-ResNet",
                                 "Conv",
                                 "KB-LSTM"],
                        default='LSTM', help="select model for experiment")

    parser.add_argument("--epochs", type=int, default=1000,
                        help="epochs to train")

    parser.add_argument("--output_n_epochs", type=int, default=20,
                        help="output loss per n epochs")

    parser.add_argument("--lstm_size", type=int, default=200,
                        help="the output size of lstm cell")

    parser.add_argument("--save_file", type=str,
                        help="the path of file to save experiment result")

    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--data_source", type=str, choices=['Lu', 'Sun'],
                        help="the source of data file to select")

    parser.add_argument("--learning_rate", type=float, default=0.001)

    args = parser.parse_args()

    experiments.ExperimentSetup.epochs = args.epochs
    experiments.ExperimentSetup.learning_rate = args.learning_rate
    experiments.ExperimentSetup.lstm_size = args.lstm_size
    experiments.ExperimentSetup.output_n_epochs = args.output_n_epochs
    experiments.ExperimentSetup.data_source = args.data_source

    if args.model == "LSTM":
        experiments.basic_lstm_model_experiments(args.save_file or "./resources/save/lstm_result.csv")
    elif args.model == "Bi-LSTM":
        experiments.bidirectional_lstm_model_experiments(args.save_file or "./resources/save/bi_lstm_result.csv")
    elif args.model == "ResNet":
        experiments.resnet_model_experiments(args.save_file or "./resources/save/resnet_result.csv")
    elif args.model == "A-LSTM":
        experiments.bi_lstm_attention_model_experiments(args.save_file or "./resources/save/a_lstm_res.csv",
                                                        True, False)
    elif args.model == "A-LSTM-ResNet":
        experiments.bi_lstm_attention_model_experiments(args.save_file or "./resources/save/a_lstm_res.csv",
                                                        True, True)
    elif args.model == "LSTM-ResNet":
        experiments.bi_lstm_attention_model_experiments(args.save_file or "./resources/save/a_lstm_res.csv",
                                                        False, True)
    elif args.model == "Conv":
        pass
    elif args.model == "KB-LSTM":
        pass


if __name__ == "__main__":
    main()
