import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import random
import collections
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')
# ==========================================================================================================
# ADD SYS PATH
# ==========================================================================================================
parent_folder = os.path.dirname((os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
path1 = os.path.join(parent_folder, 'general_functions')
sys.path.append(path1)
# ==========================================================================================================


# ==========================================================================================================
# local package import
# ==========================================================================================================
from trade_general_funcs import calculate_rmse
from trade_general_funcs import compute_average_f1
from trade_general_funcs import get_avg_price_change



# ==========================================================================================================


from mlp_trade import MlpTrade
from mlp_regressor import MlpRegressor_P



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size = 1, num_layers = 1, batch_first = True):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,     # rnn hidden unit
            num_layers=num_layers,       # number of rnn layer
            batch_first=batch_first,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)

        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size = 1, num_layers = 1, batch_first = True):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,     # rnn hidden unit
            num_layers=num_layers,       # number of rnn layer
            batch_first=batch_first,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)

        outs = []    # save all predictions
        for time_step in range(r_out.size(1)):    # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state



class RnnTradeRegressor(MlpTrade, MlpRegressor_P):

    def __init__(self):
        super().__init__()

    def set_rnn_regressor(self, input_size, hidden_size = 1, num_layers = 1, learning_rate = 0.001, batch_first = True,
                          max_iter = 2000, tol = 1e-6, random_state = 1, batch_size = None, verbose = False):
        torch.manual_seed(random_state)  # pytorch set random seed
        self.trade_rnn = RNN(input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = batch_first)
        self.optimizer = torch.optim.Adam(self.trade_rnn.parameters(), lr=learning_rate)  # optimize all cnn parameters
        self.loss_func = nn.MSELoss()
        self.h_state = None  # for initial hidden state
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.batch_size = batch_size


    def _process_input_data(self, feaure_list, value_list, stock_id_set, data_set = None, is_training = False):
        stock_set = sorted(list(set(self.training_stock_id_set)))

        stock_rnn_dict = collections.defaultdict(lambda :collections.defaultdict(lambda:[]))
        # TODO ADD data_set to make sure the order is correct
        for i, stock in enumerate(stock_id_set):
            feature_list = feaure_list[i]
            value = value_list[i] # or label
            stock_rnn_dict[stock]['x_list'].append(feature_list)
            stock_rnn_dict[stock]['y_list'].append(np.array(value))

        X_input = []
        Y_input = []
        for stock in stock_set:
            # x_list
            x_list = stock_rnn_dict[stock]['x_list']
            x_array = np.array([np.array(x) for x in x_list])
            X_input.append(x_array)
            #

            # y_list
            y_list = stock_rnn_dict[stock]['y_list']
            y_array = np.array([np.array(y) for y in y_list])
            Y_input.append(y_array)
            #

        if is_training:
            if self.batch_size and self.batch_size < len(X_input):
                X_input = random.sample(X_input, self.batch_size)
            if self.batch_size and self.batch_size < len(Y_input):
                Y_input = random.sample(Y_input, self.batch_size)

        x_array = np.array(X_input)
        y_array = np.array(Y_input)


        x_variable = Variable(torch.from_numpy(x_array[:]).float())
        y_variable = Variable(torch.from_numpy(y_array[:]).float())
        return x_variable, y_variable



    def _predict(self):
        x_variable, y_variable = self._process_input_data(self.dev_set, self.dev_value_set, self.dev_stock_id_set)
        prediction, h_state = self.trade_rnn(x_variable, self.h_state)  # rnn output

        prediction = [x[0][0] for x in prediction.data.numpy()]
        actual = [x[0] for x in y_variable.data.numpy()]
        return prediction, actual, self.dev_date_set, self.dev_stock_id_set


    def train_rnn(self):
        x_variable, y_variable = self._process_input_data(self.training_set, self.training_value_set,
                                                          self.training_stock_id_set,  is_training=True)
        #print ("x_variable: ", x_variable)



        loss_value_previous = float('inf')
        stop_training_list = [False, False]
        stop_epoch = 0

        for epoch in range(self.max_iter):
            prediction, h_state = self.trade_rnn(x_variable, self.h_state)  # rnn output
            self.h_state = h_state
            # !! next step is important !!
            self.h_state = Variable(self.h_state.data)  # repack the hidden state, break the connection from last iteration

            loss = self.loss_func(prediction, y_variable)  # cross entropy loss
            self.optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            self.optimizer.step()  # apply gradients
            loss_value = loss.data[0]

            # decide whether to stop training
            if loss_value - loss_value_previous < self.tol:
                stop_training_list[1] = True

            stop_epoch = epoch

            if self.verbose and epoch%10 == 0:
                print("epoch: {}, loss: {}".format(epoch, loss_value))
            if stop_training_list[0] == True and stop_training_list[1] == True:
                print ("Two consecutive epoches don't improve {} on loss. Training terminate".format(self.tol))
                break

            # switch position
            loss_value_previous = loss_value - loss_value_previous
            stop_training_list[0], stop_training_list[1] = stop_training_list[1], False
            #

        return stop_epoch, loss_value


    def reg_dev_for_moving_window_test(self):

        # (2.) get pred label list
        pred_value_list, actual_value_list, dev_date_set, dev_stock_id_set  = self._predict()
        # print ("pred_value_list: ", pred_value_list)
        # print ("dev_date_set: ", self.dev_date_set)
        return pred_value_list, actual_value_list, self.dev_date_set, self.dev_stock_id_set


