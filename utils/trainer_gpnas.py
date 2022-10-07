from paddleslim.nas import GPNAS
import numpy as np
import scipy
import scipy.stats


class GPNAS_train_model:
    name = None
    model = None
    model_acc = -1
    data_train_X = None
    data_train_Y = None
    data_all_X = None
    data_all_Y = None
    trained = False

    def do_train(self, input_X, input_Y):
        '''
            return: accuracy, trained modle
        '''
        # 划分训练及测试集
        if len(input_X) != len(input_Y):
            raise "length of input_X and input_Y diff"
        inpu_size = len(input_X)
        train_num = int(inpu_size*0.8)
        X_all_k, Y_all_k = np.array(input_X), np.array(input_Y)
        X_train_k, Y_train_k = X_all_k[0:train_num], Y_all_k[0:train_num]
        X_test_k, Y_test_k = X_all_k[train_num:], Y_all_k[train_num:]

        self.data_train_X, self.data_train_Y = X_train_k, Y_train_k
        self.data_all_X, self.data_all_Y = X_all_k, Y_all_k


        for c_flag in range(1,3):
            for m_flag in range(1,3):
                # 初始该任务的gpnas预测器参数
                gp = GPNAS(c_flag, m_flag)

                gp.get_initial_mean(X_train_k[0::2], Y_train_k[0::2])
                init_cov = gp.get_initial_cov(X_train_k)
                # 更新（训练）gpnas预测器超参数
                gp.get_posterior_mean(X_train_k[1::2], Y_train_k[1::2])

                # 基于测试评估预测误差
                error_list_gp = np.array(Y_test_k.reshape(
                    len(Y_test_k), 1)-gp.get_predict(X_test_k))
                error_list_gp_j = np.array(Y_test_k.reshape(len(Y_test_k), 1)-gp.get_predict_jiont(X_test_k, X_train_k, Y_train_k))
                print(f'{self.name}  AVE mean gp :',      np.mean(
                    abs(np.divide(error_list_gp, Y_test_k.reshape(len(Y_test_k), 1)))))
                print(f'{self.name}  AVE mean gp jonit :', np.mean(
                    abs(np.divide(error_list_gp_j, Y_test_k.reshape(len(Y_test_k), 1)))))

                y_predict_jiont = gp.get_predict_jiont(X_test_k, self.data_train_X, self.data_train_Y)
                y_predict = gp.get_predict(X_test_k)

                res = scipy.stats.kendalltau(y_predict, Y_test_k)
                jiont_res = scipy.stats.kendalltau(y_predict_jiont, Y_test_k)
                print(f"{self.name}  c_flag:{c_flag} m_flag:{m_flag} acc:{res.correlation} jiont_acc:{jiont_res.correlation}")

                acc = jiont_res.correlation
                if acc > self.model_acc:
                    self.model_acc = acc
                    self.model = gp

        self.trained = True
        return self.model_acc

    def do_predict(self, input_X):
        rank_all = []
        
        predict_X = np.array(input_X)
        return self.model.get_predict_jiont(predict_X, self.data_all_X, self.data_all_Y)
