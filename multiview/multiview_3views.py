from mvkm.utils import *
import numpy as np
from numpy import linalg as LA
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings

warnings.filterwarnings("error")


class MultiView(object):
    """
    pure rank based tensor factorization without non-negativity constraint on Q-matrix, and
    the temporal smoothness is enforced on the observed entries of T only.
    """

    def __init__(self, config, **kwargs):
        """
        :param config:
        :var
        """
        np.random.seed(1)
        self.verbose = config['verbose']
        self.log_file = config['log_file']
        if self.log_file:
            self.logger = create_logger(self.log_file, True)
        self.views = config['views']
        self.train_data = config['train']

        self.num_users = config['num_users']
        self.num_skills = config['num_skills']
        self.num_attempts = config['num_time_index']
        self.num_concepts = config['num_concepts']
        self.num_questions = config['num_questions']
        self.num_lectures = config['num_lectures']
        self.num_discussions = config['num_discussions']
        self.lambda_s = config['lambda_s']
        self.lambda_t = config['lambda_t']
        self.lambda_q = config['lambda_q']
        self.lambda_l = config['lambda_l']
        self.lambda_d = config['lambda_d']
        self.lambda_bias = config['lambda_bias']
        self.penalty_weight = config['penalty_weight']
        self.markovian_steps = config['markovian_steps']
        self.trade_off_l = config['trade_off_lecture']
        self.trade_off_d = config['trade_off_discussion']
        self.lr = config['lr']
        self.max_iter = config['max_iter']
        self.tol = config['tol']
        self.metrics = config['metrics']

        self.use_bias_t = False
        self.use_global_bias = True
        self.exact_penalty = False  # False if use log-sigmoid on penalty

        self.binarized_question = True
        self.binarized_lecture = True
        self.binarized_discussion = True

        self.current_test_attempt = None
        self.test_obs_list = []
        self.test_pred_list = []

        self.train_data_markovian = []
        train_data_dict = {}

        for student, attempt, question, obs, resource in self.train_data:
            key = (student, attempt, question, resource)
            if key not in train_data_dict:
                train_data_dict[key] = obs

        train_data_markovian_dict = {}
        for student, attempt, question, obs, resource in self.train_data:
            if resource == 0:
                upper_steps = min(self.num_attempts, attempt + self.markovian_steps + 1)
                for j in range(attempt + 1, upper_steps):
                    if (student, j, question, resource) not in train_data_dict:
                        if (student, j, question, resource) not in train_data_markovian_dict:
                            train_data_markovian_dict[(student, j, question, resource)] = True
                            self.train_data_markovian.append((student, j, question, resource))

        if int(self.views[0]) == 1:
            self.S = np.random.random_sample((self.num_users, self.num_skills))
            self.T = np.random.random_sample((self.num_skills, self.num_attempts,
                                              self.num_concepts))
            self.Q = np.random.random_sample((self.num_concepts, self.num_questions))
            self.L = np.zeros((self.num_concepts, self.num_lectures))
            self.D = np.zeros((self.num_concepts, self.num_discussions))
            self.bias_s = np.zeros(self.num_users)
            self.bias_t = np.zeros(self.num_attempts)
            self.bias_q = np.zeros(self.num_questions)
            self.bias_l = np.zeros(self.num_lectures)
            self.bias_d = np.zeros(self.num_discussions)
            self.global_bias = np.mean(self.train_data, axis=0)[3]
        else:
            raise AttributeError

        if int(self.views[1]) == 1:
            self.L = np.random.random_sample((self.num_concepts, self.num_lectures))
        if int(self.views[2]) == 1:
            self.D = np.random.random_sample((self.num_concepts, self.num_discussions))

    def __getstate__(self):
        """
        since the logger cannot be pickled, to avoid the pickle error, we should add this
        :return:
        """
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def _get_question_prediction(self, student, attempt, question):
        """
        predict value at tensor Y[attempt, student, question]
        :param attempt: attempt index
        :param student: student index
        :param question: question index
        :return: predicted value of tensor Y[attempt, student, question]
        """
        pred = np.dot(np.dot(self.S[student, :], self.T[:, attempt, :]), self.Q[:, question])
        if self.use_bias_t:
            if self.use_global_bias:
                pred += self.bias_s[student] + self.bias_t[attempt] + self.bias_q[question] + \
                        self.global_bias
            else:
                pred += self.bias_s[student] + self.bias_t[attempt] + self.bias_q[question]
        else:
            if self.use_global_bias:
                pred += self.bias_s[student] + self.bias_q[question] + self.global_bias
            else:
                pred += self.bias_s[student] + self.bias_q[question]

        if self.binarized_question:
            pred = sigmoid(pred)
        return pred

    def _get_lecture_prediction(self, student, attempt, lecture):
        """
        predict value at tensor Y[attempt, student, question]
        :param attempt: attempt index
        :param student: student index
        :param lecture: lecture index
        :return: predicted value of tensor Y[attempt, student, question]
        """
        pred = np.dot(np.dot(self.S[student, :], self.T[:, attempt, :]), self.L[:, lecture])
        if self.use_bias_t:
            if self.use_global_bias:
                pred += self.bias_s[student] + self.bias_t[attempt] + self.bias_l[lecture] + \
                        self.global_bias
            else:
                pred += self.bias_s[student] + self.bias_t[attempt] + self.bias_l[lecture]
        else:
            if self.use_global_bias:
                pred += self.bias_s[student] + self.bias_l[lecture] + self.global_bias
            else:
                pred += self.bias_s[student] + self.bias_l[lecture]

        if self.binarized_discussion:
            pred = sigmoid(pred)
        return pred

    def _get_discussion_prediction(self, student, attempt, discussion):
        """
        predict value at tensor Y[attempt, student, question]
        :param attempt: attempt index
        :param student: student index
        :param question: question index
        :return: predicted value of tensor Y[attempt, student, question]
        """
        pred = np.dot(np.dot(self.S[student, :], self.T[:, attempt, :]), self.D[:, discussion])
        if self.use_bias_t:
            if self.use_global_bias:
                pred += self.bias_s[student] + self.bias_t[attempt] + self.bias_d[discussion] + \
                        self.global_bias
            else:
                pred += self.bias_s[student] + self.bias_t[attempt] + self.bias_d[discussion]
        else:
            if self.use_global_bias:
                pred += self.bias_s[student] + self.bias_d[discussion] + self.global_bias
            else:
                pred += self.bias_s[student] + self.bias_d[discussion]

        if self.binarized_discussion:
            pred = sigmoid(pred)
        return pred

    def _get_loss(self):
        """
        override the function in super class
        compute the loss, which is RMSE of observed records +
        regularization + penalty of temporal non-smoothness
        :return: loss
        """
        loss, square_loss, reg_bias = 0., 0., 0.
        square_loss_q, square_loss_l, square_loss_d = 0., 0., 0.
        q_count, l_count, d_count = 0., 0., 0.
        for (student, attempt, question, obs, resource) in self.train_data:
            if resource == 0:
                pred = self._get_question_prediction(student, attempt, question)
                square_loss_q += (obs - pred) ** 2
                q_count += 1
            elif resource == 1:
                pred = self._get_lecture_prediction(student, attempt, question)
                square_loss_l += (obs - pred) ** 2
                l_count += 1
            elif resource == 2:
                pred = self._get_discussion_prediction(student, attempt, question)
                square_loss_d += (obs - pred) ** 2
                d_count += 1
        square_loss = square_loss_q + self.trade_off_l * square_loss_l + \
                      self.trade_off_d * square_loss_d
        # print("square loss {},{},{}".format(square_loss_q, square_loss_e, square_loss_l))

        reg_S = LA.norm(self.S) ** 2
        reg_T = LA.norm(self.T) ** 2  # regularization on tensor T
        reg_Q = LA.norm(self.Q) ** 2  # regularization on matrix Q
        reg_L = LA.norm(self.L) ** 2
        reg_D = LA.norm(self.D) ** 2

        reg_features = self.lambda_s * reg_S + self.lambda_q * reg_Q + self.lambda_t * reg_T + \
                       self.lambda_l * reg_L + self.lambda_d * reg_D
        q_rmse = np.sqrt(square_loss_q / q_count) if q_count != 0 else 0.
        l_rmse = np.sqrt(self.trade_off_l * square_loss_l / l_count) if l_count != 0 else 0.
        d_rmse = np.sqrt(self.trade_off_d * square_loss_d / d_count) if d_count != 0 else 0.
        if self.lambda_bias:
            if self.use_bias_t:
                reg_bias = self.lambda_bias * (
                        LA.norm(self.bias_s) ** 2 + LA.norm(self.bias_t) ** 2 +
                        LA.norm(self.bias_q) ** 2 + LA.norm(self.bias_l) ** 2 +
                        LA.norm(self.bias_d) ** 2)
            else:
                reg_bias = self.lambda_bias * (
                        LA.norm(self.bias_s) ** 2 + LA.norm(self.bias_q) ** 2 +
                        LA.norm(self.bias_l) ** 2 + LA.norm(self.bias_d) ** 2)

        penalty = self._get_penalty()
        loss = square_loss + reg_features + reg_bias + penalty
        return loss, q_count, q_rmse, l_rmse, d_rmse, penalty, reg_features, reg_bias

    def _get_penalty(self):
        """
        compute the penalty on the observations, we want all attempts before the obs has smaller
        score, and the score after obs should be greater.
        we use sigmoid to set the penalty between 0 and 1
        if knowledge at current attempt >> prev attempt, then diff is large, that mean
        sigmoid(diff) is large and close to 1., so penalty is a very small negative number
        since we aim to minimize the objective = loss + penalty, the smaller penalty is better

        #TODO if diff is positive, then the penalty should be zero, rather than a small value
        :return:
        """
        penalty = 0.
        for student, attempt, index, obs, resource in self.train_data:
            if attempt >= 1 and resource == 0:
                gap = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
                knowledge_gap = np.dot(self.S[student, :], gap)
                if self.exact_penalty:
                    knowledge_gap[knowledge_gap > 0.] = 0.
                    penalty_val = -np.dot(knowledge_gap, self.Q[:, index])
                else:
                    diff = np.dot(knowledge_gap, self.Q[:, index])
                    penalty_val = -np.log(sigmoid(diff))
                penalty += self.penalty_weight * penalty_val

        for student, attempt, index, resource in self.train_data_markovian:
            if attempt >= 1 and resource == 0:
                gap = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
                knowledge_gap = np.dot(self.S[student, :], gap)
                if self.exact_penalty:
                    knowledge_gap[knowledge_gap > 0.] = 0.
                    penalty_val = -np.dot(knowledge_gap, self.Q[:, index])
                else:
                    diff = np.dot(knowledge_gap, self.Q[:, index])
                    penalty_val = -np.log(sigmoid(diff))
                penalty += self.penalty_weight * penalty_val
        return penalty

    def _grad_S_k(self, student, attempt, index, obs=None, resource=None):
        """
        note that the penalty is actually next score is larger than previous score
        instead of saying next knowledge state is higher than previous knowledge state

        :param student:
        :param attempt:
        :param index:
        :param obs:
        :param resource:
        :return:
        """
        grad = np.zeros_like(self.S[student, :])
        if obs is not None:
            if resource == 0:
                pred = self._get_question_prediction(student, attempt, index)
                if self.binarized_question:
                    grad = -2. * (obs - pred) * pred * (1. - pred) * np.dot(self.T[:, attempt, :],
                                                                            self.Q[:, index])
                else:
                    grad = -2. * (obs - pred) * np.dot(self.T[:, attempt, :], self.Q[:, index])
            elif resource == 1:
                pred = self._get_lecture_prediction(student, attempt, index)
                if self.binarized_lecture:
                    grad = -2. * self.trade_off_l * (obs - pred) * pred * (1. - pred) * np.dot(
                        self.T[:, attempt, :], self.L[:, index])
                else:
                    grad = -2. * self.trade_off_l * (obs - pred) * np.dot(self.T[:, attempt, :],
                                                                          self.L[:, index])
            elif resource == 2:
                pred = self._get_discussion_prediction(student, attempt, index)
                if self.binarized_discussion:
                    grad = -2. * self.trade_off_d * (obs - pred) * pred * (1. - pred) * np.dot(
                        self.T[:, attempt, :], self.D[:, index])
                else:
                    grad = -2. * self.trade_off_d * (obs - pred) * np.dot(self.T[:, attempt, :],
                                                                          self.D[:, index])
        grad += 2. * self.lambda_s * self.S[student, :]

        if resource == 0:
            if attempt == 0:
                diff = self.T[:, attempt + 1, :] - self.T[:, attempt, :]
            elif attempt == self.num_attempts - 1:
                diff = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
            else:
                diff = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
                diff += self.T[:, attempt + 1, :] - self.T[:, attempt, :]

            if self.exact_penalty:
                # penalty on Q = - min(0, ST[i]Q - ST[i-1]Q)))
                # grad += self.penalty_weight * grad(penalty on Q)
                # grad(penalty on Q) = - min(0, T[i]Q - T[i-1]Q)
                TQ_diff = np.dot(diff, self.Q[:, index])
                TQ_diff[TQ_diff > 0.] = 0.
                grad += self.penalty_weight * (- TQ_diff)
            else:
                TQ_diff = np.dot(diff, self.Q[:, index])
                val = np.dot(self.S[student, :], TQ_diff)

                # penalty on S = -log(sigmoid(ST[i]Q - ST[i-1]Q))
                # grad += self.penalty_weight * grad(penalty on S)
                # grad(penalty on S) = - (1. / sigmoid(val) * sigmoid(val) * (1. - sigmoid(val)) *
                #                        TQ_diff)
                grad += - self.penalty_weight * (
                        1. / sigmoid(val) * sigmoid(val) * (1. - sigmoid(val)) * TQ_diff
                )
        return grad

    def _grad_T_ij(self, student, attempt, index, obs=None, resource=None):
        """
        compute the gradient of loss w.r.t a specific student j's knowledge at
        a specific attempt i: T_{i,j,:},
        :param attempt: index
        :param student: index
        :param obs: observation
        :return:
        """

        grad = np.zeros_like(self.T[:, attempt, :])
        if obs is not None:
            if resource == 0:
                pred = self._get_question_prediction(student, attempt, index)
                if self.binarized_question:
                    grad = -2. * (obs - pred) * pred * (1. - pred) * np.outer(
                        self.S[student, :], self.Q[:, index])
                else:
                    grad = -2. * (obs - pred) * np.outer(self.S[student, :], self.Q[:, index])
            elif resource == 1:
                pred = self._get_lecture_prediction(student, attempt, index)
                if self.binarized_lecture:
                    grad = -2. * self.trade_off_l * (obs - pred) * pred * (1. - pred) * np.outer(
                        self.S[student, :], self.L[:, index])
                else:
                    grad = -2. * self.trade_off_l * (obs - pred) * np.outer(
                        self.S[student, :], self.L[:, index])
            elif resource == 2:
                pred = self._get_discussion_prediction(student, attempt, index)
                if self.binarized_discussion:
                    grad = -2. * self.trade_off_d * (obs - pred) * pred * (1. - pred) * np.outer(
                        self.S[student, :], self.D[:, index])
                else:
                    grad = -2. * self.trade_off_d * (obs - pred) * np.outer(
                        self.S[student, :], self.D[:, index])
        grad += 2. * self.lambda_t * self.T[:, attempt, :]

        if resource == 0:
            if attempt == 0:
                diff = self.T[:, attempt + 1, :] - self.T[:, attempt, :]
                if self.exact_penalty:
                    # penalty on T = - min(0, ST[i]Q - ST[i-1]Q)))
                    # grad += self.penalty_weight * grad(penalty on T)
                    # grad(penalty on Q) = - min(0, ST[i] - ST[i-1])
                    diff[diff > 0.] = 0.
                    penalty_val = -np.dot(np.dot(self.S[student, :], diff), self.Q[:, index])
                    grad += self.penalty_weight * penalty_val * (-1.)
                else:
                    val = np.dot(np.dot(self.S[student, :], diff), self.Q[:, index])
                    # penalty on T = -log(sigmoid(ST[i]Q - ST[i-1]Q))
                    # grad += self.penalty_weight * grad(penalty on T)
                    # grad(penalty on T) = - (1. / sigmoid(val) * sigmoid(val) * (1. - sigmoid(val))
                    #                       * (-1.0)* np.outer(self.S[student,:], self.Q[:, index])

                    grad += -self.penalty_weight * (
                            1. / sigmoid(val) * sigmoid(val) * (1. - sigmoid(val)) * (-1.) *
                            np.outer(self.S[student, :], self.Q[:, index])
                    )
            elif attempt == self.num_attempts - 1:
                diff = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
                if self.exact_penalty:
                    diff[diff > 0.] = 0.
                    penalty_val = -np.dot(np.dot(self.S[student, :], diff), self.Q[:, index])
                    grad += self.penalty_weight * penalty_val
                else:
                    val = np.dot(np.dot(self.S[student, :], diff), self.Q[:, index])
                    grad += -self.penalty_weight * (
                            1. / sigmoid(val) * sigmoid(val) * (1. - sigmoid(val)) *
                            np.outer(self.S[student, :], self.Q[:, index])
                    )
            else:
                if self.exact_penalty:
                    diff = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
                    diff[diff > 0.] = 0.
                    penalty_val = -np.dot(np.dot(self.S[student, :], diff), self.Q[:, index])
                    grad += self.penalty_weight * penalty_val

                    diff = self.T[:, attempt + 1, :] - self.T[:, attempt, :]
                    diff[diff > 0.] = 0.
                    penalty_val = -np.dot(np.dot(self.S[student, :], diff), self.Q[:, index])
                    grad += self.penalty_weight * penalty_val * (-1.)

                else:
                    diff = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
                    val = np.dot(np.dot(self.S[student, :], diff), self.Q[:, index])
                    grad += -self.penalty_weight * (
                            1. / sigmoid(val) * sigmoid(val) * (1. - sigmoid(val)) *
                            np.outer(self.S[student, :], self.Q[:, index])
                    )

                    diff = self.T[:, attempt + 1, :] - self.T[:, attempt, :]
                    val = np.dot(np.dot(self.S[student, :], diff), self.Q[:, index])
                    grad += -self.penalty_weight * (
                            1. / sigmoid(val) * sigmoid(val) * (1. - sigmoid(val)) * (-1.) *
                            np.outer(self.S[student, :], self.Q[:, index])
                    )
        return grad

    def _grad_Q_k(self, student, attempt, question, obs=None):
        """
        compute the gradient of loss w.r.t a specific concept-question association
        of a question in Q-matrix,
        :param attempt: index
        :param student:  index
        :param question:  index
        :param obs: the value at Y[attempt, student, question]
        :return:
        """
        grad = np.zeros_like(self.Q[:, question])
        if obs is not None:
            pred = self._get_question_prediction(student, attempt, question)
            if self.binarized_question:
                grad = -2. * (obs - pred) * pred * (1. - pred) * np.dot(
                    self.S[student, :], self.T[:, attempt, :])
            else:
                grad = -2. * (obs - pred) * np.dot(
                    self.S[student, :], self.T[:, attempt, :])
        grad += 2. * self.lambda_q * self.Q[:, question]

        if attempt == 0:
            diff = self.T[:, attempt + 1, :] - self.T[:, attempt, :]
        elif attempt == self.num_attempts - 1:
            diff = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
        else:
            diff = self.T[:, attempt, :] - self.T[:, attempt - 1, :]
            diff += self.T[:, attempt + 1, :] - self.T[:, attempt, :]
        knowledge_gap = np.dot(self.S[student, :], diff)

        if self.exact_penalty:
            # penalty on Q = - min(0, ST[i]Q - ST[i-1]Q)))
            # grad += self.penalty_weight * grad(penalty on Q)
            # grad(penalty on Q) = - min(0, ST[i] - ST[i-1])
            knowledge_gap[knowledge_gap > 0] = 0.
            grad += self.penalty_weight * (- knowledge_gap)
        else:
            val = np.dot(knowledge_gap, self.Q[:, question])
            # penalty on Q = -log(sigmoid(ST[i]Q - ST[i-1]Q))
            # grad += self.penalty_weight * grad(penalty on Q)
            # grad(penalty on Q) = - (1. / sigmoid(val) * sigmoid(val) * (1. - sigmoid(val)) *
            #                        knowledge_diff)
            grad += - self.penalty_weight * (
                    1. / sigmoid(val) * sigmoid(val) * (1. - sigmoid(val)) * knowledge_gap
            )
        return grad

    def _grad_L_k(self, student, attempt, lecture, obs=None):
        """
        compute the gradient of loss w.r.t a specific concept-question association
        of a question in Q-matrix,
        :param attempt: index
        :param student:  index
        :param lecture:  index
        :param obs: observation
        :return:
        """
        grad = np.zeros_like(self.L[:, lecture])
        if obs is not None:
            pred = self._get_lecture_prediction(student, attempt, lecture)
            if self.binarized_lecture:
                grad = -2. * self.trade_off_l * (obs - pred) * pred * (1. - pred) * np.dot(
                    self.S[student, :], self.T[:, attempt, :])
            else:
                grad = -2. * self.trade_off_l * (obs - pred) * np.dot(
                    self.S[student, :], self.T[:, attempt, :])
        grad += 2. * self.lambda_d * self.L[:, lecture]
        return grad

    def _grad_D_k(self, student, attempt, discussion, obs=None):
        """
        compute the gradient of loss w.r.t a specific concept-question association
        of a question in Q-matrix,
        :param attempt: index
        :param student:  index
        :param discussion:  index
        :param obs: the value at Y[attempt, student, question]
        :return:
        """
        grad = np.zeros_like(self.D[:, discussion])
        if obs is not None:
            pred = self._get_discussion_prediction(student, attempt, discussion)
            if self.binarized_discussion:
                grad = -2. * self.trade_off_d * (obs - pred) * pred * (1. - pred) * np.dot(
                    self.S[student, :], self.T[:, attempt, :])
            else:
                grad = -2. * self.trade_off_d * (obs - pred) * np.dot(self.S[student, :],
                                                                      self.T[:, attempt, :])
        grad += 2. * self.lambda_d * self.D[:, discussion]
        return grad

    def _grad_bias_s(self, student, attempt, material, obs=None, resource=None):
        """
        compute the gradient of loss w.r.t a specific bias_s
        :param attempt:
        :param student:
        :param material: material material of that resource
        :param obs:
        :return:
        """
        grad = 0.
        if obs is not None:
            if resource == 0:
                pred = self._get_question_prediction(student, attempt, material)
                if self.binarized_question:
                    grad -= 2. * (obs - pred) * pred * (1. - pred)
                else:
                    grad -= 2. * (obs - pred)
            elif resource == 1:
                pred = self._get_lecture_prediction(student, attempt, material)
                if self.binarized_lecture:
                    grad -= 2. * self.trade_off_l * (obs - pred) * pred * (1. - pred)
                else:
                    grad -= 2. * self.trade_off_l * (obs - pred)
            elif resource == 2:
                pred = self._get_discussion_prediction(student, attempt, material)
                if self.binarized_discussion:
                    grad -= 2. * self.trade_off_d * (obs - pred) * pred * (1. - pred)
                else:
                    grad -= 2. * self.trade_off_d * (obs - pred)
        grad += 2.0 * self.lambda_bias * self.bias_s[student]
        return grad

    def _grad_bias_t(self, student, attempt, material, obs=None, resource=None):
        """
        compute the gradient of loss w.r.t a specific bias_a
        :param attempt:
        :param student:
        :param material: material material of that resource
        :return:
        """
        grad = 0.
        if obs is not None:
            if resource == 0:
                pred = self._get_question_prediction(student, attempt, material)
                if self.binarized_question:
                    grad -= 2. * (obs - pred) * pred * (1. - pred)
                else:
                    grad -= 2. * (obs - pred)
            elif resource == 1:
                pred = self._get_lecture_prediction(student, attempt, material)
                if self.binarized_lecture:
                    grad -= 2. * self.trade_off_l * (obs - pred) * pred * (1. - pred)
                else:
                    grad -= 2. * self.trade_off_l * (obs - pred)
            elif resource == 2:
                pred = self._get_discussion_prediction(student, attempt, material)
                if self.binarized_discussion:
                    grad -= 2. * self.trade_off_d * (obs - pred) * pred * (1. - pred)
                else:
                    grad -= 2. * self.trade_off_d * (obs - pred)
        grad += 2.0 * self.lambda_bias * self.bias_t[attempt]
        return grad

    def _grad_bias_q(self, student, attempt, question, obs=None):
        """
        compute the gradient of loss w.r.t a specific bias_q
        :param attempt:
        :param student:
        :param question:
        :param obs:
        :return:
        """
        grad = 0.
        if obs is not None:
            pred = self._get_question_prediction(student, attempt, question)
            if self.binarized_question:
                grad -= 2. * (obs - pred) * pred * (1. - pred)
            else:
                grad -= 2. * (obs - pred)
        grad += 2. * self.lambda_bias * self.bias_q[question]
        return grad

    def _grad_bias_l(self, student, attempt, lecture, obs=None):
        """
        compute the gradient of loss w.r.t a specific bias_q
        :param attempt:
        :param student:
        :param lecture:
        :param obs:
        :return:
        """
        grad = 0.
        if obs is not None:
            pred = self._get_lecture_prediction(student, attempt, lecture)
            if self.binarized_lecture:
                grad -= 2. * self.trade_off_l * (obs - pred) * pred * (1. - pred)
            else:
                grad -= 2. * self.trade_off_l * (obs - pred)
        grad += 2. * self.lambda_bias * self.bias_l[lecture]
        return grad

    def _grad_bias_d(self, student, attempt, discussion, obs=None):
        """
        compute the gradient of loss w.r.t a specific bias_q
        :param attempt:
        :param student:
        :param discussion:
        :param obs:
        :return:
        """
        grad = 0.
        if obs is not None:
            pred = self._get_discussion_prediction(student, attempt, discussion)
            if self.binarized_discussion:
                grad -= 2. * self.trade_off_d * (obs - pred) * pred * (1. - pred)
            else:
                grad -= 2. * self.trade_off_d * (obs - pred)
        grad += 2. * self.lambda_bias * self.bias_d[discussion]
        return grad

    def _optimize_sgd(self, student, attempt, material, obs=None, resource=None):
        """
        train the T and Q with stochastic gradient descent
        :param attempt:
        :param student:
        :param material: material material of that resource
        :return:
        """
        # train the bias(es)
        if resource == 0:
            self.bias_q[material] -= self.lr * self._grad_bias_q(student, attempt, material, obs)
        elif resource == 1:
            self.bias_l[material] -= self.lr * self._grad_bias_l(student, attempt, material, obs)
        elif resource == 2:
            self.bias_d[material] -= self.lr * self._grad_bias_d(student, attempt, material, obs)

        self.bias_s[student] -= self.lr * self._grad_bias_s(student, attempt, material, obs,
                                                            resource)
        if self.use_bias_t:
            self.bias_t[attempt] -= self.lr * self._grad_bias_t(student, attempt, material, obs,
                                                                resource)

        # optimize T
        grad_t = self._grad_T_ij(student, attempt, material, obs, resource)
        self.T[:, attempt, :] -= self.lr * grad_t

        # optimize S
        grad_s = self._grad_S_k(student, attempt, material, obs, resource)
        self.S[student, :] -= self.lr * grad_s
        if self.exact_penalty:
            # when self.exact_penalty == True, S should be always positive
            self.S[student, :][self.S[student, :] < 0.] = 0.
            if self.lambda_s == 0.:
                sum_val = np.sum(self.S[student, :])
                if sum_val != 0:
                    self.S[student, :] /= sum_val
        else:
            self.S[student, :][self.S[student, :] < 0.] = 0.
            if self.lambda_s == 0.:
                sum_val = np.sum(self.S[student, :])
                if sum_val != 0:
                    self.S[student, :] /= sum_val

        # update Q if current test attempt is still small, otherwise keep Q the same
        # if self.current_test_attempt < (self.num_attempts / 2):
        if resource == 0:  # optimize Q
            grad_q = self._grad_Q_k(student, attempt, material, obs)
            self.Q[:, material] -= self.lr * grad_q
            self.Q[:, material][self.Q[:, material] < 0.] = 0.
            if self.lambda_q == 0.:
                sum_val = np.sum(self.Q[:, material])
                if sum_val != 0:
                    self.Q[:, material] /= sum_val  # normalization
        elif resource == 1:  # optimize L
            grad_l = self._grad_L_k(student, attempt, material, obs)
            self.L[:, material] -= self.lr * grad_l
            self.L[:, material][self.L[:, material] < 0.] = 0.
            if self.lambda_l == 0.:
                sum_val = np.sum(self.L[:, material])
                if sum_val != 0:
                    self.L[:, material] /= sum_val  # normalization
        elif resource == 2:  # optimize D
            grad_e = self._grad_D_k(student, attempt, material, obs)
            self.D[:, material] -= self.lr * grad_e
            self.D[:, material][self.D[:, material] < 0.] = 0.
            if self.lambda_d == 0.:
                sum_val = np.sum(self.D[:, material])
                if sum_val != 0:
                    self.D[:, material] /= sum_val  # normalization

    def training(self):
        """
        minimize the loss until converged or reach the maximum iterations
        with stochastic gradient descent
        :return:
        """
        self.logger.info(strBlue("*" * 100))
        self.logger.info(strBlue('test attempt: {}, train size: {}'.format(
            self.current_test_attempt, len(self.train_data)))
        )

        loss, q_count, q_rmse, l_rmse, d_rmse, penalty, reg_features, reg_bias = self._get_loss()
        self.logger.info(strBlue("initial: lr: {:.4f}, loss: {:.2f}, q_count: {}, q_rmse: {:.5f}, "
                                 "penalty: {:.5f}, reg_features: {:.2f}, reg_bias: {:.3f}".format(
            self.lr, loss, q_count, q_rmse, penalty, reg_features, reg_bias))
        )
        loss_list = [loss]
        self.logger.info(strBlue("*" * 40 + "[ Training Results ]" + "*" * 40))

        train_perf = []
        start_time = time.time()
        converge = False
        iter_num = 0
        min_iter = 10
        best_S, best_T, best_Q, best_L, best_D = [0] * 5
        best_bias_s, best_bias_t, best_bias_q, best_bias_l, best_bias_d = [0] * 5

        while not converge:
            np.random.shuffle(self.train_data)
            np.random.shuffle(self.train_data_markovian)
            best_S = np.copy(self.S)
            best_T = np.copy(self.T)
            best_Q = np.copy(self.Q)
            best_L = np.copy(self.L)
            best_D = np.copy(self.D)
            best_bias_s = np.copy(self.bias_s)
            best_bias_t = np.copy(self.bias_t)
            best_bias_q = np.copy(self.bias_q)
            best_bias_l = np.copy(self.bias_l)
            best_bias_d = np.copy(self.bias_d)

            for (student, attempt, index, obs, resource) in self.train_data:
                self._optimize_sgd(student, attempt, index, obs, resource=resource)

            for (student, attempt, index, resource) in self.train_data_markovian:
                self._optimize_sgd(student, attempt, index, resource=resource)

            loss, q_count, q_rmse, l_rmse, d_rmse, penalty, reg_features, reg_bias = \
                self._get_loss()
            train_perf.append([q_count, q_rmse])

            run_time = time.time() - start_time
            self.logger.debug("iter: {}, lr: {:.4f}, total loss: {:.2f}, q_count: {}, "
                              "q_rmse: {:.5f}".format(iter_num, self.lr, loss, q_count, q_rmse))
            self.logger.debug("--- penalty: {:.5f}, reg_features: {:.2f}, reg_bias: {:.3f}, "
                              "run time so far: {:.2f}".format(
                penalty, reg_features, reg_bias, run_time))

            if iter_num == self.max_iter:
                self.logger.info("=" * 50)
                self.logger.info("** converged **, condition: 0, iter: {}".format(iter_num))
                loss_list.append(loss)
                converge = True
                self.logger.info("training loss: {:.5f}".format(loss))
                self.logger.info("q_rmse: {:.5f}".format(q_rmse))
                self.logger.info("penalty: {:.5f}".format(penalty))
                self.logger.info("regularization on parameters: {:.5f}".format(reg_features))
            elif iter_num >= min_iter and loss >= np.mean(loss_list[-5:]):
                self.logger.info("=" * 40)
                self.logger.info("** converged **, condition: 1, iter: {}".format(iter_num))
                converge = True
                self.logger.info("training loss: {:.5f}".format(loss))
                self.logger.info("q_rmse: {:.5f}".format(q_rmse))
                self.logger.info("penalty: {:.5f}".format(penalty))
                self.logger.info("regularization on parameters: {:.5f}".format(reg_features))
            elif loss == np.nan:
                self.lr *= 0.1
            elif loss > loss_list[-1]:
                loss_list.append(loss)
                self.lr *= 0.5
                iter_num += 1
            else:
                loss_list.append(loss)
                iter_num += 1

        # reset to previous S, T, Q
        self.S = best_S
        self.T = best_T
        self.Q = best_Q
        self.L = best_L
        self.D = best_D
        self.bias_s = best_bias_s
        self.bias_t = best_bias_t
        self.bias_q = best_bias_q
        self.bias_l = best_bias_l
        self.bias_d = best_bias_d

        return train_perf[-1]

    def testing(self, test_data, validation=False):
        """
        :return: performance metrics mean squared error, RMSE, and mean absolute error
        """
        if not validation:
            self.logger.info(strGreen("*" * 40 + "[ Testing Results ]" + "*" * 40))
            self.logger.info(strGreen("Current testing attempt: {}, Test size: {}".format(
                self.current_test_attempt, len(test_data))))

        curr_pred_list = []
        curr_obs_list = []

        for (student, attempt, question, obs, resource) in test_data:
            if resource == 0:
                curr_obs_list.append(obs)
                pred = self._get_question_prediction(student, attempt, question)

                pred = np.dot(np.dot(self.S[student, :], self.T[:, attempt, :]),
                              self.Q[:, question])
                if self.use_bias_t:
                    if self.use_global_bias:
                        pred += self.bias_s[student] + self.bias_t[attempt] + self.bias_q[
                            question] + \
                                self.global_bias
                    else:
                        pred += self.bias_s[student] + self.bias_t[attempt] + self.bias_q[question]
                else:
                    if self.use_global_bias:
                        pred += self.bias_s[student] + self.bias_q[question] + self.global_bias
                    else:
                        pred += self.bias_s[student] + self.bias_q[question]

                if self.binarized_question:
                    pred = sigmoid(pred)

                curr_pred_list.append(pred)
                self.test_obs_list.append(obs)
                self.test_pred_list.append(pred)
                self.logger.debug((strCyan("student: {}, question: {}".format(student, question))))
                self.logger.debug(strCyan("true: {:.5f}, pred: {:.5f}".format(obs, pred)))
                self.logger.debug(strPurple("S: {}".format(np.round(self.S[student, :], 5))))
                self.logger.debug(strPurple("Q: {}".format(np.round(self.Q[:, question], 5))))
                self.logger.debug(strPurple("bias s: {:.5f}".format(self.bias_s[student])))
                self.logger.debug(strPurple("bias q: {:.5f}".format(self.bias_q[question])))
                self.logger.debug(strPurple("global bias: {:.5f}\n".format(self.global_bias)))

        return self.eval(curr_obs_list, curr_pred_list)

    def eval(self, obs_list, pred_list):
        """
        evaluate the prediction performance
        :param threshold:
        :param obs_list:
        :param pred_list:
        :return:
        """
        assert len(pred_list) == len(obs_list)
        threshold_list = [0.1 * i for i in range(1, 10)]

        count = len(obs_list)
        perf_dict = {}
        if len(pred_list) == 0:
            return perf_dict
        else:
            # self.logger.info("Test Size: {}".format(count))
            perf_dict["count"] = count

        for metric in self.metrics:
            if metric == "rmse":
                rmse = mean_squared_error(obs_list, pred_list, squared=False)
                perf_dict[metric] = rmse
                self.logger.info(strGreen("RMSE: {:.5f}".format(rmse)))
            elif metric == 'mae':
                mae = mean_absolute_error(obs_list, pred_list)
                perf_dict[metric] = mae
                self.logger.info(strGreen("MAE: {:.5f}".format(mae)))
            elif metric == "auc":
                if np.sum(obs_list) == count:
                    self.logger.info(strGreen("AUC: None (all ones in true y)"))
                    perf_dict[metric] = None
                else:
                    auc = roc_auc_score(obs_list, pred_list)
                    perf_dict[metric] = auc
                    self.logger.info(strGreen("AUC: {:.5f}".format(auc)))
            elif metric == "accuracy":
                max_accuracy = 0.
                best_threshold = 0.
                for threshold in sorted(pred_list):
                    pred_list_class = np.array(pred_list) > threshold
                    accuracy = accuracy_score(obs_list, pred_list_class)
                    if accuracy > max_accuracy:
                        best_threshold = threshold
                        max_accuracy = accuracy
                perf_dict[metric] = max_accuracy
                self.logger.info(strGreen("Threshold: {:.1f}, Accuracy: {:.5f}".format(
                    best_threshold, max_accuracy)))
            elif metric == "precision":
                max_precision = 0.
                best_threshold = 0.
                for threshold in threshold_list:
                    pred_list_class = np.array(pred_list) > threshold
                    try:
                        precision = precision_score(obs_list, pred_list_class, zero_division=1)
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except:
                        precision = 0.
                    if precision > max_precision:
                        max_precision = precision
                        best_threshold = threshold
                perf_dict[metric] = max_precision
                self.logger.info(strGreen("Threshold: {:.1f}, Precision: {:.5f}".format(
                    best_threshold, max_precision)))
            elif metric == "recall":
                max_recall = 0.
                best_threshold = 0.
                for threshold in threshold_list:
                    pred_list_class = np.array(pred_list) > threshold
                    try:
                        recall = recall_score(obs_list, pred_list_class)
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except:
                        recall = 0.
                    if recall > max_recall:
                        max_recall = recall
                        best_threshold = threshold
                perf_dict[metric] = max_recall
                self.logger.info(strGreen("Threshold: {:.1f}, Recall: {:.5f}".format(
                    best_threshold, max_recall)))
            elif metric == "f1":
                max_f1 = 0.
                best_threshold = 0.
                for threshold in threshold_list:
                    pred_list_class = np.array(pred_list) > threshold
                    try:
                        f1 = f1_score(obs_list, pred_list_class)
                    except (KeyboardInterrupt, SystemExit):
                        raise
                    except:
                        f1 = 0.
                    if f1 > max_f1:
                        max_f1 = f1
                        best_threshold = threshold
                perf_dict[metric] = max_f1
                self.logger.info(strGreen("Threshold: {:.1f}, F1-Score: {:.5f}".format(
                    best_threshold, max_f1)))
        self.logger.info("\n")
        return perf_dict
