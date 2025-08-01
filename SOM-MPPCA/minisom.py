from numpy import (array, unravel_index, nditer, linalg, random, subtract, max,
                   power, exp, zeros, ones, arange, outer, meshgrid, dot,
                   logical_and, mean, cov, argsort, linspace,
                   einsum, prod, nan, sqrt, hstack, diff, argmin, multiply,
                   nanmean, nansum, tile, array_equal, isclose, bincount)
from numpy.linalg import norm
from collections import defaultdict, Counter
from warnings import warn
import warnings
from sys import stdout
from time import time
from datetime import timedelta
import pickle
import os
import numba
from numba import jit

"""
    Minimalistic implementation of the Self Organizing Maps (SOM).
"""


def _build_iteration_indexes(data_len, num_iterations,
                             verbose=False, random_generator=None,
                             use_epochs=False):
    """Returns an iterable with the indexes of the samples
    to pick at each iteration of the training.

    If random_generator is not None, it must be an instance
    of numpy.random.RandomState and it will be used
    to randomize the order of the samples."""
    if use_epochs:
        iterations_per_epoch = arange(data_len)
        if random_generator:
            random_generator.shuffle(iterations_per_epoch)
        iterations = tile(iterations_per_epoch, num_iterations)
    else:
        iterations = arange(num_iterations) % data_len
        if random_generator:
            random_generator.shuffle(iterations)
    if verbose:
        # tqdm互換の進捗表示に変更
        from tqdm import trange
        return trange(num_iterations)
    else:
        return range(num_iterations)


def _wrap_index__in_verbose(iterations):
    """(tqdmに置き換えられたため、この関数は直接は使われない)"""
    m = len(iterations)
    digits = len(str(m))
    progress = '\r [ {s:{d}} / {m} ] {s:3.0f}% - ? it/s'
    progress = progress.format(m=m, d=digits, s=0)
    stdout.write(progress)
    beginning = time()
    stdout.write(progress)
    for i, it in enumerate(iterations):
        yield it
        sec_left = ((m-i+1) * (time() - beginning)) / (i+1)
        time_left = str(timedelta(seconds=sec_left))[:7]
        progress = '\r [ {i:{d}} / {m} ]'.format(i=i+1, d=digits, m=m)
        progress += ' {p:3.0f}%'.format(p=100*(i+1)/m)
        progress += ' - {time_left} left '.format(time_left=time_left)
        stdout.write(progress)


# NumbaのJITデコレータを追加して高速化
@jit(nopython=True, cache=True)
def fast_norm(x):
    """Returns norm-2 of a 1-D numpy array.
    """
    return sqrt(dot(x, x.T))


# Numba JITコンパイル用のヘルパー関数 (distance_mapのループ部分)
@jit(nopython=True, cache=True)
def _distance_map_loop(weights, um, ii, jj):
    """Numba-accelerated loop for distance_map."""
    for x in range(weights.shape[0]):
        for y in range(weights.shape[1]):
            # nanチェックを追加
            if not isnan(weights[x, y, 0]):
                w_2 = weights[x, y]
                # ★★★ 修正点 3: ブール値のインデックスを整数に明示的に変換 ★★★
                e = y % 2 == 0
                row_index = int(e) 
                for k, (i, j) in enumerate(zip(ii[row_index], jj[row_index])):
                    if (x + i >= 0 and x + i < weights.shape[0] and
                            y + j >= 0 and y + j < weights.shape[1]):
                        # 隣接ノードの重みもnanでないことを確認
                        if not isnan(weights[x + i, y + j, 0]):
                            w_1 = weights[x + i, y + j]
                            um[x, y, k] = fast_norm(w_2 - w_1)
    return um

# isnanをnopythonモードで使えるようにJIT化
@jit(nopython=True, cache=True)
def isnan(x):
    return x != x

class MiniSom(object):
    Y_HEX_CONV_FACTOR = (3.0 / 2.0) / sqrt(3)

    def __init__(self, x, y, input_len, sigma=1, learning_rate=0.5,
                 decay_function='asymptotic_decay',
                 neighborhood_function='gaussian', topology='rectangular',
                 activation_distance='euclidean', random_seed=None,
                 sigma_decay_function='asymptotic_decay'):
        if sigma > sqrt(x*x + y*y):
            warn('Warning: sigma might be too high ' +
                 'for the dimension of the map.')

        self._random_generator = random.RandomState(random_seed)

        self._learning_rate = learning_rate
        self._sigma = sigma
        self._input_len = input_len
        self._weights = self._random_generator.rand(x, y, input_len)*2-1
        self._weights /= linalg.norm(self._weights, axis=-1, keepdims=True)

        self._activation_map = zeros((x, y))
        self._neigx = arange(x, dtype=float)
        self._neigy = arange(y, dtype=float)

        if topology not in ['hexagonal', 'rectangular']:
            msg = '%s not supported only hexagonal and rectangular available'
            raise ValueError(msg % topology)
        self.topology = topology
        self._xx, self._yy = meshgrid(self._neigx, self._neigy)
        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)
        if topology == 'hexagonal':
            self._xx[1::2] -= 0.5 # 修正: 奇数行をずらす正しい方法
            self._yy *= self.Y_HEX_CONV_FACTOR
            if neighborhood_function in ['triangle']:
                warn('triangle neighborhood function does not ' +
                     'take in account hexagonal topology')

        lr_decay_functions = {
            'inverse_decay_to_zero': self._inverse_decay_to_zero,
            'linear_decay_to_zero': self._linear_decay_to_zero,
            'asymptotic_decay': self._asymptotic_decay}

        if isinstance(decay_function, str):
            if decay_function not in lr_decay_functions:
                msg = '%s not supported. Functions available: %s'
                raise ValueError(msg % (decay_function,
                                        ', '.join(lr_decay_functions.keys())))
            self._learning_rate_decay_function = lr_decay_functions[decay_function]
        elif callable(decay_function):
            self._learning_rate_decay_function = decay_function

        sig_decay_functions = {
            'inverse_decay_to_one': self._inverse_decay_to_one,
            'linear_decay_to_one': self._linear_decay_to_one,
            'asymptotic_decay': self._asymptotic_decay}

        if sigma_decay_function not in sig_decay_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (sigma_decay_function,
                                    ', '.join(sig_decay_functions.keys())))
        self._sigma_decay_function = sig_decay_functions[sigma_decay_function]

        neig_functions = {'gaussian': self._gaussian,
                          'mexican_hat': self._mexican_hat,
                          'bubble': self._bubble,
                          'triangle': self._triangle}

        if neighborhood_function not in neig_functions:
            msg = '%s not supported. Functions available: %s'
            raise ValueError(msg % (neighborhood_function,
                                    ', '.join(neig_functions.keys())))
        self.neighborhood = neig_functions[neighborhood_function]

        distance_functions = {'euclidean': self._euclidean_distance,
                              'cosine': self._cosine_distance,
                              'manhattan': self._manhattan_distance,
                              'chebyshev': self._chebyshev_distance}

        if isinstance(activation_distance, str):
            if activation_distance not in distance_functions:
                msg = '%s not supported. Distances available: %s'
                raise ValueError(msg % (activation_distance,
                                        ', '.join(distance_functions.keys())))
            self._activation_distance = distance_functions[activation_distance]
        elif callable(activation_distance):
            self._activation_distance = activation_distance

    def get_weights(self):
        return self._weights

    def get_euclidean_coordinates(self):
        return self._xx.T, self._yy.T

    def convert_map_to_euclidean(self, xy):
        return self._xx.T[xy], self._yy.T[xy]

    def _activate(self, x):
        self._activation_map = self._activation_distance(x, self._weights)

    def activate(self, x):
        self._activate(x)
        return self._activation_map

    def _inverse_decay_to_zero(self, learning_rate, t, max_iter):
        C = max_iter / 100.0
        return learning_rate * C / (C + t)

    def _linear_decay_to_zero(self, learning_rate, t, max_iter):
        return learning_rate * (1 - t / max_iter)

    def _inverse_decay_to_one(self, sigma, t, max_iter):
        C = (sigma - 1) / max_iter
        return sigma / (1 + (t * C))

    def _linear_decay_to_one(self, sigma, t, max_iter):
        return sigma + (t * (1 - sigma) / max_iter)

    def _asymptotic_decay(self, dynamic_parameter, t, max_iter):
        return dynamic_parameter / (1 + t / (max_iter / 2))

    def _gaussian(self, c, sigma):
        d = 2*sigma*sigma
        ax = exp(-power(self._xx-self._xx.T[c], 2)/d)
        ay = exp(-power(self._yy-self._yy.T[c], 2)/d)
        return (ax * ay).T

    def _mexican_hat(self, c, sigma):
        p = power(self._xx-self._xx.T[c], 2) + power(self._yy-self._yy.T[c], 2)
        d = 2*sigma*sigma
        return (exp(-p/d)*(1-2/d*p)).T

    def _bubble(self, c, sigma):
        ax = logical_and(self._neigx > c[0]-sigma,
                         self._neigx < c[0]+sigma)
        ay = logical_and(self._neigy > c[1]-sigma,
                         self._neigy < c[1]+sigma)
        return outer(ax, ay)*1.

    def _triangle(self, c, sigma):
        triangle_x = (-abs(c[0] - self._neigx)) + sigma
        triangle_y = (-abs(c[1] - self._neigy)) + sigma
        triangle_x[triangle_x < 0] = 0.
        triangle_y[triangle_y < 0] = 0.
        return outer(triangle_x, triangle_y)

    def _cosine_distance(self, x, w):
        num = (w * x).sum(axis=2)
        denum = multiply(linalg.norm(w, axis=2), linalg.norm(x))
        return 1 - num / (denum+1e-8)

    def _euclidean_distance(self, x, w):
        return linalg.norm(subtract(x, w), axis=-1)

    def _manhattan_distance(self, x, w):
        return linalg.norm(subtract(x, w), ord=1, axis=-1)

    def _chebyshev_distance(self, x, w):
        return max(subtract(x, w), axis=-1)

    def _check_iteration_number(self, num_iteration):
        if num_iteration < 1:
            raise ValueError('num_iteration must be > 1')

    def _check_input_len(self, data):
        data_len = len(data[0])
        if self._input_len != data_len:
            msg = 'Received %d features, expected %d.' % (data_len,
                                                          self._input_len)
            raise ValueError(msg)

    def winner(self, x):
        self._activate(x)
        # 修正点: 2次元の activation_map を .flatten() で1次元配列に変換する
        return unravel_index(nanargmin(self._activation_map.flatten()),
                             self._activation_map.shape)

    def update(self, x, win, t, max_iteration):
        eta = self._learning_rate_decay_function(self._learning_rate,
                                                 t, max_iteration)
        sig = self._sigma_decay_function(self._sigma, t, max_iteration)
        g = self.neighborhood(win, sig)*eta
        self._weights += einsum('ij, ijk->ijk', g, x-self._weights)

    def quantization(self, data):
        self._check_input_len(data)
        # 修正点: nanargmin を NumPy の argmin に変更
        winners_coords = argmin(self._distance_from_weights(data), axis=1)
        return self._weights[unravel_index(winners_coords,
                                           self._weights.shape[:2])]

    def random_weights_init(self, data):
        self._check_input_len(data)
        map_shape = self._weights.shape[:2]
        rand_indices = self._random_generator.randint(len(data),
                                                      size=map_shape)
        self._weights = array(data)[rand_indices]

    def pca_weights_init(self, data):
        if self._input_len == 1:
            msg = 'The data needs at least 2 features for pca initialization'
            raise ValueError(msg)
        self._check_input_len(data)
        if len(self._neigx) == 1 or len(self._neigy) == 1:
            msg = 'PCA initialization inappropriate:' + \
                  'One of the dimensions of the map is 1.'
            warn(msg)
        pc_length, pc_vecs = linalg.eigh(cov(data.T))
        pc_order = argsort(-pc_length)
        pc_vecs = pc_vecs[:, pc_order]
        mx, my = self._weights.shape[0], self._weights.shape[1]
        c1 = linspace(-1, 1, mx)[:, None]
        c2 = linspace(-1, 1, my)
        pc1 = data.mean(axis=0) + c1 * sqrt(pc_length[pc_order[0]]) * pc_vecs[:, 0]
        pc2 = c2 * sqrt(pc_length[pc_order[1]]) * pc_vecs[:, 1]
        self._weights = pc1[:, None, :] + pc2[None, :, :]

    def train(self, data, num_iteration,
              random_order=False, verbose=False,
              use_epochs=False):
        data = array(data)
        self._check_input_len(data)
        # nanが含まれている場合、学習に影響が出るため警告
        if isnan(data.sum()):
             warn('Input data contains nan values. This will influence the learning process.')

        random_generator = self._random_generator if random_order else None
        
        # ログとtqdmの重複を避けるため、verbose=Trueでもtqdmを使うように
        iterations = _build_iteration_indexes(len(data), num_iteration,
                                              True, random_generator,
                                              use_epochs)

        if use_epochs:
            def get_decay_rate(iteration_index, data_len):
                return int(iteration_index / data_len)
        else:
            def get_decay_rate(iteration_index, data_len):
                return int(iteration_index)

        data_len = len(data)
        for t, iteration_index in enumerate(iterations):
            # 実際のデータインデックスを取得
            data_idx = iteration_index % data_len
            x = data[data_idx]
            # 入力データがnanの場合はスキップ
            if isnan(x.sum()):
                continue
            
            decay_rate = get_decay_rate(t, data_len)
            self.update(x, self.winner(x), decay_rate, num_iteration)
        
        # tqdmを使う場合、quantization errorの表示はmain側で行う
        q_error = self.quantization_error(data)
        if verbose:
            print(f'\n quantization error: {q_error}')

    def train_batch(self, data, num_iteration, verbose=False, log_interval=1000):
        """
        バッチ学習を用いてSOMを訓練します。
        
        :param data: NxD のNumpy配列。Nはサンプル数、Dは特徴次元数。
        :param num_iteration: 学習の反復回数。
        :param verbose: 進捗を表示するかどうか。
        :param log_interval: 量子化誤差を記録する間隔。
        :return: 量子化誤差の履歴リスト。
        """
        self._check_iteration_number(num_iteration)
        self._check_input_len(data)

        quantization_errors = [] # 誤差を記録するリストを初期化

        # verboseがTrueの場合、tqdmによるプログレスバーを表示
        iterations = _build_iteration_indexes(len(data), num_iteration,
                                                verbose=verbose, use_epochs=False)

        for i in iterations:
            # 学習率と近傍半径を更新
            eta = self._learning_rate_decay_function(self._learning_rate, i, num_iteration)
            sig = self._sigma_decay_function(self._sigma, i, num_iteration)

            # E-step: 各データに最も近いノード（勝者ノード）を見つける
            win_map = self.win_map(data)

            # M-step: 重みを更新する
            numerator = zeros(self._weights.shape)
            denominator = zeros(self._weights.shape[:2])

            # バッチ更新のための分子と分母を計算
            for win_pos, data_points in win_map.items():
                h = self.neighborhood(win_pos, sig)
                numerator += h[:, :, None] * mean(data_points, axis=0)
                denominator += h
            
            # 重みを更新
            self._weights = numerator / (denominator[:, :, None] + 1e-9)
            
            # 指定された間隔で量子化誤差を記録
            if i % log_interval == 0:
                q_error = self.quantization_error(data)
                quantization_errors.append(q_error)
                # tqdmの進捗バーに現在の誤差を表示
                if verbose and hasattr(iterations, 'set_postfix'):
                    iterations.set_postfix(q_error=f"{q_error:.5f}")

        if verbose:
            q_error = self.quantization_error(data)
            print(f'\n final quantization error: {q_error}')
        
        # 量子化誤差の履歴を返す
        return quantization_errors


    def distance_map(self, scaling='sum'):
        if scaling not in ['sum', 'mean']:
            raise ValueError(f'scaling should be either "sum" or "mean" ('
                             f'"{scaling}" not valid)')
        
        weights = self._weights
        um = nan * zeros((weights.shape[0], weights.shape[1], 8))

        if self.topology == 'hexagonal':
            ii = array([[1, 1, 1, 0, -1, 0], [0, 1, 0, -1, -1, -1]], dtype=int)
            jj = array([[1, 0, -1, -1, 0, 1], [1, 0, -1, -1, 0, 1]], dtype=int)
        else: # rectangular
            ii = array([[0, -1, -1, -1, 0, 1, 1, 1], [0, -1, -1, -1, 0, 1, 1, 1]], dtype=int)
            jj = array([[-1, -1, 0, 1, 1, 1, 0, -1], [-1, -1, 0, 1, 1, 1, 0, -1]], dtype=int)
        
        um = _distance_map_loop(weights, um, ii, jj)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if scaling == 'mean':
                um = nanmean(um, axis=2)
            if scaling == 'sum':
                um = nansum(um, axis=2)
            
        # nanが含まれる場合、最大値計算でエラーになるのを防ぐ
        max_val = nanmax(um)
        if max_val == 0 or isnan(max_val):
            return um
        else:
            return um / max_val

    def activation_response(self, data):
        self._check_input_len(data)
        data_clean = data[~isnan(data).any(axis=1)]
        if len(data_clean) == 0:
            return zeros(self._weights.shape[:2])
            
        map_shape = self._weights.shape[:2]
        winners_flat = self._distance_from_weights(data_clean).argmin(axis=1)
        win_counts = bincount(winners_flat, minlength=prod(map_shape))
        return win_counts.reshape(map_shape)

    def _distance_from_weights(self, data):
        input_data = array(data)
        weights_flat = self._weights.reshape(-1, self._weights.shape[2])
        input_data_sq = power(input_data, 2).sum(axis=1, keepdims=True)
        weights_flat_sq = power(weights_flat, 2).sum(axis=1, keepdims=True)
        cross_term = dot(input_data, weights_flat.T)
        return sqrt(maximum(0.0, -2 * cross_term + input_data_sq + weights_flat_sq.T))

    def quantization_error(self, data):
        self._check_input_len(data)
        data_clean = data[~isnan(data).any(axis=1)]
        if len(data_clean) == 0:
            return nan
        return norm(data_clean-self.quantization(data_clean), axis=1).mean()
        
    def win_map(self, data, return_indices=False):
        self._check_input_len(data)
        winmap = defaultdict(list)
        # nanデータを除外して処理
        valid_indices = where(~isnan(data).any(axis=1))[0]
        if len(valid_indices) == 0:
            return winmap

        winners = self._winners_from_weights(data[valid_indices])
        
        data_to_append = valid_indices if return_indices else data[valid_indices]

        for i, win_pos in enumerate(winners):
            winmap[win_pos].append(data_to_append[i])
        return winmap
        
    def _winners_from_weights(self, data):
        """Helper function to return the winner coordinates for all data points."""
        distances = self._distance_from_weights(data)
        # 修正点: nanargmin を NumPy の argmin に変更
        winner_indices = argmin(distances, axis=1)
        return [unravel_index(i, self._weights.shape[:2]) for i in winner_indices]

# nanを無視するargminとnanmaxをJIT化
@jit(nopython=True, cache=True)
def nanargmin(arr):
    min_val = float('inf')
    min_idx = -1
    for i in range(arr.shape[0]):
        if not isnan(arr[i]) and arr[i] < min_val:
            min_val = arr[i]
            min_idx = i
    return min_idx

from numpy import nanmax, where, maximum