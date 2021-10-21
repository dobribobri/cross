# -*- coding: utf-8 -*-
from builtins import float
from typing import Union, Tuple, List, Any, SupportsFloat
import numpy as np
import dill
import os
import struct
import warnings
from enum import Enum
import PIL.Image
import scipy.stats
from scipy import signal


class Band(Enum):
    P = 'P'
    L = 'L'


class ReadMode(Enum):
    CMP = 'cmp'
    FLT = 'flt'


class DType(Enum):
    AMPLITUDES = 'амплитуды'
    PHASES = 'фазы'
    INTENSITIES = 'интенсивности'
    REAL_PARTS = 'действ. часть'
    IMAG_PARTS = 'мнимая часть'
    AS_IS = 'as_is'


class Direction(Enum):
    X = 'x'
    Y = 'y'
    BOTH = 'both'


class CorrelationMode(Enum):
    MODE1 = 'Метод 1'
    MODE2 = 'Метод 2'


class Sizes:
    def __init__(self, band: Band):
        if band == Band.P:
            self.width, self.height = 1500, 3840
        if band == Band.L:
            self.width, self.height = 1499, 7778

    @property
    def tuple(self) -> Tuple[int, int]:
        return self.width, self.height


def pearsonr(x: np.ndarray, y: np.ndarray) -> SupportsFloat:
    x, y = np.ravel(x), np.conjugate(np.ravel(y))
    r, _ = scipy.stats.pearsonr(x, y)
    return r


class do:
    @staticmethod
    def nothing(x: np.ndarray) -> np.ndarray:
        return x


class Weighting:
    pass


class CommonOperations:
    @staticmethod
    def to_PIL_image(data: np.ndarray, operations=do.nothing) -> PIL.Image:
        data = operations(data)
        return PIL.Image.fromarray(np.uint8(data), mode='L')

    @staticmethod
    def save_png(data: np.ndarray, path: str, operations=do.nothing) -> None:
        img = CommonOperations.to_PIL_image(data, operations)
        img.save(fp=path)

    @staticmethod
    def convolve(data: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return signal.convolve2d(data, kernel, mode='same')

    @staticmethod
    def smoothing(data: np.ndarray, n: int, exclusive=False) -> np.ndarray:
        N = 2 * n + 1
        kernel = np.ones((N, N), dtype=data.dtype)
        if not exclusive:
            kernel /= (N * N)
        else:
            kernel[n, n] = 0
            kernel /= (N * N - 1)
        return CommonOperations.convolve(data, kernel)

    @staticmethod
    def __smoothing(data: np.ndarray, n: int, weights: np.ndarray = None, exclusive=False) -> np.ndarray:
        if weights is None:
            weights = np.ones((2 * n, 2 * n), dtype=data.dtype)
            weights /= (4*n*n)
        if exclusive:
            weights[n, n] = 0
        w, h = data.shape
        res = np.zeros(data.shape, data.dtype)
        for i in range(n, w - n):
            for j in range(n, h - n):
                res[i, j] = np.average(data[i - n:i + n, j - n:j + n], weights=weights)
        for i in range(n):
            for j in range(n):
                res[i, j] = np.average(data[:i + n, :j + n], weights=weights[n - i:, n - j:])
            for j in range(h - n, h):
                res[i, j] = np.average(data[:i + n, j - n:], weights=weights[n - i:, :n - j + h])
        for i in range(w - n, w):
            for j in range(n):
                res[i, j] = np.average(data[i - n:, :j + n], weights=weights[:n - i + w, n - j:])
            for j in range(h - n, h):
                res[i, j] = np.average(data[i - n:, j - n:], weights=weights[:n - i + w, :n - j + h])
        return res

    @staticmethod
    def scaling(data: np.ndarray, scale: Union[Tuple[int, int], List[int]] = (0, 255)):
        _min, _max = np.min(data), np.max(data)
        return scale[0] + (data - _min) / (_max - _min) * (scale[1] - scale[0])

    @staticmethod
    def elim_outliers(data: np.ndarray) -> np.ndarray:
        s = np.sort(np.ravel(data))
        q2 = np.median(s)
        q1, q3 = np.median(s[s < q2]), np.median(s[s >= q2])
        shift = (q3 - q1) * 3
        l_lim, r_lim = q1 - shift, q3 + shift
        x = data.copy()
        x[x < l_lim] = l_lim
        x[x > r_lim] = r_lim
        return x

    @staticmethod
    def correlate(imgs: Union[Tuple[np.ndarray, np.ndarray], List[np.ndarray]],
                  window: Union[Tuple[Tuple[int, int], Tuple[int, int]], List[Tuple[int, int]], List[List[int]]],
                  direction: Direction = Direction.X,
                  correlation_function=pearsonr, weighting=do.nothing,
                  shift: Union[Tuple[int, int], List[int]] = (0, 0),
                  mode: CorrelationMode = CorrelationMode.MODE1, verbose=True) -> dict:
        if imgs[0].shape != imgs[1].shape:
            raise Exception('images should have the same shape')
        h, w = imgs[0].shape
        x, y = [window[i][0] for i in range(2)], [window[i][1] for i in range(2)]
        wx, wy = x[1] - x[0] + 1, y[1] - y[0] + 1
        sx, sy = shift[0], shift[1]
        if y[0] - sy < 0 or x[0] - sx < 0 or y[1] - sy > h or x[1] - sx > w:
            return {}
        area = weighting(imgs[1][y[0]-sy:y[0]+wy-sy, x[0]-sx:x[0]+wx-sx])
        if verbose:
            print('Изображение #2 (выделенная область). Анализ')
            print('\tMin: {:.2f}'.format(np.min(area)), end='')
            print('\tAvg: {:.2f}'.format(np.mean(area)), end='')
            print('\tMax: {:.2f}'.format(np.max(area)))
            # print('\tДисперсия: {:.2f}'.format(np.var(area)), end='')
            print('\tСтд. отклонение: {:.2f}'.format(np.std(area)))
            print('Подождите... ')
        res = {}
        if direction in [Direction.X, Direction.BOTH]:
            R = []
            for i in range(0, w - wx):
                temp = weighting(imgs[0][y[0]:y[1]+1, i:i+wx])
                if mode == CorrelationMode.MODE2:
                    if i - sx < 0 or i + wx - sx > w:
                        continue
                    area = weighting(imgs[1][y[0]-sy:y[1]+1-sy, i-sx:i+wx-sx])
                r = correlation_function(area, temp)
                R.append([i-x[0], r])
            res[Direction.X.value] = np.array(R)
        if direction in [Direction.Y, Direction.BOTH]:
            R = []
            for j in range(0, h - wy):
                temp = weighting(imgs[0][j:j+wy, x[0]:x[1]+1])
                if mode == CorrelationMode.MODE2:
                    if j - sy < 0 or j + wy - sy > h:
                        continue
                    area = weighting(imgs[1][j-sy:j+wy-sy, x[0]-sx:x[1]+1-sx])
                r = correlation_function(area, temp)
                R.append([j-y[0], r])
            res[Direction.Y.value] = np.array(R)
        if verbose:
            print(' OK.')
        return res


class RawReader:
    @staticmethod
    def read_raw(path: str, sizes: Union[Tuple[int, int], List[int]],
                 mode: ReadMode = None, dump=True) -> np.ndarray:
        w, h = sizes
        if mode is None:
            mode = path[-3:]
        if dump and os.path.exists('{}.dump'.format(path)):
            with open('{}.dump'.format(path), 'rb') as file:
                raw = dill.load(file)
        else:
            with open(path, 'rb') as file:
                if mode in [ReadMode.CMP, ReadMode.CMP.value]:
                    raw = np.zeros((h, w), dtype=np.complex)
                    for i in range(h - 1, -1, -1):
                        for j in range(w):
                            re = struct.unpack('f', file.read(4))[0]
                            im = struct.unpack('f', file.read(4))[0]
                            raw[i, j] = np.complex(real=re, imag=im)
                elif mode in [ReadMode.FLT, ReadMode.FLT.value]:
                    raw = np.zeros((h, w), dtype=np.float)
                    for i in range(h - 1, -1, -1):
                        for j in range(w):
                            raw[i, j] = struct.unpack('f', file.read(4))[0]
                else:
                    raise Exception('\'mode\' should be in {}'.format(list(ReadMode)))
        if dump and not os.path.exists('{}.dump'.format(path)):
            with open('{}.dump'.format(path), 'wb') as file:
                dill.dump(raw, file, recurse=True)
        return raw

    @staticmethod
    def load_raw(path: str) -> Any:
        with open(path, 'rb') as file:
            raw = dill.load(file)
        return raw


class RawConverter:
    @staticmethod
    def __checks(raw: np.ndarray) -> None:
        if raw is None:
            raise Exception('\'raw\' is not set')
        if raw.dtype not in [complex, np.complex]:
            warnings.warn('\'raw\' is not a complex matrix')

    @staticmethod
    def amplitude(raw: np.ndarray) -> np.ndarray:
        RawConverter.__checks(raw)
        if raw.dtype in [float, np.float]:
            warnings.warn('amplitude() of a real number interpreted as its absolute value')
        return np.absolute(raw)

    @staticmethod
    def phase(raw: np.ndarray, deg=False) -> np.ndarray:
        RawConverter.__checks(raw)
        if raw.dtype in [float, np.float]:
            raise Exception('can\'t calculate phase of a real number')
        return np.angle(raw, deg)

    @staticmethod
    def real(raw: np.ndarray) -> np.ndarray:
        RawConverter.__checks(raw)
        if raw.dtype == np.float:
            warnings.warn('nothing done')
        return np.real(raw)

    @staticmethod
    def imaginary(raw: np.ndarray) -> np.ndarray:
        RawConverter.__checks(raw)
        if raw.dtype in [float, np.float]:
            raise Exception('can\'t take imaginary part of a real number')
        return np.imag(raw)

    @staticmethod
    def intensity(raw: np.ndarray) -> np.ndarray:
        ampl = RawConverter.amplitude(raw)
        return ampl * ampl

    @staticmethod
    def convert(raw: np.ndarray, _return_type: DType = DType.AMPLITUDES,
                operations=do.nothing) -> np.ndarray:
        if _return_type == DType.AMPLITUDES:
            return operations(RawConverter.amplitude(raw))
        if _return_type == DType.PHASES:
            return operations(RawConverter.phase(raw))
        if _return_type == DType.REAL_PARTS:
            return operations(RawConverter.real(raw))
        if _return_type == DType.IMAG_PARTS:
            return operations(RawConverter.imaginary(raw))
        if _return_type == DType.INTENSITIES:
            return operations(RawConverter.intensity(raw))
        if _return_type == DType.AS_IS:
            return operations(raw)
        raise Exception('no such DType exists')


class RadarImage(RawReader, RawConverter, CommonOperations):
    def __init__(self, band: Band = Band.P,
                 path: str = None, read_mode: ReadMode = None, use_dumps: bool = True,
                 raw: np.ndarray = None):
        self.sizes = Sizes(band).tuple
        self.path, self.read_mode, self.use_dumps = path, read_mode, use_dumps
        self.raw = None
        if raw is not None:
            self.raw = raw
        elif path:
            self.read()

    def get(self):
        return self.raw

    def read(self) -> None:
        if not self.path:
            raise Exception('\'path\' is not set')
        self.raw = super(RadarImage, self).read_raw(self.path, self.sizes, self.read_mode, self.use_dumps)

    def load(self, path: str) -> None:
        self.raw = super(RadarImage, self).load_raw(path)

    def amplitude(self) -> np.ndarray:
        return super(RadarImage, self).amplitude(self.raw)

    @property
    def A(self) -> np.ndarray:
        return self.amplitude()

    def phase(self, deg=False) -> np.ndarray:
        return super(RadarImage, self).phase(self.raw, deg)

    @property
    def phi(self) -> np.ndarray:
        return self.phase()

    @property
    def phi_deg(self) -> np.ndarray:
        return self.phase(deg=True)

    def real(self) -> np.ndarray:
        return super(RadarImage, self).real(self.raw)

    @property
    def Re(self) -> np.ndarray:
        return self.real()

    def imaginary(self) -> np.ndarray:
        return super(RadarImage, self).imaginary(self.raw)

    @property
    def Im(self) -> np.ndarray:
        return self.imaginary()

    def intensity(self) -> np.ndarray:
        return super(RadarImage, self).intensity(self.raw)

    def convert(self, _return_type: DType = DType.AMPLITUDES, operations=do.nothing) -> np.ndarray:
        return super(RadarImage, self).convert(self.raw, _return_type, operations)

    def dump(self, path: str, _as_type: DType = DType.AS_IS) -> None:
        with open(path, 'wb') as file:
            dill.dump(self.convert(_as_type), file, recurse=True)

    def to_PIL_image(self, mode: DType = DType.AMPLITUDES, operations=do.nothing) -> PIL.Image:
        data = self.convert(_return_type=mode)
        return super(RadarImage, self).to_PIL_image(data, operations)

    def get_default_path_to_png(self, mode: DType = DType.AMPLITUDES) -> str:
        if self.path:
            return '{}.{}.png'.format(self.path, mode.value)
        return '{}.png'.format(mode.value)

    def save_png(self, mode: DType = DType.AMPLITUDES, path: str = None, operations=do.nothing) -> None:
        if not path:
            path = self.get_default_path_to_png(mode)
        data = self.convert(_return_type=mode)
        super(RadarImage, self).save_png(data, path, operations)

    def smoothing(self, n: int, exclusive=False) -> np.ndarray:
        return super(RadarImage, self).smoothing(self.raw, n, exclusive)

    def correlate(self, img: np.ndarray,
                  window: Union[Tuple[Tuple[int, int], Tuple[int, int]], List[Tuple[int, int]], List[List[int]]],
                  direction: Direction = Direction.X,
                  correlation_function=pearsonr, weighting=do.nothing,
                  shift: Union[Tuple[int, int], List[int]] = (0, 0),
                  mode: CorrelationMode = CorrelationMode.MODE1, verbose=True) -> dict:
        return super(RadarImage, self).correlate((self.raw, img),
                                                 window, direction, correlation_function, weighting, shift, mode,
                                                 verbose)
