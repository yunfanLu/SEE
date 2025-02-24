from functools import partial

import numba
import numpy as np
from absl.logging import info

DEBUG = False


class VoxelGridConfig:
    def __init__(self, channel, H, W, to_bin, scale):
        self.type = "voxel_grid"
        self.channel = channel
        self.H = H
        self.W = W
        self.to_bin = to_bin
        self.scale = scale


class TemporalPyramidConfig:
    def __init__(self, H, W, pyramid_level, pyramid_moments, reduction_factor):
        self.type = "temporal_pyramid_centered"
        self.H = H
        self.W = W
        self.pyramid_level = pyramid_level
        self.pyramid_moments = pyramid_moments
        self.reduction_factor = reduction_factor


class EventRepresentationBuilder:
    def __init__(self, event_representation_config):
        self._cfg = event_representation_config
        if self._cfg.type == "voxel_grid":
            self._builder = partial(
                self._voxel_grid,
                channel=self._cfg.channel,
                H=self._cfg.H,
                W=self._cfg.W,
                to_bin=self._cfg.to_bin,
                scale=self._cfg.scale,
            )
        elif self._cfg.type == "temporal_pyramid_centered":
            self._builder = partial(
                self._temporal_pyramid,
                H=self._cfg.H,
                W=self._cfg.W,
                pyramid_level=self._cfg.pyramid_level,
                pyramid_moments=self._cfg.pyramid_moments,
                reduction_factor=self._cfg.reduction_factor,
            )
        elif self._cfg.type == "zeros":
            self._builder = partial(
                self._zeros,
                channel=self._cfg.channel,
                H=self._cfg.H,
                W=self._cfg.W,
            )
        elif self._cfg.type == "empty":
            self._builder = lambda es: None
        else:
            raise NotImplementedError

    def __call__(self, es, *args, **kwargs):
        """
        es: event stream np.ndarray, shape=(N, 4), (t, x, y, p) dtype=np.float32
        """
        es = es[es[:, 0].argsort()]
        return self._builder(es, *args, **kwargs)

    def _temporal_pyramid_centered(self, es, H, W, pyramid_level, pyramid_moments, reduction_factor):
        temporal_pyramid = np.zeros(shape=[pyramid_level, pyramid_moments, H, W], dtype=np.float32)
        if es.shape[0] == 0:
            return temporal_pyramid
        es = es[es[:, 0].argsort()]
        t_s = es[:, 0].min()
        t_e = es[:, 0].max()
        during_time = t_e - t_s
        time_start_end_list = [[0, 1]]
        for i in range(pyramid_level):
            l, r = time_start_end_list[-1]
            during = r - l
            deta = (during - during / reduction_factor) / 2
            l = l + deta
            r = r - deta
            time_start_end_list.append([l, r])
        for i in range(pyramid_level):
            l, r = time_start_end_list[i]
            l = l * during_time + t_s
            r = r * during_time + t_s
            moment_during_time = r - l
            for j in range(pyramid_moments):
                m_t_l = l + moment_during_time * j / pyramid_moments
                m_t_r = l + moment_during_time * (j + 1) / pyramid_moments
                left_index = np.searchsorted(es[:, 0], m_t_l, side="left")
                right_index = np.searchsorted(es[:, 0], m_t_r, side="right")
                li, ri = left_index, right_index
                x, y, p = es[li:ri, 1], es[li:ri, 2], es[li:ri, 3]
                x = x.astype(np.int32)
                y = y.astype(np.int32)
                temporal_pyramid[i, j] = self._render(x=x, y=y, p=p, shape=(H, W))
        return temporal_pyramid

    def _zeros(self, es, channel, H, W):
        """This function are used for empty representation for ablation studies to proof effect of event representation"""
        voxel_grid = np.zeros((channel, H, W), dtype=np.float32)
        return voxel_grid

    def _voxel_grid(self, es, channel, H, W, to_bin, scale):
        voxel_grid = np.zeros((channel, H, W), dtype=np.float32)
        if es.shape[0] == 0:
            return voxel_grid
        es = es[es[:, 0].argsort()]
        t, x, y, p = es[:, 0], es[:, 1], es[:, 2], es[:, 3]
        t_s = t.min()
        t_e = t.max()
        t_step = (t_e - t_s) / channel
        p = p * 2 - 1
        for i in range(channel):
            t_l = t_s + i * t_step
            t_r = t_s + (i + 1) * t_step
            t_l_i = np.searchsorted(t, t_l)
            t_r_i = np.searchsorted(t, t_r)
            x_i, y_i, p_i = x[t_l_i:t_r_i], y[t_l_i:t_r_i], p[t_l_i:t_r_i]
            x_i = x_i.astype(np.int32)
            y_i = y_i.astype(np.int32)
            voxel_grid[i] = self._render(x=x_i, y=y_i, p=p_i, shape=(H, W), to_bin=to_bin, scale=scale)
        return voxel_grid

    def _render(self, x, y, p, shape, to_bin, scale):
        moments = np.zeros(shape=shape, dtype=np.float32)
        np.add.at(moments, (y, x), p)
        if scale > 1:
            moments = moments / scale
        if to_bin:
            moments[moments > 0] = 1
            moments[moments < 0] = -1
        return moments
