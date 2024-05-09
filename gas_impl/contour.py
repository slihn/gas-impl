import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import List
from skimage.measure import find_contours


def get_xy_grid(x1, x2, y1, y2, n):
    x, y = np.meshgrid(np.linspace(x1, x2, n), np.linspace(y1, y2, n))
    return x, y

def get_mesh(x, y, fn, min_cap=-100.0, max_cap=100.0) -> np.ndarray:
    z = np.empty_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            p = fn(x[i, j], y[i, j])
            # print('get_mesh:', str(fn), [x[i, j], y[i, j], p])
            if not np.isfinite(p): 
                p = max_cap # if p > 0 else min_cap 
            z[i, j] = np.clip(p, a_min=min_cap, a_max=max_cap)

    return z

def add_contour_level(contour_levels: np.ndarray, more: List[float]) -> np.ndarray:
    for i in more:
        if i not in contour_levels:
            contour_levels = np.append(contour_levels, [i])
    contour_levels = np.sort(contour_levels)
    return contour_levels


class ContourPlotUtil:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def calculate_mesh(self, fn, min_cap=-100.0, max_cap=100.0) -> np.ndarray:
        self.min_cap = min_cap 
        self.max_cap = max_cap 
        self.z = get_mesh(self.x, self.y, fn, min_cap=min_cap, max_cap=max_cap)
        self.z_min, self.z_max = np.min(self.z), np.max(self.z)
        return self.z
    
    def plot_color_mesh(self, fig: Figure, ax: Axes, cmap='RdBu_r'):
        c = ax.pcolormesh(self.x, self.y, self.z, cmap=cmap, vmin=self.z_min, vmax=self.z_max, shading='auto')  # type: ignore
        fig.colorbar(c, ax=ax)

    def plot_contour_levels(self, ax, contour_levels, colors='blue', linewidths=0.25):
        contour = ax.contour(self.x, self.y, self.z, levels=contour_levels, colors=colors, linewidths=linewidths)
        ax.axis([self.x.min(), self.x.max(), self.y.min(), self.y.max()])
        ax.clabel(contour, inline=True, fontsize=6)


class ContourPlotBase:
    def __init__(self, name, mode, dict_settings, min_cap, max_cap, cmap='RdBu_r', contour_colors='blue', contour_linewidths=0.25, dist_family='GSaS'):
        self.name: str = name
        self.mode: str = mode  
        self.dist_family: str = dist_family
        self.debug_fn = False
        assert self.mode in ['ak', 'sk', 'ka', 'ks']

        self.dict_settings = dict_settings
        self.grid_settings = convert_dict_to_grid_settings(self.mode, dict_settings)
        self.min_cap = min_cap
        self.max_cap = max_cap
        self.cmap = cmap
        self.contour_levels: np.ndarray = np.ndarray([])
        self.contour_colors = contour_colors
        self.contour_linewidths = contour_linewidths
        self.cpu: ContourPlotUtil

    @property
    def z(self): return self.cpu.z

    @property
    def z_min(self): return self.cpu.z_min

    @property
    def z_max(self): return self.cpu.z_max

    def set_grid_and_mesh(self):
        x, y = get_xy_grid(*self.grid_settings)
        self.cpu = ContourPlotUtil(x, y)
        self.set_mesh()
    
    def calculate_mesh(self, mesh_fn):
        self.cpu.calculate_mesh(mesh_fn, min_cap=self.min_cap, max_cap=self.max_cap)

    def set_mesh(self):
        assert self.mesh_fn is not None
        # you can override this if you want to, but it should be fine for most cases
        self.calculate_mesh(self.mesh_fn)  

    def set_contour_levels(self): pass

    def generate_std_plot(self, fig, ax):    
        self.set_grid_and_mesh()
        self.set_contour_levels()

        cplot = self.cpu
        cplot.plot_color_mesh(fig, ax, cmap=self.cmap)
        cplot.plot_contour_levels(ax, self.contour_levels, colors=self.contour_colors, linewidths=self.contour_linewidths)

        self.additional_plot(ax)

    def additional_plot(self, ax):  # override this with your own, if so desired
        self.set_std_title(ax)
        self.std_xy_axes_by_mode(ax)

    ### mode
    def call_alpha_k_func_by_mode(self, fn, x, y):
            if self.mode in ['ka', 'ks']:
                x, y = y, x

            if 'a' in self.mode: 
                alpha = x
            elif 's' in self.mode: 
                alpha = 1/x 
            else:
                raise Exception(f"ERROR: undefined mode: {self.mode}")
            p = fn(float(alpha), float(y))  # float is slightly different from np.float64 for implicit complex conversion
            if self.debug_fn == True:
                print('call_alpha_k_func_by_mode:', str(fn), [alpha, y, p])
            return p

    def set_std_title(self, ax):
        a_str = "alpha" if 'a' in self.mode else "s (1/alpha)"
        ax.set_title(f'{self.name} of {self.dist_family} by {a_str}: capped by [{self.z_min:.1f}, {self.z_max:.1f}]')

    def std_xy_axes_by_mode(self, ax):
        if 'ak' in self.mode:
            std_alpha_xy_axes(ax)
        elif 'sk' in self.mode:
            std_inv_alpha_xy_axes(ax)
        elif 'ka' in self.mode:
            std_k_alpha_xy_axes(ax)
        elif 'ks' in self.mode:
            std_k_inv_alpha_xy_axes(ax)
        else:
            raise Exception(f"ERROR: undefined mode: {self.mode}")

    def get_default_contour_levels(self):
        # use this to experiment, but this is not for production levels
        z_min, z_max = self.z_min, self.z_max
        if z_min > 0:
            contour_levels = np.linspace(z_min, z_max, 20)  # Adjust as needed
        else:
            contour_levels = np.linspace(z_max/1000, z_max, 20) + np.linspace(z_min, z_min/1000, 20)
        return add_contour_level(contour_levels, [0.0, 1.0])


# ----------------------------------------------------------
def convert_dict_to_grid_settings(mode: str, dict_settings):
    d = dict_settings
    a_key = 'alpha' if 'a' in mode else 's'
    if mode.startswith('k'):
        grid_settings = d['k'] + d[a_key] + (d['n'],)
    else:
        grid_settings = d[a_key] + d['k'] + (d['n'],)
    return grid_settings


def set_std_fig_size(mode):
    if mode.startswith('k'):
        plt.rcParams['figure.figsize'] = [6.5, 4.5]
    else:
        plt.rcParams['figure.figsize'] = [5.5, 5.0]

# ----------------------------------------------------------
def std_x_axis(ax: Axes, label, lines):
    ax.set_xlabel(label)
    for x in lines:
        ax.axvline(x, color="orange", alpha=0.5, linestyle='--', linewidth=1.5)  # type: ignore


def std_y_axis(ax: Axes, label, lines):
    ax.set_ylabel(label)
    for x in lines:
        ax.axhline(x, color="orange", alpha=0.5, linestyle='--', linewidth=1.5)  # type: ignore


def _std_k_axis(ax, fn):
    fn(ax, 'k (degree of freedom)', [ 1, 2, 3, 4, 5, 10])

def _std_a_axis(ax, fn, inv=False):
    label = 'alpha' if not inv else 's (inverse of alpha)'
    fn(ax, label, [ 0.5, 1.0, 2.0])


def std_alpha_xy_axes(ax: Axes):
    _std_a_axis(ax, std_x_axis)
    _std_k_axis(ax, std_y_axis)

def std_inv_alpha_xy_axes(ax: Axes):
    _std_a_axis(ax, std_x_axis, inv=True)
    _std_k_axis(ax, std_y_axis)

def std_k_alpha_xy_axes(ax: Axes):
    _std_a_axis(ax, std_y_axis)
    _std_k_axis(ax, std_x_axis)

def std_k_inv_alpha_xy_axes(ax: Axes):
    _std_a_axis(ax, std_y_axis, inv=True)
    _std_k_axis(ax, std_x_axis)


# contour solver: find the intersection between kurtosis path and pdf0 path
class CountourSolver:
    def __init__(self, kurt_target, pdf0_target, kurt_fn, pdf0_fn, points=101):
        self.alpha_min, self.alpha_max = 0.1, 1.0
        # a_min, a_max = 0.55, 0.9
        # k_min, k_max = 2.95, 3.5
        # k_min, k_max = 0.95, 1.5
        self.k_min, self.k_max = 1.0, 10.05
        self.points = points

        self.kurt_target = kurt_target
        self.pdf0_target = pdf0_target

        self.kurt_cap = 30.0  # max kurt cap
        self.pdf0_cap = 1.5  # max pdf0 cap
        
        self.kurt_fn = kurt_fn
        self.pdf0_fn = pdf0_fn
        
        self.alpha = None
        self.k = None 

        self.kurt_contours = None
        self.pdf0_contours = None

    def _eval_point(self, i, j, fn, cap):
        v = fn(self.alpha[i, j], k=self.k[i, j])
        if not np.isfinite(v): v = cap 
        return np.clip(v, a_min=-cap, a_max=cap)
        
    def find_contours(self, debug=False):
        self.alpha, self.k = np.meshgrid(
            np.linspace(self.alpha_min, self.alpha_max, self.points), 
            np.linspace(self.k_min, self.k_max, self.points)
        )
    
        zk = np.empty_like(self.k)
        zp = np.empty_like(self.k)
        for i in range(self.k.shape[0]):
            if i % int(self.points / 10) == 0 and debug == True: print(f"kurt i={i}")
            for j in range(self.k.shape[1]):
                zk[i, j] = self._eval_point(i, j, self.kurt_fn, self.kurt_cap) 
                zp[i, j] = self._eval_point(i, j, self.pdf0_fn, self.pdf0_cap)

        self.kurt_contours = find_contours(zk, self.kurt_target)
        self.pdf0_contours = find_contours(zp, self.pdf0_target)
        print(f"kr_contours found: {len(self.kurt_contours)}; pdf_contours found: {len(self.pdf0_contours)}")

    def a_coords(self, contour):
        return self.alpha[0, contour[:, 1].astype(int)]
    
    def k_coords(self, contour):
        return self.k[contour[:, 0].astype(int), 0]

    def _get_idx_of_contours(self, contour, fn, target, tol):
        a_coords = self.a_coords(contour)
        k_coords = self.k_coords(contour)
        arr = np.array([fn(a_coords[i], k=k_coords[i]) for i in np.arange(len(a_coords))])
        idx = [i for i in np.arange(len(arr)) if target-tol < arr[i] and arr[i] < target+tol]  # filter the good result
        return idx, arr

    def plot_solution(self, save_fig_file=None):
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams.update({'font.size': 6})
        plt.rcParams['figure.figsize'] = [10, 4]

        assert self.kurt_target > 0
        assert self.pdf0_target > 0
        fig, (ax1, ax2) = plt.subplots(1, 2)

        label = 'kurt'
        for contour in self.kurt_contours:
            idx, _ = self._get_idx_of_contours(contour, self.kurt_fn, self.kurt_target, tol=5.0)
            a_coords = self.a_coords(contour)
            k_coords = self.k_coords(contour)
            ax1.plot(k_coords[idx], a_coords[idx], color="blue", linewidth=1, label=label)
            ax2.plot(k_coords[idx], 1/a_coords[idx], color="blue", linewidth=1, label=label)
            label = None  # only show this once

        for contour in self.pdf0_contours:
            idx, p0 = self._get_idx_of_contours(contour, self.pdf0_fn, self.pdf0_target, tol=0.1)
            a_coords = self.a_coords(contour)
            k_coords = self.k_coords(contour)
            ax1.plot(k_coords[idx], a_coords[idx], color="red", linewidth=1, label='pdf0')
            ax2.plot(k_coords[idx], 1/a_coords[idx], color="red", linewidth=1, label='pdf0')
            print(f"debug pdf0 mean: {p0[idx].mean()}")

        ax1.set_xlabel('k (degree of freedom)')
        ax1.set_ylabel('alpha')
        ax1.legend(loc="lower left")
        
        ax2.set_xlabel('k (degree of freedom)')
        ax2.set_ylabel('s (1/alpha)')
        ax2.legend(loc="upper left")
        
        fig.suptitle(f"find solution for pdf0_target = {self.pdf0_target}, kurt_target = {self.kurt_target}")

        plt.show()
        if save_fig_file is not None:
            fig.savefig(save_fig_file, format='png', dpi=300)

