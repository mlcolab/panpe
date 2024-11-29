# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt


def plot_data(
    data: "MeasuredData", color: str = "m", lw: float = 2.0, show: bool = True, **kwargs
):
    plt.errorbar(
        x=data.q.squeeze().cpu().numpy(),
        y=data.data.squeeze().cpu().numpy(),
        yerr=data.sigmas.squeeze().cpu().numpy(),
        color=color,
        lw=lw,
        **kwargs
    )

    plt.grid()
    plt.gca().set_yscale("log")
    plt.xlabel(r"$q$ (Å$^{-1}$)")
    plt.ylabel(r"$R(q)$")
    if show:
        plt.show()
