"""Test of data set functionalities"""
import pathlib

import lmfit
import numpy as np

from nanite import IndentationGroup
from nanite.model import model_sneddon_spherical as hertzSpherical


data_path = pathlib.Path(__file__).resolve().parent / "data"
jpkfile = data_path / "fmt-jpk-fd_spot3-0192.jpk-force"


def test_app_ret():
    ds = IndentationGroup(jpkfile)
    ar = ds[0]
    ar.apply_preprocessing(["compute_tip_position",
                            "correct_force_offset"])
    idp = ar.estimate_contact_point_index()

    aprid = ar["segment"] == 0
    x = ar["tip position"][aprid]
    y = ar["force"][aprid]
    contact_point = x[idp]

    # crop x and y around contact_point
    distcp = x.shape[0]-idp
    x = x[-3*distcp:]
    y = y[-3*distcp:]

    params = lmfit.Parameters()
    params.add("contact_point", value=contact_point)
    params.add("baseline", value=0, vary=False)
    params.add("E", value=15e3, min=0)
    params.add("nu", value=.5, vary=False)
    params.add("R", value=10e-6, vary=False)

    fit_n = lmfit.minimize(hertzSpherical.residual, params, args=(x, y, False))
    # Correctly reproduces fit results in the JPK analysis software
    # with "Vertical Tip Position", "Switchable Baseline Operation",
    # and a spherical indenter with the "Hertz/Sneddon" "Model type".
    # The contact point was calculated through Hertz model and not adjusted
    # before fitting by JPK Software.
    # Set tip radius to 10um.
    E_jpk = 14.94e3
    cp_jpk = 18.04e-6
    assert np.allclose(fit_n.params["E"].value, E_jpk, rtol=1e-1, atol=0)
    assert np.allclose(
        fit_n.params["contact_point"].value, cp_jpk, rtol=4e-3, atol=0)


def test_fit_apret():
    ds = IndentationGroup(jpkfile)
    ar = ds[0]
    ar.apply_preprocessing(["compute_tip_position",
                            "correct_force_offset"])
    ar.fit_model(model_key="sneddon_spher",
                 params_initial=None,
                 x_axis="tip position",
                 y_axis="force",
                 weight_cp=False,
                 segment="retract")
