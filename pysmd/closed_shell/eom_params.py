#
# Copyright (c) 2026. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001
# for Los Alamos National Laboratory (LANL), which is operated by Triad
# National Security, LLC for the U.S. Department of Energy/National Nuclear
# Security Administration. All rights in the program are reserved by Triad
# National Security, LLC, and the U.S. Department of Energy/National Nuclear
# Security Administration. The Government is granted for itself and others
# acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
# in this material to reproduce, prepare derivative works, distribute copies
# to the public, perform publicly and display publicly, and to permit others
# to do so.

"""EOM parameters and dissipation coefficients for MD integration.

Values for dissipative electronic force terms are optimized
for different values of K_max, which controls how many previous
timesteps are stored.

Key references for optimized dissipative term alpha/kappa values:
    - Niklasson et al., JCP 130, 214109 (2009);
    - Steneteg et al.,  PRB  82, 075110 (2010);
    - Niklasson,        JCP 147, 054103 (2017);

Key references for optimized coordinate/velocity coefficient values:
    - Odell et al., JCP 131, 244106 (2009);
    - Odell et al., JCP 135, 224105 (2011);
    - Albaugh et al., JCTC 14, 499-511 (2018);
"""

# Pre-compute coefficents, optimal; l_max = 2
_a1_opt_lmax2 = 2.0 ** -0.5
_a2_opt_lmax2 = 1.0 - _a1_opt_lmax2

# Pre-compute coefficents, optimal; l_max = 3
_a1_opt_lmax3 = 0.919661523017399857
_a2_opt_lmax3 = 0.25 * (1.0 / _a1_opt_lmax3) - (0.5 * _a1_opt_lmax3)
_a3_opt_lmax3 = 1.0 - _a1_opt_lmax3 - _a2_opt_lmax3

# Optimal integration parameters; (keys: scheme, l_max)
EOM_INTEGRATION_COEFFS: dict[str, dict[int, dict[str, tuple[float, ...]]]] = {
    "verlet": { # scheme

        2: { # l_max = 2 (only l_max for verlet scheme)
            "coord_coeffs": (1.0, 0.0),
            "veloc_coeffs": (0.5, 0.5)
        }
    },

    "optimal": { # scheme

        2: { # l_max = 2
            "coord_coeffs": (_a1_opt_lmax2, _a2_opt_lmax2),
            "veloc_coeffs": (_a1_opt_lmax2, _a2_opt_lmax2)
        },

        3: { # l_max = 3
            "coord_coeffs": (_a1_opt_lmax3, _a2_opt_lmax3, _a3_opt_lmax3),
            "veloc_coeffs": (_a3_opt_lmax3, _a2_opt_lmax3, _a1_opt_lmax3)
        },

        4: { # l_max = 4
            "coord_coeffs": (0.5153528374311229364, -0.085782019412973646,
                             0.4415830236164665242,  0.1288461583653841854),

            "veloc_coeffs": (0.1344961992774310892, -0.2248198030794208058,
                             0.7563200005156682911,  0.3340036032863214255),
        },

        6: { # l_max (fifth-order optimal; l_max = 6)
            "coord_coeffs": (0.339839625839110000,  -0.088601336903027329,
                             0.5858564768259621188, -0.603039356536491888,
                             0.3235807965546976394,  0.4423637942197494587),

            "veloc_coeffs": (0.1193900292875672758,  0.6989273703824752308,
                            -0.1713123582716007754,  0.4012695022513534480,
                             0.0107050818482359840, -0.0589796254980311632),
        }
    }
}

# Dissipation expansion coefficients; (key: k_max)
EOM_DISS_COEFFS: dict[int, tuple[float, ...]] = {
    3: (-2.0, 3.0, 0.0, -1.0),
    4: (-3.0, 6.0, -2.0, -2.0, 1.0),
    5: (-6.0, 14.0, -8.0, -3.0, 4.0, -1.0),
    6: (-14.0, 36.0, -27.0, -2.0, 12.0, -6.0, 1.0),
    7: (-36.0, 99.0, -88.0, 11.0, 32.0, -25.0, 8.0, -1.0),
    8: (-99.0, 286.0, -286.0, 78.0, 78.0, -90.0, 42.0, -10.0, 1.0),
    9: (-286.0, 858.0, -936.0, 364.0, 168.0, -300.0, 184.0, -63.0, 12.0, -1.0)
}

# Dissipation kappa/alpha values; (keys: scheme, l_max, k_max)
EOM_DISS_KAPPA_ALPHA_PARAMS: dict[str, dict[int, dict[int, dict[str, float]]]] = {
    "verlet": { # scheme

        2: { # l_max = 2 (only l_max for verlet scheme)

            3: { # k_max = 3
                "kappa": 1.69,
                "alpha": 0.150
            },

            4: { # k_max = 4
                "kappa": 1.75,
                "alpha": 0.057
            },

            5: { # k_max = 5
                "kappa": 1.82,
                "alpha": 0.018
            },

            6: { # k_max = 6
                "kappa": 1.84,
                "alpha": 0.0055
            },

            7: { # k_max = 7
                "kappa": 1.86,
                "alpha": 0.0016
            },

            8: { # k_max = 8
                "kappa": 1.88,
                "alpha": 0.00044
            },

            9: { # k_max = 9
                "kappa": 1.89,
                "alpha": 0.00012
            },
        },
    },

    "optimal": { # scheme

        2: { # l_max = 2

            3: { # k_max = 3
                "kappa": 2.183,
                "alpha": 0.190
            },

            4: { # k_max = 4
                "kappa": 2.279,
                "alpha": 0.0712
            },

            5: { # k_max = 5
                "kappa": 2.271,
                "alpha": 0.0292
            },

            6: { # k_max = 6
                "kappa": 2.281,
                "alpha": 0.0101
            },

            7: { # k_max = 7
                "kappa": 2.295,
                "alpha": 0.00320
            },

            8: { # k_max = 8
                "kappa": 2.311,
                "alpha": 0.000958
            },

            9: { # k_max = 9
                "kappa": 2.327,
                "alpha": 0.000276
            },
        },

        3: { # l_max = 3

            3: { # k_max = 3
                "kappa": 3.856,
                "alpha": 0.403
            },

            4: { # k_max = 4
                "kappa": 4.025,
                "alpha": 0.155
            },

            5: { # k_max = 5
                "kappa": 4.172,
                "alpha": 0.0475
            },

            6: { # k_max = 6
                "kappa": 4.312,
                "alpha": 0.0125
            },

            7: { # k_max = 7
                "kappa": 4.376,
                "alpha": 0.00349
            },

            8: { # k_max = 8
                "kappa": 4.350,
                "alpha": 0.00117
            },

            9: { # k_max = 9
                "kappa": 4.323,
                "alpha": 0.000382
            },
        },

        4: { # l_max = 4

            3: { # k_max = 3
                "kappa": 3.891,
                "alpha": 0.363
            },

            4: { # k_max = 4
                "kappa": 4.061,
                "alpha": 0.139
            },

            5: { # k_max = 5
                "kappa": 4.187,
                "alpha": 0.0430
            },

            6: { # k_max = 6
                "kappa": 4.292,
                "alpha": 0.0116
            },

            7: { # k_max = 7
                "kappa": 4.332,
                "alpha": 0.00340
            },

            8: { # k_max = 8
                "kappa": 4.298,
                "alpha": 0.00121
            },

            9: { # k_max = 9
                "kappa": 4.288,
                "alpha": 0.000384
            },
        },

        6: { # l_max = 6 (FIFTH-order optimal)

            3: { # k_max = 3
                "kappa": 3.973,
                "alpha": 0.358
            },

            4: { # k_max = 4
                "kappa": 4.133,
                "alpha": 0.139
            },

            5: { # k_max = 5
                "kappa": 4.255,
                "alpha": 0.0434
            },

            6: { # k_max = 6
                "kappa": 4.361,
                "alpha": 0.0117
            },

            7: { # k_max = 7
                "kappa": 4.424,
                "alpha": 0.00316
            },

            8: { # k_max = 8
                "kappa": 4.385,
                "alpha": 0.00115
            },

            9: { # k_max = 9
                "kappa": 4.371,
                "alpha": 0.000371
            }
        }
    }
}


class EOMParams:
    """Base class for equation of motion integration parameters.

    Args:
        scheme: Type of integration scheme (default="verlet").
        l_max: Maximum number of sub-integration steps (default=2).
        k_max: Maximum order for dissipative terms (default=6).

    Attributes:
        coords_coeffs: optimal coefficients for coordinate update.
        veloc_coeffs: optimal coefficients for velocity update.
        diss_coeffs: optimal dissipative coefficient terms.
        diss_kappa: optimal dissipative kappa value.
        diss_alpha: optimal dissipative alpha value.
    """


    def __init__(
        self,
        scheme: str = "verlet",
        l_max: int = 2,
        k_max: int = 6,
    ) -> None:
        """Create object to store EOM parameters."""
        ### Validate EOM parameters
        # Integration scheme
        if scheme.lower() not in ("verlet", "optimal"):
            raise Exception(f"Unsupported integration scheme: {scheme}.")

        # Number of sub-integration steps requested for verlet scheme
        if scheme == "verlet" and l_max != 2:
            raise Exception("Verlet integration schemes only available " + \
                           f"for L_max = 2. Current L_max = {l_max}.")

        # Number of sub-integration steps requested for optimal scheme
        if scheme == "optimal" and l_max not in (2, 3, 4, 6):
            raise Exception("Optimal integration schemes only available " + \
                           f"for L_max in [2, 3, 4, 6]. Current L_max = {l_max}.")

        # Number of previous timesteps to include in dissipation terms
        if k_max not in range(3, 10):
            raise Exception("Optimal dissipative values only available " + \
                           f"for K_max in [3, 9]. Current K_max = {k_max}.")

        ### EOM scheme integration parameters
        self.coord_coeffs: tuple[float, ...] | None = None
        self.veloc_coeffs: tuple[float, ...] | None = None

        ### Dissipation coefficients and parameters
        self.diss_coeffs: tuple[float, ...] | None = None
        self.diss_kappa: float | None = None
        self.diss_alpha: float | None = None

        ### Update parameter attributes
        self.get_integration_params(scheme, l_max)
        self.get_dissipation_params(scheme, l_max, k_max)

        return


    def get_integration_params(
        self,
        scheme: str,
        l_max: int,
    ) -> None:
        """Update coordinate and velocity update coefficients."""
        self.coord_coeffs = EOM_INTEGRATION_COEFFS[scheme][l_max]["coord_coeffs"]
        self.veloc_coeffs = EOM_INTEGRATION_COEFFS[scheme][l_max]["veloc_coeffs"]
        return


    def get_dissipation_params(
        self,
        scheme: str,
        l_max: int,
        k_max: int,
    ) -> None:
        """Update dissipative term coefficients."""
        self.diss_coeffs = EOM_DISS_COEFFS[k_max]
        self.diss_kappa = EOM_DISS_KAPPA_ALPHA_PARAMS[scheme][l_max][k_max]["kappa"]
        self.diss_alpha = EOM_DISS_KAPPA_ALPHA_PARAMS[scheme][l_max][k_max]["alpha"]
        return
