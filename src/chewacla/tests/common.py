"""Used in various tests."""

import numpy as np

# https://physics.nist.gov/cgi-bin/cuu/Value?asil
SI_A0 = 5.431020511
SI_A0_UNCERTAINTY = 0.000000089

TWO_PI = 2 * np.pi

# lattice constants: a, b, c, alpha, beta, gamma
CUBIC_5A_LATTICE = 5, 5, 5, 90, 90, 90
DOLOMITE_LATTICE = 4.8012, 4.8012, 16.002, 90, 90, 120
EUPTIN4_EH1_VER_LATTICE = 4.542, 16.955, 7.389, 90, 90, 90
SILICON_LATTICE = SI_A0, SI_A0, SI_A0, 90, 90, 90
VIBRANIUM_LATTICE = TWO_PI, TWO_PI, TWO_PI, 90, 90, 90

APS_FOURC_GEOMETRY = dict(
    # E4CV
    incident_beam="z+",  # APS coordinate system
    sample_stage=dict(omega="x-", chi="z+", phi="x+"),
    detector_stage=dict(ttheta="x-"),
)
APS_PSIC_GEOMETRY = dict(
    # E6C with psic names, H. You
    incident_beam="z+",  # APS coordinate system
    sample_stage=dict(mu="y+", eta="x-", chi="z+", phi="x+"),
    detector_stage=dict(nu="y+", delta="x-"),
)
APS_SIXC_GEOMETRY = dict(
    # IUCr 6-circle, Lohmeier & Vlieg
    incident_beam="z+",  # APS coordinate system
    sample_stage=dict(alpha="y+", omega="x-", chi="z+", phi="x+"),
    detector_stage=dict(alpha="y+", delta="x-", gamma="y+"),
)
E4CH_GEOMETRY = dict(
    # https://people.debian.org/~picca/hkl/hkl.html#org800c866
    incident_beam="x+",  # z+: anti-gravity
    sample_stage=dict(omega="z+", chi="x+", phi="z+"),
    detector_stage=dict(tth="z+"),
)
E4CV_GEOMETRY = dict(
    # https://people.debian.org/~picca/hkl/hkl.html#org1a91260
    incident_beam="x+",
    sample_stage=dict(omega="y-", chi="x+", phi="y-"),
    detector_stage=dict(tth="y-"),
)
E6C_GEOMETRY = dict(
    # https://people.debian.org/~picca/hkl/hkl.html#orgf48ceba
    incident_beam="x+",
    sample_stage=dict(mu="z+", omega="y-", chi="x+", phi="y-"),
    detector_stage=dict(gamma="z+", delta="y-"),
)
KAPPA_6_CIRCLE_GEOMETRY = dict(
    incident_beam="x+",
    # kappa angle: alpha = 50 degrees
    sample_stage=dict(mu="z+", keta="y-", kappa=[0.0, -0.6427876096865394, -0.766044443118978], kphi="y-"),
    detector_stage=dict(nu="z+", tth="y-"),
)
K4CV_GEOMETRY = dict(
    # https://people.debian.org/~picca/hkl/hkl.html#org182695d
    incident_beam="x+",
    # kappa angle: alpha = 50 degrees
    sample_stage=dict(komega="y-", kappa=[0.0, -0.6427876096865394, -0.766044443118978], kphi="y-"),
    detector_stage=dict(tth="y-"),
)
