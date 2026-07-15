"""Curated Raman-band registry used for pre-specified EV interpretation.

Assignments describe plausible biochemical context, not molecule-specific
quantitation or diagnostic evidence.  The windows are intentionally identical
to the registered ``whole_spectra.bands`` configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RamanBand:
    peak: int
    bounds: tuple[int, int]
    category: str
    assignment: str


RAMAN_BANDS: dict[str, RamanBand] = {
    "cytosine_786": RamanBand(786, (773, 796), "nucleic_acids", "Cytosine/uracil breathing/stretching modes"),
    "o_p_o_821": RamanBand(821, (812, 832), "nucleic_acids", "O-P-O stretch"),
    "tyrosine_851": RamanBand(851, (836, 874), "mixed", "Tyrosine doublet/glycan signature"),
    "c_o_c_sugar_925": RamanBand(925, (900, 938), "nucleic_acids", "C-O-C stretch in sugars"),
    "beta_sheet_980": RamanBand(980, (965, 995), "proteins", "C-C stretch beta-sheet"),
    "c_o_c_n_1060": RamanBand(1060, (1044, 1078), "proteins", "C-O/C-N stretch"),
    "lipid_cc_1089": RamanBand(1089, (1082, 1098), "lipids", "C-C stretch"),
    "po2_cn_1112": RamanBand(1112, (1105, 1120), "mixed", "PO2- stretch/C-N stretch"),
    "carotenoid_1156": RamanBand(1156, (1145, 1165), "carotenoids", "C-C stretch carotene"),
    "amide_iii_1258": RamanBand(1258, (1240, 1280), "mixed", "CH bend/amide III"),
    "ch2_twist_1301": RamanBand(1301, (1290, 1325), "lipids", "CH2 twist"),
    "ch2_ch3_scissor_1440": RamanBand(1440, (1430, 1445), "lipids", "CH2/CH3 scissor deformation"),
    "ch2_ch3_bend_1465": RamanBand(1465, (1450, 1475), "lipids", "CH2/CH3 bend"),
    "carotenoid_1525": RamanBand(1525, (1510, 1535), "carotenoids", "C=C stretch carotene"),
    "tryptophan_1555": RamanBand(1555, (1550, 1560), "proteins", "Tryptophan signature"),
    "amide_i_1657": RamanBand(1657, (1640, 1675), "mixed", "C=C stretch/amide I"),
    "lipid_carbonyl_1721": RamanBand(1721, (1710, 1730), "lipids", "C=O stretch (hydrogen bonded)"),
    "lipid_ester_1744": RamanBand(1744, (1735, 1755), "lipids", "C=O stretch"),
    "ch2_sym_stretch_2853": RamanBand(2853, (2845, 2860), "lipids", "CH2 symmetric stretch"),
    "ch2_asym_stretch_2890": RamanBand(2890, (2870, 2910), "mixed", "CH2 asymmetric stretch"),
    "ch3_asym_stretch_2937": RamanBand(2937, (2925, 2945), "mixed", "CH3 asymmetric stretch"),
}


def validate_configured_bands(bands: Mapping[str, object]) -> None:
    """Fail closed if analysis windows diverge from the pre-specified registry."""
    observed = {str(name): tuple(map(int, values)) for name, values in bands.items()}
    expected = {name: band.bounds for name, band in RAMAN_BANDS.items()}
    if observed != expected:
        raise ValueError("whole_spectra.bands does not match the curated 21-band Raman registry")
