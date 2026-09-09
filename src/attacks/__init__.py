from .lcg_attack import (
    recover_parameters,
    LCGPredictor,
    recover_modulus_and_parameters,
    attack_lcg,
)
from .mt19937_attack import (
    inverse_right_shift_xor,
    inverse_left_shift_xor_mask,
    untemper,
    recover_mt19937_state,
    attack_mt19937,
)
