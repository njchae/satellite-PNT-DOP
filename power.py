import numpy as np
def power():
    Pt = ... #power of transmitter
    Gt = ... #gain of transmitter
    boltz = ... #target cross section
    R1 = ... #range from GPS satellites to target
    R2 = ... #range from target to receiver
    Aef = ... #receiving antenna effective area


    Pr = (Pt * Gt) * (boltz * Aef) / (4 * np.pi * R1 ** 2 * 4 * np.pi * R2 ** 2)