# DEFINITIONS OF THE G, O, T, S parameters as functions of no, zo, ni, zi




def GOTS_params(no, zo, ni,zi): 

    G = (ni**2*zi - no**2*zo)**2 / (ni*no*(ni*zi - no*zo)*(ni*zo - no*zi))
    O = (ni*zo - no*zi) / (zi*zo*(ni - no))
    T = (ni - no)*(ni + no)**2 / (4*ni*no*zi*zo*(ni*zi - no*zo))
    S = (ni + no)*(ni**2*zi - no**2*zo) / (2*ni*no*zi*zo*(ni*zi - no*zo))

    return G, O, T, S
