#!/usr/bin/env python
#
# Code to compute the 1-loop EPT power spectrum for
# and EdS Universe and a power-law power spectrum.
# Appendix B of https://arxiv.org/pdf/astro-ph/9602070
# This paper uses the 4pi k^3 P convention for Delta2.
#
import numpy as np
from scipy.special import gamma



def compute_q13(n):
    """
    Computes the q_13(n) dimensional regularization factor
    for the 1-loop power spectrum.
    """
    # Calculate individual terms for readability
    t1 = - (gamma((n + 1)/2) * gamma((1 - n)/2)) / (84 * gamma(1 - n/2) * gamma(1 + n/2))
    t2 = - (19 * gamma(-(n + 3)/2) * gamma((n + 5)/2)) / (84 * gamma(-1 - n/2) * gamma(3 + n/2))
    t3 =   (gamma(-(n + 5)/2) * gamma((n + 7)/2)) / (12 * gamma(-2 - n/2) * gamma(4 + n/2))
    t4 =   (5 * gamma(-(n + 1)/2) * gamma((n + 3)/2)) / (28 * gamma(2 + n/2) * gamma(-n/2))
    t5 = - (gamma((n - 1)/2) * gamma((3 - n)/2)) / (42 * gamma(2 - n/2) * gamma(n/2))
    return np.pi**2 * (t1+t2+t3+t4+t5)


def compute_q22_old(n):
    """
    Computes the q_22(n) dimensional regularization factor
    for the 1-loop power spectrum.
    """
    # Calculate individual terms for readability
    t1 =   (gamma(2.5 - n) * gamma((n - 1)/2)**2) / (2 * gamma(2 - n/2)**2 * gamma(n - 1))
    t2 =   (3 * gamma(1.5 - n) * gamma((n - 1)/2) * gamma((n + 1)/2)) / (gamma(1 - n/2) * gamma(2 - n/2) * gamma(n))
    t3 =   (29 * gamma(0.5 - n) * gamma((n + 1)/2)**2) / (4 * gamma(1 - n/2)**2 * gamma(n + 1))
    t4 =   (11 * gamma(0.5 - n) * gamma((n - 1)/2) * gamma((n + 3)/2)) / (4 * gamma(2 - n/2) * gamma(-n/2) * gamma(n + 1))
    t5 = - (15 * gamma(-0.5 - n) * gamma((n - 1)/2) * gamma((n + 5)/2)) / (2 * gamma(-1 - n/2) * gamma(2 - n/2) * gamma(n + 2))
    t6 =   (15 * gamma(-0.5 - n) * gamma((n + 1)/2) * gamma((n + 3)/2)) / (2 * gamma(1 - n/2) * gamma(-n/2) * gamma(n + 2))
    t7 = - (25 * gamma(-1.5 - n) * gamma((n + 1)/2) * gamma((n + 5)/2)) / (gamma(-1 - n/2) * gamma(1 - n/2) * gamma(n + 3))
    t8 =   (25 * gamma(-1.5 - n) * gamma((n - 1)/2) * gamma((n + 7)/2)) / (4 * gamma(-2 - n/2) * gamma(2 - n/2) * gamma(n + 3))
    t9 =   (75 * gamma(-1.5 - n) * gamma((n + 3)/2)**2) / (4 * gamma(-n/2)**2 * gamma(n + 3))
    return (np.pi**1.5 / 49.0) * (t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8 + t9)




def compute_q22(n):
    """
    Computes the q_22(n) dimensional regularization factor
    for the 1-loop power spectrum.
    """
    pr = np.pi**1.5 / 49.0
    t1 = (gamma(5/2 - n) * gamma((n - 1)/2)**2) / \
         (2 * gamma(2 - n/2)**2 * gamma(n - 1))
    t2 = (3 * gamma(3/2 - n) * gamma((n - 1)/2) * gamma((n + 1)/2)) / \
         (gamma(1 - n/2) * gamma(2 - n/2) * gamma(n))
    t3 = (29 * gamma(1/2 - n) * gamma((n + 1)/2)**2) / \
         (4 * gamma(1 - n/2)**2 * gamma(n + 1))
    t4 = (11 * gamma(1/2 - n) * gamma((n - 1)/2) * gamma((n + 3)/2)) / \
         (4 * gamma(2 - n/2) * gamma(-n/2) * gamma(n + 1))
    t5 = (15 * gamma(-1/2 - n) * gamma((n - 1)/2) * gamma((n + 5)/2)) / \
         (2 * gamma(-1 - n/2) * gamma(2 - n/2) * gamma(n + 2))
    t6 = ((15 * gamma(-1/2 - n) * gamma((n + 1)/2)) / (2 * gamma(1 - n/2) * gamma(-n/2))) * \
         (gamma((n + 3)/2) / gamma(n + 2))
    t7 = (25 * gamma(-3/2 - n) * gamma((n + 1)/2) * gamma((n + 5)/2)) / \
         (gamma(-1 - n/2) * gamma(1 - n/2) * gamma(n + 3))
    t8 = (25 * gamma(-3/2 - n) / (4 * gamma(-2 - n/2))) * \
         (gamma((n - 1)/2) * gamma((n + 7)/2) / (gamma(2 - n/2) * gamma(n + 3)))
    t9 = (75 * gamma(-3/2 - n) * gamma((n + 3)/2)**2) / \
         (4 * gamma(-n/2)**2 * gamma(n + 3))
    # Combine all terms with their respective signs from the equation
    result = pr*(t1+t2+t3+t4-t5+t6-t7+t8+t9)
    return result





def P_linear(k, A, n):
    """
    Computes the linear theory power spectrum P_11(k).
    """
    return A * k**n




def P_1loop(k, A, n):
    """
    Computes the 1-loop power spectrum P_13(k) + P_22(k).
    """
    q13_val = compute_q13(n)
    q22_val = compute_q22(n)
    # Calculate P13 and P22
    P13 = (A**2) * (k**(2*n + 3)) * q13_val
    P22 = (A**2) * (k**(2*n + 3)) * q22_val
    return P13 + P22





if __name__ == "__main__":
    # Note: Avoid spectral indices n that cause divergences
    # (e.g., n >= -1 or n <= -3) and integer or half-integer values
    # of n to avoid poles in the Gamma functions.
    #
    # Look at the size of the "correction" prefactor.
    print("# Scale free power spectra.")
    print(f"# {'n':>8} {'q(n)':>8} {'q(n)/4pi':>10}")
    for n_val in [-1.25,-1.50001,-1.75,-2.0001,-17./8.,-2.25,-19./8.,-2.50001]:
        qtot = compute_q13(n_val)+compute_q22(n_val)
        qdel = qtot / (4*np.pi)
        print(f"# {n_val:8.4f} {qtot:8.2f} {qdel:10.4f}") 
    print("#",flush=True)
    #
    # Define parameters for a scale-free universe
    # We want 4pi k^3P=1 at knl, say k=1.
    n_val = -1.50001
    A_val = 1/(4*np.pi)
    # Create an array of k values -- really k/knl.
    ks = np.geomspace(1e-2,10.,11)
    # 
    # Compute the components of the power spectrum
    q13_result = compute_q13(n_val)
    q22_result = compute_q22(n_val)
    P11_res    = P_linear(ks,A_val,n_val)
    P1loop_res = P_1loop( ks,A_val,n_val)
    ratio      = np.abs(P1loop_res)/P11_res
    # and print the results.
    print(f"# Spectral Index (n) = {n_val}, q(n)={q13_result+q22_result:.4f}")
    print(f"# {'k/knl':>6} {'Delta2_11':>15}"+\
          f" {'P_11(k)':>15} {'P_1loop(k)':>15} {'Ratio':>15}")
    for i in range(len(ks)):
        print(f"{ks[i]:8.3f} {4*np.pi*ks[i]**3*P11_res[i]:15.6e}"+\
              f" {P11_res[i]:15.6e} {P1loop_res[i]:15.6e}"+\
              f" {ratio[i]:15.6e}")
    #
    # Now work out what value of k/knl leads to P1loop/Ptree=thresh.
    thresh,ks = 0.10,np.geomspace(1e-4,1e1,1501)
    print("#")
    print(f"# |P1loop|/Ptree={thresh} for k/knl of:",flush=True)
    #for n_val in [-1.1,-1.3,-1.50001,-1.7,-1.9,-2.1,-2.3]:
    for n_val in [-1.25,-1.50001,-1.75,-2.0001,-17./8.,-2.25,-19./8.,-2.50001]:
        ratio   = np.abs(P_1loop( ks,A_val,n_val))/P_linear(ks,A_val,n_val)
        kthresh = ks[np.nonzero(ratio<thresh)[0][-1]]
        print(f"# n={n_val:8.4f}, k/knl={kthresh:8.5f}") 
