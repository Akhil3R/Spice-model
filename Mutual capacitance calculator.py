import numpy as np

def calculate_coupling_coefficient(capacitance_matrix):
    """
    Calculate the mutual coupling coefficient (k) between conductors using TEM approximation.
    
    Parameters:
    -----------
    capacitance_matrix : 2D numpy array
        A 2x2 capacitance matrix [C] where:
        - C[0,0] = C11 (self-capacitance of conductor 1)
        - C[0,1] = C12 (mutual capacitance between conductors 1 and 2)
        - C[1,0] = C21 (mutual capacitance between conductors 2 and 1, should equal C12)
        - C[1,1] = C22 (self-capacitance of conductor 2)
    
    Returns:
    --------
    tuple: (k, L11, L22, M)
        - k: mutual coupling coefficient
        - L11: self-inductance of conductor 1
        - L22: self-inductance of conductor 2
        - M: mutual inductance between conductors 1 and 2
    """
    # Physical constants
    mu0 = 4 * np.pi * 1e-7       # Permeability of free space (H/m)
    epsilon0 = 8.85418782e-12    # Permittivity of free space (F/m)
    c = 299792458                # Speed of light (m/s)
    
    # Verify that μ₀ε₀ = 1/c²
    mu0_epsilon0 = mu0 * epsilon0
    inv_c_squared = 1 / (c * c)
    print(f"μ₀ε₀ = {mu0_epsilon0:.17e}")
    print(f"1/c² = {inv_c_squared:.17e}")
    
    # Convert to numpy array if it's not already
    C = np.array(capacitance_matrix)
    
    # Calculate the inverse of the capacitance matrix
    # In TEM approximation: [L] = μ₀ε₀[C]⁻¹
    try:
        C_inv = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        print("Error: Capacitance matrix is singular and cannot be inverted.")
        return None
    
    # Calculate the inductance matrix by scaling with μ₀ε₀
    L = mu0_epsilon0 * C_inv
    
    # Extract self-inductances and mutual inductance
    L11 = L[0, 0]  # Self-inductance of conductor 1
    L22 = L[1, 1]  # Self-inductance of conductor 2
    M = L[0, 1]    # Mutual inductance (should equal L[1, 0])
    
    # Verify symmetry of mutual inductance
    if not np.isclose(L[0, 1], L[1, 0]):
        print("Warning: Mutual inductances L12 and L21 are not equal.")
        print(f"L12 = {L[0, 1]}, L21 = {L[1, 0]}")
    
    # Calculate the coupling coefficient
    k = M / np.sqrt(L11 * L22)
    
    return k, L11, L22, M

# Example usage with the given capacitance values
if __name__ == "__main__":
    # Given capacitance values (F)
    C11 = 1.25e-10  # Self-capacitance of conductor 1
    C12 = -4.90e-16  # Mutual capacitance between conductors 1 and 2
    C13 = -1.25e-10  # (Not used in this calculation)
    
    C22 = 1.23e-10  # Self-capacitance of conductor 2
    C21 = -4.90e-16  # Mutual capacitance between conductors 2 and 1 (should equal C12)
    C23 = -1.22e-10  # (Not used in this calculation)
    
    # Create the 2x2 capacitance matrix
    # We only need the submatrix related to conductors 1 and 2
    C = np.array([
        [C11, C12],
        [C21, C22]
    ])
    
    print("Capacitance Matrix (2×2 submatrix):")
    print(C)
    
    # Calculate the coupling coefficient and related values
    result = calculate_coupling_coefficient(C)
    
    if result:
        k, L11, L22, M = result
        
        print("\nResults:")
        print(f"Self-inductance of conductor 1 (L11) = {L11:.6e} H")
        print(f"Self-inductance of conductor 2 (L22) = {L22:.6e} H")
        print(f"Mutual inductance (M) = {M:.6e} H")
        print(f"Coupling coefficient (k) = {k:.6e}")
        
        # Provide an interpretation of the k value
        if abs(k) < 0.01:
            print("\nInterpretation: Very weak coupling between the conductors.")
        elif abs(k) < 0.3:
            print("\nInterpretation: Weak coupling between the conductors.")
        elif abs(k) < 0.7:
            print("\nInterpretation: Moderate coupling between the conductors.")
        elif abs(k) < 0.9:
            print("\nInterpretation: Strong coupling between the conductors.")
        else:
            print("\nInterpretation: Very strong coupling, approaching ideal coupling.")
        
        # Verify that k is within the valid range [-1, 1]
        if abs(k) > 1:
            print("\nWarning: |k| > 1, which is physically impossible.")
            print("This suggests an error in the data or calculations.")
