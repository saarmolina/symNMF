import sys
import numpy as np
import symnmf  # C extension module

def read_data_file(file_name):
    """
    Read and parse data from input file.
    Args:
        file_name: Path to input file
    Returns:
        data: List of lists containing the data points
        n: Number of points
        d: Number of dimensions
    """
    try:
       data = np.loadtxt(file_name, delimiter=',')
       # Handle 1D case - convert to 2D array
       if len(data.shape) == 1:
           data = data.reshape(-1, 1)
       data = data.tolist()
       n = len(data)
       d = len(data[0]) 
       return data, n, d
    except Exception:
       print("An Error Has Occurred") 
       sys.exit(1)

def validate_args():
    """
    Validate command line arguments.
    Returns:
        k: Number of clusters
        goal: Type of calculation to perform
        file_name: Input file path
    """
    try:
        if len(sys.argv) != 4:
            print("An Error Has Occurred")
            sys.exit(1)
        return int(sys.argv[1]), sys.argv[2], sys.argv[3]
    except ValueError:
        print("An Error Has Occurred")
        sys.exit(1)

def initialize_h(W, n, k):
    """
    Initialize H matrix for symNMF.
    Args:
        W: Normalized similarity matrix
        n: Number of points
        k: Number of clusters
    Returns:
        H: Initial H matrix
    """
    np.random.seed(1234)
    m = np.mean(W)
    return np.random.uniform(0, 2 * np.sqrt(m/k), size=(n, k)).tolist()

def main():
    """
    Main function to handle the symNMF program execution.
    Reads input, performs calculations based on goal,
    and outputs results.
    """
    k, goal, file_name = validate_args()
    data, n, d = read_data_file(file_name)

    if goal == "symnmf":
        W = symnmf.norm(data)
        if W is None:
            print("An Error Has Occurred")
            sys.exit(1)
        H = initialize_h(W, n, k)
        result = symnmf.symnmf(W, H, n, k)
        
    elif goal == "sym":
        result = symnmf.sym(data)
       
    elif goal == "ddg":
        result = symnmf.ddg(data)
        
    elif goal == "norm":
        result = symnmf.norm(data)
        
    else:
        print("An Error Has Occurred")
        sys.exit(1)

    if result is None:
        print("An Error Has Occurred")
        sys.exit(1)
    
    print_matrix(result)

def print_matrix(matrix):
    # Format and print matrix with 4 decimal places
    for row in matrix:
        print(','.join(f'{x:.4f}' for x in row))

if __name__ == "__main__":
    main()