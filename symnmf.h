#ifndef SYMNMF_H
#define SYMNMF_H

/* Core algorithm functions */

/*
 * Calculate similarity matrix from input points
 * @param points: Input data points as n x d matrix
 * @param n: Number of data points
 * @param d: Number of dimensions
 * @return: n x n similarity matrix, or NULL if error occurs
 */
double** sym(double** points, int n, int d);

/*
 * Calculate diagonal degree matrix
 * @param points: Input data points as n x d matrix
 * @param n: Number of data points
 * @param d: Number of dimensions
 * @return: n x n diagonal degree matrix, or NULL if error occurs
 */
double** ddg(double** points, int n, int d);

/*
 * Calculate normalized similarity matrix
 * @param points: Input data points as n x d matrix
 * @param n: Number of data points
 * @param d: Number of dimensions
 * @return: n x n normalized similarity matrix, or NULL if error occurs
 */
double** norm(double** points, int n, int d);

/*
 * Perform Symmetric NMF algorithm
 * @param W: Input normalized similarity matrix (n x n)
 * @param H: Initial H matrix (n x k)
 * @param n: Number of data points
 * @param k: Number of clusters
 * @return: Final H matrix (n x k), or NULL if error occurs
 */
double** symnmf(double** W, double** H, int n, int k);

/* Matrix operation functions */

/*
 * Multiply two matrices: A(n x m) and B(m x p)
 * @param A: First matrix
 * @param B: Second matrix
 * @param n: Number of rows in A
 * @param m: Number of columns in A (and rows in B)
 * @param p: Number of columns in B
 * @return: Result matrix (n x p), or NULL if error occurs
 */
double** matrix_multiply(double** A, double** B, int n, int m, int p);

/*
 * Create transpose of a matrix
 * @param A: Input matrix
 * @param n: Number of rows in A
 * @param m: Number of columns in A
 * @return: Transposed matrix (m x n), or NULL if error occurs
 */
double** transpose_matrix(double** A, int n, int m);

/*
 * Calculate Frobenius norm of the difference between two matrices
 * @param A: First matrix
 * @param B: Second matrix
 * @param n: Number of rows
 * @param k: Number of columns
 * @return: Frobenius norm value
 */
double calculate_frobenius_norm(double** A, double** B, int n, int k);

/*
 * Copy contents of one matrix to another
 * @param dest: Destination matrix
 * @param src: Source matrix
 * @param n: Number of rows
 * @param k: Number of columns
 */
void copy_matrix(double** dest, double** src, int n, int k);

/*
 * Update H matrix according to symNMF update rule
 * @param W: Normalized similarity matrix
 * @param H: H matrix to update
 * @param n: Number of rows
 * @param k: Number of columns in H
 */
int update_H(double** W, double** H, int n, int k);

/*
 * Free memory allocated for 2D array
 * @param array: The array to free
 * @param n: Number of rows in the array
 */
void free_c_array(double** array, int n);

/*
 * Read data from file into matrix
 * @param filename: Name of input file
 * @param n: Pointer to store number of rows
 * @param d: Pointer to store number of columns
 * @return: Data matrix, or NULL if error occurs
 */
double** read_data_from_file(const char* filename, int* n, int* d);

/*
 * Print matrix to stdout
 * @param matrix: Matrix to print
 * @param n: Number of rows
 * @param m: Number of columns
 */
void print_matrix(double** matrix, int n, int m);

#endif /* SYMNMF_H */