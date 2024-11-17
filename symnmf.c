/*
 * Symmetric Non-negative Matrix Factorization (symNMF) Implementation
 * This file contains both the core symNMF algorithm and utility functions
 */

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "symnmf.h"

#define MAX_ITER 300
#define EPSILON 1e-4
#define MAX_LINE_LENGTH 1024


/* Matrix multiplication: multiply matrices A(n x m) and B(m x p) */
double** matrix_multiply(double** A, double** B, int n, int m, int p) {
    double** C = NULL;
    int i, j, k;
    
    C = (double**)malloc(n * sizeof(double*));
    if (!C) return NULL;
    
    for (i = 0; i < n; i++) {
        C[i] = (double*)calloc(p, sizeof(double)); /* Initialize to zero */
        if (!C[i]) {
            free_c_array(C, i);
            return NULL;
        }
        for (j = 0; j < p; j++) {
            C[i][j] = 0.0;  /* Explicit initialization */
            for (k = 0; k < m; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

/* Create transpose of matrix A */
double** transpose_matrix(double** A, int n, int m) {
    double** result;
    int i, j;
    
    result = (double**)malloc(m * sizeof(double*));
    if (!result) return NULL;
    
    for (i = 0; i < m; i++) {
        result[i] = (double*)malloc(n * sizeof(double));
        if (!result[i]) {
            free_c_array(result, i);
            return NULL;
        }
        for (j = 0; j < n; j++) {
            result[i][j] = A[j][i];
        }
    }
    return result;
}

/* Compute Frobenius norm of difference between matrices A and B */
double calculate_frobenius_norm(double** A, double** B, int n, int k) {
    double sum, diff;
    int i, j;
    
    sum = 0.0;
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            diff = A[i][j] - B[i][j];
            sum += diff * diff;
        }
    }
    return sum;
}

/* Copy contents of matrix src to dest */
void copy_matrix(double** dest, double** src, int n, int k) {
    int i, j;
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            dest[i][j] = src[i][j];
        }
    }
}

/* Update H matrix according to symNMF update rule */
int update_H(double** W, double** H, int n, int k) {
    double** WH = NULL;     /* W*H */
    double** Ht = NULL;     /* H^T */
    double** HHt = NULL;    /* H*H^T */
    double** HHtH = NULL;   /* (H*H^T)*H */
    const double beta = 0.5;
    int i, j;
    WH = matrix_multiply(W, H, n, n, k);
    if (!WH) {
        return 0;
    }
    Ht = transpose_matrix(H, n, k);
    if (!Ht) {
        free_c_array(WH, n);
        return 0;
    }
    HHt = matrix_multiply(H, Ht, n, k, n);
    if (!HHt) {
        free_c_array(WH, n);
        free_c_array(Ht, k);
        return 0;
    }
    HHtH = matrix_multiply(HHt, H, n, n, k);
    if (!HHtH) {
        free_c_array(WH, n);
        free_c_array(Ht, k);
        free_c_array(HHt, n);
        return 0;
    }
    /* Update each element of H */
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            H[i][j] *= (1 - beta + beta * (WH[i][j]/HHtH[i][j]));
        }
    }
    /* Free temporary matrices */
    free_c_array(WH, n); free_c_array(Ht, k); free_c_array(HHt, n); free_c_array(HHtH, n);
    return 1;  /* Success */ 
}

/* Free a 2D array and handle NULL pointers safely */
void free_c_array(double** array, int n) {
    int i;
    
    if (array) {
        for (i = 0; i < n; i++) {
            if (array[i]) {
                free(array[i]);
            }
        }
        free(array);
    }
}

/* Calculate similarity matrix from input points */
double** sym(double** points, int n, int d) {
    double** similarity;
    double sum;
    double diff;
    int i, j, k;
    
    similarity = (double**)malloc(n * sizeof(double*));
    if (!similarity) return NULL;
    
    for (i = 0; i < n; i++) {
        similarity[i] = (double*)malloc(n * sizeof(double));
        if (!similarity[i]) {
            free_c_array(similarity, i);
            return NULL;
        }
        for (j = 0; j < n; j++) {
            if (i == j) 
                similarity[i][j] = 0.0;  /* Diagonal elements are 0 */
            else {
                /* Calculate squared Euclidean distance */
                sum = 0.0;
                for (k = 0; k < d; k++) {
                    diff = points[i][k] - points[j][k];
                    sum += diff * diff;
                }
                similarity[i][j] = exp(-sum / 2.0);
            }
        }
    }
    return similarity;
}

/* Calculate diagonal degree matrix using similarity matrix */
double** ddg(double** points, int n, int d) {
    double** similarity;
    double** degree;
    int i, j;
    
    similarity = sym(points, n, d);
    if (!similarity) return NULL;
    
    degree = (double**)malloc(n * sizeof(double*));
    if (!degree) {
        free_c_array(similarity, n);
        return NULL;
    }
    
    for (i = 0; i < n; i++) {
        degree[i] = (double*)calloc(n, sizeof(double));
        if (!degree[i]) {
            free_c_array(similarity, n);
            free_c_array(degree, i);
            return NULL;
        }
        /* Sum row i to get degree */
        for (j = 0; j < n; j++) {
            degree[i][i] += similarity[i][j];
        }
    }
    free_c_array(similarity, n);
    return degree;
}

/* Calculate normalized similarity matrix */
double** norm(double** points, int n, int d) {
    double** similarity;
    double* degree_diag;
    double** normalized;
    int i, j;
    similarity = sym(points, n, d);
    if (!similarity) return NULL;
    /* Calculate diagonal degree values */
    degree_diag = (double*)calloc(n, sizeof(double));
    if (!degree_diag) {
        free_c_array(similarity, n);
        return NULL;
    }
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            degree_diag[i] += similarity[i][j];
        }
    }
    /* Create normalized matrix */
    normalized = (double**)malloc(n * sizeof(double*));
    if (!normalized) {
        free(degree_diag);
        free_c_array(similarity, n);
        return NULL;
    }
    for (i = 0; i < n; i++) {
        normalized[i] = (double*)malloc(n * sizeof(double));
        if (!normalized[i]) {
            free(degree_diag);
            free_c_array(similarity, n);
            free_c_array(normalized, i);
            return NULL;
        }
        /* Normalize using degree values */
        for (j = 0; j < n; j++) {
            normalized[i][j] = similarity[i][j] / sqrt(degree_diag[i] * degree_diag[j]);
        }
    }
    free(degree_diag); free_c_array(similarity, n);
    return normalized;
}

/* Perform symNMF algorithm */
double** symnmf(double** W, double** H, int n, int k) {
    double** H_prev = NULL; 
    double** result = NULL;
    int i, iter;
    /* Allocate memory for matrices */
    H_prev = (double**)malloc(n * sizeof(double*));
    if (!H_prev) { return NULL;}
    result = (double**)malloc(n * sizeof(double*));
    if (!result) {
        free_c_array(H_prev, n); return NULL;
    }
    /* Allocate memory for each row */
    for (i = 0; i < n; i++) {
        H_prev[i] = (double*)malloc(k * sizeof(double));
        if (!H_prev[i]) {
            free_c_array(H_prev, i); free_c_array(result, i);
            return NULL;
        }
        result[i] = (double*)malloc(k * sizeof(double));
        if (!result[i]) {
            free_c_array(H_prev, i+1); free_c_array(result, i);
            return NULL;
        }
    }
    /* Initialize result with input H */
    copy_matrix(result, H, n, k);
    /* Main iteration loop */
    for (iter = 0; iter < MAX_ITER; iter++) {
        copy_matrix(H_prev, result, n, k);
        if (!update_H(W, result, n, k)) {
            free_c_array(H_prev, n);
            free_c_array(result, n);
            return NULL;
        }
        if (calculate_frobenius_norm(result, H_prev, n, k) < EPSILON) {
            break;
        }
    }  
    free_c_array(H_prev, n);
    return result;
}

/* Read input data from file and convert to matrix form */
double** read_data_from_file(const char* filename, int* n, int* d) {
    FILE* file; char line[MAX_LINE_LENGTH]; char* token; double** data; int i, j;
    /* Open and validate file */
    file = fopen(filename, "r");
    if (!file) return NULL;
    /* Count dimensions */
    *n = 0;
    *d = 0;
    while (fgets(line, MAX_LINE_LENGTH, file)) {
        (*n)++;
        if (*d == 0) {
            token = strtok(line, ",");
            while (token) {
                (*d)++;
                token = strtok(NULL, ",");
            }
        }
    }
    /* Reset to file start */
    rewind(file);
    /* Allocate and read data */
    data = (double**)malloc(*n * sizeof(double*));
    for (i = 0; i < *n; i++) {
        data[i] = (double*)malloc(*d * sizeof(double));
        if (!data[i]) {
            free_c_array(data, i); fclose(file);
            return NULL;
        }
        if (fgets(line, MAX_LINE_LENGTH, file) == NULL) {
            free_c_array(data, i + 1); fclose(file);
            return NULL;
        }
        token = strtok(line, ",");
        for (j = 0; j < *d; j++) {
            data[i][j] = atof(token);
            token = strtok(NULL, ",");
        }
    }
    fclose(file);
    return data;
}

/* Print matrix to stdout with specified format */
void print_matrix(double** matrix, int n, int m) {
    int i, j;
    
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            printf("%.4f", matrix[i][j]);
            if (j < m - 1) printf(",");
        }
        printf("\n");
    }
}

/* Main function: handle arguments and execute requested operation */
int main(int argc, char* argv[]) {
    const char* goal; const char* filename;
    int n, d;
    double** data; double** result;
    
    /* Validate arguments */
    if (argc != 3) {
        printf("An Error Has Occurred\n"); return 1;
    }
    
    goal = argv[1];
    filename = argv[2];
    data = read_data_from_file(filename, &n, &d);
    
    if (!data) {
        printf("An Error Has Occurred\n"); return 1;
    }
    
    /* Execute requested operation */
    result = NULL;
    if (strcmp(goal, "sym") == 0) {
        result = sym(data, n, d);
    } else if (strcmp(goal, "ddg") == 0) {
        result = ddg(data, n, d);
    } else if (strcmp(goal, "norm") == 0) {
        result = norm(data, n, d);
    } else {
        printf("An Error Has Occurred\n");
        free_c_array(data, n); return 1;
    }
    
    /* Handle result and cleanup */
    if (!result) {
        printf("An Error Has Occurred\n");
        free_c_array(data, n); return 1;
    }
    
    print_matrix(result, n, n);
    free_c_array(data, n); free_c_array(result, n);
    return 0;
}