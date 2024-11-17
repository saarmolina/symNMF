#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/* Convert Python list to C array 
 * Input: Python list and its dimensions
 * Output: Dynamically allocated 2D array or NULL if memory allocation fails
 */
static double** py_list_to_c_array(PyObject* py_list, int n, int d) {
    /* Allocate memory for array of pointers */
    double** array = (double**)malloc(n * sizeof(double*));
    if (!array) {
        return NULL;
    }
    /* Allocate memory for each row and copy data */
    for (int i = 0; i < n; i++) {
        array[i] = (double*)malloc(d * sizeof(double));
        if (!array[i]) {
            free_c_array(array, i);  /* Free previously allocated memory */
            return NULL;
        }
        PyObject* py_row = PyList_GetItem(py_list, i);
        for (int j = 0; j < d; j++) {
            array[i][j] = PyFloat_AsDouble(PyList_GetItem(py_row, j));
        }
    }
    return array;
}

/* Convert C array to Python list 
 * Input: C 2D array and its dimensions
 * Output: Python list or NULL if creation fails
 */
static PyObject* c_array_to_py_list(double** array, int n, int d) {
    /* Create new Python list */
    PyObject* py_list = PyList_New(n);
    if (!py_list) {
        return NULL;
    }
    /* Create each row and copy data */
    for (int i = 0; i < n; i++) {
        PyObject* py_row = PyList_New(d);
        if (!py_row) {
            Py_DECREF(py_list);
            return NULL;
        }
        for (int j = 0; j < d; j++) {
            PyObject* py_float = PyFloat_FromDouble(array[i][j]);
            if (!py_float) {
                Py_DECREF(py_row);
                Py_DECREF(py_list);
                return NULL;
            }
            PyList_SET_ITEM(py_row, j, py_float);
        }
        PyList_SET_ITEM(py_list, i, py_row);
    }
    return py_list;
}

/* Python wrapper for sym function
 * Converts Python input to C, calls sym, converts result back to Python
 */
static PyObject* py_sym(PyObject* self, PyObject* args) {
    PyObject *py_points;
    /* Parse Python arguments */
    if (!PyArg_ParseTuple(args, "O", &py_points)) return NULL;
    int n = PyList_Size(py_points);
    int d = PyList_Size(PyList_GetItem(py_points, 0));
    
    /* Convert input to C array */
    double **points = py_list_to_c_array(py_points, n, d);
    if (!points) {
        Py_RETURN_NONE;
    }
    
    /* Call C function */
    double **result = sym(points, n, d);
    if (!result) {
        free_c_array(points, n);
        Py_RETURN_NONE;
    }
    
    /* Convert result back to Python */
    PyObject* py_result = c_array_to_py_list(result, n, n);
    free_c_array(points, n);
    free_c_array(result, n);
    if (!py_result) {
        Py_RETURN_NONE;
    }
    return py_result;
}

/* Python wrapper for ddg function
 * Converts Python input to C, calls ddg, converts result back to Python
 */
static PyObject* py_ddg(PyObject* self, PyObject* args) {
    PyObject *py_points;
    /* Parse Python arguments */
    if (!PyArg_ParseTuple(args, "O", &py_points)) return NULL;
    int n = PyList_Size(py_points);
    int d = PyList_Size(PyList_GetItem(py_points, 0));
    
    /* Convert input to C array */
    double **points = py_list_to_c_array(py_points, n, d);
    if (!points) {
        Py_RETURN_NONE;
    }
    
    /* Call C function */
    double **result = ddg(points, n, d);
    if (!result) {
        free_c_array(points, n);
        Py_RETURN_NONE;
    }
    
    /* Convert result back to Python */
    PyObject* py_result = c_array_to_py_list(result, n, n);
    free_c_array(points, n);
    free_c_array(result, n);
    if (!py_result) {
        Py_RETURN_NONE;
    }
    return py_result;
}

/* Python wrapper for norm function
 * Converts Python input to C, calls norm, converts result back to Python
 */
static PyObject* py_norm(PyObject* self, PyObject* args) {
    PyObject *py_points;
    /* Parse Python arguments */
    if (!PyArg_ParseTuple(args, "O", &py_points)) return NULL;
    int n = PyList_Size(py_points);
    int d = PyList_Size(PyList_GetItem(py_points, 0));
    
    /* Convert input to C array */
    double **points = py_list_to_c_array(py_points, n, d);
    if (!points) {
        Py_RETURN_NONE;
    }
    
    /* Call C function */
    double **result = norm(points, n, d);
    if (!result) {
        free_c_array(points, n);
        Py_RETURN_NONE;
    }
    
    /* Convert result back to Python */
    PyObject* py_result = c_array_to_py_list(result, n, n);
    free_c_array(points, n);
    free_c_array(result, n);
    if (!py_result) {
        Py_RETURN_NONE;
    }
    return py_result;
}

/* Python wrapper for symnmf function
 * Converts Python input to C, calls symnmf, converts result back to Python
 */
static PyObject* py_symnmf(PyObject* self, PyObject* args) {
    PyObject *py_W, *py_H;
    int n, k;
    /* Parse Python arguments */
    if (!PyArg_ParseTuple(args, "OOii", &py_W, &py_H, &n, &k)) return NULL;
    
    /* Convert inputs to C arrays */
    double **W = py_list_to_c_array(py_W, n, n);
    if (!W) {
        Py_RETURN_NONE;
    }
    
    double **H = py_list_to_c_array(py_H, n, k);
    if (!H) {
        free_c_array(W, n);
        Py_RETURN_NONE;
    }
    
    /* Call C function */
    double **result = symnmf(W, H, n, k);
    if (!result) {
        free_c_array(W, n);
        free_c_array(H, n);
        Py_RETURN_NONE;
    }
    
    /* Convert result back to Python */
    PyObject* py_result = c_array_to_py_list(result, n, k);
    free_c_array(W, n);
    free_c_array(H, n);
    free_c_array(result, n);
    if (!py_result) {
        Py_RETURN_NONE;
    }
    return py_result;
}

/* Module method definitions */
static PyMethodDef SymNMFMethods[] = {
    {"symnmf", py_symnmf, METH_VARARGS, "Execute the symNMF algorithm."},
    {"sym", py_sym, METH_VARARGS, "Calculate the similarity matrix."},
    {"ddg", py_ddg, METH_VARARGS, "Calculate the Diagonal Degree Matrix."},
    {"norm", py_norm, METH_VARARGS, "Calculate the normalized similarity matrix."},
    {NULL, NULL, 0, NULL}
};

/* Module definition */
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmf",
    "Symmetric Non-negative Matrix Factorization implementation",
    -1,
    SymNMFMethods
};

/* Module initialization function */
PyMODINIT_FUNC PyInit_symnmf(void) {
    return PyModule_Create(&symnmfmodule);
}