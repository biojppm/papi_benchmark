#ifndef _B5_MATRIX_HPP_
#define _B5_MATRIX_HPP_

#include <stddef.h> // for size_t

/** a simple dense matrix class for creating some token benchmarks */
template< class T >
struct matrix
{
    int nrows;
    int ncols;
    T * vals;

    ~matrix() { release(); }

    matrix() : nrows(0), ncols(0), vals(nullptr) {}
    matrix(int nr, int nc) : nrows(0), ncols(0), vals(nullptr) { resize(nr, nc); }
    matrix(int nr, int nc, T v) : nrows(0), ncols(0), vals(nullptr) { resize(nr, nc); set_diag(v); }

    matrix(matrix const& other) : nrows(0), ncols(0), vals(nullptr) { *this = other; }
    matrix(matrix     && other) : nrows(0), ncols(0), vals(nullptr) { *this = std::move(other); }
    matrix& operator= (matrix const& other)
    {
        if(&other == this) return;
        resize(other.nrows, other.ncols);
        memcpy(vals, other.vals, size_bytes());
        return *this;
    }
    matrix& operator= (matrix && other)
    {
        if(&other == this) return;
        release();
        nrows = other.nrows;
        ncols = other.ncols;
        vals  = other.vals;
        other.nrows = 0;
        other.ncols = 0;
        other.vals = nullptr;
        other.alloc = {};
    }

    size_t size_bytes() const { return nrows * ncols * sizeof(T); }

    void release()
    {
        if(vals == nullptr) return;
        delete [] vals;
        vals = nullptr;
        nrows = 0;
        ncols = 0;
    }

    void resize(int nr, int nc)
    {
        if(nr*nc > nrows*ncols)
        {
            release();
            vals  = new T [nr * nc];
        }
        else
        {
            for(int i = 0, e = nr * nc; i < e; ++i)
            {
                vals[i] = T(0);
            }
        }
        nrows = nr;
        ncols = nc;
    }

    inline T  operator() (int i, int j) const { return vals[i * ncols + j]; }
    inline T& operator() (int i, int j)       { return vals[i * ncols + j]; }

    inline T const* operator[] (int i) const { return vals + i*ncols; }
    inline T      * operator[] (int i)       { return vals + i*ncols; }

    void set_diag(T v)
    {
        int n = nrows < ncols ? nrows : ncols;
        for(int i = 0; i < n; i++)
        {
            (*this)(i, i) = v;
        }
    }

    void set_all(T v)
    {
        for(int i = 0; i < nrows; i++)
        {
            for(int j = 0; j < ncols; j++)
            {
                (*this)(i, j) = v;
            }
        }
    }

    /** naive matrix multiplication *C = A*B */
    static void mult_naive(matrix const& A, matrix const& B, matrix *C)
    {
        C->resize(A.nrows, B.ncols);
        for(int i = 0; i < A.nrows; i++)
        {
            T const* ra = A[i];
            T *rc = (*C)[i];
            for(int j = 0; j < B.ncols; j++)
            {
                T tmp = T(0);
                for(int k = 0; k < A.ncols; k++)
                {
                    tmp += ra[k] * B[k][j];
                }
                rc[j] = tmp;
            }
        }
    }
    /** @see http://functionspace.com/articles/40/Cache-aware-Matrix-Multiplication---Naive-isn--039;t-that-bad- */
    static void mult_naive_better(matrix const& A, matrix const& B, matrix *C)
    {
        C->resize(A.nrows, B.ncols);
        for(int i = 0; i < A.nrows; i++)
        {
            T const* ra = A[i];
            T *rc = (*C)[i];
            for(int k = 0; k < A.ncols; k++)
            {
                for(int j = 0; j < B.ncols; j++)
                {
                    rc[j] += ra[k] * B[k][j];
                }
            }
        }
    }
    static void mult_naive_bad(matrix const& A, matrix const& B, matrix *C)
    {
        C->resize(A.nrows, B.ncols);
        for(int j = 0; j < B.ncols; j++)
        {
            for(int i = 0; i < A.nrows; i++)
            {
                T tmp = T(0);
                for(int k = 0; k < A.ncols; k++)
                {
                    tmp += A[i][k] * B[k][j];
                }
                (*C)[i][j] = tmp;
            }
        }
    }
    static void mult_naive_transposed(matrix const& A, matrix const& B, matrix *C, matrix *workspace)
    {
        C->resize(A.nrows, B.ncols);
        workspace->transpose_recursive(B);
        for(int i = 0; i < A.nrows; i++)
        {
            for(int j = 0; j < workspace->nrows; j++)
            {
                T tmp = T(0);
                for(int k = 0; k < A.ncols; k++)
                {
                    tmp += A[i][k] * (*workspace)[j][k];
                }
                (*C)[i][j] = tmp;
            }
        }
    }

    /** regarding transposition:
     * @see http://stackoverflow.com/questions/16737298/what-is-the-fastest-way-to-transpose-a-matrix-in-c
     * @see http://stackoverflow.com/questions/11413855/why-is-transposing-a-matrix-of-512x512-much-slower-than-transposing-a-matrix-of?lq=1
     * @see http://stackoverflow.com/questions/5200338/a-cache-efficient-matrix-transpose-program
     */
    void transpose_naive()
    {
        T tmp;
        for(int i = 0; i < nrows; ++i)
        {
            T *rowi = (*this)[i];
            for(int j = 0; j <= i; ++j)
            {
                tmp = rowi[j];
                auto &r = (*this)[j][i];
                rowi[j] = r;
                r = tmp;
            }
        }
        tmp = ncols;
        ncols = nrows;
        nrows = tmp;
    }
    void transpose_naive(matrix const& that)
    {
        resize(that.ncols, that.nrows);
        for(int i = 0; i < that.nrows; ++i)
        {
            for(int j = 0; j < that.ncols; ++j)
            {
                (*this)[j][i] = that[i][j];
            }
        }
    }
    void transpose_recursive(matrix *workspace)
    {
        *workspace = this;
        transpose_recursive_(0, nrows, 0, ncols, *workspace);
        T tmp = ncols;
        ncols = nrows;
        nrows = tmp;
    }
    void transpose_recursive(matrix const& that)
    {
        resize(that.ncols, that.nrows);
        transpose_recursive_(0, that.nrows, 0, that.ncols, that);
        T tmp = ncols;
        ncols = nrows;
        nrows = tmp;
    }
    /** taken from http://stackoverflow.com/questions/5200338/a-cache-efficient-matrix-transpose-program?lq=1 */
    void transpose_recursive_(int rb, int re, int cb, int ce, matrix const& that)
    {
        int r = re - rb, c = ce - cb;
        if(r <= 16 && c <= 16)
        {
            for (int i = rb; i < re; i++)
            {
                for (int j = cb; j < ce; j++)
                {
                    (*this)[j][i] = that[i][j];
                }
            }
        }
        else if(r >= c)
        {
            transpose_recursive_(rb, rb + (r / 2), cb, ce, that);
            transpose_recursive_(rb + (r / 2), re, cb, ce, that);
        }
        else
        {
            transpose_recursive_(rb, re, cb, cb + (c / 2), that);
            transpose_recursive_(rb, re, cb + (c / 2), ce, that);
        }
    }
};

#endif // _B5_MATRIX_HPP_
