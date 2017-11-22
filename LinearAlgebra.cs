using System;
using System.Numerics;
using System.Runtime.InteropServices;

namespace LAPack
{
    static class LinearAlgebra
    {
        const int LAPACK_ROW_MAJOR = 101;
        const int LAPACK_COL_MAJOR = 102;

        /// <summary>
        /// Solves the system of linear equations AX = B for X, where A, B, and X are general matrices.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static DenseTensor<double> Solve(DenseTensor<double> a, DenseTensor<double> b)
        {
            if (a.Rank != 2) throw new ArgumentException("a must be a square matrix", nameof(a));
            if (a.Dimensions[0] != a.Dimensions[1]) throw new ArgumentException("a must be a square matrix", nameof(a));
            if (b.Rank != 2) throw new ArgumentException("b must be a matrix", nameof(b));
            if (a.Dimensions[0] != b.Dimensions[0]) throw new ArgumentException("The number of rows in b must match the number of rows in a", nameof(b));

            // need to clone the inputs because LAPack will mutate the values
            var aClone = (DenseTensor<double>)a.Clone();
            var bClone = (DenseTensor<double>)b.Clone();

            unsafe
            {
                Span<int> pivotIntegers = stackalloc int[a.Dimensions[1]];
                fixed (double* aPtr = &aClone.Buffer.Span.DangerousGetPinnableReference())
                fixed (double* bPtr = &bClone.Buffer.Span.DangerousGetPinnableReference())
                fixed (int* ipiv = &pivotIntegers.DangerousGetPinnableReference())
                {
                    LAPACKE_dgesv(
                        a.IsReversedStride ? LAPACK_COL_MAJOR : LAPACK_ROW_MAJOR, 
                        a.Dimensions[0], 
                        b.Dimensions[1], 
                        aPtr, 
                        a.Dimensions[1], 
                        ipiv, 
                        bPtr, 
                        b.Dimensions[1]);
                }
            }

            return bClone;
        }

        [DllImport("liblapacke.dll")]
        static extern unsafe int LAPACKE_dgesv(int matrix_layout, int n, int nrhs, double* a, int lda, int* ipvt, double* bx, int ldb);
    }
}
