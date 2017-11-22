using System;
using System.Numerics;

namespace LAPack
{
    class Program
    {
        static void Main(string[] args)
        {
            // sample taken from https://software.intel.com/sites/products/documentation/doclib/mkl_sa/11/mkl_lapack_examples/lapacke_dgesv_row.c.htm
            DenseTensor<double> a = new[,]
            {
                { 6.80, -6.05, -0.45,  8.32, -9.67, },
                {-2.11, -3.30,  2.58,  2.71, -5.14, },
                { 5.66,  5.36, -2.70,  4.35, -7.26, },
                { 5.97, -4.44,  0.27, -7.17,  6.08, },
                { 8.23,  1.08,  9.04,  2.14, -6.87  },
            }.ToTensor();

            DenseTensor<double> b = new[,]
            {
                { 4.02, -1.56,  9.81,},
                { 6.19,  4.00, -4.09,},
                {-8.22, -8.67, -4.57,},
                {-7.57,  1.75, -8.61,},
                {-3.03,  2.86,  8.99 },
            }.ToTensor();

            Solve(a, b);


            // 3*x0 + x1 = 9 
            // x0 + 2*x1 = 8
            a = new[,]
            {
                { 3.0, 1.0 },
                { 1.0 , 2.0 },
            }.ToTensor();

            b = new[,]
            {
                { 9.0 },
                { 8.0 },
            }.ToTensor();

            Solve(a, b);
        }

        static void Solve(DenseTensor<double> a, DenseTensor<double> b)
        {
            Console.WriteLine("Entry Matrix A");
            Console.WriteLine(a.GetArrayString());
            Console.WriteLine("Right Hand Side");
            Console.WriteLine(b.GetArrayString());

            var solution = LinearAlgebra.Solve(a, b);

            Console.WriteLine("Solution");
            Console.WriteLine(solution.GetArrayString());
        }
    }
}
