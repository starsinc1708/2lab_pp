#include <omp.h>
#include "stdio.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#define CHUNK 100
#define NMAX 254000
#define OpenMP
#define Reduction

double reduction(int* a, double sum, int i) {
#pragma omp parallel for shared(a) private(i) reduction(+: sum)
    for (i = 0; i < NMAX; i++)
    {
        sum = sum + a[i];
    }
    return sum;
}

double critial(int* a, double sum, int i) {
#pragma omp parallel for
    for (i = 0; i < NMAX; i++)
    {
#pragma omp critical
        {
            sum = sum + a[i];
        }
    }
    return sum;
}

double atomic(int* a, double sum, int i) {
#pragma omp parallel for
    for (i = 0; i < NMAX; i++)
    {
#pragma omp atomic
        sum += a[i];
    }
    return sum;
}

#ifdef OpenMP
int main(int argc, char** argv) {
    omp_set_num_threads(2);
    int i = 0;
    int a[NMAX];
    double sum;

    for (int i = 0; i < NMAX; i++) {
        a[i] = 1;
    }

    double st_time, end_time;
    st_time = omp_get_wtime();
    for (int i = 0; i < 22; i++)
    {
        sum = 0;
#ifdef Reduction
        sum = reduction(a, sum, i);
#endif // Rduction
#ifdef Critical
        sum = critial(a, sum, i);
#endif // Critical
#ifdef Atomic
        sum = atomic(a, sum, i);
#endif // Atomic
    }
    end_time = omp_get_wtime();
    end_time = end_time - st_time;
    printf("Total Sum = %10.2f", sum);
    printf("\nTIME OF WORK IS %f ", end_time);
    return 0;
}
#endif // OpenMp



#ifdef MPI
int main(int argc, char** argv)
{
    double* x = 0;
    double TotalSum = 0.0;
    double ProcSum = 0.0;
    int ProcRank, ProcNum, N = 254000, i;

    MPI_Status Status;

    double st_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    if (ProcRank == 0)
    {
        x = (double*)malloc(sizeof(double) * N);
        for (int i = 0; i < N; i++) {
            x[i] = 1;
        }
    }
    int k = N / ProcNum;
    int i1 = k * ProcRank;
    int i2 = k * (ProcRank + 1);

    double* loc = (double*)malloc(sizeof(double) * k);

    MPI_Scatter(x, k, MPI_DOUBLE, loc, k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (ProcRank == 0) free(x);

    st_time = MPI_Wtime();

    for (i = 0; i < k; i++) ProcSum += loc[i];
    free(loc);

#ifdef Point
    if (ProcRank == 0)
    {
        printf("Point to point\n");
        TotalSum = ProcSum;
        for (i = 1; i < ProcNum; i++)
        {
            // Получение лок. сумм от других процессов
            MPI_Recv(&ProcSum, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &Status);
            TotalSum = TotalSum + ProcSum;
        }
    }
    else
        MPI_Send(&ProcSum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); // Локальные суммы отправляются нулевому процессу
#endif // Point
#ifdef Collect
    if (ProcRank == 0) printf("Collective operations\n");
    MPI_Reduce(&ProcSum, &TotalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif // Collect


    MPI_Barrier(MPI_COMM_WORLD);

    end_time = MPI_Wtime();
    end_time = end_time - st_time;

    if (ProcRank == 0)
    {
        printf("Total Sum = %10.2f", TotalSum);
        printf("\nTIME OF WORK IS %f", end_time);
    }

    MPI_Finalize();
    return 0;
}
#endif // MPI
