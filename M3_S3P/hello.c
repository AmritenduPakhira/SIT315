#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    int rank, size;
    char msg[6] = "Hello";

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Broadcast message from rank 0 to all other processes
        MPI_Bcast(&msg, 6, MPI_CHAR, 0, MPI_COMM_WORLD);
        printf("Rank %d broadcasting message: %s\n", rank, msg);
    } else {
        // Receive broadcast message from rank 0
        MPI_Bcast(&msg, 6, MPI_CHAR, 0, MPI_COMM_WORLD);
        printf("Rank %d received message: %s\n", rank, msg);
    }

    MPI_Finalize();
    return 0;
}
