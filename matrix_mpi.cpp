#include <mpi.h>
#include <algorithm>
#include <random>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using Matrix = std::vector<double>;

int element_count(int rows, int cols) {
    if (rows <= 0 || cols <= 0 || rows > (std::numeric_limits<int>::max)() / cols)
        throw std::runtime_error("Invalid dimensions or MPI count limit exceeded");
    return rows * cols;
}

Matrix read_matrix(std::istream& input, int& rows, int& cols) {
    if (!(input >> rows >> cols))
        throw std::runtime_error("Expected row and column counts");
    Matrix values(element_count(rows, cols));
    for (double& value : values)
        if (!(input >> value))
            throw std::runtime_error("Missing or invalid matrix element");
    return values;
}

void run(int argc, char** argv, int rank, int processes) {
    int dims[3] = {};  // A: m x n, B: n x k
    Matrix a, b, c;
    if (rank == 0) {
        if (argc == 2) {
            std::ifstream input(argv[1]);
            if (!input) throw std::runtime_error("Cannot open input file");
            int b_rows;
            a = read_matrix(input, dims[0], dims[1]);
            b = read_matrix(input, b_rows, dims[2]);
            if (b_rows != dims[1])
                throw std::runtime_error("Columns of A must equal rows of B");
        }
        else {
            // Команда и интерактивный ввод используют порядок n m k.
            int& m = dims[0];
            int& n = dims[1];
            int& k = dims[2];
            if (argc == 1) {
                std::cerr << "Enter n m k (A: m x n, B: n x k): " << std::flush;
                if (!(std::cin >> n >> m >> k))
                    throw std::runtime_error("Expected three integer dimensions");
            }
            else if (argc == 4) {
                int* arguments[] = { &n, &m, &k };
                for (int i = 1; i <= 3; ++i) {
                    std::istringstream arg(argv[i]);
                    int& value = *arguments[i - 1];
                    char extra;
                    if (!(arg >> value) || (arg >> extra))
                        throw std::runtime_error("Expected three integer dimensions: n m k");
                }
            }
            else {
                throw std::runtime_error("Usage: MatrixMPI [n m k | matrices.txt]");
            }
            const int size_a = element_count(m, n);
            const int size_b = element_count(n, k);
            element_count(m, k);
            a.resize(size_a);
            b.resize(size_b);
            std::mt19937 generator(42);
            for (double& x : a) x = generator() % 10;
            for (double& x : b) x = generator() % 10;
        }
        c.resize(element_count(dims[0], dims[2]));
    }
    MPI_Bcast(dims, 3, MPI_INT, 0, MPI_COMM_WORLD);
    const int m = dims[0], n = dims[1], k = dims[2];
    if (rank != 0) b.resize(element_count(n, k));

    // Scatterv и Gatherv считают элементы, а не строки матрицы.
    std::vector<int> a_counts(processes), a_offsets(processes);
    std::vector<int> c_counts(processes), c_offsets(processes);
    int first_row = 0;
    for (int p = 0; p < processes; ++p) {
        const int rows = m / processes + (p < m % processes);
        a_counts[p] = rows * n;
        a_offsets[p] = first_row * n;
        c_counts[p] = rows * k;
        c_offsets[p] = first_row * k;
        first_row += rows;
    }

    Matrix local_a(a_counts[rank]), local_c(c_counts[rank], 0.0);
    const int local_rows = a_counts[rank] / n;
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    MPI_Bcast(b.data(), n * k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(a.data(), a_counts.data(), a_offsets.data(), MPI_DOUBLE,
        local_a.data(), a_counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Каждый процесс вычисляет все столбцы для своих строк.
    const double compute_start = MPI_Wtime();
    for (int i = 0; i < local_rows; ++i)
        for (int t = 0; t < n; ++t)
            for (int j = 0; j < k; ++j)
                local_c[i * k + j] += local_a[i * n + t] * b[t * k + j];

    const double compute = MPI_Wtime() - compute_start;
    MPI_Gatherv(local_c.data(), c_counts[rank], MPI_DOUBLE, c.data(),
        c_counts.data(), c_offsets.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    const double elapsed = MPI_Wtime() - start;
    std::vector<double> times(processes);
    MPI_Gather(&compute, 1, MPI_DOUBLE, times.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    char host[MPI_MAX_PROCESSOR_NAME] = {};
    int length;
    MPI_Get_processor_name(host, &length);
    std::vector<char> hosts;
    if (rank == 0) hosts.resize(static_cast<std::size_t>(processes) * sizeof(host));
    MPI_Gather(host, sizeof(host), MPI_CHAR, hosts.data(), sizeof(host),
        MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (int p = 0; p < processes; ++p)
            std::cerr << "Rank " << p << " on "
            << &hosts[static_cast<std::size_t>(p) * sizeof(host)]
            << ": " << a_counts[p] / n << " rows, compute " << times[p] << " s\n";
        const double slowest = *std::max_element(times.begin(), times.end());
        const double giga_ops = 2.0 * m * n * k / 1e9;
        std::cerr << "A: " << m << " x " << n << ", B: " << n << " x " << k
            << ", C: " << m << " x " << k << ", processes: " << processes
            << "\nTotal (distribution + compute + gather): " << elapsed << " s"
            << "\nCompute (max rank): " << slowest << " s\n";
        if (elapsed > 0) std::cerr << "Total GFLOPS: " << giga_ops / elapsed << '\n';
        if (slowest > 0) std::cerr << "Compute GFLOPS: " << giga_ops / slowest << '\n';
        if (argc != 2 && c.size() > 100) {
            std::cerr << "C[0,0] = " << c.front() << ", C[last,last] = " << c.back() << '\n';
            return;
        }
        // Формат результата совпадает с форматом одной входной матрицы.
        std::cout << m << ' ' << k << '\n' << std::setprecision(17);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < k; ++j) {
                if (j) std::cout << ' ';
                std::cout << c[i * k + j];
            }
            std::cout << '\n';
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    try {
        run(argc, argv, rank, processes);
    }
    catch (const std::exception& e) {
        std::cerr << "Rank " << rank << ": " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
}
