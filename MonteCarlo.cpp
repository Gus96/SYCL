#include <iostream>
#include <CL/sycl.hpp>
#include "oneapi/mkl/rng.hpp" 

#include <ctime>
using namespace cl::sycl;

#define CPU "CPU"
#define HOST "HOST"

constexpr double res = 0.3868223;

void monteCarlo_par(const char* device, int N, int num_points_for_thread, float& res, std::pair<float, float>& time) {
	try {
		auto queue_property = property_list{ property::queue::enable_profiling() };

		queue q;

		if (device == "CPU")
		{
			q = queue{ cpu_selector{}, async_handler{},queue_property };
			std::cout << "Device: " << q.get_device().get_info<info::device::name>() << '\n';
		}
		else if (device == "HOST")
		{
			q = queue{ host_selector{}, async_handler{},queue_property };
		}


		int n = 0;
		size_t num_thread = ceil(N / num_points_for_thread);

		buffer<int, 1> buf_n(&n, 1);
		buffer<int, 1> buf_N(&N, 1);
		buffer<int, 1> buf_num_points_for_th(&num_points_for_thread, 1);

		std::vector<float> vec_points(3 * N);
		buffer<float, 1> buf_points(vec_points.data(), vec_points.size());

		oneapi::mkl::rng::sobol engine(q, 3);
		oneapi::mkl::rng::uniform<float, oneapi::mkl::rng::uniform_method::standard> distribution(0.0f, 1.0f);

		try {
			auto start = std::chrono::steady_clock::now();
			oneapi::mkl::rng::generate(distribution, engine, vec_points.size(), buf_points);
			auto end = std::chrono::steady_clock::now();
			time.first = std::chrono::duration<double>(end - start).count();
		}
		catch (std::exception e) {
			std::cout << e.what() << std::endl;
		}

		sycl::event e = q.submit([&](handler& cgh) {
			auto N_points = buf_N.get_access<access::mode::read>(cgh);//количество всех точек
			auto count = buf_num_points_for_th.get_access<access::mode::read>(cgh);//количество точек на поток
			auto points = buf_points.get_access<access::mode::read>(cgh);//сгенерированные точки
			auto n_accessor = buf_n.get_access<access::mode::read_write>(cgh);//число точек под графиком
			auto n_sum_reduction = ONEAPI::reduction(n_accessor, 0, ONEAPI::plus<>());


			cl::sycl::stream output(1024, 256, cgh);

			cgh.parallel_for(nd_range<1>{num_thread, 1}, n_sum_reduction, [=](nd_item<1> it, auto& n_sum) {
				int start = it.get_global_linear_id() * count[0] * 3;
				int ind = start;
				int n_under = 0;
				float x = 0;
				float y = 0;
				float z = 0;
				while (ind < cl::sycl::min(start + count[0] * 3, N_points[0] * 3)) {
					x = points[ind];
					y = points[ind + 1];
					z = points[ind + 2];
					n_under += (z < sin(x) * cos(y)) ? 1 : 0;
					ind += 3;
				}
				n_sum += n_under;
				//output << " " << it.get_global_linear_id() << '\n';
				//output << " " << it.get_local_range(0) << '\n';
				//output << " " << it.get_global_range() << '\n';
				//output << " " << n << '\n';
				//output << " " << n << '\n';
			});
			//std::cout << n << std::endl;
		});

		std::cout << n << std::endl;

		e.wait_and_throw();
		q.wait_and_throw();


		double start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
		double end = e.get_profiling_info<sycl::info::event_profiling::command_end>();

		time.second = 1e-9 * (end - start);//из наносекунд в секунды
		res = float(n) / float(N);
	}
	catch (invalid_parameter_error& E) {
		std::cout << E.what() << std::endl;
		std::cout << "With OpenCL error code : " << E.get_cl_code() << std::endl;
	}
}



void monteCarlo_seq(const int N, float& res, std::pair<float, float>& time) {
	std::vector<float> vec_points(3 * N);
	buffer<float, 1> buf_points(vec_points.data(), vec_points.size());

	queue q(cpu_selector{});
	oneapi::mkl::rng::sobol engine(q, 3);
	oneapi::mkl::rng::uniform<float> distribution(0.0f, 1.0f);//случайное распределение(равномерное от 0 до 1)

	int n = 0;
	float x = 0;
	float y = 0;
	float z = 0;

	auto start = std::chrono::steady_clock::now();
	oneapi::mkl::rng::generate(distribution, engine, vec_points.size(), buf_points);//генерируем последовательность случ. чисел на устройстве
	auto end = std::chrono::steady_clock::now();
	time.first = std::chrono::duration<double>(end - start).count();


	start = std::chrono::steady_clock::now();
	{
		for (int i = 0; i < vec_points.size(); i += 3) {
			x = vec_points[i];
			y = vec_points[i + 1];
			z = vec_points[i + 2];
			n += (z < sin(x) * cos(y)) ? 1 : 0;
		}
		res = float(n) / float(N);
	}
	end = std::chrono::steady_clock::now();
	time.second = std::chrono::duration<double>(end - start).count();
}




int main() {
	bool repeat = true;
	while (repeat) {
		int N, points_for_thread;
		std::cout << "N points: ";
		std::cin >> N;
		std::cout << "Num of point for thread: ";
		std::cin >> points_for_thread;
		std::cout << "\n";

		float res_seq{};
		float res_CPU{};
		float res_HOST{};


		std::pair<float, float> time_seq;
		std::pair<float, float> time_CPU;
		std::pair<float, float> time_HOST;


		monteCarlo_seq(N, res_seq, time_seq);
		monteCarlo_par(CPU, N, points_for_thread, res_CPU, time_CPU);
		monteCarlo_par(HOST, N, points_for_thread, res_HOST, time_HOST);

		std::cout << "\n";
		std::cout << "  SEQ\t" << time_seq.first + time_seq.second << " sec" << std::endl;
		std::cout << "  CPU\t" << time_CPU.first + time_CPU.second << " sec" << std::endl;
		std::cout << "  HOST\t" << time_HOST.first + time_HOST.second << " sec" << std::endl;

		std::cout << "\n";
		std::cout << "Accuracy:" << std::endl;
		std::cout << "  SEQ\t" << abs(res - res_seq) << std::endl;
		std::cout << "  CPU_\t" << abs(res - res_CPU) << std::endl;
		std::cout << "  HOST\t" << abs(res - res_HOST) << std::endl;

	}

	return 0;
}