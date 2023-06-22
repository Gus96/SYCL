#include <iostream>
#include "image.h"
#include <CL/sycl.hpp>


using namespace cl::sycl;

const size_t filter_size = 16;

void blur_Par(const std::vector<float>& src, std::vector<float>& res,
	const int h, const int w, const int num_ch, const float part_cpu, const int f_size, double& time) {
	try {
		auto queue_property = property_list{ property::queue::enable_profiling() };

		queue q_cpu(cpu_selector{}, async_handler{}, queue_property);
		queue q_host(host_selector{}, async_handler{}, queue_property);

		std::cout << "Device: " << q_cpu.get_device().get_info<info::device::name>() << std::endl;


		int split = h * part_cpu;

		buffer<float, 1> src_buf(src.data(), src.size());
		buffer<float, 1> res_buf(res.data(), res.size());

		buffer<int, 1> f_size_buf(&f_size, 1);
		buffer<int, 1> w_buf(&w, 1);
		buffer<int, 1> h_buf(&h, 1);

		sycl::event e_cpu = q_cpu.submit([&](handler& cgh) {
			auto src_ = src_buf.get_access<access::mode::read>(cgh);
			auto w_ = w_buf.get_access<access::mode::read>(cgh);
			auto h_ = h_buf.get_access<access::mode::read>(cgh);
			auto f_size_ = f_size_buf.get_access<access::mode::read>(cgh);
			auto res_ = res_buf.get_access<access::mode::write>(cgh);

			//cl::sycl::stream output(5024, 256, cgh);
			cgh.parallel_for(num_ch * w * split, [=](nd_item<1> item) {
				int offset = f_size_[0] / 2;
				int h = h_[0];
				int w = w_[0];

				int id = item.get_global_id(0);
				int channel = id / (split * w);//ищем, какому каналу принадлежит поток (split * w) - часть картинки одного канала
				int pos = id % (split * w);//смотрим, в какой позиции находится в этом канале
				int x = pos / w;//получаем номер строки
				int y = pos % w;//получаем номер столбца

				float res_px = 0;
				float num_pxl = ((x + offset) - sycl::max(0, x - offset)) *
					(sycl::min(w - 1, y + offset) - sycl::max(0, y - offset));

				for (int ii = sycl::max(0, x - offset); ii <= x + offset; ++ii)
					for (int jj = sycl::max(0, y - offset); jj <= sycl::min(w - 1, y + offset); ++jj)
						res_px += src_[channel * h * w + ii * w + jj];

				res_px /= num_pxl;
				res_[channel * h * w + x * w + y] = res_px;
			});
		});



		sycl::event e_host = q_host.submit([&](handler& cgh) {
			auto src_ = src_buf.get_access<access::mode::read>(cgh);
			auto f_size_ = f_size_buf.get_access<access::mode::read>(cgh);
			auto w_ = w_buf.get_access<access::mode::read>(cgh);
			auto h_ = h_buf.get_access<access::mode::read>(cgh);
			auto res_ = res_buf.get_access<access::mode::write>(cgh);

			//cl::sycl::stream output(1024, 256, cgh);
			cgh.parallel_for(num_ch * w * (h - split), [=](nd_item<1> item) {

				int offset = f_size_[0] / 2;
				int h = h_[0];
				int w = w_[0];
				int h_cpu = h - split;

				int id = item.get_global_id(0);
				int channel = id / (split * w);
				int pos = id % (split * w);
				int x = pos / w + h_cpu;
				int y = pos % w;

				float res_px = 0;
				float num_pxl = (sycl::min(h - 1, x + offset) - (x - offset)) *
					(sycl::min(w - 1, y + offset) - sycl::max(0, y - offset));

				for (int ii = x - offset; ii <= sycl::min(h - 1, x + offset); ++ii)
					for (int jj = sycl::max(0, y - offset); jj <= sycl::min(w - 1, y + offset); ++jj)
						res_px += src_[channel * h * w + ii * w + jj];

				res_px /= num_pxl;
				res_[channel * h * w + x * w + y] = res_px;
			});
		});

		//e_host.wait_and_throw();
		//e_cpu.wait_and_throw();

		q_host.wait_and_throw();
		q_cpu.wait_and_throw();


		double time_host = 1e-9 * (e_host.get_profiling_info<sycl::info::event_profiling::command_end>() -
			e_host.get_profiling_info<sycl::info::event_profiling::command_start>());//из наносекунд в секунды
		double time_cpu = 1e-9 * (e_cpu.get_profiling_info<sycl::info::event_profiling::command_end>() -
			e_cpu.get_profiling_info<sycl::info::event_profiling::command_start>());

		std::cout << std::endl;
		std::cout << "Time CPU: " << time_cpu+ time_host << std::endl;
		std::cout << "Time Host: " << time_host + time_host << std::endl;


		time = std::max(time_host, time_cpu);
	}
	catch (invalid_parameter_error& E) {
		std::cout << E.what() << std::endl;
		std::cout << "With OpenCL error code : " << E.get_cl_code() << std::endl;
	}
}


void blurParallel(const char* device, const std::vector<float>& src, std::vector<float>& dst,
	const int h, const int w, const int n, const int fs, double& time) {
	try {
		auto queue_property = property_list{ property::queue::enable_profiling() };
		queue q;
		if (device == "CPU")
		{
			q = queue{ cpu_selector{}, async_handler{},queue_property };
			//std::cout << "Device: " << q.get_device().get_info<info::device::name>() << '\n';
		}
		else if (device == "HOST")
		{
			q = queue{ host_selector{}, async_handler{},queue_property };
		}

		range<1> img_shape{ n * h * w };
		buffer<float, 1> src_buf(src.data(), img_shape);
		buffer<float, 1> dst_buf(dst.data(), img_shape);
		buffer<int, 1> fs_buf(&fs, range<1> {1});
		buffer<int, 1> h_buf(&h, range<1> {1});
		buffer<int, 1> w_buf(&w, range<1> {1});

		sycl::event e = q.submit([&](handler& cgh) {
			auto src_ = src_buf.get_access<access::mode::read>(cgh);
			auto fs_ = fs_buf.get_access<access::mode::read>(cgh);
			auto h_ = h_buf.get_access<access::mode::read>(cgh);
			auto w_ = w_buf.get_access<access::mode::read>(cgh);
			auto dst_ = dst_buf.get_access<access::mode::write>(cgh);

			cgh.parallel_for(img_shape, [=](nd_item<1> item) {
				int ofs = fs_[0] / 2;
				int h = h_[0];
				int w = w_[0];

				int c = item.get_global_id(0) / (h * w);
				int i = item.get_global_id(0) % (h * w);
				int x = i / w;
				int y = i % w;

				float dst_px = 0;
				float divisor = (sycl::min(h - 1, x + ofs) - sycl::max(0, x - ofs)) *
					(sycl::min(w - 1, y + ofs) - sycl::max(0, y - ofs));

				for (int ii = sycl::max(0, x - ofs); ii <= sycl::min(h - 1, x + ofs); ++ii)
					for (int jj = sycl::max(0, y - ofs); jj <= sycl::min(w - 1, y + ofs); ++jj)
						dst_px += src_[c * h * w + ii * w + jj];

				dst_px /= divisor;
				dst_[item.get_global_id(0)] = dst_px;
			});
		});

		//e.wait_and_throw();
		q.wait_and_throw();

		double start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
		double end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
		time = 1e-9 * (end - start);
	}
	catch (invalid_parameter_error& E) {
		std::cout << E.what() << std::endl;
		std::cout << "With OpenCL error code : " << E.get_cl_code() << std::endl;
	}
}

void blur_Seq(const std::vector<float>& src, std::vector<float>& res,
	const int h, const int w, const int num_ch, const int f_size, double& time) {
	const int offset = f_size / 2;
	float pxl = 0.f;
	float num_pxl = 0.f;

	auto start = std::chrono::steady_clock::now();
	for (int c = 0; c < num_ch; ++c)
		for (int i = 0; i < h; ++i)
			for (int j = 0; j < w; ++j) {
				pxl = 0.f;
				for (int ii = std::max(0, i - offset); ii < std::min(h, i + offset); ++ii)// i - центр кернела
					for (int jj = std::max(0, j - offset); jj < std::min(w, j + offset); ++jj)
						pxl += src[c * h * w + ii * w + jj];

				num_pxl = (std::min(h - 1, i + offset) - std::max(0, i - offset)) *
					(std::min(w - 1, j + offset) - std::max(0, j - offset));

				res[c * h * w + i * w + j] = pxl / num_pxl;
			}
	auto end = std::chrono::steady_clock::now();

	time = std::chrono::duration<double>(end - start).count();
}

std::vector<float> img_To_Vec(const img::Image& img) {
	std::vector<float> v = img.r;
	v.insert(v.end(), img.g.begin(), img.g.end());
	v.insert(v.end(), img.b.begin(), img.b.end());
	return v;
}


img::Image vec_To_Img(const std::vector<float>& v, int h, int w) {
	img::Image img;
	img.r = std::vector<float>(v.begin(), v.begin() + h * w);
	img.g = std::vector<float>(v.begin() + h * w, v.begin() + 2 * h * w);
	img.b = std::vector<float>(v.begin() + 2 * h * w, v.end());
	img.h = h;
	img.w = w;
	return img;
}


int main() {
	img::Image src_img;
	if (!img::Load("forest.jpg", src_img)) std::cout << "Image load error" << std::endl;
	int h = src_img.h;
	int w = src_img.w;
	int c = 3;

	std::vector<float> src = img_To_Vec(src_img);

	std::vector<float> res_seq_cpu(src.size(), 0);
	std::vector<float> res_seq_host(src.size(), 0);
	std::vector<float> res_seq(src.size(), 0);
	std::vector<float> res_parallel(src.size(), 0);

	double time_par_cpu{};
	double time_par_host{};
	double time_seq{};
	double time_paralell{};

	float part_cpu = 0.5;


	blurParallel("CPU", src, res_seq_cpu, h, w, c, filter_size, time_par_cpu);
	std::cout << "CPU time:\t" << time_par_cpu << "s" << std::endl;
	blurParallel("HOST", src, res_seq_host, h, w, c, filter_size, time_par_host);
	std::cout << "Host time:\t" << time_par_host << "s" << std::endl;

	blur_Seq(src, res_seq, h, w, c, filter_size, time_seq);
	blur_Par(src, res_parallel, h, w, c, part_cpu, filter_size, time_paralell);

	std::cout << std::endl;
	std::cout << "Seq time:\t" << time_seq << "s" << std::endl;
	std::cout << "Parallel time:\t" << time_paralell << "s" << std::endl;
	std::cout << "boost: " << time_seq / time_paralell << std::endl;


	img::Image img_seq = vec_To_Img(res_seq, h, w);
	img::Image img_parallel = vec_To_Img(res_parallel, h, w);

	if (!img::SavePng("Seq.jpg", img_seq)) std::cout << "Image save error" << std::endl;
	if (!img::SavePng("Parallel.jpg", img_parallel)) std::cout << "Image save error" << std::endl;

	return 0;
}