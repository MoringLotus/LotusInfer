#include <iostream>
#include <memory>
#include "memory_pool.hpp"

template <typename T>
class Tensor {
	public:
		std::vector<size_t> shape;
		std::vector<size_t> stride;
		int ndim;
		int nelements;
		std::shared_ptr<mm::MemoryPool> pool;

		Tensor (std::shared_ptr<mm::MemoryPool> pool, std::vector<size_t> shape_vec): shape(shape_vec), ndim(shape_vec.size()){
			calcuteStrides(shape_vec);
			nelements = calcuteElements(shape_vec);
			data_ptr = (T *)pool->alloc(sizeof(T) * nelements);
		}

		~ Tensor() {
			pool->free(data_ptr);
		}

		void calcuteStrides(std::vector<size_t> shape){
			int n = static_cast<int>(shape.size());
			stride.clear();  // 先清空
			
			if (n == 0) return;  // 处理0维情况
			
			// 行主序（C风格）：最后一个维度步长为1
			stride.resize(n);
			stride[n - 1] = 1;
			
			for (int i = n - 2; i >= 0; --i) {
				stride[i] = stride[i + 1] * shape[i + 1];
			}
		}

		int calcuteElements(std::vector<size_t> shape) {
			int n = 1;
			for(auto i : shape){
				n *= i;
			}
			return n;
		}

		int getElements() {
			return nelements;
		}

		std::vector<size_t> getStrides(){
			return stride;
		}

		int getDims(){
			return this->ndim;
		}

		
		void loadData(T * source_data_ptr){
			// this copy function use iterator
			std::copy(source_data_ptr, source_data_ptr + nelements, data_ptr);
		}

		void debug(){
			for(int i = 0; i < nelements; i ++){
				std::cout << data_ptr[i] << " ";
			}
		}
	private:
		T * data_ptr;	
};	
