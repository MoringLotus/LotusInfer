#pragma once
#include <vector>
#include <memory>
#include <random>
#include <iostream>
#include <stdexcept>
template <typename T>
class Tensor
{
private:
	std::vector<int64_t> shape_;
	std::vector<int64_t> strides_;
	std::size_t size_;
	std::shared_ptr<T[]> data_; // 不用手动管理data_
	void compute_strides()
	{
		strides_.resize(shape_.size());
		size_t stride = 1;
		for (int64_t i = static_cast<int64_t>(shape_.size()) - 1; i >= 0; i--)
		{
			strides_[i] = stride;
			stride *= static_cast<size_t>(shape_[i]);
		}
	}

public:
	Tensor(const std::vector<int64_t> &shape) : shape_(shape)
	{
		for (int64_t ndim : shape)
		{
			if (ndim < 0)
			{
				throw std::invalid_argument("negative dimension is not allowed");
			}
		}
		size_ = 1;
		for (int64_t ndim : shape_)
		{
			size_ *= ndim;
		}
		data_ = std::shared_ptr<T[]>(new T[size_]);
		compute_strides();
	}
	Tensor() : shape_{}, strides_{}, size_(0), data_(nullptr) {};

	// 获取Tensor信息
	size_t size() const { return size_; }
	size_t ndim() const { return shape_.size(); }
	const std::vector<int64_t> &shape() const { return shape_; }
	const std::vector<int64_t> &strides() const { return strides_; }
	const T *data() const { return data_.get(); }
	T *data() { return data_.get(); } // 可修改数据
	// 重载操作符
	T &operator[](size_t index)
	{
		if (this->isContiguous())
		{
			return data_[index];
		}
		else
		{
			throw std::runtime_error("tensor is not supported currently");
		}
	}

	const T &operator[](size_t index) const
	{
		if (this->isContiguous())
		{
			return data_[index];
		}
		else
		{
			throw std::runtime_error("tensor is not supported currently");
		}
	}
	// 数据填充
	void fill(const T &value)
	{
		T *ptr = data();
		for (size_t i = 0; i < size_; i++)
		{
			ptr[i] = value;
		}
	}
	void randomize(T mean = 0.0, T stddev = 1.0)
	{
		static std::random_device rd;
		static std::mt19937 gen(rd());
		std::normal_distribution<T> dist(mean, stddev);
		T *ptr = data();
		for (size_t i = 0; i < size_; i++)
		{
			ptr[i] = dist(gen);
		}
	}
	// 张量检查
	bool isContiguous() const
	{
		const auto &shape = this->shape();
		const auto &strides = this->strides();
		size_t ndim = shape.size();
		if (ndim == 0)
		{
			return true;
		}
		int64_t expected_stride = 1;
		for (int64_t i = static_cast<int64_t>(ndim) - 1; i >= 0; --i)
		{
			if (shape[i] == 1)
				continue;
			else if (strides[i] != expected_stride)
				return false;
			expected_stride *= shape[i];
		}
		return true;
	}
	// 形状变化
	Tensor view(const std::vector<int64_t> &new_shape) const
	{
		// 先判断数量是否匹配
		size_t target_numel = 1;
		for (int64_t s : new_shape)
		{
			target_numel *= static_cast<size_t>(s);
		}
		if (target_numel != this->size())
		{
			throw std::invalid_argument("View failed: target shape has different number of elements.");
		}
		// 先判断能不能合并
		if (this->isContiguous())
		{
			std::vector<int64_t> new_strides(new_shape.size());
			size_t new_stride = 1;
			for (int64_t i = static_cast<int64_t>(new_shape.size()) - 1; i >= 0; i--)
			{
				new_strides[i] = new_stride;
				new_stride *= new_shape[i];
			}
			Tensor new_tensor; // 默认构造函数，避免多余开辟data
			new_tensor.data_ = data_;
			new_tensor.shape_ = new_shape;
			new_tensor.strides_ = new_strides;
			new_tensor.size_ = size_;
			return new_tensor;
		}
		// 不能
		throw std::runtime_error("tensor is not contiguous.");
	}

	Tensor permute(const std::vector<size_t> &order) const
	{
		size_t ndims = this->ndim();
		if (order.size() != ndims)
		{
			throw std::invalid_argument("order size does not match tensor dimensions.");
		}
		std::vector<bool> used(ndims, false);
		for (size_t i = 0; i < ndims; i++)
		{
			size_t d = order[i];
			if (d >= ndims || used[d])
			{
				throw std::invalid_argument("invalid permutation order");
			}
			used[d] = true;
		}
		std::vector<int64_t> new_shape(ndims);
		std::vector<int64_t> new_strides(ndims);
		for (size_t i = 0; i < ndims; i++)
		{
			size_t d = order[i];
			new_shape[i] = this->shape()[d];
			new_strides[i] = this->strides()[d];
		}
		Tensor new_tensor;
		new_tensor.data_ = this->data_;
		new_tensor.shape_ = new_shape;
		new_tensor.strides_ = new_strides;
		new_tensor.size_ = this->size_;
		return new_tensor;
	}

	Tensor transpose(size_t dim1, size_t dim2) const
	{
		size_t ndims = this->ndim();
		if (dim1 >= ndims || dim2 >= ndims)
		{
			throw std::out_of_range("transpose dim out of range");
		}
		std::vector<size_t> dims(ndims);
		for (size_t i = 0; i < ndims; i++)
		{
			dims[i] = i;
		}
		std::swap(dims[dim1], dims[dim2]);
		return this->permute(dims);
	}

	Tensor reshape(const std::vector<int64_t> &new_shape) const
	{
		size_t target_numel = 1;
		for (int64_t s : new_shape)
		{
			target_numel *= static_cast<size_t>(s);
		}
		if (target_numel != this->size())
		{
			throw std::invalid_argument("target shape has different number of elements.");
		}
		if (this->isContiguous())
		{
			return this->view(new_shape);
		}
		else
		{
			Tensor new_tensor(new_shape);
			std::vector<int64_t> current_coord(this->shape_.size(), 0);
			for (size_t i = 0; i < target_numel; i++)
			{
				size_t old_physical_idx = 0;
				for (size_t j = 0; j < this->shape_.size(); j++)
				{
					old_physical_idx += current_coord[j] * strides_[j];
				}

				new_tensor.data_[i] = this->data_[old_physical_idx];
				for (int64_t d = static_cast<int64_t>(this->shape_.size()) - 1; d >= 0; d--)
				{
					current_coord[d]++;
					if (current_coord[d] < this->shape_[d])
					{
						break;
					}
					else
					{
						current_coord[d] = 0;
					}
				}
			}
			return new_tensor;
		}
	}
	// 打印调试
	void print_info() const
	{
		std::cout << "Shape: [";
		for (auto s : shape_)
		{
			std::cout << s << " ";
		}
		std::cout << "]\n";

		std::cout << "Strides: [";
		for (auto s : strides_)
		{
			std::cout << s << " ";
		}
		std::cout << "]\n";

		std::cout << "Numel: " << size_ << "\n";
		std::cout << "Contiguous: " << (this->isContiguous() ? "true" : "false") << "\n";
	}
};