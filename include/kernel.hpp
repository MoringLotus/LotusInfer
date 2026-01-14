#include <iostream>

template<typename T>
class Kernel {
    public:
        Tensor<T> input_tensor;
        Tensor<T> output_tensor;
        void forward();
};