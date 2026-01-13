#include "tensor.hpp"
#include "memory.h"

int main () {

    auto pool = std::make_shared<mm::MemoryPool>();
    std::vector<size_t> shape{3, 4, 5};
    std::shared_ptr<Tensor<int>> t1 = std::make_shared<Tensor<int>>(pool, shape);
    std::cout << "create end" << std::endl;
    size_t dims = t1->getDims();
    if(dims == 3){
        std::cout << "dim is right" << std::endl;
    }
    size_t nelements = t1->getElements();
    int * data_dst = (int *)malloc(sizeof(int) * nelements);
    for(int i  = 0; i < nelements; i ++){
        data_dst[i] = i;
    }
    t1->loadData(data_dst);
    t1->debug();
    std::vector<size_t> stride = t1->getStrides();
    std::cout << "Stride is ";
    for(auto i : stride){
        std::cout << i << " ";
    }
    std::cout << std::endl;
}