#include "tensor.hpp"
#include "kernel.hpp"

template<typename T>
class RMSNorm : public Kernel<T>{
    public:
        T gemma;
        T epsilon;

        RMSNorm (Tensor input_tensor, Tensor output_tensor) {
            this->input_tensor = input_tensor;
            this->output_tensor = output_tensor;
        }

        void loadWeight(T gemma, T epsilon){
            this->gemma = gemma;
            this->epsilon = epsilon;
        }
    
        void forward(){
            
        }
        
};