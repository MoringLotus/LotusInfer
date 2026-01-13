#include <iostream>
#include "tensor.hpp"
int main () {
  std::cout << "Test Entry" << std::endl;
  mm::MemoryPool mp;
  mp.alloc(sizeof(float) * 256);
  mp.debugInfo();
}
