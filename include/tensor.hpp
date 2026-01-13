#include <iostream>
<template T>
class Tensor {
	public:
		Tensor() {};
		~Tensor() {};
		vector<int> shape;
		vector<int> stride;
	private:
		T * data_ptr;	
}	
