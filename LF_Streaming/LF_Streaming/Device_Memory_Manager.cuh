#pragma once
#include "LF_Utils.cuh"

class DeviceMemoryManager
{
public:
	DeviceMemoryManager(const size_t& num_of_slice, const size_t& slice_size);
	~DeviceMemoryManager();
	int rent_access_number();
	void return_access_number(const size_t& num);
	void init();
	uint8_t* get_empty_space(const size_t& access_number);
	uint8_t* dev_slice_buffer;

private:
	int num_of_slice;
	size_t slice_size;
	int last_access_number;
	std::vector<size_t> access_number_set;
};