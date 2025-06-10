/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const uint* aabbs,
	const uint32_t* offsets,
	uint32_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Find this Gaussian's offset in buffer for writing keys/values.
	uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
	for (int z = aabbs[idx * 6 + 2]; z < aabbs[idx * 6 + 5]; z++) {
		for (int y = aabbs[idx * 6 + 1]; y < aabbs[idx * 6 + 4]; y++) {
			for (int x = aabbs[idx * 6]; x < aabbs[idx * 6 + 3]; x++) {
				gaussian_keys_unsorted[off] =  z * grid.x * grid.y + y * grid.x + x;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint32_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint32_t currcell = point_list_keys[idx];
	if (idx == 0)
		ranges[currcell].x = 0;
	else
	{
		uint32_t prevcell = point_list_keys[idx - 1];
		if (currcell != prevcell)
		{
			ranges[prevcell].y = idx;
			ranges[currcell].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currcell].y = L;
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.clamped, P, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.values, P, 128);
	obtain(chunk, geom.volumes, P, 128);
	obtain(chunk, geom.means, P, 128);
	obtain(chunk, geom.conic, P * 6, 128);
	obtain(chunk, geom.aabbs, P * 6, 128);
	obtain(chunk, geom.cells_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.cells_touched, geom.cells_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P,
	const float* means3D,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* values,
	const float3 volume_mins,
	const float3 volume_maxes,
	const uint3 num_cells,
	const float cell_size,
	float* out_cells,
	int* radii,
	bool debug)
{
	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 block_grid((num_cells.x + BLOCK_X - 1) / BLOCK_X, (num_cells.y + BLOCK_Y - 1) / BLOCK_Y, (num_cells.z + BLOCK_Z - 1) / BLOCK_Z);
	dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(num_cells.x * num_cells.y * num_cells.z);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, num_cells.x * num_cells.y * num_cells.z);

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of values to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		values,
		geomState.clamped,
		volume_mins, volume_maxes,
		num_cells,
		cell_size,
		radii,
		geomState.means,
		geomState.values, geomState.volumes,
		geomState.conic,
		geomState.aabbs,
		block_grid,
		geomState.cells_touched
	), debug)

	// if (debug) {
	// 	int* host_cells_touched = new int[P];
	// 	cudaMemcpy(host_cells_touched, geomState.cells_touched, P * sizeof(int), cudaMemcpyDeviceToHost);
	// 	std::cout << "cells_touched:" << std::endl;
	// 	for (int i = 0; i < 1000; ++i) {
	// 		std::cout << host_cells_touched[i] << " ";
	// 	}
	// 	std::cout << std::endl;
	// 	delete[] host_cells_touched;
	// }

	// uint* host_aabbs = new uint[P * 6];
	// cudaMemcpy(host_aabbs, geomState.aabbs, P * 6 * sizeof(uint), cudaMemcpyDeviceToHost);
	// std::cout << "aabbs:" << std::endl;
	// for (int i = 0; i < 1000; ++i) {
	// 	std::cout << host_aabbs[i] << " ";
	// }
	// std::cout << std::endl;
	// delete[] host_aabbs;

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.cells_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_intersections;
	CHECK_CUDA(cudaMemcpy(&num_intersections, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);
	if (debug) {
		std::cout << "Num Intersections " << num_intersections << "\n";
	}

	size_t binning_chunk_size = required<BinningState>(num_intersections);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_intersections);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding duplicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.aabbs,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		block_grid)
	CHECK_CUDA(, debug)

	// cudaDeviceSynchronize();
    // uint32_t* host_keys = new uint32_t[100];
    // cudaMemcpy(host_keys, binningState.point_list_keys_unsorted, 
    //            100 * sizeof(uint32_t), 
    //            cudaMemcpyDeviceToHost);
    // // Print and verify the keys are within expected range
    // for (int i = 0; i < 100; i++) {
	// 	std::cout << host_keys[i] << " ";
    //     if (host_keys[i] >= block_grid.x * block_grid.y * block_grid.z) {
    //         printf("Invalid key at %d: %u\n", i, host_keys[i]);
    //     }
    // }
    // delete[] host_keys;
	// std::cout << "\n";

	int bit = getHigherMsb(block_grid.x * block_grid.y * block_grid.z);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_intersections, 0, bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, num_cells.x * num_cells.y * num_cells.z * sizeof(uint2)), debug);


	// Identify start and end of per-tile workloads in sorted list
	if (num_intersections > 0)
		identifyTileRanges << <(num_intersections + 255) / 256, 256 >> > (
			num_intersections,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each cell blend its range of Gaussians independently in parallel
	const float* feature_ptr = geomState.values;
	CHECK_CUDA(FORWARD::render(
		block_grid, block,
		imgState.ranges,
		binningState.point_list,
		volume_mins,
		num_cells,
		cell_size,
		geomState.means,
		feature_ptr,
		geomState.volumes,
		geomState.conic,
		imgState.accum_alpha,
		imgState.n_contrib,
		out_cells), 
		debug)


	return num_intersections;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int R,
	const float* means3D,
	const float* scales,
	const float scale_modifier,
	const uint3 num_cells,
	const float3 volume_mins, const float3 volume_maxes,
	const float cell_size,
	const float* rotations,
	const float* values,
	const float* out_cells,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dcells,
	float* dL_dconic,
	float* dL_dmean3D,
	float* dL_dscale,
	float* dL_drot,
	float* dL_dvalue,
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, num_cells.x * num_cells.y * num_cells.z);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 block_grid((num_cells.x + BLOCK_X - 1) / BLOCK_X, (num_cells.y + BLOCK_Y - 1) / BLOCK_Y, (num_cells.z + BLOCK_Z - 1) / BLOCK_Z);
	dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);

	// Compute loss gradients w.r.t. mean position, conic matrix,
	// opacity and value of Gaussians from per-cell loss gradients.
	CHECK_CUDA(BACKWARD::render(
		block_grid, block,
		imgState.ranges,
		binningState.point_list,
		volume_mins,
		num_cells,
		cell_size,
		geomState.clamped,
		geomState.means,
		geomState.values,
		out_cells,
		geomState.volumes,
		geomState.conic,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dcells,
		(float3*)dL_dmean3D,
		dL_dconic,
		dL_dvalue), debug);


	// Take care of the rest of preprocessing, compute loss w.r.t
	// scales and rotation from conic gradients.
	CHECK_CUDA(BACKWARD::preprocess(P,
		radii,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		geomState.conic,
		scale_modifier,
		dL_dconic,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug);
}
