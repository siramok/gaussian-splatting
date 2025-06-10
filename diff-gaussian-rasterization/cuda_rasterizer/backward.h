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

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, const dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		const float3 volume_mins,
		const uint3 num_cells,
		const float cell_size,
		const bool* clamped,
		const float3* means3D,
		const float* values,
		const float* out_cells,
		const float* volumes,
		const float* conic,
		const float* accumulated_weights,
		const uint32_t* n_contrib,
		const float* dL_dcells,
		float3* dL_dmean3D,
		float* dL_dconic,
		float* dL_dvalue);

	void preprocess(
		int P,
		const int* radii,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float* conics,
		const float scale_modifier,
		const float* dL_dconics,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot);
}

#endif
