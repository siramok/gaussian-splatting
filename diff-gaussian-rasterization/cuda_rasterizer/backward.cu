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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

__device__ __forceinline__ float sq(float x) { return x * x; }

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P,
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float* conics,
	const float scale_modifier,
	const float* dL_dconics,
	glm::vec3* dL_dscales,
	glm::vec4* dL_drots)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	auto scale = scales[idx];
	auto rot = rotations[idx];
	
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = scale_modifier * scale.x;
	S[1][1] = scale_modifier * scale.y;
	S[2][2] = scale_modifier * scale.z;

	// Normalize quaternion to get valid rotation (commented out for some reason?)
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	float conic[6] = {
		conics[6 * idx],
		conics[6 * idx + 1],
		conics[6 * idx + 2],
		conics[6 * idx + 3],
		conics[6 * idx + 4],
		conics[6 * idx + 5],
	};
	float dL_dconic[6] = {
		dL_dconics[6 * idx],
		dL_dconics[6 * idx + 1],
		dL_dconics[6 * idx + 2],
		dL_dconics[6 * idx + 3],
		dL_dconics[6 * idx + 4],
		dL_dconics[6 * idx + 5],
	};

	// Compute dL_dcov as -conic * dL_dconic * conic
	// since conic is inverse of cov
	const float dL_dconic_conic[6] = {
		dL_dconic[0]*conic[0] + dL_dconic[1]*conic[1] + dL_dconic[2]*conic[2],
    	dL_dconic[0]*conic[1] + dL_dconic[1]*conic[3] + dL_dconic[2]*conic[4],
    	dL_dconic[0]*conic[2] + dL_dconic[1]*conic[4] + dL_dconic[2]*conic[5],
    	dL_dconic[1]*conic[1] + dL_dconic[3]*conic[3] + dL_dconic[4]*conic[4],
    	dL_dconic[1]*conic[2] + dL_dconic[3]*conic[4] + dL_dconic[4]*conic[5],
    	dL_dconic[2]*conic[2] + dL_dconic[4]*conic[4] + dL_dconic[5]*conic[5]
	};
	const float dL_dcov[6] = {
		-1.0f * (conic[0]*dL_dconic_conic[0] + conic[1]*dL_dconic_conic[1] + conic[2]*dL_dconic_conic[2]),
    	-1.0f * (conic[0]*dL_dconic_conic[1] + conic[1]*dL_dconic_conic[3] + conic[2]*dL_dconic_conic[4]),
    	-1.0f * (conic[0]*dL_dconic_conic[2] + conic[1]*dL_dconic_conic[4] + conic[2]*dL_dconic_conic[5]),
    	-1.0f * (conic[1]*dL_dconic_conic[1] + conic[3]*dL_dconic_conic[3] + conic[4]*dL_dconic_conic[4]),
    	-1.0f * (conic[1]*dL_dconic_conic[2] + conic[3]*dL_dconic_conic[4] + conic[4]*dL_dconic_conic[5]),
    	-1.0f * (conic[2]*dL_dconic_conic[2] + conic[4]*dL_dconic_conic[4] + conic[5]*dL_dconic_conic[5])
	};
	float abs_Sigma00 = abs(Sigma[0][0]);
	float abs_Sigma11 = abs(Sigma[1][1]);
	float abs_Sigma22 = abs(Sigma[2][2]);
	float depsilon_dSigma00 = (abs_Sigma00 >= abs_Sigma11 && abs_Sigma00 >= abs_Sigma22) ? 1e-5 * glm::sign(Sigma[0][0]) : 0.0f;
	float depsilon_dSigma11 = (abs_Sigma11 >= abs_Sigma00 && abs_Sigma11 >= abs_Sigma22) ? 1e-5 * glm::sign(Sigma[1][1]) : 0.0f;
	float depsilon_dSigma22 = (abs_Sigma22 >= abs_Sigma00 && abs_Sigma22 >= abs_Sigma11) ? 1e-5 * glm::sign(Sigma[2][2]) : 0.0f;
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov[0] + (dL_dcov[0] + dL_dcov[3] + dL_dcov[5]) * depsilon_dSigma00, dL_dcov[1], dL_dcov[2],
		dL_dcov[1], dL_dcov[3] + (dL_dcov[0] + dL_dcov[3] + dL_dcov[5]) * depsilon_dSigma11, dL_dcov[4],
		dL_dcov[2], dL_dcov[4], dL_dcov[5] + (dL_dcov[0] + dL_dcov[3] + dL_dcov[5]) * depsilon_dSigma22
	);
	glm::mat3 dL_dM = 2.f * dL_dSigma * M;
	glm::mat3 dL_dS = dL_dM * glm::transpose(R);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = dL_dS[0][0] * scale_modifier;
	dL_dscale->y = dL_dS[1][1] * scale_modifier;
	dL_dscale->z = dL_dS[2][2] * scale_modifier;

	dL_dM[0] *= scale_modifier * scale.x;
	dL_dM[1] *= scale_modifier * scale.y;
	dL_dM[2] *= scale_modifier * scale.z;
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dM[1][0] - dL_dM[0][1]) + 2 * y * (dL_dM[0][2] - dL_dM[2][0]) + 2 * x * (dL_dM[2][1] - dL_dM[1][2]);
	dL_dq.y = 2 * y * (dL_dM[0][1] + dL_dM[1][0]) + 2 * z * (dL_dM[0][2] + dL_dM[2][0]) + 2 * r * (dL_dM[2][1] - dL_dM[1][2]) - 4 * x * (dL_dM[2][2] + dL_dM[1][1]);
	dL_dq.z = 2 * x * (dL_dM[0][1] + dL_dM[1][0]) + 2 * r * (dL_dM[0][2] - dL_dM[2][0]) + 2 * z * (dL_dM[2][1] + dL_dM[1][2]) - 4 * y * (dL_dM[2][2] + dL_dM[0][0]);
	dL_dq.w = 2 * r * (dL_dM[1][0] - dL_dM[0][1]) + 2 * x * (dL_dM[0][2] + dL_dM[2][0]) + 2 * y * (dL_dM[2][1] + dL_dM[1][2]) - 4 * z * (dL_dM[1][1] + dL_dM[0][0]);
	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y * BLOCK_Z)
renderCUDA(
	const dim3 grid,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const float3 volume_mins,
	const uint3 num_cells,
	const float cell_size,
	const bool* __restrict__ clamped,
	const float3* __restrict__ means3D,
	const float* __restrict__ values,
	const float* __restrict__ out_cells,
	const float* __restrict__ volumes,
	const float* __restrict__ conic,
	const float* __restrict__ accumulated_weights,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dcells,
	float3* __restrict__ dL_dmeans,
	float* __restrict__ dL_dconic,
	float* __restrict__ dL_dvalues
)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	uint3 cell_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y, block.group_index().z * BLOCK_Z};
	uint3 cell_max = { min(cell_min.x + BLOCK_X, num_cells.x), min(cell_min.y + BLOCK_Y , num_cells.y), min(cell_min.z + BLOCK_Z , num_cells.z) };
	uint3 cell = { cell_min.x + block.thread_index().x, cell_min.y + block.thread_index().y, cell_min.z + block.thread_index().z  };
	float3 cell_pos =  make_float3(
		(static_cast<float>(cell.x) + 0.5) * cell_size + volume_mins.x, 
		(static_cast<float>(cell.y) + 0.5) * cell_size + volume_mins.y, 
		(static_cast<float>(cell.z) + 0.5) * cell_size + volume_mins.z
	);
	uint32_t cell_id = cell.z * num_cells.x * num_cells.y + cell.y * num_cells.x + cell.x;

	// Check if this thread is associated with a valid cell or outside.
	bool inside = cell.x < num_cells.x && cell.y < num_cells.y && cell.z < num_cells.z;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().z * grid.y * grid.x + block.group_index().y * grid.x + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float3 collected_means[BLOCK_SIZE];
	__shared__ float collected_volumes[BLOCK_SIZE];
	__shared__ float collected_values[BLOCK_SIZE];
	__shared__ float collected_clamped[BLOCK_SIZE];
	__shared__ float collected_conic[BLOCK_SIZE * 6];

	float acc_weight = accumulated_weights[cell_id];
	float dl_dout = dL_dcells[cell_id];
	
	// Iterate over batches
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Collective fetch similar to forward pass
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_means[block.thread_rank()] = means3D[coll_id];
			collected_volumes[block.thread_rank()] = volumes[coll_id];
			collected_values[block.thread_rank()] = values[coll_id];
			collected_clamped[block.thread_rank()] = clamped[coll_id];
			for (int k = 0; k < 6; k++)
				collected_conic[block.thread_rank() * 6 + k] = conic[coll_id * 6 + k];
		}
		block.sync();

		// Process current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Only process if we have meaningful gradients
			if (inside && accumulated_weights[cell_id] > 1e-5)
			{
                int point_idx = collected_id[j];
				float3 d = make_float3(cell_pos.x - collected_means[j].x, cell_pos.y - collected_means[j].y, cell_pos.z - collected_means[j].z);
                
                // Compute quadratic form and weight as in forward pass
                float quad_form = (
                    d.x * (collected_conic[j * 6] * d.x + collected_conic[j * 6 + 1] * d.y + collected_conic[j * 6 + 2] * d.z) +
                    d.y * (collected_conic[j * 6 + 1] * d.x + collected_conic[j * 6 + 3] * d.y + collected_conic[j * 6 + 4] * d.z) +
                    d.z * (collected_conic[j * 6 + 2] * d.x + collected_conic[j * 6 + 4] * d.y + collected_conic[j * 6 + 5] * d.z)
                );
                float normalize_factor = 1.0f;
                float weight = normalize_factor * exp(-0.5f * quad_form);

                if (exp(-0.5f * quad_form) > 1.0f)
                    continue;

                // Compute gradients
                // dl_dvalue = dl_dout * dout_dvalue
                float dl_dvalue = dl_dout * weight / acc_weight;
				// If clamped don't add gradient (Pytorch rules)
				if (!collected_clamped[j]) {
                	atomicAdd(&dL_dvalues[point_idx], dl_dvalue);
				}

                // Gradient for weight terms
                float dL_dweight = dl_dout * (collected_values[j] / acc_weight - out_cells[cell_id] / acc_weight);
                float dweight_dquad = -0.5f * weight;
				float dL_dquad = dL_dweight * dweight_dquad;
                
                // Gradients for means
                float3 dl_dmean;
                dl_dmean.x = dL_dquad * 2 * -1 *
                    (collected_conic[j * 6] * d.x + collected_conic[j * 6 + 1] * d.y + collected_conic[j * 6 + 2] * d.z);
                dl_dmean.y = dL_dquad * 2 * -1 *
                    (collected_conic[j * 6 + 1] * d.x + collected_conic[j * 6 + 3] * d.y + collected_conic[j * 6 + 4] * d.z);
                dl_dmean.z = dL_dquad * 2 * -1 *
                    (collected_conic[j * 6 + 2] * d.x + collected_conic[j * 6 + 4] * d.y + collected_conic[j * 6 + 5] * d.z);
                
                atomicAdd(&dL_dmeans[point_idx].x, dl_dmean.x);
                atomicAdd(&dL_dmeans[point_idx].y, dl_dmean.y);
                atomicAdd(&dL_dmeans[point_idx].z, dl_dmean.z);

                // Gradients for conic matrix
                atomicAdd(&dL_dconic[point_idx * 6], dL_dquad * d.x * d.x);      // xx
                atomicAdd(&dL_dconic[point_idx * 6 + 1], dL_dquad * d.x * d.y);  // xy
                atomicAdd(&dL_dconic[point_idx * 6 + 2], dL_dquad * d.x * d.z);  // xz
                atomicAdd(&dL_dconic[point_idx * 6 + 3], dL_dquad * d.y * d.y);  // yy
                atomicAdd(&dL_dconic[point_idx * 6 + 4], dL_dquad * d.y * d.z);  // yz
                atomicAdd(&dL_dconic[point_idx * 6 + 5], dL_dquad * d.z * d.z);  // zz
            }
        }
	}
}

void BACKWARD::preprocess(
	int P,
	const int* radii,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float* conics,
	const float scale_modifier,
	const float* dL_dconic,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	// Propagate gradients for remaining steps: using dl_dconics
	// propagate back to scales and rotations
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P,
		radii,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		conics,
		scale_modifier,
		dL_dconic,
		dL_dscale,
		dL_drot);
}

void BACKWARD::render(
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
	float* dL_dvalue)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		grid,
		ranges,
		point_list,
		volume_mins,
		num_cells,
		cell_size,
		clamped,
		means3D,
		values,
		out_cells,
		volumes,
		conic,
		accumulated_weights,
		n_contrib,
		dL_dcells,
		dL_dmean3D,
		dL_dconic,
		dL_dvalue);
}
