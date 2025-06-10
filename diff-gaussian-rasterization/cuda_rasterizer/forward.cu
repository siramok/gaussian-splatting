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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* values,
	bool* clamped,
	const float3 volume_mins,
	const float3 volume_maxes,
	const uint3 num_cells,
	const float cell_size,
	int* radii,
	float3* means,
	float* values_out, float* volumes,
	float* conic,
	uint* aabbs,
	const dim3 grid,
	uint32_t* cells_touched)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize touched cells to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	cells_touched[idx] = 0;
	radii[idx] = 0;

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

	// Normalize by epsilon to prevent numerical issues
	const float epsilon = max(max(abs(Sigma[0][0]), abs(Sigma[1][1])), abs(Sigma[2][2])) * 1e-5;
	const float cov[6] = {
		Sigma[0][0] + epsilon,
        Sigma[0][1],
        Sigma[0][2],
		Sigma[1][1] + epsilon,
        Sigma[1][2],
        Sigma[2][2] + epsilon,
	};

	// Use 3D covariance to compute and store 3D conic
	const float a = cov[0]; // Sigma[0][0]
    const float b = cov[1]; // Sigma[0][1]
    const float c = cov[2]; // Sigma[0][2]
    const float d = cov[3]; // Sigma[1][1]
    const float e = cov[4]; // Sigma[1][2]
    const float f = cov[5]; // Sigma[2][2]
    const float det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
    const float det_inv = 1.0 / det;
    conic[idx * 6] = (d * f - e * e) * det_inv;
    conic[idx * 6 + 1] = (c * e - b * f) * det_inv;
    conic[idx * 6 + 2] = (b * e - c * d) * det_inv;
    conic[idx * 6 + 3] = (a * f - c * c) * det_inv;
    conic[idx * 6 + 4] = (b * c - a * e) * det_inv;
    conic[idx * 6 + 5] = (a * d - b * b) * det_inv;

	// Scale S by 3 to include up to three std from Gaussian position
	const float m = 2.0;
	const float3 scaled_S = { S[0][0] * m, S[1][1] * m, S[2][2] * m };

 	// Create array for corner computations
    const float n[2] = {-1.0f, 1.0f};
    
    // Initialize mins and maxes with gaussian position
	const float3 position = { means3D[3 * idx], means3D[3 * idx + 1], means3D[3 * idx + 2] };
	means[idx] = position;
    float3 mins = position;
    float3 maxes = position;

	// Compute corners using vector operations
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = 0; k < 2; k++) {
                float3 corner = make_float3(
					position.x + n[i] * R[0].x * scaled_S.x + n[j] * R[1].x * scaled_S.y +  n[k] * R[2].x * scaled_S.z,
					position.y + n[i] * R[0].y * scaled_S.x + n[j] * R[1].y * scaled_S.y +  n[k] * R[2].y * scaled_S.z,
					position.z + n[i] * R[0].z * scaled_S.x + n[j] * R[1].z * scaled_S.y +  n[k] * R[2].z * scaled_S.z
				);
                    
                mins = make_float3(
					min(mins.x, corner.x),
					min(mins.y, corner.y),
					min(mins.z, corner.z)
				);
                maxes = make_float3(
					max(maxes.x, corner.x),
					max(maxes.y, corner.y),
					max(maxes.z, corner.z)
				);
            }
        }
    }

	uint3 start_cell = make_uint3(
		max(0u, static_cast<unsigned int>(floor((mins.x - volume_mins.x) / cell_size))),
		max(0u, static_cast<unsigned int>(floor((mins.y - volume_mins.y) / cell_size))),
		max(0u, static_cast<unsigned int>(floor((mins.z - volume_mins.z) / cell_size)))
    );    
	uint3 end_cell = make_uint3(
		min(num_cells.x, static_cast<unsigned int>(ceil((maxes.x - volume_mins.x) / cell_size))),
		min(num_cells.y, static_cast<unsigned int>(ceil((maxes.y - volume_mins.y) / cell_size))),
		min(num_cells.z, static_cast<unsigned int>(ceil((maxes.z - volume_mins.z) / cell_size)))
    );
    uint3 cell_dims = make_uint3(
		end_cell.x - start_cell.x,
		end_cell.y - start_cell.y,
		end_cell.z - start_cell.z
	);

    // Store results
    cells_touched[idx] = static_cast<int>(cell_dims.x * cell_dims.y * cell_dims.z);
	radii[idx] = 1;
    aabbs[idx * 6] = start_cell.x;
	aabbs[idx * 6 + 1] = start_cell.y;
    aabbs[idx * 6 + 2] = start_cell.z;
    aabbs[idx * 6 + 3] = end_cell.x;
    aabbs[idx * 6 + 4] = end_cell.y;
	aabbs[idx * 6 + 5] = end_cell.z;
	clamped[idx] = (values[idx] < 0.0f) || (values[idx] > 1.0f);
    values_out[idx] = glm::clamp(values[idx], 0.0f, 1.0f);
    volumes[idx] = static_cast<float>(cell_dims.x * cell_dims.y * cell_dims.z);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y * BLOCK_Z)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	const dim3 grid,
	const float3 volume_mins,
	const uint3 num_cells,
	const float cell_size,
	const float3* __restrict__ means,
	const float* __restrict__ values,
	const float* __restrict__ volumes,
	const float* __restrict__ conic,
	float* __restrict__ accumulated_weights,
	uint32_t* __restrict__ n_contrib,
	float* __restrict__ out_cells)
{
	// Identify current tile and associated min/max pixel range.
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
	__shared__ float collected_conic[BLOCK_SIZE * 6];

	// Initialize helper variables
	float accumulated_weight = 0;
	float accumulated_value = 0;
	uint32_t n_contributor = 0;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_means[block.thread_rank()] = means[coll_id];
			collected_volumes[block.thread_rank()] = volumes[coll_id];
			collected_values[block.thread_rank()] = values[coll_id];
			for (int k = 0; k < 6; k++)
                collected_conic[block.thread_rank() * 6 + k] = conic[coll_id * 6 + k];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			n_contributor++;

			float3 d = make_float3(cell_pos.x - collected_means[j].x, cell_pos.y - collected_means[j].y, cell_pos.z - collected_means[j].z);
			float quad_form = (
				d.x * (collected_conic[j * 6] * d.x + collected_conic[j * 6 + 1] * d.y + collected_conic[j * 6 + 2] * d.z) +
				d.y * (collected_conic[j * 6 + 1] * d.x + collected_conic[j * 6 + 3] * d.y + collected_conic[j * 6 + 4] * d.z) +
				d.z * (collected_conic[j * 6 + 2] * d.x + collected_conic[j * 6 + 4] * d.y + collected_conic[j * 6 + 5] * d.z)
			);
			// float normalize_factor = 1.0 / collected_volumes[j];
			float normalize_factor = 1.0;
			float weight = normalize_factor * exp(-0.5 * quad_form);

			if (exp(-0.5 * quad_form) > 1.0f)
				continue;

			accumulated_value += collected_values[j] * weight;
			accumulated_weight += weight;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		// This both gives a dropoff where we have to have a certain weight to set a value
		// and prevents numerical issues of dividing by something close to 0
		if (accumulated_weight > 1e-5) {
			out_cells[cell_id] = accumulated_value / accumulated_weight;
			accumulated_weights[cell_id] = accumulated_weight;
			n_contrib[cell_id] = n_contributor;

		} else {
			out_cells[cell_id] = 0.0;
			accumulated_weights[cell_id] = 0.0;
			n_contrib[cell_id] = n_contributor;
		}
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const float3 volume_mins,
	const uint3 num_cells,
	const float cell_size,
	const float3* means,
	const float* values,
	const float* volumes,
	const float* conic,
	float* accumulated_weights,
	uint32_t* n_contrib,
	float* out_cells)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		grid,
		volume_mins,
		num_cells,
		cell_size,
		means,
		values,
		volumes,
		conic,
		accumulated_weights,
		n_contrib,
		out_cells);
}

void FORWARD::preprocess(int P,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* values,
	bool* clamped,
	const float3 volume_mins,
	const float3 volume_maxes,
	const uint3 num_cells,
	const float cell_size,
	int* radii,
	float3* means,
	float* values_out, float* volumes,
	float* conic,
	uint* aabbs,
	const dim3 grid,
	uint32_t* cells_touched)
{
	preprocessCUDA<NUM_CHANNELS> <<<(P + 255) / 256, 256>>> (
		P,
		means3D,
		scales,
		scale_modifier,
		rotations,
		values,
		clamped,
		volume_mins,
		volume_maxes,
		num_cells,
		cell_size,
		radii,
		means,
		values_out, volumes,
		conic,
		aabbs,
		grid,
		cells_touched
	);
}
