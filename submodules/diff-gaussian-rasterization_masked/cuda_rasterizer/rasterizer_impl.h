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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

// CudaRasterizer 命名空间
namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	} // 从内存块中获取对齐的内存指针

	struct GeometryState
	{
		size_t scan_size;  			// 扫描大小
		float* depths;     			// 深度值数组
		char* scanning_space; 		// 扫描空间
		bool* clamped;        		// 标记是否裁剪数组
		int* internal_radii;  		// 内部半径数组
		float2* means2D;     		// 2D 均值数组
		float* cov3D;         		// 3D协方差数组
		float4* conic_opacity; 		// 锥形透明度数组
		float* rgb;           		// RGB值数组
		uint32_t* point_offsets; 	// 点偏移数组
		uint32_t* tiles_touched; 	// 触摸的瓦片数组

		static GeometryState fromChunk(char*& chunk, size_t P); // 静态方法，从内存块创建GeometryState实例
	}; // 几何状态结构体

	struct ImageState
	{
		uint2* ranges;
		uint32_t* n_contrib;
		float* accum_alpha;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	template<typename T> 
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}
};