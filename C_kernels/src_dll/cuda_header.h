#pragma once

#ifdef _WIN32
  #ifdef MYCUDA_EXPORTS
    #define MYCUDA_API __declspec(dllexport)
  #else
    #define MYCUDA_API __declspec(dllimport)
  #endif
#else
  #define MYCUDA_API
#endif

extern "C" {
    MYCUDA_API void launch_context_kernel(
        const float* a,
        const float* fy,
        const float* fz,
        float* context,
        int D
    );

    MYCUDA_API void launch_denominator_kernel(
        const float* a,
        const float* fy,
        const float* fz,
        float* denom,
        int D
    );
    MYCUDA_API void launch_transform_tensor_kernel(
        const float* U,
        const float* a,
        float* new_a,
        int D
    );
    MYCUDA_API void launch_ema_update_kernel(
        const float* x,
        const float* y,
        const float* z,
        float* a,
        float ema_lambda,
        int D
    );
    MYCUDA_API void launch_entropy_kernel(
        const float* activations,
        float* entropy_out,
        int B, int D
    );

    MYCUDA_API void launch_mi_kernel(
        const float* actX,
        const float* actY,
        float* mi_out,
        int B, int D
    );

    MYCUDA_API void launch_mean_estimation_kernel(
    const float* fx,
    const float* fy,
    const float* fz,
    float* out,
    int D,
    int N
    );
    MYCUDA_API void launch_bilinear_kernel(
    const float* a0,
    const float* a1,
    const float* fy,
    const float* fz,
    float* denom,
    float* num,
    int D
    );
}
