// ============================================================================
//  lbm3d_vortex_final.cu  —  3D Karman Vortex Simulation & Visualizer
//  [корекции: (1) входният слой x<2 изключен от визуализацията — Dirichlet
//   артефакти; (2) --pipeline отново измерва и печата резултат; (3) върнат
//   самоописващият се конфигурационен ред при старт]
// ============================================================================

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GL/glu.h>

#include <cuda_runtime.h>

#define NX 300
#define NY 200
#define NZ 100

#define CYL_X (NX / 5.0f)
#define CYL_Y (NY / 2.0f)
#define CYL_Z (NZ / 2.0f)
#define CYL_R 12.5f

#define RE   180.0f
#define U_IN 0.08f

#define SPONGE_W   30
#define SPONGE_AMP 20.0f

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t e = (call);                                                \
        if (e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(e));                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

__host__ __device__ inline size_t idx3d(int x, int y, int z) {
    return (size_t)z * (NX * NY) + (size_t)y * NX + (size_t)x;
}

__constant__ int   d_cx[19];
__constant__ int   d_cy[19];
__constant__ int   d_cz[19];
__constant__ float d_w[19];
__constant__ int   d_OPP[19];

const int h_cx[19] = {0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0};
const int h_cy[19] = {0, 0, 0, 1,-1, 0, 0, 1, 1,-1,-1, 0, 0, 0, 0, 1,-1, 1,-1};
const int h_cz[19] = {0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1, 1,-1,-1, 1, 1,-1,-1};
const float h_w[19] = {
    1.0f/3.0f,
    1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f, 1.0f/18.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f,
    1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f, 1.0f/36.0f
};
const int h_OPP[19] = {0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11,
                       18, 17, 16, 15};

__device__ inline float feq3d(int i, float rho, float ux, float uy, float uz) {
    float cu = (float)d_cx[i] * ux + (float)d_cy[i] * uy + (float)d_cz[i] * uz;
    float u2 = ux * ux + uy * uy + uz * uz;
    return d_w[i] * rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * u2);
}

__global__ void initKernel3D(float* d_F, unsigned char* d_mask) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;

    size_t n = idx3d(x, y, z);
    size_t total = (size_t)NX * NY * NZ;

    float dx = (float)x - CYL_X;
    float dy = (float)y - CYL_Y;
    d_mask[n] = ((dx * dx + dy * dy) <= CYL_R * CYL_R) ? 1 : 0;

    float ux = d_mask[n] ? 0.0f : U_IN;
    float uy = d_mask[n] ? 0.0f
             : (0.02f * U_IN * sinf(2.0f * 3.14159265f * z / (float)NZ));
    float uz = 0.0f;

    for (int i = 0; i < 19; i++) {
        d_F[i * total + n] = feq3d(i, 1.0f, ux, uy, uz);
    }
}

__global__ void lbmStepKernel3D(
    const float* __restrict__ d_F,
    float* __restrict__ d_F2,
    float* __restrict__ d_ux,
    float* __restrict__ d_uy,
    float* __restrict__ d_uz,
    const unsigned char* __restrict__ d_mask,
    float omega,
    float nu)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;

    size_t n = idx3d(x, y, z);
    size_t total = (size_t)NX * NY * NZ;

    if (d_mask[n]) {
        for (int i = 0; i < 19; i++) {
            d_F2[i * total + n] = d_F[d_OPP[i] * total + n];
        }
        d_ux[n] = 0.0f; d_uy[n] = 0.0f; d_uz[n] = 0.0f;
        return;
    }

    float f[19];
    for (int i = 0; i < 19; i++) {
        int xs = (x - d_cx[i] + NX) % NX;
        int ys = (y - d_cy[i] + NY) % NY;
        int zs = (z - d_cz[i] + NZ) % NZ;
        f[i] = d_F[i * total + idx3d(xs, ys, zs)];
    }

    float rho = 0.0f, ux = 0.0f, uy = 0.0f, uz = 0.0f;
    for (int i = 0; i < 19; i++) {
        rho += f[i];
        ux  += f[i] * d_cx[i];
        uy  += f[i] * d_cy[i];
        uz  += f[i] * d_cz[i];
    }
    if (rho > 0.0f) { ux /= rho; uy /= rho; uz /= rho; }

    if (x == 0) { ux = U_IN; uy = 0.0f; uz = 0.0f; rho = 1.0f; }

    d_ux[n] = ux; d_uy[n] = uy; d_uz[n] = uz;

    float omega_loc = omega;
    if (x >= NX - SPONGE_W) {
        float s = (float)(x - (NX - SPONGE_W)) / (float)(SPONGE_W - 1);
        float nu_loc = nu * (1.0f + SPONGE_AMP * s * s);
        omega_loc = 1.0f / (3.0f * nu_loc + 0.5f);
    }

    for (int i = 0; i < 19; i++) {
        float feq = feq3d(i, rho, ux, uy, uz);
        d_F2[i * total + n] = f[i] + omega_loc * (feq - f[i]);
    }
}

__global__ void generateVortexCloudKernel(
    const float* d_ux, const float* d_uy, const float* d_uz,
    const unsigned char* mask,
    float3* d_points,
    uchar3* d_colors,
    int* d_pointCount,
    float threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;
    if (x >= NX - SPONGE_W) return;   // sponge зоната не се рисува
    if (x < 2) return;                // [КОРЕКЦИЯ] входен слой: Dirichlet артефакти

    size_t n = idx3d(x, y, z);

    if (mask[n]) {
        if (z % 2 == 0 && y % 2 == 0) {
            int writeIdx = atomicAdd(d_pointCount, 1);
            d_points[writeIdx] = make_float3((float)x, (float)y, (float)z);
            d_colors[writeIdx] = make_uchar3(180, 180, 180);
        }
        return;
    }

    int xm = (x - 1 + NX) % NX;
    int xp = (x + 1) % NX;
    int ym = (y - 1 + NY) % NY;
    int yp = (y + 1) % NY;

    float dvx_dy = (d_ux[idx3d(x, yp, z)] - d_ux[idx3d(x, ym, z)]) * 0.5f;
    float dvy_dx = (d_uy[idx3d(xp, y, z)] - d_uy[idx3d(xm, y, z)]) * 0.5f;
    float wz = dvy_dx - dvx_dy;

    if (fabsf(wz) > threshold) {
        int writeIdx = atomicAdd(d_pointCount, 1);
        d_points[writeIdx] = make_float3((float)x, (float)y, (float)z);

        float t = wz / (threshold * 3.0f);
        t = fmaxf(-1.0f, fminf(1.0f, t));

        if (t > 0.0f) {
            d_colors[writeIdx] = make_uchar3(
                255, (unsigned char)(255 * (1.0f - t)), 30);
        } else {
            d_colors[writeIdx] = make_uchar3(
                30, (unsigned char)(255 * (1.0f + t)), 255);
        }
    }
}

__global__ void probeKernel(const float* d_uy_field, float* d_probe, size_t n) {
    *d_probe = d_uy_field[n];
}

__global__ void forceKernel(const float* __restrict__ d_F,
                            const unsigned char* __restrict__ d_mask,
                            float* d_force)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= NX || y >= NY || z >= NZ) return;

    size_t n = idx3d(x, y, z);
    if (d_mask[n]) return;
    size_t total = (size_t)NX * NY * NZ;

    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    for (int i = 1; i < 19; i++) {
        int xn = (x + d_cx[i] + NX) % NX;
        int yn = (y + d_cy[i] + NY) % NY;
        int zn = (z + d_cz[i] + NZ) % NZ;
        if (d_mask[idx3d(xn, yn, zn)]) {
            float fsum = d_F[i        * total + n]
                       + d_F[d_OPP[i] * total + n];
            fx += fsum * (float)d_cx[i];
            fy += fsum * (float)d_cy[i];
            fz += fsum * (float)d_cz[i];
        }
    }
    if (fx != 0.0f || fy != 0.0f || fz != 0.0f) {
        atomicAdd(&d_force[0], -fx);
        atomicAdd(&d_force[1], -fy);
        atomicAdd(&d_force[2], -fz);
    }
}

// ══════════════ Benchmark Modes ═════════════════════════════════════════════
static void runBenchmark(int nsteps, float*& d_F, float*& d_F2,
                         float* d_ux, float* d_uy, float* d_uz,
                         unsigned char* d_mask, float omega, float nu,
                         dim3 blocks, dim3 threads)
{
    size_t Ncells = (size_t)NX * NY * NZ;

    for (int t = 0; t < 200; ++t) {
        lbmStepKernel3D<<<blocks, threads>>>(d_F, d_F2, d_ux, d_uy, d_uz, d_mask, omega, nu);
        std::swap(d_F, d_F2);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));

    for (int t = 0; t < nsteps; ++t) {
        lbmStepKernel3D<<<blocks, threads>>>(d_F, d_F2, d_ux, d_uy, d_uz, d_mask, omega, nu);
        std::swap(d_F, d_F2);
    }

    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));

    double mlups = ((double)Ncells * nsteps) / ((double)ms * 1000.0);
    double bw    = mlups * 1e6 * 152.0 / 1e9;

    printf("=====================================================\n");
    printf("  MLUPS BENCHMARK  (grid %dx%dx%d, %d steps)\n", NX, NY, NZ, nsteps);
    printf("  Total time      : %.2f ms\n", ms);
    printf("  Time per step   : %.4f ms\n", ms / nsteps);
    printf("  Performance     : %.1f MLUPS\n", mlups);
    printf("  Eff. bandwidth  : %.1f GB/s\n", bw);
    printf("=====================================================\n");

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
}

static void runMeasurement(int nsteps, float*& d_F, float*& d_F2,
                            float* d_ux, float* d_uy, float* d_uz,
                            unsigned char* d_mask, float omega, float nu,
                            dim3 blocks, dim3 threads)
{
    int px = (int)(CYL_X + 4.0f * 2.0f * CYL_R);
    int py = (int)(CYL_Y + CYL_R);
    int pz = NZ / 2;
    if (px >= NX - SPONGE_W) px = NX - SPONGE_W - 5;
    size_t pn = idx3d(px, py, pz);

    float *d_probe, *d_force;
    CUDA_CHECK(cudaMalloc(&d_probe, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_force, 3 * sizeof(float)));

    FILE* fp = fopen("probe_signal.csv", "w");
    FILE* ff = fopen("forces.csv", "w");
    fprintf(fp, "step,uy\n");
    fprintf(ff, "step,Fx,Fy,Fz\n");

    const int TRANSIENT = 10000;
    const int SAMPLE_EVERY = 10;

    for (int t = 0; t < TRANSIENT + nsteps; ++t) {
        lbmStepKernel3D<<<blocks, threads>>>(d_F, d_F2, d_ux, d_uy, d_uz, d_mask, omega, nu);
        std::swap(d_F, d_F2);

        if (t >= TRANSIENT && (t % SAMPLE_EVERY) == 0) {
            float h_probe, h_force[3];
            probeKernel<<<1, 1>>>(d_uy, d_probe, pn);
            CUDA_CHECK(cudaMemset(d_force, 0, 3 * sizeof(float)));
            forceKernel<<<blocks, threads>>>(d_F, d_mask, d_force);
            CUDA_CHECK(cudaMemcpy(&h_probe, d_probe, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_force, d_force, 3 * sizeof(float), cudaMemcpyDeviceToHost));
            fprintf(fp, "%d,%.8f\n", t - TRANSIENT, h_probe);
            fprintf(ff, "%d,%.8f,%.8f,%.8f\n", t - TRANSIENT, h_force[0], h_force[1], h_force[2]);
        }
    }

    fclose(fp); fclose(ff);
    cudaFree(d_probe); cudaFree(d_force);
}

static void runPipeline(int nframes, float*& d_F, float*& d_F2,
                        float* d_ux, float* d_uy, float* d_uz,
                        unsigned char* d_mask, float omega, float nu,
                        float vortexThreshold, dim3 blocks, dim3 threads)
{
    size_t N = (size_t)NX * NY * NZ;

    float3 *d_points;  uchar3 *d_colors;  int *d_pointCount;
    CUDA_CHECK(cudaMalloc(&d_points, N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_colors, N * sizeof(uchar3)));
    CUDA_CHECK(cudaMalloc(&d_pointCount, sizeof(int)));

    std::vector<float3> h_points(N);
    std::vector<uchar3> h_colors(N);

    for (int t = 0; t < 10000; ++t) {
        lbmStepKernel3D<<<blocks, threads>>>(d_F, d_F2, d_ux, d_uy, d_uz, d_mask, omega, nu);
        std::swap(d_F, d_F2);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // [ВЪЗСТАНОВЕНО] измерване и отчет — без тях режимът не връща резултат
    const char* names[2] = {
        "A: host-mediated (D2H copy)",
        "B: zero-copy     (no D2H) "
    };
    double frame_ms[2] = {0, 0};
    double avg_pts [2] = {0, 0};

    for (int variant = 0; variant < 2; ++variant) {
        bool doCopy = (variant == 0);
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        long long ptsSum = 0;

        CUDA_CHECK(cudaEventRecord(e0));
        for (int fr = 0; fr < nframes; ++fr) {
            for (int sub = 0; sub < 50; ++sub) {
                lbmStepKernel3D<<<blocks, threads>>>(d_F, d_F2, d_ux, d_uy, d_uz, d_mask, omega, nu);
                std::swap(d_F, d_F2);
            }
            CUDA_CHECK(cudaMemset(d_pointCount, 0, sizeof(int)));
            generateVortexCloudKernel<<<blocks, threads>>>(
                d_ux, d_uy, d_uz, d_mask, d_points, d_colors, d_pointCount, vortexThreshold);

            int h_pointCount = 0;
            CUDA_CHECK(cudaMemcpy(&h_pointCount, d_pointCount, sizeof(int), cudaMemcpyDeviceToHost));
            ptsSum += h_pointCount;

            if (doCopy && h_pointCount > 0) {
                CUDA_CHECK(cudaMemcpy(h_points.data(), d_points, h_pointCount * sizeof(float3), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_colors.data(), d_colors, h_pointCount * sizeof(uchar3), cudaMemcpyDeviceToHost));
            }
        }
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        frame_ms[variant] = ms / nframes;
        avg_pts [variant] = (double)ptsSum / nframes;
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }

    double mbFrame = avg_pts[0] * 15.0 / 1e6;
    double dltaMs  = frame_ms[0] - frame_ms[1];

    printf("=====================================================\n");
    printf("  VISUALIZATION-PATH EXPERIMENT (%d frames x 50 steps, grid %dx%dx%d)\n",
           nframes, NX, NY, NZ);
    for (int v = 0; v < 2; ++v) {
        printf("  %s : %8.3f ms/frame  (%6.1f FPS eq.)  avg points %.0f\n",
               names[v], frame_ms[v], 1000.0 / frame_ms[v], avg_pts[v]);
    }
    printf("  D2H payload      : %.2f MB/frame\n", mbFrame);
    printf("  Transfer cost    : %.3f ms/frame  =  %.1f%% of frame\n",
           dltaMs, dltaMs / frame_ms[0] * 100.0);
    printf("=====================================================\n");

    cudaFree(d_points); cudaFree(d_colors); cudaFree(d_pointCount);
}

// ════════════ Billboarded Vector Font & Wireframe Box ══════════════════════
static void drawCharStroke(char c, float x, float y, float scale) {
    glBegin(GL_LINES);
    switch(c) {
        case '-':
            glVertex2f(x, y + scale); glVertex2f(x + scale, y + scale);
            break;
        case '+':
            glVertex2f(x, y + scale); glVertex2f(x + scale, y + scale);
            glVertex2f(x + scale * 0.5f, y + scale * 0.5f); glVertex2f(x + scale * 0.5f, y + scale * 1.5f);
            break;
        case '0':
            glVertex2f(x, y);                 glVertex2f(x + scale, y);
            glVertex2f(x + scale, y);         glVertex2f(x + scale, y + 2 * scale);
            glVertex2f(x + scale, y + 2 * scale); glVertex2f(x, y + 2 * scale);
            glVertex2f(x, y + 2 * scale);     glVertex2f(x, y);
            break;
        case '1':
            glVertex2f(x + scale, y); glVertex2f(x + scale, y + 2 * scale);
            break;
        case '2':
            glVertex2f(x, y + 2 * scale);     glVertex2f(x + scale, y + 2 * scale);
            glVertex2f(x + scale, y + 2 * scale); glVertex2f(x + scale, y + scale);
            glVertex2f(x + scale, y + scale); glVertex2f(x, y + scale);
            glVertex2f(x, y + scale);         glVertex2f(x, y);
            glVertex2f(x, y);                 glVertex2f(x + scale, y);
            break;
        case '3':
            glVertex2f(x, y + 2 * scale);     glVertex2f(x + scale, y + 2 * scale);
            glVertex2f(x + scale, y + 2 * scale); glVertex2f(x + scale, y);
            glVertex2f(x + scale, y);         glVertex2f(x, y);
            glVertex2f(x, y + scale);         glVertex2f(x + scale, y + scale);
            break;
        case '4':
            glVertex2f(x, y + 2 * scale);     glVertex2f(x, y + scale);
            glVertex2f(x, y + scale);         glVertex2f(x + scale, y + scale);
            glVertex2f(x + scale, y + 2 * scale); glVertex2f(x + scale, y);
            break;
        case '5':
            glVertex2f(x + scale, y + 2 * scale); glVertex2f(x, y + 2 * scale);
            glVertex2f(x, y + 2 * scale);     glVertex2f(x, y + scale);
            glVertex2f(x, y + scale);         glVertex2f(x + scale, y + scale);
            glVertex2f(x + scale, y + scale); glVertex2f(x + scale, y);
            glVertex2f(x + scale, y);         glVertex2f(x, y);
            break;
        case '6':
            glVertex2f(x + scale, y + 2 * scale); glVertex2f(x, y + 2 * scale);
            glVertex2f(x, y + 2 * scale);     glVertex2f(x, y);
            glVertex2f(x, y);                 glVertex2f(x + scale, y);
            glVertex2f(x + scale, y);         glVertex2f(x + scale, y + scale);
            glVertex2f(x + scale, y + scale); glVertex2f(x, y + scale);
            break;
        case '7':
            glVertex2f(x, y + 2 * scale);     glVertex2f(x + scale, y + 2 * scale);
            glVertex2f(x + scale, y + 2 * scale); glVertex2f(x + scale, y);
            break;
        case '8':
            glVertex2f(x, y);                 glVertex2f(x + scale, y);
            glVertex2f(x + scale, y);         glVertex2f(x + scale, y + 2 * scale);
            glVertex2f(x + scale, y + 2 * scale); glVertex2f(x, y + 2 * scale);
            glVertex2f(x, y + 2 * scale);     glVertex2f(x, y);
            glVertex2f(x, y + scale);         glVertex2f(x + scale, y + scale);
            break;
        case '9':
            glVertex2f(x + scale, y);         glVertex2f(x + scale, y + 2 * scale);
            glVertex2f(x + scale, y + 2 * scale); glVertex2f(x, y + 2 * scale);
            glVertex2f(x, y + 2 * scale);     glVertex2f(x, y + scale);
            glVertex2f(x, y + scale);         glVertex2f(x + scale, y + scale);
            break;
    }
    glEnd();
}

static void drawBillboardText(const char* str, float worldX, float worldY, float worldZ, float cameraYaw, float cameraPitch, float scale) {
    glPushMatrix();
    glTranslatef(worldX, worldY, worldZ);
    glRotatef(-cameraYaw, 0.0f, 1.0f, 0.0f);
    glRotatef(-cameraPitch, 1.0f, 0.0f, 0.0f);

    float offset = 0.0f;
    for (int i = 0; str[i] != '\0'; i++) {
        drawCharStroke(str[i], offset, 0.0f, scale);
        offset += scale * 1.6f;
    }

    glPopMatrix();
}

static void drawWireBox(float maxX, float maxY, float maxZ, float cameraYaw, float cameraPitch) {
    float minX = 0.0f, minY = 0.0f, minZ = 0.0f;

    // 1. Draw outer wireframe box
    glLineWidth(1.5f);
    glColor3f(0.3f, 0.3f, 0.4f);
    glBegin(GL_LINES);
    // Bottom
    glVertex3f(minX, minY, minZ); glVertex3f(maxX, minY, minZ);
    glVertex3f(maxX, minY, minZ); glVertex3f(maxX, maxY, minZ);
    glVertex3f(maxX, maxY, minZ); glVertex3f(minX, maxY, minZ);
    glVertex3f(minX, maxY, minZ); glVertex3f(minX, minY, minZ);
    // Top
    glVertex3f(minX, minY, maxZ); glVertex3f(maxX, minY, maxZ);
    glVertex3f(maxX, minY, maxZ); glVertex3f(maxX, maxY, maxZ);
    glVertex3f(maxX, maxY, maxZ); glVertex3f(minX, maxY, maxZ);
    glVertex3f(minX, maxY, maxZ); glVertex3f(minX, minY, maxZ);
    // Pillars
    glVertex3f(minX, minY, minZ); glVertex3f(minX, minY, maxZ);
    glVertex3f(maxX, minY, minZ); glVertex3f(maxX, minY, maxZ);
    glVertex3f(maxX, maxY, minZ); glVertex3f(maxX, maxY, maxZ);
    glVertex3f(minX, maxY, minZ); glVertex3f(minX, maxY, maxZ);
    glEnd();

    // 2. Draw Ticks along bottom front edges
    glLineWidth(2.0f);
    glColor3f(0.6f, 0.6f, 0.7f);
    glBegin(GL_LINES);
    for (float x = 0; x <= maxX; x += 50.0f) {
        glVertex3f(x, minY, minZ); glVertex3f(x, minY - 3.0f, minZ);
    }
    for (float y = 0; y <= maxY; y += 50.0f) {
        glVertex3f(minX, y, minZ); glVertex3f(minX - 3.0f, y, minZ);
    }
    for (float z = 0; z <= maxZ; z += 25.0f) {
        glVertex3f(minX, minY, z); glVertex3f(minX - 3.0f, minY, z);
    }
    glEnd();

    // 3. Render Billboarded Labels facing camera
    glColor3f(0.85f, 0.85f, 0.95f);
    float fontScale = 2.0f;
    char labelBuf[16];

    // X axis labels
    for (int x = 0; x <= (int)maxX; x += 50) {
        snprintf(labelBuf, sizeof(labelBuf), "%d", x);
        drawBillboardText(labelBuf, (float)x, minY - 10.0f, minZ, cameraYaw, cameraPitch, fontScale);
    }

    // Y axis labels
    for (int y = 0; y <= (int)maxY; y += 50) {
        snprintf(labelBuf, sizeof(labelBuf), "%d", y);
        drawBillboardText(labelBuf, minX - 18.0f, (float)y, minZ, cameraYaw, cameraPitch, fontScale);
    }

    // Z axis labels
    for (int z = 0; z <= (int)maxZ; z += 25) {
        snprintf(labelBuf, sizeof(labelBuf), "%d", z);
        drawBillboardText(labelBuf, minX - 18.0f, minY, (float)z, cameraYaw, cameraPitch, fontScale);
    }
}

static void drawCoordinateAxes() {
    float axisLen = 40.0f;

    glLineWidth(3.0f);
    glBegin(GL_LINES);
    // X axis - Red
    glColor3f(1.0f, 0.2f, 0.2f);
    glVertex3f(0.0f, 0.0f, 0.0f); glVertex3f(axisLen, 0.0f, 0.0f);

    // Y axis - Green
    glColor3f(0.2f, 1.0f, 0.2f);
    glVertex3f(0.0f, 0.0f, 0.0f); glVertex3f(0.0f, axisLen, 0.0f);

    // Z axis - Blue
    glColor3f(0.2f, 0.4f, 1.0f);
    glVertex3f(0.0f, 0.0f, 0.0f); glVertex3f(0.0f, 0.0f, axisLen);
    glEnd();

    // Axis indicator stroke letters X, Y, Z
    glLineWidth(2.0f);
    glBegin(GL_LINES);
    // 'X' label
    glColor3f(1.0f, 0.2f, 0.2f);
    float px = axisLen + 5.0f;
    glVertex3f(px - 2, -2, 0); glVertex3f(px + 2, 2, 0);
    glVertex3f(px - 2, 2, 0);  glVertex3f(px + 2, -2, 0);

    // 'Y' label
    glColor3f(0.2f, 1.0f, 0.2f);
    float py = axisLen + 5.0f;
    glVertex3f(-2, py + 2, 0); glVertex3f(0, py, 0);
    glVertex3f(2, py + 2, 0);  glVertex3f(0, py, 0);
    glVertex3f(0, py, 0);      glVertex3f(0, py - 2, 0);

    // 'Z' label
    glColor3f(0.2f, 0.4f, 1.0f);
    float pz = axisLen + 5.0f;
    glVertex3f(-2, 2, pz); glVertex3f(2, 2, pz);
    glVertex3f(2, 2, pz);  glVertex3f(-2, -2, pz);
    glVertex3f(-2, -2, pz); glVertex3f(2, -2, pz);
    glEnd();
}

// ═════════════════════════════════ MAIN ═════════════════════════════════════
int main(int argc, char** argv) {
    bool benchMode = false, measureMode = false, pipelineMode = false;
    int  nsteps = 0;
    if (argc >= 3 && strcmp(argv[1], "--bench") == 0)
        { benchMode = true;    nsteps = atoi(argv[2]); }
    if (argc >= 3 && strcmp(argv[1], "--measure") == 0)
        { measureMode = true;  nsteps = atoi(argv[2]); }
    if (argc >= 3 && strcmp(argv[1], "--pipeline") == 0)
        { pipelineMode = true; nsteps = atoi(argv[2]); }

    CUDA_CHECK(cudaMemcpyToSymbol(d_cx,  h_cx,  19 * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_cy,  h_cy,  19 * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_cz,  h_cz,  19 * sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_w,   h_w,   19 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_OPP, h_OPP, 19 * sizeof(int)));

    size_t N = (size_t)NX * NY * NZ;
    size_t fBytes      = 19 * N * sizeof(float);
    size_t fieldBytes  = N * sizeof(float);
    size_t maskBytes   = N * sizeof(unsigned char);

    float *d_F, *d_F2, *d_ux, *d_uy, *d_uz;
    unsigned char *d_mask;
    CUDA_CHECK(cudaMalloc(&d_F,   fBytes));
    CUDA_CHECK(cudaMalloc(&d_F2,  fBytes));
    CUDA_CHECK(cudaMalloc(&d_ux,  fieldBytes));
    CUDA_CHECK(cudaMalloc(&d_uy,  fieldBytes));
    CUDA_CHECK(cudaMalloc(&d_uz,  fieldBytes));
    CUDA_CHECK(cudaMalloc(&d_mask, maskBytes));

    dim3 threads(32, 4, 4);
    dim3 blocks((NX + threads.x - 1) / threads.x,
                (NY + threads.y - 1) / threads.y,
                (NZ + threads.z - 1) / threads.z);
    printf("threads(%d,%d,%d) blocks(%d,%d,%d) grid %dx%dx%d\n",
           threads.x, threads.y, threads.z,
           blocks.x, blocks.y, blocks.z, NX, NY, NZ);   // [самоописание]

    initKernel3D<<<blocks, threads>>>(d_F, d_mask);
    CUDA_CHECK(cudaDeviceSynchronize());

    float nu    = U_IN * (2.0f * CYL_R) / RE;
    float tau   = 3.0f * nu + 0.5f;
    float omega = 1.0f / tau;
    float vortexThreshold = 0.008f;

    if (benchMode || measureMode || pipelineMode) {
        if (benchMode)
            runBenchmark(nsteps, d_F, d_F2, d_ux, d_uy, d_uz, d_mask, omega, nu, blocks, threads);
        else if (measureMode)
            runMeasurement(nsteps, d_F, d_F2, d_ux, d_uy, d_uz, d_mask, omega, nu, blocks, threads);
        else
            runPipeline(nsteps, d_F, d_F2, d_ux, d_uy, d_uz, d_mask, omega, nu, vortexThreshold, blocks, threads);

        cudaFree(d_F); cudaFree(d_F2);
        cudaFree(d_ux); cudaFree(d_uy); cudaFree(d_uz); cudaFree(d_mask);
        return 0;
    }

    // ════════════════ Interactive OpenGL Window Mode ════════════════════════
    if (!glfwInit()) return -1;
    GLFWwindow* window = glfwCreateWindow(1400, 800, "3D Karman Vortex Street - CUDA V100", NULL, NULL);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);

    glewExperimental = GL_TRUE;
    glewInit();

    glEnable(GL_DEPTH_TEST);
    glPointSize(3.0f);
    glClearColor(0.05f, 0.05f, 0.08f, 1.0f);

    float3 *d_points;
    uchar3 *d_colors;
    int    *d_pointCount;
    CUDA_CHECK(cudaMalloc(&d_points, N * sizeof(float3)));
    CUDA_CHECK(cudaMalloc(&d_colors, N * sizeof(uchar3)));
    CUDA_CHECK(cudaMalloc(&d_pointCount, sizeof(int)));

    std::vector<float3> h_points(N);
    std::vector<uchar3> h_colors(N);

    GLuint vbo_points, vbo_colors;
    glGenBuffers(1, &vbo_points);
    glGenBuffers(1, &vbo_colors);

    float cameraPitch = 30.0f;
    float cameraYaw   = 25.0f;
    int   frameCnt    = 0;
    double tPrev      = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        for (int sub = 0; sub < 50; ++sub) {
            lbmStepKernel3D<<<blocks, threads>>>(d_F, d_F2, d_ux, d_uy, d_uz, d_mask, omega, nu);
            std::swap(d_F, d_F2);
        }

        CUDA_CHECK(cudaMemset(d_pointCount, 0, sizeof(int)));
        generateVortexCloudKernel<<<blocks, threads>>>(
            d_ux, d_uy, d_uz, d_mask, d_points, d_colors, d_pointCount, vortexThreshold);

        int h_pointCount = 0;
        CUDA_CHECK(cudaMemcpy(&h_pointCount, d_pointCount, sizeof(int), cudaMemcpyDeviceToHost));

        if (h_pointCount > 0) {
            CUDA_CHECK(cudaMemcpy(h_points.data(), d_points, h_pointCount * sizeof(float3), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_colors.data(), d_colors, h_pointCount * sizeof(uchar3), cudaMemcpyDeviceToHost));

            glBindBuffer(GL_ARRAY_BUFFER, vbo_points);
            glBufferData(GL_ARRAY_BUFFER, h_pointCount * sizeof(float3), h_points.data(), GL_DYNAMIC_DRAW);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
            glBufferData(GL_ARRAY_BUFFER, h_pointCount * sizeof(uchar3), h_colors.data(), GL_DYNAMIC_DRAW);
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45.0, 1400.0 / 800.0, 1.0, 2000.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glTranslatef(0.0f, 0.0f, -420.0f);                       // Camera Zoom
        glRotatef(cameraPitch, 1.0f, 0.0f, 0.0f);                // Camera Pitch
        glRotatef(cameraYaw,   0.0f, 1.0f, 0.0f);                // Camera Yaw
        cameraYaw += 0.2f;

        // Shift origin so center (150, 100, 50) is the rotation pivot
        glTranslatef(-NX / 2.0f, -NY / 2.0f, -NZ / 2.0f);

        // Render Wireframe Box with screen-facing billboard labels
        drawWireBox(NX, NY, NZ, cameraYaw, cameraPitch);
        drawCoordinateAxes();

        if (h_pointCount > 0) {
            glEnableClientState(GL_VERTEX_ARRAY);
            glEnableClientState(GL_COLOR_ARRAY);

            glBindBuffer(GL_ARRAY_BUFFER, vbo_points);
            glVertexPointer(3, GL_FLOAT, 0, NULL);

            glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
            glColorPointer(3, GL_UNSIGNED_BYTE, 0, NULL);

            glDrawArrays(GL_POINTS, 0, h_pointCount);

            glDisableClientState(GL_COLOR_ARRAY);
            glDisableClientState(GL_VERTEX_ARRAY);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }

        glfwSwapBuffers(window);

        if (++frameCnt == 30) {
            double tNow = glfwGetTime();
            double fps  = 30.0 / (tNow - tPrev);
            double mlups = fps * 50.0 * (double)N / 1e6;
            char title[256];
            snprintf(title, sizeof(title),
                "3D Karman | %.1f FPS | %.0f MLUPS | %d points | Domain: [%d,%d,%d]",
                fps, mlups, h_pointCount, NX, NY, NZ);
            glfwSetWindowTitle(window, title);
            frameCnt = 0;
            tPrev = tNow;
        }
    }

    glDeleteBuffers(1, &vbo_points);
    glDeleteBuffers(1, &vbo_colors);

    cudaFree(d_F); cudaFree(d_F2);
    cudaFree(d_ux); cudaFree(d_uy); cudaFree(d_uz);
    cudaFree(d_mask); cudaFree(d_points); cudaFree(d_colors);
    cudaFree(d_pointCount);

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
