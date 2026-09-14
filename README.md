# Gaussian Splatting Viewer

A real-time **3D Gaussian Splatting** viewer written in Vulkan (C++20). Loads a pretrained `.ply` capture and renders it with free-fly navigation.

![Cactus capture rendered by the viewer](assets/cactusGS.webp)

## Features

- Parses binary 3DGS `.ply` files (position, SH degree-0 color, opacity, scale, rotation)
  into GPU-ready splat data, with covariance and activations precomputed on the CPU.
- Renders each Gaussian as an instanced quad via the graphics pipeline, with EWA covariance
  projection and premultiplied-alpha back-to-front blending.
- Depth sorting is done either on the CPU (`std::sort`) or on the GPU via a compute
  bitonic sort — toggle at runtime.
- Free-fly (FPS-style) camera: WASD movement, mouse look, Space/Shift for up/down.
- Vulkan 1.3 dynamic rendering, `synchronization2`, RAII (`vk::raii`), shaders written in
  Slang. Targets macOS/MoltenVK, kept portable to Windows/Linux.

## Requirements

- Vulkan SDK 1.4.335+ (provides `slangc`)
- `glfw`
- `glm`

## Build & run

```bash
./run.sh          # configure + build (incremental) + run (loads bundled models/cactus.ply)
./run.sh --clean  # wipe build/ first, then build + run
```

To view a different capture, pass a path:

```bash
build/GaussianSplatting/GaussianSplatting /path/to/scene.ply
```

The loader handles uncompressed float32 3DGS `.ply` files. The app expects to run with
`build/GaussianSplatting/` as its working directory (it reads shaders and models relative
to cwd) — `run.sh` takes care of this for you.

## Controls

| Key | Action |
|---|---|
| `W` `A` `S` `D` | Move |
| `Space` / `Shift` | Move up / down |
| Mouse | Look around |
| `F` | Toggle up-axis correction (COLMAP Y-down → Y-up) |
| `G` | Toggle sort mode (CPU `std::sort` ↔ GPU bitonic) |
| `Esc` | Quit |

## Project layout

```
src/main.cpp          # application: Vulkan setup, splat pipeline, CPU/GPU sort, input loop
src/ply_loader.{hpp,cpp} # binary 3DGS .ply -> GpuSplat[] + AABB
src/camera.hpp         # free-fly / FPS camera
shaders/shader.slang   # EWA splat vertex+fragment shader (compiled to slang.spv)
shaders/compute.slang  # GPU bitonic depth sort (compiled to compute.spv)
assets/cactus.ply      # pretrained capture (copied to run dir as models/cactus.ply)
CMakeLists.txt         # build + shader compilation + asset copy
run.sh                 # configure, build, run
```

## Roadmap

- **Phase 0 — baseline (done):** Vulkan instance/device/swapchain, dynamic rendering, Slang
  shader pipeline, GLFW window.
- **Phase 1 — minimal viewer (done):** `.ply` loading, instanced-quad splat rendering, CPU
  depth sort, fly camera.
- **Phase 2 — compute acceleration (in progress):** move projection, sorting, and
  tile-based rasterization into compute shaders.
  - GPU bitonic depth sort (done), toggle with `G`.
  - Next: compute covariance-projection/culling preprocess, a radix sort, then a full
    tile-binning compute rasterizer.

