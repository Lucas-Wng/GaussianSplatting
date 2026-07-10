// GPU bitonic depth sort. In-place: each pass compares (gid, gid ^ j) over key/value arrays
// padded to a power of two (padding keyed 0xFFFFFFFF sorts to the end). Ascending key ==
// ascending camera-space z == back-to-front; only splatCount instances are drawn.

// Must match GpuSplat in src/ply_loader.hpp.
struct Splat
{
    float3 position;
    float  opacity;
    float4 color;
    float4 cov_a;
    float4 cov_b;
};

struct SortPush
{
    float4 viewRow2;      // z = dot(viewRow2.xyz, pos) + viewRow2.w
    uint   splatCount;
    uint   paddedCount;
    uint   j;             // bitonic partner stride
    uint   k;             // bitonic subsequence size; ascending when (gid & k) == 0
};

[[vk::binding(0)]] StructuredBuffer<Splat>   splats;
[[vk::binding(1)]] RWStructuredBuffer<uint>  keys;
[[vk::binding(2)]] RWStructuredBuffer<uint>  values;

[[vk::push_constant]] SortPush pc;

// uint whose unsigned order matches the float's signed order.
uint orderableFloat(float f)
{
    uint u = asuint(f);
    return u ^ ((u >> 31) != 0u ? 0xFFFFFFFFu : 0x80000000u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sortInitMain(uint3 tid : SV_DispatchThreadID)
{
    uint gid = tid.x;
    if (gid >= pc.paddedCount)
        return;

    if (gid < pc.splatCount)
    {
        float3 p    = splats[gid].position;
        keys[gid]   = orderableFloat(dot(pc.viewRow2.xyz, p) + pc.viewRow2.w);
        values[gid] = gid;
    }
    else
    {
        keys[gid]   = 0xFFFFFFFFu;
        values[gid] = gid;
    }
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sortBitonicMain(uint3 tid : SV_DispatchThreadID)
{
    uint gid = tid.x;
    if (gid >= pc.paddedCount)
        return;

    uint ixj = gid ^ pc.j;
    if (ixj <= gid)
        return;

    bool ascending = (gid & pc.k) == 0u;
    if ((keys[gid] > keys[ixj]) == ascending)
    {
        uint tk = keys[gid];   keys[gid]   = keys[ixj];   keys[ixj]   = tk;
        uint tv = values[gid]; values[gid] = values[ixj]; values[ixj] = tv;
    }
}
