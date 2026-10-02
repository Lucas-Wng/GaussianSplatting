// LSD radix sort building blocks (8-bit digit, stable) plus a multi-level exclusive scan.
// Sorts (key, value) pairs ascending: 64-bit key packed as uint2 (x=lo, y=hi), uint value.
// One pass per digit ping-pongs A<->B; the host drives passes 0..N and the scan recursion.
//
// Global offsets come from a per-block histogram laid out bin-major
// (blockHist[bin * numBlocks + block]); a plain exclusive prefix sum over that array yields,
// for each (block, bin), the global destination base of that block's elements of that digit.

struct RadixPush
{
    uint count;        // active elements (radix) / scan length
    uint digit;        // 0..7: which 8-bit digit of the 64-bit key
    uint numBlocks;    // ceil(count / 256)
};

[[vk::push_constant]] RadixPush pc;

[[vk::binding(0)]] RWStructuredBuffer<uint2> keysIn;
[[vk::binding(1)]] RWStructuredBuffer<uint>  valsIn;
[[vk::binding(2)]] RWStructuredBuffer<uint2> keysOut;
[[vk::binding(3)]] RWStructuredBuffer<uint>  valsOut;
[[vk::binding(4)]] RWStructuredBuffer<uint>  blockHist;

[[vk::binding(5)]] RWStructuredBuffer<uint>  scanData;
[[vk::binding(6)]] RWStructuredBuffer<uint>  scanSums;

uint keyDigit(uint2 key, uint digit)
{
    uint word  = (digit < 4u) ? key.x : key.y;
    uint shift = (digit & 3u) * 8u;
    return (word >> shift) & 0xFFu;
}

// --- Radix pass, step 1: per-block digit histogram (bin-major) ---
groupshared uint sHist[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void radixHistogramMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID, uint3 dtid : SV_DispatchThreadID)
{
    uint t = gtid.x;
    sHist[t] = 0u;
    GroupMemoryBarrierWithGroupSync();

    uint i = dtid.x;
    if (i < pc.count)
        InterlockedAdd(sHist[keyDigit(keysIn[i], pc.digit)], 1u);
    GroupMemoryBarrierWithGroupSync();

    blockHist[t * pc.numBlocks + gid.x] = sHist[t];
}

// --- Radix pass, step 3: stable scatter using the scanned (exclusive) blockHist ---
//
// Two-level counting sort over the 256-element block, instead of one thread doing a single
// 256-iteration serial scan: split the block into kScatterGroups equal slices, each with its
// own serial local-rank pass (phase A, parallel across group leaders -- 32 steps instead of
// 256), then exclusive-scan each digit's per-group counts across groups (phase B, one thread
// per digit value, fully parallel -- 8 steps). Stability is preserved because both phases
// process elements/groups in increasing index order, same as the original single-thread scan.
groupshared uint sDigit[256];
groupshared uint sLocalRank[256];         // rank within this thread's own group, thread order
groupshared uint sGroupCount[8 * 256];    // [group][digit]; count, then exclusive group-base

static const uint kScatterGroups    = 8;
static const uint kScatterGroupSize = 256u / kScatterGroups;        // 32

[shader("compute")]
[numthreads(256, 1, 1)]
void radixScatterMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID, uint3 dtid : SV_DispatchThreadID)
{
    uint t = gtid.x;
    uint i = dtid.x;
    bool active = i < pc.count;

    uint2 key = active ? keysIn[i] : uint2(0u, 0u);
    uint  val = active ? valsIn[i] : 0u;
    sDigit[t] = active ? keyDigit(key, pc.digit) : 0xFFFFFFFFu;

    for (uint k = 0u; k < kScatterGroups; ++k)
        sGroupCount[k * 256u + t] = 0u;
    GroupMemoryBarrierWithGroupSync();

    // Phase A: each group leader serially scans its own 32-wide slice.
    uint myGroup = t / kScatterGroupSize;
    if (t % kScatterGroupSize == 0u)
    {
        uint base = myGroup * kScatterGroupSize;
        for (uint e = 0u; e < kScatterGroupSize; ++e)
        {
            uint d = sDigit[base + e];
            if (d != 0xFFFFFFFFu)
            {
                sLocalRank[base + e]     = sGroupCount[myGroup * 256u + d];
                sGroupCount[myGroup * 256u + d] += 1u;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Phase B: one thread per digit value (t == digit) exclusive-scans that digit's counts
    // across the kScatterGroups groups, turning them into per-group base offsets in place.
    {
        uint digit = t;
        uint run   = 0u;
        for (uint g = 0u; g < kScatterGroups; ++g)
        {
            uint c                      = sGroupCount[g * 256u + digit];
            sGroupCount[g * 256u + digit] = run;
            run += c;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (active)
    {
        uint d    = sDigit[t];
        uint dest = blockHist[d * pc.numBlocks + gid.x] + sGroupCount[myGroup * 256u + d] + sLocalRank[t];
        keysOut[dest] = key;
        valsOut[dest] = val;
    }
}

// --- Exclusive scan (Hillis-Steele within a block; host recurses over block sums) ---
groupshared uint sScan[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void scanBlockMain(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID, uint3 dtid : SV_DispatchThreadID)
{
    uint t   = gtid.x;
    uint i   = dtid.x;
    uint val = (i < pc.count) ? scanData[i] : 0u;
    sScan[t] = val;
    GroupMemoryBarrierWithGroupSync();

    for (uint off = 1u; off < 256u; off <<= 1)
    {
        uint add = (t >= off) ? sScan[t - off] : 0u;
        GroupMemoryBarrierWithGroupSync();
        sScan[t] += add;
        GroupMemoryBarrierWithGroupSync();
    }

    uint incl = sScan[t];
    if (i < pc.count)
        scanData[i] = incl - val;          // inclusive -> exclusive
    if (t == 255u)
        scanSums[gid.x] = incl;            // block total
}

[shader("compute")]
[numthreads(256, 1, 1)]
void scanAddMain(uint3 gid : SV_GroupID, uint3 dtid : SV_DispatchThreadID)
{
    uint i = dtid.x;
    if (i < pc.count)
        scanData[i] += scanSums[gid.x];
}

// --- Depth-key init: seeds keyBufferA/valBufferA from splat positions (own set/layout). ---

// Must match GpuSplat in src/ply_loader.hpp (only `position` is read here, but the struct's
// total size must still match so indexing into the shared splat buffer stays aligned).
struct Splat
{
    float3 position;
    float  opacity;
    float4 color;
    float4 cov_a;
    float4 cov_b;
    float4 shRest[15];
};

struct RadixInitPush
{
    float4 viewRow2;      // z = dot(viewRow2.xyz, pos) + viewRow2.w
    uint   splatCount;
    uint   paddedCount;
};

[[vk::push_constant]] RadixInitPush initPc;

[[vk::binding(7)]] StructuredBuffer<Splat>   initSplats;
[[vk::binding(8)]] RWStructuredBuffer<uint2> initKeysOut;
[[vk::binding(9)]] RWStructuredBuffer<uint>  initValsOut;

// uint whose unsigned order matches the float's signed order.
uint orderableFloatInit(float f)
{
    uint u = asuint(f);
    return u ^ ((u >> 31) != 0u ? 0xFFFFFFFFu : 0x80000000u);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void radixDepthInitMain(uint3 tid : SV_DispatchThreadID)
{
    uint gid = tid.x;
    if (gid >= initPc.paddedCount)
        return;

    if (gid < initPc.splatCount)
    {
        float3 p        = initSplats[gid].position;
        uint   lo       = orderableFloatInit(dot(initPc.viewRow2.xyz, p) + initPc.viewRow2.w);
        initKeysOut[gid] = uint2(lo, 0u);
        initValsOut[gid] = gid;
    }
    else
    {
        initKeysOut[gid] = uint2(0xFFFFFFFFu, 0xFFFFFFFFu);
        initValsOut[gid] = gid;
    }
}
