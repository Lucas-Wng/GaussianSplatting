// Graphics-pipeline EWA splatting. Each Gaussian is one instanced quad (4-vert triangle strip,
// no vertex buffer): the vertex shader projects the 3D covariance to a 2D screen-space
// covariance sized to ~3 sigma, the fragment shader evaluates the Gaussian falloff and outputs
// premultiplied color for back-to-front alpha blending.

struct CameraUBO
{
    float4x4 view;        // world -> camera (glm::lookAt, -Z forward)
    float4x4 proj;        // camera -> clip
    float2   viewport;    // framebuffer size in pixels
    float2   focal;       // fx, fy in pixels
};

// Matches GpuSplat in src/ply_loader.hpp (std430, 64 bytes).
struct Splat
{
    float3 position;
    float  opacity;
    float4 color;         // rgb used
    float4 cov_a;         // Sigma[0][0], Sigma[0][1], Sigma[0][2], Sigma[1][1]
    float4 cov_b;         // Sigma[1][2], Sigma[2][2], pad, pad
};

[[vk::binding(0)]] ConstantBuffer<CameraUBO>    ubo;
[[vk::binding(1)]] StructuredBuffer<Splat>      splats;
[[vk::binding(2)]] StructuredBuffer<uint>       sortedIndices;

struct VSOutput
{
    float4 pos : SV_Position;
    float3 color;
    float  opacity;
    float3 conic;         // inverse 2D covariance: (Sigma^-1.xx, .xy, .yy)
    float2 d;             // pixel offset from splat center
};

// Project the 3D covariance (world space) to a 2D screen-space covariance (pixels).
float3 computeCov2D(float3 viewPos, float3x3 Vrk)
{
    // Clamp the position to a reasonable field of view so the Jacobian stays sane
    // near the screen edges (avoids splats exploding at grazing angles).
    float limx = 1.3f * (ubo.viewport.x / (2.0f * ubo.focal.x));
    float limy = 1.3f * (ubo.viewport.y / (2.0f * ubo.focal.y));
    float tx = clamp(viewPos.x / viewPos.z, -limx, limx) * viewPos.z;
    float ty = clamp(viewPos.y / viewPos.z, -limy, limy) * viewPos.z;
    float tz = viewPos.z;

    // Jacobian of the perspective projection at the splat center. The overall sign
    // relative to the -Z-forward view convention cancels out of J * Sigma * J^T.
    float3x3 J = float3x3(
        ubo.focal.x / tz, 0.0f,             -(ubo.focal.x * tx) / (tz * tz),
        0.0f,             ubo.focal.y / tz, -(ubo.focal.y * ty) / (tz * tz),
        0.0f,             0.0f,              0.0f);

    float3x3 W    = (float3x3) ubo.view;                 // world -> camera rotation
    float3x3 cov  = mul(J, mul(mul(W, mul(Vrk, transpose(W))), transpose(J)));

    // Low-pass filter: dilate by ~0.3px so tiny splats stay at least one pixel wide.
    return float3(cov[0][0] + 0.3f, cov[0][1], cov[1][1]);
}

[shader("vertex")]
VSOutput vertMain(uint vid : SV_VertexID, uint iid : SV_InstanceID)
{
    VSOutput o;
    o.color   = float3(0.0f);
    o.opacity = 0.0f;
    o.conic   = float3(0.0f);
    o.d       = float2(0.0f);

    Splat s = splats[sortedIndices[iid]];

    float4 viewPos = mul(ubo.view, float4(s.position, 1.0f));
    float4 clip    = mul(ubo.proj, viewPos);

    // Cull anything behind the camera.
    if (clip.w <= 0.0f)
    {
        o.pos = float4(0.0f, 0.0f, 2.0f, 1.0f);        // outside NDC -> clipped
        return o;
    }

    float3x3 Vrk = float3x3(
        s.cov_a.x, s.cov_a.y, s.cov_a.z,
        s.cov_a.y, s.cov_a.w, s.cov_b.x,
        s.cov_a.z, s.cov_b.x, s.cov_b.y);

    float3 cov2d = computeCov2D(viewPos.xyz, Vrk);      // (a, b, c) => [[a,b],[b,c]]
    float  det   = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det <= 0.0f)
    {
        o.pos = float4(0.0f, 0.0f, 2.0f, 1.0f);
        return o;
    }

    // Inverse covariance (conic) for the fragment falloff.
    o.conic = float3(cov2d.z, -cov2d.y, cov2d.x) / det;

    // Screen-space radius from the larger eigenvalue (~3 sigma).
    float mid    = 0.5f * (cov2d.x + cov2d.z);
    float lambda = mid + sqrt(max(0.1f, mid * mid - det));
    float radius = ceil(3.0f * sqrt(lambda));           // pixels

    // Quad corner in {-1,+1}^2 from the vertex id (triangle strip).
    float2 corner = float2((vid & 1u) != 0u ? 1.0f : -1.0f,
                           (vid & 2u) != 0u ? 1.0f : -1.0f);
    o.d = corner * radius;                               // pixel offset from center

    float2 centerNdc = clip.xy / clip.w;
    float2 offsetNdc = o.d * 2.0f / ubo.viewport;        // pixels -> NDC
    o.pos     = float4((centerNdc + offsetNdc) * clip.w, clip.z, clip.w);
    o.color   = s.color.rgb;
    o.opacity = s.opacity;
    return o;
}

[shader("fragment")]
float4 fragMain(VSOutput i) : SV_Target
{
    // Gaussian power = -0.5 * d^T Sigma^-1 d.
    float power = -0.5f * (i.conic.x * i.d.x * i.d.x + i.conic.z * i.d.y * i.d.y)
                  - i.conic.y * i.d.x * i.d.y;
    if (power > 0.0f)
        discard;

    float alpha = min(0.99f, i.opacity * exp(power));
    if (alpha < 1.0f / 255.0f)
        discard;

    return float4(i.color * alpha, alpha);              // premultiplied
}
