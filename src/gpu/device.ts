/** WebGPU device acquisition and adapter description. */

export interface GpuInfo {
  vendor: string;
  architecture: string;
  device: string;
  description: string;
  /** best-effort human name */
  name: string;
  maxWorkgroupsPerDimension: number;
  timestampQuery: boolean;
}

export interface Gpu {
  adapter: GPUAdapter;
  device: GPUDevice;
  info: GpuInfo;
}

let cached: Promise<Gpu | null> | null = null;

export function hasWebGpu(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator;
}

export function getGpu(onLost?: (reason: string) => void): Promise<Gpu | null> {
  if (cached) return cached;
  cached = (async () => {
    if (!hasWebGpu()) return null;
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' }).catch(() => null);
    if (!adapter) return null;
    const features: GPUFeatureName[] = [];
    if (adapter.features.has('timestamp-query')) features.push('timestamp-query');
    const device = await adapter.requestDevice({
      requiredFeatures: features,
      requiredLimits: {
        maxComputeWorkgroupsPerDimension: adapter.limits.maxComputeWorkgroupsPerDimension,
        maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 256 * 1024 * 1024),
      },
    }).catch(() => null);
    if (!device) return null;
    const ai = (adapter as any).info ?? {};
    const info: GpuInfo = {
      vendor: ai.vendor ?? '',
      architecture: ai.architecture ?? '',
      device: ai.device ?? '',
      description: ai.description ?? '',
      name: prettyName(ai),
      maxWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
      timestampQuery: features.includes('timestamp-query'),
    };
    device.lost.then((l) => {
      cached = null;
      onLost?.(`${l.reason}: ${l.message}`);
    });
    return { adapter, device, info };
  })();
  return cached;
}

function prettyName(ai: any): string {
  const desc: string = ai.description || '';
  if (desc) return desc.replace(/\s*\(.*?\)\s*$/, '');
  const parts = [ai.vendor, ai.architecture, ai.device].filter(Boolean);
  return parts.length ? parts.join(' ') : 'unknown GPU';
}

export function resetGpuCache(): void {
  cached = null;
}
