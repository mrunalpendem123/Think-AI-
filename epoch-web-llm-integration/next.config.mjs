/** @type {import('next').NextConfig} */
const isExtension = process.env.EXTENSION_BUILD === 'true';

/** @type {import('next').NextConfig} */
const nextConfig = {
  output: isExtension ? 'export' : undefined,
  images: {
    unoptimized: isExtension,
    remotePatterns: [
      {
        protocol: "https",
        hostname: "**",
      },
      {
        protocol: "http",
        hostname: "**",
      },
    ],
  },
  // Headers are ignored in export mode
  async headers() {
    if (isExtension) return [];
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'Cross-Origin-Embedder-Policy',
            value: 'require-corp',
          },
          {
            key: 'Cross-Origin-Opener-Policy',
            value: 'same-origin',
          },
        ],
      },
    ];
  },
  typescript: {
    // !! WARN !!
    // Dangerously allow production builds to successfully complete even if
    // your project has type errors.
    // !! WARN !!
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
