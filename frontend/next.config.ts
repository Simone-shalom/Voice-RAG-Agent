import path from "node:path";
import type { NextConfig } from "next";

const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const nextConfig: NextConfig = {
  // Pins the workspace root to this directory — without it, Next.js walks up
  // looking for the nearest lockfile and can latch onto an unrelated one
  // outside the repo, which then makes it trace (and sometimes fail to
  // access) files far outside this project.
  outputFileTracingRoot: path.join(__dirname),
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
