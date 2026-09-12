/** @type {import('next').NextConfig} */
const isDev = process.env.NODE_ENV === "development";

const nextConfig = {
  reactStrictMode: true,
  // In development the Python API is served by uvicorn on :5328.
  // In production Vercel rewrites /api/py/* to the api/index.py serverless function
  // (see vercel.json), so the destination is a no-op passthrough.
  async rewrites() {
    return [
      {
        source: "/api/py/:path*",
        destination: isDev
          ? "http://127.0.0.1:5328/api/py/:path*"
          : "/api/py/:path*",
      },
    ];
  },
};

export default nextConfig;
