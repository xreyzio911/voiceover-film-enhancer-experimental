/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    proxyClientMaxBodySize: "2gb",
  },
  turbopack: {
    root: __dirname,
  },
};

module.exports = nextConfig;
