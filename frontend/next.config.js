/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // standalone only for Docker (Koyeb/Render), not needed on Vercel
  output: process.env.VERCEL ? undefined : 'standalone',
}

module.exports = nextConfig
