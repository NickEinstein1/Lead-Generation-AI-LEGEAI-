"use client";
import { useState, useEffect, useRef } from "react";
import { login, setSession } from "@/lib/auth";
import { useRouter } from "next/navigation";
import Link from "next/link";

export default function LoginPage() {
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);

    // Particle system for futuristic background
    const particles: Array<{
      x: number;
      y: number;
      vx: number;
      vy: number;
      size: number;
      opacity: number;
    }> = [];

    for (let i = 0; i < 80; i++) {
      particles.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        size: Math.random() * 2 + 1,
        opacity: Math.random() * 0.5 + 0.2,
      });
    }

    const animate = () => {
      ctx.fillStyle = "rgba(10, 15, 30, 0.1)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      particles.forEach((p, i) => {
        p.x += p.vx;
        p.y += p.vy;

        if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
        if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

        // Draw particle
        ctx.fillStyle = `rgba(0, 255, 255, ${p.opacity})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fill();

        // Draw connections
        particles.slice(i + 1).forEach((p2) => {
          const dx = p.x - p2.x;
          const dy = p.y - p2.y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 150) {
            ctx.strokeStyle = `rgba(0, 255, 255, ${0.2 * (1 - dist / 150)})`;
            ctx.lineWidth = 0.5;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.stroke();
          }
        });
      });

      requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener("resize", resizeCanvas);
    };
  }, []);

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError("");
    setLoading(true);
    const fd = new FormData(e.currentTarget);
    const username = String(fd.get("username") || "");
    const password = String(fd.get("password") || "");
    try {
      const res = await login({ username, password });
      setSession({ sessionId: res.session_id, token: res.token, userId: res.user_id, role: res.role });
      router.push("/dashboard");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="relative min-h-screen flex items-center justify-center bg-gradient-to-br from-[#0a0f1e] via-[#0d1425] to-[#1a1f35] overflow-hidden">
      {/* Animated Background Canvas */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 w-full h-full"
        style={{ opacity: 0.6 }}
      />

      {/* Grid Overlay */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(0,255,255,0.03)_1px,transparent_1px),linear-gradient(to_bottom,rgba(0,255,255,0.03)_1px,transparent_1px)] bg-[size:40px_40px]" />

      {/* Glowing Orbs */}
      <div className="absolute top-20 right-20 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-20 left-20 w-80 h-80 bg-blue-600/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: "1s" }} />

      {/* Back to Home Link */}
      <Link
        href="/"
        className="absolute top-6 left-6 z-20 flex items-center gap-2 text-cyan-200 hover:text-cyan-300 font-medium transition-colors group"
      >
        <span className="text-xl group-hover:-translate-x-1 transition-transform">←</span>
        <span>Back to home</span>
      </Link>

      {/* Login Form */}
      <div className="relative z-10 w-full max-w-md px-6">
        {/* Logo */}
        <div className="flex justify-center mb-8">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="absolute inset-0 bg-cyan-400 blur-md opacity-50 rounded-lg"></div>
              <div className="relative h-12 w-12 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center border border-cyan-300/50 shadow-lg shadow-cyan-500/50">
                <span className="font-bold text-white text-2xl">L</span>
              </div>
            </div>
            <span className="font-bold text-3xl bg-gradient-to-r from-cyan-300 via-blue-300 to-cyan-400 bg-clip-text text-transparent tracking-wider">
              LEGEAI
            </span>
          </div>
        </div>

        {/* Form Container */}
        <form
          onSubmit={onSubmit}
          className="relative p-8 rounded-2xl bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-cyan-500/20 backdrop-blur-sm shadow-2xl shadow-cyan-500/10"
        >
          <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-blue-600/5 rounded-2xl"></div>

          <div className="relative space-y-6">
            {/* Title */}
            <div className="text-center mb-8">
              <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-300 to-blue-400 bg-clip-text text-transparent mb-2">
                Welcome Back
              </h1>
              <p className="text-cyan-200/60 text-sm">Sign in to access your dashboard</p>
            </div>

            {/* Username Input */}
            <div>
              <label className="block text-sm font-medium text-cyan-200 mb-2">
                Email or Username
              </label>
              <input
                name="username"
                placeholder="Enter your email or username"
                className="w-full bg-slate-900/50 border border-cyan-500/30 px-4 py-3 rounded-lg text-cyan-100 placeholder-cyan-200/40 focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:border-transparent transition-all backdrop-blur-sm"
              />
            </div>

            {/* Password Input */}
            <div>
              <label className="block text-sm font-medium text-cyan-200 mb-2">
                Password
              </label>
              <input
                name="password"
                type="password"
                placeholder="Enter your password"
                className="w-full bg-slate-900/50 border border-cyan-500/30 px-4 py-3 rounded-lg text-cyan-100 placeholder-cyan-200/40 focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:border-transparent transition-all backdrop-blur-sm"
              />
            </div>

            {/* Error Message */}
            {error && (
              <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 backdrop-blur-sm">
                <p className="text-sm text-red-300">{error}</p>
              </div>
            )}

            {/* Submit Button */}
            <button
              disabled={loading}
              className="w-full py-3 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold hover:from-cyan-400 hover:to-blue-500 transition-all shadow-lg shadow-cyan-500/30 hover:shadow-cyan-500/50 disabled:opacity-50 disabled:cursor-not-allowed hover:scale-[1.02] transform duration-200"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                  Signing in...
                </span>
              ) : (
                "Sign in"
              )}
            </button>

            {/* Register Link */}
            <div className="text-center text-sm text-cyan-200/60">
              Don't have an account?{" "}
              <Link
                href="/register"
                className="text-cyan-300 hover:text-cyan-200 font-semibold transition-colors underline decoration-cyan-400/30 hover:decoration-cyan-300"
              >
                Create one
              </Link>
            </div>
          </div>
        </form>

        {/* Additional Info */}
        <p className="text-center text-cyan-200/40 text-xs mt-6">
          Secured with enterprise-grade encryption
        </p>
      </div>
    </div>
  );
}

