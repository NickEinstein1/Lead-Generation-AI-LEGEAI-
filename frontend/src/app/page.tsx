"use client";
import Link from "next/link";
import { useEffect, useRef } from "react";

export default function Landing() {
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

    for (let i = 0; i < 100; i++) {
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

  return (
    <div className="relative min-h-screen bg-gradient-to-br from-[#0a0f1e] via-[#0d1425] to-[#1a1f35] overflow-hidden">
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
      <div className="absolute top-1/2 left-1/2 w-64 h-64 bg-green-400/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: "2s" }} />

      {/* Header */}
      <header className="relative z-10 px-6 py-6 border-b border-cyan-500/20 backdrop-blur-sm bg-slate-900/30">
        <div className="mx-auto max-w-7xl flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="absolute inset-0 bg-cyan-400 blur-md opacity-50 rounded-lg"></div>
              <div className="relative h-10 w-10 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center border border-cyan-300/50 shadow-lg shadow-cyan-500/50">
                <span className="font-bold text-white text-xl">L</span>
              </div>
            </div>
            <span className="font-bold text-2xl bg-gradient-to-r from-cyan-300 via-blue-300 to-cyan-400 bg-clip-text text-transparent tracking-wider">
              LEGEAI
            </span>
          </div>
          <nav className="hidden md:flex items-center gap-8 text-sm">
            <Link href="#features" className="text-cyan-200 hover:text-cyan-300 font-medium transition-colors relative group">
              Features
              <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-cyan-400 group-hover:w-full transition-all duration-300"></span>
            </Link>
            <Link href="#how-it-works" className="text-cyan-200 hover:text-cyan-300 font-medium transition-colors relative group">
              How it works
              <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-cyan-400 group-hover:w-full transition-all duration-300"></span>
            </Link>
            <Link href="#pricing" className="text-cyan-200 hover:text-cyan-300 font-medium transition-colors relative group">
              Pricing
              <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-cyan-400 group-hover:w-full transition-all duration-300"></span>
            </Link>
            <Link href="#contact" className="text-cyan-200 hover:text-cyan-300 font-medium transition-colors relative group">
              Contact
              <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-cyan-400 group-hover:w-full transition-all duration-300"></span>
            </Link>
          </nav>
          <div className="flex items-center gap-4 text-sm">
            <Link href="/login" className="text-cyan-200 hover:text-cyan-300 font-medium transition-colors">
              Login
            </Link>
            <Link
              href="/register"
              className="px-6 py-2 rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold hover:from-cyan-400 hover:to-blue-500 transition-all shadow-lg shadow-cyan-500/30 hover:shadow-cyan-500/50"
            >
              Get Started
            </Link>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 flex-1">
        {/* Hero Section */}
        <section className="mx-auto max-w-7xl px-6 py-20 md:py-32 text-center">
          {/* Holographic LEGEAI Logo */}
          <div className="mb-8 flex justify-center">
            <div className="relative">
              <div className="absolute inset-0 bg-cyan-400 blur-2xl opacity-40 rounded-full"></div>
              <h1 className="relative text-7xl md:text-9xl font-black bg-gradient-to-r from-cyan-300 via-blue-400 to-cyan-300 bg-clip-text text-transparent tracking-tight animate-pulse">
                LEGEAI
              </h1>
            </div>
          </div>

          {/* Tagline */}
          <p className="text-2xl md:text-4xl font-light text-cyan-100 mb-4 tracking-wide">
            Revolutionize Your Leads with
          </p>
          <p className="text-3xl md:text-5xl font-bold bg-gradient-to-r from-green-300 via-cyan-300 to-blue-400 bg-clip-text text-transparent mb-12">
            Intelligent Automation
          </p>

          {/* CTA Button */}
          <Link
            href="/register"
            className="inline-block px-12 py-5 text-lg font-bold rounded-xl bg-gradient-to-r from-cyan-500 via-blue-600 to-cyan-500 text-white hover:from-cyan-400 hover:via-blue-500 hover:to-cyan-400 transition-all shadow-2xl shadow-cyan-500/50 hover:shadow-cyan-400/70 hover:scale-105 transform duration-300 border border-cyan-300/30"
          >
            Start Generating Leads Now →
          </Link>

          {/* Trust Indicators */}
          <div className="mt-16 flex flex-wrap justify-center gap-8 text-sm text-cyan-200/80">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span>99.9% Uptime</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span>SOC2 Compliant</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span>API-First</span>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section id="features" className="mx-auto max-w-7xl px-6 py-20">
          <h2 className="text-4xl md:text-5xl font-bold text-center mb-4 bg-gradient-to-r from-cyan-300 to-blue-400 bg-clip-text text-transparent">
            Powered by Advanced AI
          </h2>
          <p className="text-center text-cyan-200/70 mb-16 text-lg">
            Transform your lead generation with cutting-edge technology
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Feature 1 */}
            <div className="group relative p-6 rounded-2xl bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-cyan-500/20 backdrop-blur-sm hover:border-cyan-400/50 transition-all duration-300 hover:shadow-xl hover:shadow-cyan-500/20 hover:-translate-y-1">
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-blue-600/5 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
              <div className="relative">
                <div className="w-14 h-14 mb-4 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-600/20 flex items-center justify-center border border-cyan-400/30 shadow-lg shadow-cyan-500/20">
                  <span className="text-3xl">🎯</span>
                </div>
                <h3 className="text-xl font-bold text-cyan-100 mb-3">AI-Powered Targeting</h3>
                <p className="text-cyan-200/60 text-sm leading-relaxed">
                  Machine learning algorithms identify and score high-quality leads with precision accuracy
                </p>
              </div>
            </div>

            {/* Feature 2 */}
            <div className="group relative p-6 rounded-2xl bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-cyan-500/20 backdrop-blur-sm hover:border-cyan-400/50 transition-all duration-300 hover:shadow-xl hover:shadow-cyan-500/20 hover:-translate-y-1">
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-blue-600/5 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
              <div className="relative">
                <div className="w-14 h-14 mb-4 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-600/20 flex items-center justify-center border border-cyan-400/30 shadow-lg shadow-cyan-500/20">
                  <span className="text-3xl">📊</span>
                </div>
                <h3 className="text-xl font-bold text-cyan-100 mb-3">Real-Time Analytics</h3>
                <p className="text-cyan-200/60 text-sm leading-relaxed">
                  Monitor performance metrics and conversion rates with live dashboards and insights
                </p>
              </div>
            </div>

            {/* Feature 3 */}
            <div className="group relative p-6 rounded-2xl bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-cyan-500/20 backdrop-blur-sm hover:border-cyan-400/50 transition-all duration-300 hover:shadow-xl hover:shadow-cyan-500/20 hover:-translate-y-1">
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-blue-600/5 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
              <div className="relative">
                <div className="w-14 h-14 mb-4 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-600/20 flex items-center justify-center border border-cyan-400/30 shadow-lg shadow-cyan-500/20">
                  <span className="text-3xl">🔗</span>
                </div>
                <h3 className="text-xl font-bold text-cyan-100 mb-3">Seamless Integration</h3>
                <p className="text-cyan-200/60 text-sm leading-relaxed">
                  Connect with your existing CRM, marketing tools, and workflows via powerful APIs
                </p>
              </div>
            </div>

            {/* Feature 4 */}
            <div className="group relative p-6 rounded-2xl bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-cyan-500/20 backdrop-blur-sm hover:border-cyan-400/50 transition-all duration-300 hover:shadow-xl hover:shadow-cyan-500/20 hover:-translate-y-1">
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-blue-600/5 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity"></div>
              <div className="relative">
                <div className="w-14 h-14 mb-4 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-600/20 flex items-center justify-center border border-cyan-400/30 shadow-lg shadow-cyan-500/20">
                  <span className="text-3xl">🚀</span>
                </div>
                <h3 className="text-xl font-bold text-cyan-100 mb-3">Boost Conversion Rates</h3>
                <p className="text-cyan-200/60 text-sm leading-relaxed">
                  Increase ROI by up to 300% with intelligent lead nurturing and automated follow-ups
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* How It Works Section */}
        <section id="how-it-works" className="mx-auto max-w-7xl px-6 py-20">
          <h2 className="text-4xl md:text-5xl font-bold text-center mb-4 bg-gradient-to-r from-cyan-300 to-blue-400 bg-clip-text text-transparent">
            How It Works
          </h2>
          <p className="text-center text-cyan-200/70 mb-16 text-lg">
            Three simple steps to supercharge your lead generation
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Step 1 */}
            <div className="relative text-center">
              <div className="mb-6 flex justify-center">
                <div className="relative">
                  <div className="absolute inset-0 bg-cyan-400 blur-xl opacity-40 rounded-full"></div>
                  <div className="relative w-20 h-20 rounded-full bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center border-2 border-cyan-300/50 shadow-2xl shadow-cyan-500/50">
                    <span className="text-3xl font-bold text-white">1</span>
                  </div>
                </div>
              </div>
              <h3 className="text-2xl font-bold text-cyan-100 mb-3">Ingest</h3>
              <p className="text-cyan-200/60 leading-relaxed">
                Connect your data sources and let our AI ingest leads from multiple channels in real-time
              </p>
            </div>

            {/* Step 2 */}
            <div className="relative text-center">
              <div className="mb-6 flex justify-center">
                <div className="relative">
                  <div className="absolute inset-0 bg-blue-400 blur-xl opacity-40 rounded-full"></div>
                  <div className="relative w-20 h-20 rounded-full bg-gradient-to-br from-blue-500 to-cyan-600 flex items-center justify-center border-2 border-blue-300/50 shadow-2xl shadow-blue-500/50">
                    <span className="text-3xl font-bold text-white">2</span>
                  </div>
                </div>
              </div>
              <h3 className="text-2xl font-bold text-cyan-100 mb-3">Score</h3>
              <p className="text-cyan-200/60 leading-relaxed">
                Advanced ML models analyze and score each lead based on conversion probability and value
              </p>
            </div>

            {/* Step 3 */}
            <div className="relative text-center">
              <div className="mb-6 flex justify-center">
                <div className="relative">
                  <div className="absolute inset-0 bg-green-400 blur-xl opacity-40 rounded-full"></div>
                  <div className="relative w-20 h-20 rounded-full bg-gradient-to-br from-green-500 to-cyan-600 flex items-center justify-center border-2 border-green-300/50 shadow-2xl shadow-green-500/50">
                    <span className="text-3xl font-bold text-white">3</span>
                  </div>
                </div>
              </div>
              <h3 className="text-2xl font-bold text-cyan-100 mb-3">Convert</h3>
              <p className="text-cyan-200/60 leading-relaxed">
                Automated workflows nurture high-value leads and route them to your sales team at the perfect moment
              </p>
            </div>
          </div>
        </section>

        {/* Pricing Section */}
        <section id="pricing" className="mx-auto max-w-7xl px-6 py-20">
          <h2 className="text-4xl md:text-5xl font-bold text-center mb-4 bg-gradient-to-r from-cyan-300 to-blue-400 bg-clip-text text-transparent">
            Simple, Transparent Pricing
          </h2>
          <p className="text-center text-cyan-200/70 mb-16 text-lg">
            Choose the plan that fits your business needs
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Starter Plan */}
            <div className="relative p-8 rounded-2xl bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-cyan-500/20 backdrop-blur-sm hover:border-cyan-400/50 transition-all duration-300 hover:shadow-xl hover:shadow-cyan-500/20">
              <h3 className="text-2xl font-bold text-cyan-100 mb-2">Starter</h3>
              <div className="mb-6">
                <span className="text-5xl font-black text-white">$0</span>
                <span className="text-cyan-200/60 ml-2">/month</span>
              </div>
              <ul className="space-y-3 mb-8">
                <li className="flex items-start gap-2 text-cyan-200/70">
                  <span className="text-green-400 mt-1">✓</span>
                  <span>Up to 10 leads/month</span>
                </li>
                <li className="flex items-start gap-2 text-cyan-200/70">
                  <span className="text-green-400 mt-1">✓</span>
                  <span>Basic AI scoring</span>
                </li>
                <li className="flex items-start gap-2 text-cyan-200/70">
                  <span className="text-green-400 mt-1">✓</span>
                  <span>Email support</span>
                </li>
                <li className="flex items-start gap-2 text-cyan-200/70">
                  <span className="text-green-400 mt-1">✓</span>
                  <span>Dashboard analytics</span>
                </li>
              </ul>
              <Link
                href="/register"
                className="block w-full py-3 text-center rounded-lg bg-gradient-to-r from-cyan-500/20 to-blue-600/20 text-cyan-200 font-semibold hover:from-cyan-500/30 hover:to-blue-600/30 transition-all border border-cyan-500/30"
              >
                Get Started Free
              </Link>
            </div>

            {/* Growth Plan */}
            <div className="relative p-8 rounded-2xl bg-gradient-to-br from-cyan-900/30 to-blue-900/30 border-2 border-cyan-400/50 backdrop-blur-sm shadow-2xl shadow-cyan-500/30 transform scale-105">
              <div className="absolute -top-4 left-1/2 -translate-x-1/2 px-4 py-1 rounded-full bg-gradient-to-r from-cyan-500 to-blue-600 text-white text-sm font-bold">
                POPULAR
              </div>
              <h3 className="text-2xl font-bold text-cyan-100 mb-2">Growth</h3>
              <div className="mb-6">
                <span className="text-5xl font-black text-white">$99</span>
                <span className="text-cyan-200/60 ml-2">/month</span>
              </div>
              <ul className="space-y-3 mb-8">
                <li className="flex items-start gap-2 text-cyan-200/70">
                  <span className="text-green-400 mt-1">✓</span>
                  <span>Up to 5,000 leads/month</span>
                </li>
                <li className="flex items-start gap-2 text-cyan-200/70">
                  <span className="text-green-400 mt-1">✓</span>
                  <span>Advanced AI scoring</span>
                </li>
                <li className="flex items-start gap-2 text-cyan-200/70">
                  <span className="text-green-400 mt-1">✓</span>
                  <span>Priority support</span>
                </li>
                <li className="flex items-start gap-2 text-cyan-200/70">
                  <span className="text-green-400 mt-1">✓</span>
                  <span>API access</span>
                </li>
                <li className="flex items-start gap-2 text-cyan-200/70">
                  <span className="text-green-400 mt-1">✓</span>
                  <span>Custom integrations</span>
                </li>
              </ul>
              <Link
                href="/register"
                className="block w-full py-3 text-center rounded-lg bg-gradient-to-r from-cyan-500 to-blue-600 text-white font-semibold hover:from-cyan-400 hover:to-blue-500 transition-all shadow-lg shadow-cyan-500/50"
              >
                Start Free Trial
              </Link>
            </div>

            {/* Enterprise Plan */}
            <div className="relative p-8 rounded-2xl bg-gradient-to-br from-slate-900/50 to-slate-800/30 border border-cyan-500/20 backdrop-blur-sm hover:border-cyan-400/50 transition-all duration-300 hover:shadow-xl hover:shadow-cyan-500/20">
              <h3 className="text-2xl font-bold text-cyan-100 mb-2">Enterprise</h3>
              <div className="mb-6">
                <span className="text-5xl font-black text-white">Custom</span>
              </div>
              <ul className="space-y-3 mb-8">
                <li className="flex items-start gap-2 text-cyan-200/70">
                  <span className="text-green-400 mt-1">✓</span>
                  <span>Unlimited leads</span>
                </li>
                <li className="flex items-start gap-2 text-cyan-200/70">
                  <span className="text-green-400 mt-1">✓</span>
                  <span>Custom AI models</span>
                </li>
                <li className="flex items-start gap-2 text-cyan-200/70">
                  <span className="text-green-400 mt-1">✓</span>
                  <span>24/7 dedicated support</span>
                </li>
                <li className="flex items-start gap-2 text-cyan-200/70">
                  <span className="text-green-400 mt-1">✓</span>
                  <span>White-label options</span>
                </li>
                <li className="flex items-start gap-2 text-cyan-200/70">
                  <span className="text-green-400 mt-1">✓</span>
                  <span>SLA guarantees</span>
                </li>
              </ul>
              <Link
                href="#contact"
                className="block w-full py-3 text-center rounded-lg bg-gradient-to-r from-cyan-500/20 to-blue-600/20 text-cyan-200 font-semibold hover:from-cyan-500/30 hover:to-blue-600/30 transition-all border border-cyan-500/30"
              >
                Contact Sales
              </Link>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer id="contact" className="relative z-10 border-t border-cyan-500/20 backdrop-blur-sm bg-slate-900/30 mt-20">
        <div className="mx-auto max-w-7xl px-6 py-12">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-8">
            {/* Logo and Description */}
            <div className="md:col-span-1">
              <div className="flex items-center gap-3 mb-4">
                <div className="relative">
                  <div className="absolute inset-0 bg-cyan-400 blur-md opacity-50 rounded-lg"></div>
                  <div className="relative h-10 w-10 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center border border-cyan-300/50 shadow-lg shadow-cyan-500/50">
                    <span className="font-bold text-white text-xl">L</span>
                  </div>
                </div>
                <span className="font-bold text-xl bg-gradient-to-r from-cyan-300 via-blue-300 to-cyan-400 bg-clip-text text-transparent tracking-wider">
                  LEGEAI
                </span>
              </div>
              <p className="text-cyan-200/60 text-sm">
                Revolutionizing lead generation with AI-powered automation
              </p>
            </div>

            {/* Navigation Links */}
            <div>
              <h4 className="font-bold text-cyan-100 mb-4">Product</h4>
              <ul className="space-y-2 text-sm">
                <li>
                  <Link href="#features" className="text-cyan-200/60 hover:text-cyan-300 transition-colors">
                    Features
                  </Link>
                </li>
                <li>
                  <Link href="#pricing" className="text-cyan-200/60 hover:text-cyan-300 transition-colors">
                    Pricing
                  </Link>
                </li>
                <li>
                  <Link href="#how-it-works" className="text-cyan-200/60 hover:text-cyan-300 transition-colors">
                    How it works
                  </Link>
                </li>
              </ul>
            </div>

            {/* Company Links */}
            <div>
              <h4 className="font-bold text-cyan-100 mb-4">Company</h4>
              <ul className="space-y-2 text-sm">
                <li>
                  <Link href="#" className="text-cyan-200/60 hover:text-cyan-300 transition-colors">
                    About
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-cyan-200/60 hover:text-cyan-300 transition-colors">
                    Blog
                  </Link>
                </li>
                <li>
                  <Link href="#" className="text-cyan-200/60 hover:text-cyan-300 transition-colors">
                    Careers
                  </Link>
                </li>
              </ul>
            </div>

            {/* Social Links */}
            <div>
              <h4 className="font-bold text-cyan-100 mb-4">Connect</h4>
              <div className="flex gap-4">
                <a href="#" className="w-10 h-10 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center text-cyan-300 hover:text-cyan-200 transition-all">
                  <span className="text-xl">𝕏</span>
                </a>
                <a href="#" className="w-10 h-10 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center text-cyan-300 hover:text-cyan-200 transition-all">
                  <span className="text-xl">in</span>
                </a>
                <a href="#" className="w-10 h-10 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/30 flex items-center justify-center text-cyan-300 hover:text-cyan-200 transition-all">
                  <span className="text-xl">📧</span>
                </a>
              </div>
            </div>
          </div>

          {/* Copyright */}
          <div className="pt-8 border-t border-cyan-500/10 text-center text-sm text-cyan-200/40">
            <p>&copy; 2025LEGEAI. All rights reserved. Built with infinite love by the LEAGAI team.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

