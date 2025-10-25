"use client";
import Link from "next/link";
import dynamic from "next/dynamic";

const Hero3D = dynamic(() => import("../components/Hero3D"), { ssr: false });

export default function Landing() {
  return (
    <div className="relative min-h-screen flex flex-col bg-gradient-to-b from-[#f8fafc] via-[#f3f6fb] to-[#eef2f7]">
      {/* Background: soft grid + radial glows + subtle 3D */}
      <div className="absolute inset-0 -z-10 overflow-hidden">
        <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,rgba(0,0,0,0.03)_1px,transparent_1px),linear-gradient(to_bottom,rgba(0,0,0,0.03)_1px,transparent_1px)] bg-[size:24px_24px]" />
        <div className="pointer-events-none absolute -top-32 right-[-10%] h-[480px] w-[480px] opacity-50 blur-3xl rounded-full bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-blue-300/20 via-blue-400/10 to-transparent" />
        <div className="pointer-events-none absolute -bottom-40 left-[-10%] h-[420px] w-[420px] opacity-40 blur-3xl rounded-full bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-sky-200/25 via-sky-300/10 to-transparent" />
        <div className="mx-auto mt-10 max-w-6xl h-[280px] md:h-[360px] opacity-70">
          <Hero3D />
        </div>
      </div>

      {/* Header */}
      <header className="px-6 py-4">
        <div className="mx-auto max-w-6xl flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-md bg-primary/10 border border-primary/20 flex items-center justify-center text-primary font-bold">L</div>
            <span className="font-semibold text-neutral-900">LEGEAI</span>
          </div>
          <nav className="hidden md:flex items-center gap-6 text-sm">
            <Link href="#features" className="text-neutral-900 hover:text-primary">Features</Link>
            <Link href="#how-it-works" className="text-neutral-900 hover:text-primary">How it works</Link>
            <Link href="#pricing" className="text-neutral-900 hover:text-primary">Pricing</Link>
          </nav>
          <div className="flex items-center gap-3 text-sm">
            <Link href="/login" className="text-neutral-900 hover:text-primary">Login</Link>
            <Link href="/register" className="bg-primary text-white px-3 py-2 rounded shadow-sm hover:shadow-md transition">Get started</Link>
          </div>
        </div>
      </header>

      {/* Hero */}
      <main className="flex-1">
        <section className="mx-auto max-w-6xl px-6 py-10 md:py-16 grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
          <div>
            <h1 className="text-4xl md:text-6xl font-semibold leading-tight text-neutral-900">
              Smarter lead generation for modern insurance teams
            </h1>
            <p className="mt-4 text-neutral-700 text-lg max-w-prose">
              Capture, score, and route leads in real time. Built-in analytics, auditability, and tasteful 3D visuals—without the noise.
            </p>
            <div className="mt-6 flex flex-wrap gap-3">
              <Link href="/dashboard" className="bg-primary text-white px-4 py-2 rounded shadow-sm">Open dashboard</Link>
              <Link href="/register" className="border border-neutral-300 bg-white px-4 py-2 rounded hover:border-primary">Create an account</Link>
            </div>
            <div className="mt-6 flex items-center gap-6 text-sm text-neutral-600">
              <div className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-green-500" />99.9% uptime</div>
              <div className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-blue-500" />SOC2-ready</div>
              <div className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-sky-500" />API-first</div>
            </div>
          </div>
          <div className="rounded-xl border bg-white/80 backdrop-blur p-4 shadow-sm">
            <div className="text-sm text-neutral-700 mb-2">Live preview</div>
            <div className="aspect-video rounded border bg-gradient-to-br from-white to-neutral-50" />
            <ul className="mt-4 grid grid-cols-2 gap-3 text-sm">
              <li className="rounded border bg-white p-3 hover:shadow-md transition">Real-time routing</li>
              <li className="rounded border bg-white p-3 hover:shadow-md transition">ML scoring</li>
              <li className="rounded border bg-white p-3 hover:shadow-md transition">Analytics</li>
              <li className="rounded border bg-white p-3 hover:shadow-md transition">Role-based access</li>
            </ul>
          </div>
        </section>

        {/* Features */}
        <section id="features" className="mx-auto max-w-6xl px-6 pb-12">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="rounded-lg border bg-white p-5 shadow-sm">
              <div className="font-medium">Lead intake</div>
              <p className="text-sm text-neutral-600 mt-1">Flexible sources, idempotent ingestion, and privacy-by-design.</p>
            </div>
            <div className="rounded-lg border bg-white p-5 shadow-sm">
              <div className="font-medium">Scoring & routing</div>
              <p className="text-sm text-neutral-600 mt-1">Model-driven scores with dynamic banding and assignment.</p>
            </div>
            <div className="rounded-lg border bg-white p-5 shadow-sm">
              <div className="font-medium">Dashboards</div>
              <p className="text-sm text-neutral-600 mt-1">Funnel, sources, and performance with tasteful 3D accents.</p>
            </div>
          </div>
        </section>

        {/* How it works */}
        <section id="how-it-works" className="mx-auto max-w-6xl px-6 pb-12">
          <h2 className="text-2xl font-semibold text-neutral-900 mb-4">How it works</h2>
          <ol className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <li className="rounded-lg border bg-white p-5 shadow-sm"><span className="font-medium">1) Ingest.</span> Connect Facebook, Webflow, Zapier, or API.</li>
            <li className="rounded-lg border bg-white p-5 shadow-sm"><span className="font-medium">2) Score.</span> ML-driven scoring and banding in real-time.</li>
            <li className="rounded-lg border bg-white p-5 shadow-sm"><span className="font-medium">3) Route.</span> Auto-assign to agents and track outcomes.</li>
          </ol>
        </section>

        {/* Pricing */}
        <section id="pricing" className="mx-auto max-w-6xl px-6 pb-12">
          <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Pricing</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="rounded-lg border bg-white p-5 shadow-sm">
              <div className="font-medium">Starter</div>
              <div className="text-2xl font-semibold mt-1">$0</div>
              <p className="text-sm text-neutral-600 mt-1">Up to 100 leads/mo</p>
              <Link href="/register" className="inline-block mt-3 bg-primary text-white px-3 py-2 rounded">Get started</Link>
            </div>
            <div className="rounded-lg border bg-white p-5 shadow-sm">
              <div className="font-medium">Growth</div>
              <div className="text-2xl font-semibold mt-1">$99</div>
              <p className="text-sm text-neutral-600 mt-1">Up to 5k leads/mo</p>
              <Link href="/register" className="inline-block mt-3 bg-primary text-white px-3 py-2 rounded">Start trial</Link>
            </div>
            <div className="rounded-lg border bg-white p-5 shadow-sm">
              <div className="font-medium">Scale</div>
              <div className="text-2xl font-semibold mt-1">Custom</div>
              <p className="text-sm text-neutral-600 mt-1">Unlimited with SLAs</p>
              <Link href="/register" className="inline-block mt-3 bg-primary text-white px-3 py-2 rounded">Contact sales</Link>
            </div>
          </div>
        </section>

        {/* Security */}
        <section id="security" className="mx-auto max-w-6xl px-6 pb-12">
          <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Security</h2>
          <ul className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <li className="rounded-lg border bg-white p-5 shadow-sm">SOC2-ready controls</li>
            <li className="rounded-lg border bg-white p-5 shadow-sm">PII encryption at rest/in transit</li>
            <li className="rounded-lg border bg-white p-5 shadow-sm">Audit logs and RBAC</li>
          </ul>
        </section>

        {/* Contact */}
        <section id="contact" className="mx-auto max-w-6xl px-6 pb-16">
          <h2 className="text-2xl font-semibold text-neutral-900 mb-4">Contact</h2>
          <div className="rounded-lg border bg-white p-6 shadow-sm flex flex-col md:flex-row items-center justify-between gap-4">
            <p className="text-neutral-700">Questions or enterprise needs? We’d love to help.</p>
            <div className="flex items-center gap-3">
              <a href="mailto:hello@example.com" className="border border-neutral-300 bg-white px-4 py-2 rounded hover:border-primary">Email us</a>
              <Link href="/register" className="bg-primary text-white px-4 py-2 rounded">Start free</Link>
            </div>
          </div>
        </section>

      </main>

      {/* Footer */}
      <footer className="border-t bg-white/70 backdrop-blur">
        <div className="mx-auto max-w-6xl px-6 py-6 flex flex-col md:flex-row items-center justify-between gap-3 text-sm text-neutral-700">
          <div>© {new Date().getFullYear()} LEGEAI. All rights reserved.</div>
          <nav className="flex items-center gap-4">
            <Link href="#pricing" className="hover:text-primary">Pricing</Link>
            <Link href="#security" className="hover:text-primary">Security</Link>
            <Link href="#contact" className="hover:text-primary">Contact</Link>
          </nav>
        </div>
      </footer>
    </div>
  );
}
