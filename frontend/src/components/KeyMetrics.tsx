"use client";
import React from "react";

interface Metric {
  label: string;
  value: string | number;
  change: number;
  icon: string;
  color: string;
}

interface KeyMetricsProps {
  metrics?: Metric[];
}

export default function KeyMetrics({ metrics }: KeyMetricsProps) {
  const keyMetrics = metrics ?? [];

  if (keyMetrics.length === 0) {
    return (
      <div className="rounded-2xl border border-blue-500/30 bg-slate-900/60 backdrop-blur-xl p-6 text-cyan-200 shadow-2xl">
        No metrics available.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
      {keyMetrics.map((metric, idx) => (
        <div
          key={idx}
          className={`group relative bg-slate-900/60 backdrop-blur-xl rounded-2xl p-4 sm:p-6 text-white shadow-2xl hover:shadow-cyan-500/20 transition-all duration-300 hover:-translate-y-2 overflow-hidden border border-blue-500/30 hover:border-cyan-400/60`}
        >
          {/* Animated gradient background */}
          <div className={`absolute inset-0 bg-gradient-to-br ${metric.color} opacity-20 group-hover:opacity-30 transition-opacity duration-300`}></div>

          {/* Glow effect */}
          <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 via-transparent to-orange-500/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>

          {/* Content */}
          <div className="relative z-10">
            <div className="flex items-start justify-between mb-3 sm:mb-4">
              <div>
                <p className="text-xs sm:text-sm text-cyan-300 font-bold tracking-wider uppercase">{metric.label}</p>
                <p className="text-2xl sm:text-3xl lg:text-4xl font-bold mt-1 sm:mt-2 tracking-tight bg-gradient-to-r from-white to-cyan-200 bg-clip-text text-transparent">{metric.value}</p>
              </div>
              <div className="text-3xl sm:text-4xl opacity-80 group-hover:scale-110 group-hover:rotate-12 transition-all duration-300 filter drop-shadow-lg">
                {metric.icon}
              </div>
            </div>

            <div className="flex items-center gap-1.5 text-xs sm:text-sm bg-cyan-500/20 backdrop-blur-sm rounded-full px-3 py-1.5 w-fit border border-cyan-400/30">
              <span className={`font-bold ${metric.change >= 0 ? "text-green-300" : "text-red-300"}`}>
                {metric.change >= 0 ? "↑" : "↓"} {Math.abs(metric.change)}%
              </span>
              <span className="text-blue-200 font-medium">vs last month</span>
            </div>
          </div>

          {/* Decorative corner accent */}
          <div className="absolute -bottom-6 -right-6 w-24 h-24 bg-cyan-500/20 rounded-full blur-2xl group-hover:scale-150 transition-transform duration-500"></div>

          {/* Border glow */}
          <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 shadow-[inset_0_0_20px_rgba(34,211,238,0.3)]"></div>
        </div>
      ))}
    </div>
  );
}

