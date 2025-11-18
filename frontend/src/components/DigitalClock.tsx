"use client";
import React, { useState, useEffect } from 'react';

interface DigitalClockProps {
  showDate?: boolean;
  showSeconds?: boolean;
  className?: string;
  use24Hour?: boolean;
}

export default function DigitalClock({
  showDate = true,
  showSeconds = true,
  className = "",
  use24Hour = false
}: DigitalClockProps) {
  const [currentTime, setCurrentTime] = useState<Date | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // Set mounted to true and initialize time on client side only
    setMounted(true);
    setCurrentTime(new Date());

    // Update time every second
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);

    // Cleanup interval on component unmount
    return () => clearInterval(timer);
  }, []);

  // Don't render anything until mounted (prevents hydration mismatch)
  if (!mounted || !currentTime) {
    return (
      <div className={`flex flex-col items-end ${className}`}>
        <div className="flex items-center gap-2">
          <span className="text-lg">🕐</span>
          <span className="text-lg font-mono font-semibold text-slate-800 tabular-nums">
            --:--:--
          </span>
        </div>
        {showDate && (
          <span className="text-xs text-slate-500 font-medium mt-0.5">
            Loading...
          </span>
        )}
      </div>
    );
  }

  const formatTime = (date: Date): string => {
    let hours = date.getHours();
    const minutes = date.getMinutes();
    const seconds = date.getSeconds();
    let period = '';

    if (!use24Hour) {
      period = hours >= 12 ? ' PM' : ' AM';
      hours = hours % 12 || 12; // Convert to 12-hour format
    }

    const formattedHours = hours.toString().padStart(2, '0');
    const formattedMinutes = minutes.toString().padStart(2, '0');
    const formattedSeconds = seconds.toString().padStart(2, '0');

    if (showSeconds) {
      return `${formattedHours}:${formattedMinutes}:${formattedSeconds}${period}`;
    }
    return `${formattedHours}:${formattedMinutes}${period}`;
  };

  const formatDate = (date: Date): string => {
    const options: Intl.DateTimeFormatOptions = { 
      weekday: 'short', 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric' 
    };
    return date.toLocaleDateString('en-US', options);
  };

  return (
    <div className={`flex flex-col items-end bg-gradient-to-br from-blue-50 to-indigo-50 px-4 py-2 rounded-lg border border-blue-200 shadow-sm ${className}`}>
      <div className="flex items-center gap-2">
        <span className="text-lg">🕐</span>
        <span className="text-lg font-mono font-bold text-blue-900 tabular-nums tracking-tight">
          {formatTime(currentTime)}
        </span>
      </div>
      {showDate && (
        <span className="text-xs text-blue-700 font-medium mt-0.5">
          {formatDate(currentTime)}
        </span>
      )}
    </div>
  );
}

