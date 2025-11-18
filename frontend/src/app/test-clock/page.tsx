"use client";
import DigitalClock from "@/components/DigitalClock";

export default function TestClockPage() {
  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        <h1 className="text-3xl font-bold text-slate-900">Digital Clock Test Page</h1>
        
        <div className="bg-white p-8 rounded-lg shadow-lg space-y-6">
          <h2 className="text-xl font-semibold text-slate-800">Clock Variations</h2>
          
          <div className="space-y-4">
            <div className="border-b pb-4">
              <p className="text-sm text-slate-600 mb-2">Default (12-hour with date and seconds):</p>
              <DigitalClock showDate={true} showSeconds={true} use24Hour={false} />
            </div>
            
            <div className="border-b pb-4">
              <p className="text-sm text-slate-600 mb-2">24-hour format:</p>
              <DigitalClock showDate={true} showSeconds={true} use24Hour={true} />
            </div>
            
            <div className="border-b pb-4">
              <p className="text-sm text-slate-600 mb-2">Without seconds:</p>
              <DigitalClock showDate={true} showSeconds={false} use24Hour={false} />
            </div>
            
            <div className="border-b pb-4">
              <p className="text-sm text-slate-600 mb-2">Without date:</p>
              <DigitalClock showDate={false} showSeconds={true} use24Hour={false} />
            </div>
            
            <div>
              <p className="text-sm text-slate-600 mb-2">Minimal (no date, no seconds):</p>
              <DigitalClock showDate={false} showSeconds={false} use24Hour={false} />
            </div>
          </div>
        </div>
        
        <div className="bg-blue-50 p-6 rounded-lg border border-blue-200">
          <h3 className="font-semibold text-blue-900 mb-2">Instructions:</h3>
          <ul className="list-disc list-inside text-blue-800 space-y-1">
            <li>All clocks should update every second</li>
            <li>Time should match your system time</li>
            <li>If you see "--:--:--" it means the component is loading</li>
            <li>Check browser console for any errors</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

