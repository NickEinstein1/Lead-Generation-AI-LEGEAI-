'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

interface Platform {
  id: string;
  name: string;
  icon: string;
  requires_auth: boolean;
  type: string;
}

export default function CreateMeetingPage() {
  const router = useRouter();
  const [platforms, setPlatforms] = useState<Platform[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [checkingAvailability, setCheckingAvailability] = useState(false);
  const [availabilityResult, setAvailabilityResult] = useState<any>(null);

  const [formData, setFormData] = useState({
    title: '',
    description: '',
    platform: 'zoom',
    start_time: '',
    end_time: '',
    attendees: '',
    organizer_email: '',
    location: '',
    reminder_minutes: 15,
    notes: '',
    customer_name: '',
  });

  useEffect(() => {
    fetchPlatforms();
    
    // Set default start time to 1 hour from now
    const now = new Date();
    now.setHours(now.getHours() + 1);
    now.setMinutes(0);
    const startTime = now.toISOString().slice(0, 16);
    
    // Set default end time to 1 hour after start
    const endDate = new Date(now);
    endDate.setHours(endDate.getHours() + 1);
    const endTime = endDate.toISOString().slice(0, 16);
    
    setFormData(prev => ({
      ...prev,
      start_time: startTime,
      end_time: endTime
    }));
  }, []);

  const fetchPlatforms = async () => {
    try {
      const response = await fetch('http://localhost:8000/v1/scheduler/platforms/list');
      const data = await response.json();
      setPlatforms(data.platforms);
    } catch (error) {
      console.error('Error fetching platforms:', error);
    }
  };

  const checkAvailability = async () => {
    if (!formData.start_time || !formData.end_time) {
      setError('Please select start and end times');
      return;
    }

    setCheckingAvailability(true);
    setAvailabilityResult(null);

    try {
      const startISO = new Date(formData.start_time).toISOString();
      const endISO = new Date(formData.end_time).toISOString();
      
      const params = new URLSearchParams({
        start_time: startISO,
        end_time: endISO,
      });

      if (formData.organizer_email) {
        params.append('organizer_email', formData.organizer_email);
      }

      const response = await fetch(`http://localhost:8000/v1/scheduler/availability/check?${params}`);
      const data = await response.json();
      setAvailabilityResult(data);
    } catch (error) {
      console.error('Error checking availability:', error);
      setError('Failed to check availability');
    } finally {
      setCheckingAvailability(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      // Convert attendees string to array
      const attendeesArray = formData.attendees
        .split(',')
        .map(email => email.trim())
        .filter(email => email.length > 0);

      // Convert datetime-local to ISO format
      const startISO = new Date(formData.start_time).toISOString();
      const endISO = new Date(formData.end_time).toISOString();

      const payload = {
        title: formData.title,
        description: formData.description || null,
        platform: formData.platform,
        start_time: startISO,
        end_time: endISO,
        attendees: attendeesArray,
        organizer_email: formData.organizer_email,
        location: formData.location || null,
        reminder_minutes: formData.reminder_minutes,
        notes: formData.notes || null,
        customer_name: formData.customer_name || null,
      };

      const response = await fetch('http://localhost:8000/v1/scheduler', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': 'dev-api-key-12345',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to create meeting');
      }

      const data = await response.json();
      setSuccess(true);
      
      // Redirect after 2 seconds
      setTimeout(() => {
        router.push('/dashboard/scheduler');
      }, 2000);
    } catch (err: any) {
      setError(err.message || 'Failed to create meeting');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-6">
      {/* Header */}
      <div className="max-w-4xl mx-auto mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 mb-2">📅 Schedule New Meeting</h1>
            <p className="text-gray-600">Create a meeting across multiple platforms</p>
          </div>
          <Link
            href="/dashboard/scheduler"
            className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
          >
            ← Back to Scheduler
          </Link>
        </div>
      </div>

      {/* Success Message */}
      {success && (
        <div className="max-w-4xl mx-auto mb-6">
          <div className="bg-green-100 border-2 border-green-500 text-green-800 px-6 py-4 rounded-xl">
            <div className="flex items-center gap-3">
              <span className="text-2xl">✅</span>
              <div>
                <p className="font-bold">Meeting created successfully!</p>
                <p className="text-sm">Redirecting to scheduler...</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="max-w-4xl mx-auto mb-6">
          <div className="bg-red-100 border-2 border-red-500 text-red-800 px-6 py-4 rounded-xl">
            <div className="flex items-center gap-3">
              <span className="text-2xl">❌</span>
              <div>
                <p className="font-bold">Error</p>
                <p className="text-sm">{error}</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Availability Check Result */}
      {availabilityResult && (
        <div className="max-w-4xl mx-auto mb-6">
          <div className={`border-2 px-6 py-4 rounded-xl ${
            availabilityResult.available
              ? 'bg-green-50 border-green-500 text-green-800'
              : 'bg-yellow-50 border-yellow-500 text-yellow-800'
          }`}>
            <div className="flex items-center gap-3">
              <span className="text-2xl">{availabilityResult.available ? '✅' : '⚠️'}</span>
              <div>
                <p className="font-bold">{availabilityResult.message}</p>
                {availabilityResult.conflicts && availabilityResult.conflicts.length > 0 && (
                  <div className="mt-2">
                    <p className="text-sm font-medium mb-1">Conflicts:</p>
                    {availabilityResult.conflicts.map((conflict: any, idx: number) => (
                      <p key={idx} className="text-sm">
                        • {conflict.title} ({new Date(conflict.start_time).toLocaleString()})
                      </p>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Form */}
      <div className="max-w-4xl mx-auto">
        <form onSubmit={handleSubmit} className="bg-white rounded-xl shadow-lg p-8">
          <div className="space-y-6">
            {/* Meeting Title */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Meeting Title <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                required
                value={formData.title}
                onChange={(e) => setFormData({ ...formData, title: e.target.value })}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500"
                placeholder="e.g., Client Consultation - Auto Insurance"
              />
            </div>

            {/* Description */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Description
              </label>
              <textarea
                value={formData.description}
                onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                rows={3}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500"
                placeholder="Meeting agenda and details..."
              />
            </div>

            {/* Platform Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Meeting Platform <span className="text-red-500">*</span>
              </label>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                {platforms.map((platform) => (
                  <button
                    key={platform.id}
                    type="button"
                    onClick={() => setFormData({ ...formData, platform: platform.id })}
                    className={`p-4 border-2 rounded-lg transition-all ${
                      formData.platform === platform.id
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="text-3xl mb-2">{platform.icon}</div>
                    <div className="text-sm font-medium text-gray-900">{platform.name}</div>
                    <div className="text-xs text-gray-500 mt-1">{platform.type}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Date and Time */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Start Time <span className="text-red-500">*</span>
                </label>
                <input
                  type="datetime-local"
                  required
                  value={formData.start_time}
                  onChange={(e) => setFormData({ ...formData, start_time: e.target.value })}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  End Time <span className="text-red-500">*</span>
                </label>
                <input
                  type="datetime-local"
                  required
                  value={formData.end_time}
                  onChange={(e) => setFormData({ ...formData, end_time: e.target.value })}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500"
                />
              </div>
            </div>

            {/* Check Availability Button */}
            <div>
              <button
                type="button"
                onClick={checkAvailability}
                disabled={checkingAvailability}
                className="w-full px-4 py-3 bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200 transition-colors disabled:opacity-50"
              >
                {checkingAvailability ? 'Checking...' : '🔍 Check Availability'}
              </button>
            </div>

            {/* Organizer Email */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Organizer Email <span className="text-red-500">*</span>
              </label>
              <input
                type="email"
                required
                value={formData.organizer_email}
                onChange={(e) => setFormData({ ...formData, organizer_email: e.target.value })}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500"
                placeholder="organizer@company.com"
              />
            </div>

            {/* Attendees */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Attendees (comma-separated emails) <span className="text-red-500">*</span>
              </label>
              <input
                type="text"
                required
                value={formData.attendees}
                onChange={(e) => setFormData({ ...formData, attendees: e.target.value })}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500"
                placeholder="john@example.com, jane@example.com"
              />
              <p className="text-sm text-gray-500 mt-1">
                Separate multiple email addresses with commas
              </p>
            </div>

            {/* Customer Name (Optional) */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Customer Name (Optional)
              </label>
              <input
                type="text"
                value={formData.customer_name}
                onChange={(e) => setFormData({ ...formData, customer_name: e.target.value })}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500"
                placeholder="Link this meeting to a customer"
              />
            </div>

            {/* Location (for in-person meetings) */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Location (Optional)
              </label>
              <input
                type="text"
                value={formData.location}
                onChange={(e) => setFormData({ ...formData, location: e.target.value })}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500"
                placeholder="Office address or meeting room"
              />
              <p className="text-sm text-gray-500 mt-1">
                For in-person meetings or additional location details
              </p>
            </div>

            {/* Reminder */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Reminder (minutes before meeting)
              </label>
              <select
                value={formData.reminder_minutes}
                onChange={(e) => setFormData({ ...formData, reminder_minutes: parseInt(e.target.value) })}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500"
              >
                <option value={5}>5 minutes</option>
                <option value={10}>10 minutes</option>
                <option value={15}>15 minutes</option>
                <option value={30}>30 minutes</option>
                <option value={60}>1 hour</option>
                <option value={1440}>1 day</option>
              </select>
            </div>

            {/* Notes */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Additional Notes
              </label>
              <textarea
                value={formData.notes}
                onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
                rows={3}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500"
                placeholder="Any additional information..."
              />
            </div>

            {/* Submit Buttons */}
            <div className="flex gap-4 pt-6 border-t border-gray-200">
              <Link
                href="/dashboard/scheduler"
                className="flex-1 px-6 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors text-center"
              >
                Cancel
              </Link>
              <button
                type="submit"
                disabled={loading}
                className="flex-1 px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all shadow-lg disabled:opacity-50"
              >
                {loading ? 'Creating Meeting...' : '📅 Schedule Meeting'}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}

