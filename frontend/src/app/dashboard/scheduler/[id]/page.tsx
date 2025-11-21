'use client';

import { useState, useEffect } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';

interface Meeting {
  id: string;
  title: string;
  description?: string;
  platform: string;
  platform_name: string;
  platform_icon: string;
  start_time: string;
  end_time: string;
  duration_minutes: number;
  attendees: string[];
  organizer_email: string;
  location?: string;
  meeting_url?: string;
  status: string;
  customer_name?: string;
  notes?: string;
  reminder_minutes: number;
  created_at: string;
  updated_at?: string;
}

export default function MeetingDetailsPage() {
  const params = useParams();
  const router = useRouter();
  const meetingId = params.id as string;
  
  const [meeting, setMeeting] = useState<Meeting | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showCancelModal, setShowCancelModal] = useState(false);
  const [showCompleteModal, setShowCompleteModal] = useState(false);
  const [completionNotes, setCompletionNotes] = useState('');

  useEffect(() => {
    fetchMeeting();
  }, [meetingId]);

  const fetchMeeting = async () => {
    try {
      const response = await fetch(`http://localhost:8000/v1/scheduler/${meetingId}`);
      if (!response.ok) {
        throw new Error('Meeting not found');
      }
      const data = await response.json();
      setMeeting(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load meeting');
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = async () => {
    try {
      const response = await fetch(`http://localhost:8000/v1/scheduler/${meetingId}`, {
        method: 'DELETE',
        headers: {
          'X-API-Key': 'dev-api-key-12345',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to cancel meeting');
      }

      router.push('/dashboard/scheduler');
    } catch (err: any) {
      setError(err.message || 'Failed to cancel meeting');
    }
  };

  const handleComplete = async () => {
    try {
      const params = new URLSearchParams();
      if (completionNotes) {
        params.append('notes', completionNotes);
      }

      const response = await fetch(`http://localhost:8000/v1/scheduler/${meetingId}/complete?${params}`, {
        method: 'POST',
        headers: {
          'X-API-Key': 'dev-api-key-12345',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to mark meeting as complete');
      }

      fetchMeeting();
      setShowCompleteModal(false);
    } catch (err: any) {
      setError(err.message || 'Failed to complete meeting');
    }
  };

  const formatDateTime = (isoString: string) => {
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
      weekday: 'long',
      month: 'long',
      day: 'numeric',
      year: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'scheduled': return 'bg-blue-100 text-blue-800';
      case 'completed': return 'bg-green-100 text-green-800';
      case 'cancelled': return 'bg-red-100 text-red-800';
      case 'rescheduled': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const isUpcoming = (startTime: string) => {
    return new Date(startTime) > new Date();
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading meeting details...</p>
        </div>
      </div>
    );
  }

  if (error || !meeting) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-6">
        <div className="max-w-4xl mx-auto">
          <div className="bg-red-100 border-2 border-red-500 text-red-800 px-6 py-8 rounded-xl text-center">
            <div className="text-6xl mb-4">❌</div>
            <h2 className="text-2xl font-bold mb-2">Error</h2>
            <p className="mb-4">{error || 'Meeting not found'}</p>
            <Link
              href="/dashboard/scheduler"
              className="inline-block px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
            >
              Back to Scheduler
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-6">
      {/* Header */}
      <div className="max-w-4xl mx-auto mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-4xl font-bold text-gray-900 mb-2">📅 Meeting Details</h1>
            <p className="text-gray-600">View and manage meeting information</p>
          </div>
          <Link
            href="/dashboard/scheduler"
            className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
          >
            ← Back to Scheduler
          </Link>
        </div>
      </div>

      {/* Meeting Card */}
      <div className="max-w-4xl mx-auto">
        <div className="bg-white rounded-xl shadow-lg overflow-hidden">
          {/* Header Section */}
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-8">
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-4">
                <span className="text-6xl">{meeting.platform_icon}</span>
                <div>
                  <h2 className="text-3xl font-bold mb-2">{meeting.title}</h2>
                  <p className="text-blue-100">{meeting.platform_name}</p>
                </div>
              </div>
              <span className={`px-4 py-2 rounded-full text-sm font-medium ${getStatusColor(meeting.status)}`}>
                {meeting.status}
              </span>
            </div>
          </div>

          {/* Details Section */}
          <div className="p-8 space-y-6">
            {/* Description */}
            {meeting.description && (
              <div>
                <h3 className="text-sm font-medium text-gray-500 mb-2">Description</h3>
                <p className="text-gray-900">{meeting.description}</p>
              </div>
            )}

            {/* Date and Time */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-sm font-medium text-gray-500 mb-2">Start Time</h3>
                <p className="text-lg text-gray-900 flex items-center gap-2">
                  <span>🕐</span>
                  {formatDateTime(meeting.start_time)}
                </p>
              </div>
              <div>
                <h3 className="text-sm font-medium text-gray-500 mb-2">End Time</h3>
                <p className="text-lg text-gray-900 flex items-center gap-2">
                  <span>🕐</span>
                  {formatDateTime(meeting.end_time)}
                </p>
              </div>
            </div>

            {/* Duration */}
            <div>
              <h3 className="text-sm font-medium text-gray-500 mb-2">Duration</h3>
              <p className="text-lg text-gray-900 flex items-center gap-2">
                <span>⏱️</span>
                {meeting.duration_minutes} minutes
              </p>
            </div>

            {/* Organizer */}
            <div>
              <h3 className="text-sm font-medium text-gray-500 mb-2">Organizer</h3>
              <p className="text-lg text-gray-900 flex items-center gap-2">
                <span>👤</span>
                {meeting.organizer_email}
              </p>
            </div>

            {/* Attendees */}
            <div>
              <h3 className="text-sm font-medium text-gray-500 mb-2">
                Attendees ({meeting.attendees.length})
              </h3>
              <div className="space-y-2">
                {meeting.attendees.map((email, idx) => (
                  <p key={idx} className="text-gray-900 flex items-center gap-2">
                    <span>👥</span>
                    {email}
                  </p>
                ))}
              </div>
            </div>

            {/* Meeting URL */}
            {meeting.meeting_url && (
              <div>
                <h3 className="text-sm font-medium text-gray-500 mb-2">Meeting Link</h3>
                <a
                  href={meeting.meeting_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  🔗 Join Meeting
                </a>
              </div>
            )}

            {/* Location */}
            {meeting.location && (
              <div>
                <h3 className="text-sm font-medium text-gray-500 mb-2">Location</h3>
                <p className="text-lg text-gray-900 flex items-center gap-2">
                  <span>📍</span>
                  {meeting.location}
                </p>
              </div>
            )}

            {/* Customer */}
            {meeting.customer_name && (
              <div>
                <h3 className="text-sm font-medium text-gray-500 mb-2">Customer</h3>
                <p className="text-lg text-gray-900 flex items-center gap-2">
                  <span>🏢</span>
                  {meeting.customer_name}
                </p>
              </div>
            )}

            {/* Reminder */}
            <div>
              <h3 className="text-sm font-medium text-gray-500 mb-2">Reminder</h3>
              <p className="text-lg text-gray-900 flex items-center gap-2">
                <span>🔔</span>
                {meeting.reminder_minutes} minutes before
              </p>
            </div>

            {/* Notes */}
            {meeting.notes && (
              <div>
                <h3 className="text-sm font-medium text-gray-500 mb-2">Notes</h3>
                <p className="text-gray-900">{meeting.notes}</p>
              </div>
            )}

            {/* Metadata */}
            <div className="pt-6 border-t border-gray-200">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-600">
                <div>
                  <span className="font-medium">Created:</span> {new Date(meeting.created_at).toLocaleString()}
                </div>
                {meeting.updated_at && (
                  <div>
                    <span className="font-medium">Updated:</span> {new Date(meeting.updated_at).toLocaleString()}
                  </div>
                )}
              </div>
            </div>

            {/* Action Buttons */}
            {meeting.status === 'scheduled' && (
              <div className="flex gap-4 pt-6 border-t border-gray-200">
                <button
                  onClick={() => setShowCompleteModal(true)}
                  className="flex-1 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                >
                  ✅ Mark as Complete
                </button>
                <button
                  onClick={() => setShowCancelModal(true)}
                  className="flex-1 px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                >
                  ❌ Cancel Meeting
                </button>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Cancel Confirmation Modal */}
      {showCancelModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-2xl max-w-md w-full p-6">
            <h3 className="text-2xl font-bold text-gray-900 mb-4">Cancel Meeting?</h3>
            <p className="text-gray-600 mb-6">
              Are you sure you want to cancel this meeting? This action cannot be undone.
            </p>
            <div className="flex gap-4">
              <button
                onClick={() => setShowCancelModal(false)}
                className="flex-1 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
              >
                No, Keep It
              </button>
              <button
                onClick={handleCancel}
                className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
              >
                Yes, Cancel Meeting
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Complete Confirmation Modal */}
      {showCompleteModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-2xl max-w-md w-full p-6">
            <h3 className="text-2xl font-bold text-gray-900 mb-4">Mark as Complete?</h3>
            <p className="text-gray-600 mb-4">
              Add any notes or summary from the meeting:
            </p>
            <textarea
              value={completionNotes}
              onChange={(e) => setCompletionNotes(e.target.value)}
              rows={4}
              className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:outline-none focus:border-blue-500 mb-4"
              placeholder="Meeting summary, action items, etc..."
            />
            <div className="flex gap-4">
              <button
                onClick={() => setShowCompleteModal(false)}
                className="flex-1 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleComplete}
                className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
              >
                Mark Complete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

