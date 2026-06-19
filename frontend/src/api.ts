import axios, { AxiosError } from 'axios';
import type { schemas } from './types';

const apiUrl = import.meta.env.VITE_API_URL;
const baseURL = apiUrl ? `${apiUrl.replace(/\/$/, '')}/api/v1` : '/api/v1';

const api = axios.create({ baseURL });

export default api;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export type StructuredChatRequest =
  | {
      message: string;
      session_id?: string;
      history?: schemas.ChatMessage[];
    }
  | schemas.ChatMessage[];

export type TrainSearchRequest = {
  from_station: string;
  to_station: string;
  date?: string;
};

// ---------------------------------------------------------------------------
// Normalizers
// ---------------------------------------------------------------------------
function isObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function normalizeChatMessage(value: unknown): schemas.ChatMessage | null {
  if (!isObject(value)) return null;
  const { role, content } = value;
  if ((role === 'user' || role === 'assistant') && typeof content === 'string') {
    return { role, content };
  }
  return null;
}

function normalizeTrain(value: unknown): schemas.Train | null {
  if (!isObject(value)) return null;
  const { train_number, train_name, source_station_code, destination_station_code } = value;
  if (
    typeof train_number === 'string' &&
    typeof train_name === 'string' &&
    typeof source_station_code === 'string' &&
    typeof destination_station_code === 'string'
  ) {
    return { train_number, train_name, source_station_code, destination_station_code };
  }
  return null;
}

function normalizeStructuredChatResponse(data: unknown): schemas.StructuredChatResponse {
  if (Array.isArray(data)) {
    return {
      messages: data.map(normalizeChatMessage).filter(Boolean) as schemas.ChatMessage[],
      missing_required_fields: [],
    };
  }

  if (!isObject(data)) {
    return { messages: [], missing_required_fields: [] };
  }

  const messages = Array.isArray(data.messages)
    ? (data.messages.map(normalizeChatMessage).filter(Boolean) as schemas.ChatMessage[])
    : [];

  const missingRequiredFields = Array.isArray(data.missing_required_fields)
    ? data.missing_required_fields.filter((item): item is string => typeof item === 'string')
    : [];

  const bookingDraft = isObject(data.booking_draft) ? (data.booking_draft as schemas.BookingDraft) : null;
  const diagnostics = isObject(data.diagnostics) ? (data.diagnostics as schemas.ChatDiagnostics) : null;

  const action =
    typeof data.action === 'string' || data.action === null
      ? (data.action as schemas.UiAction | string | null)
      : undefined;

  return { messages, action, booking_draft: bookingDraft, missing_required_fields: missingRequiredFields, diagnostics };
}

// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------

/**
 * POST /chat/structured
 *
 * On network / server errors this now THROWS so the caller (ChatAssistant)
 * can append an error bubble to existing messages instead of wiping history.
 */
export async function postStructuredChat(
  request: StructuredChatRequest
): Promise<schemas.StructuredChatResponse> {
  const response = await api.post('/chat/structured', request);
  return normalizeStructuredChatResponse(response.data);
}

/**
 * GET /trains/search
 * Returns empty array on any error (non-critical, UI shows fallback).
 */
export async function searchTrains(request: TrainSearchRequest): Promise<schemas.Train[]> {
  try {
    const params: Record<string, string> = {
      from_station: request.from_station,
      to_station: request.to_station,
    };
    if (request.date) params.date = request.date;

    const response = await api.get('/trains/search', { params });
    if (Array.isArray(response.data)) {
      return response.data.map(normalizeTrain).filter(Boolean) as schemas.Train[];
    }
    return [];
  } catch (error) {
    const axiosError = error as AxiosError<unknown>;
    console.error('[searchTrains] failed', {
      status: axiosError.response?.status,
      message: axiosError.message,
    });
    return [];
  }
}

/**
 * Legacy /chat endpoint — kept for backward compatibility.
 */
export async function postLegacyChat(
  request:
    | { message: string; session_id?: string; history?: schemas.ChatMessage[] }
    | schemas.ChatMessage[]
): Promise<schemas.ChatMessage[]> {
  const response = await api.post('/chat', request);
  const data = response.data;
  if (Array.isArray(data)) {
    return data.map(normalizeChatMessage).filter(Boolean) as schemas.ChatMessage[];
  }
  return [];
}
