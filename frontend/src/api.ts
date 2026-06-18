import axios, { AxiosError } from 'axios';
import type { schemas } from './types';

const apiUrl = import.meta.env.VITE_API_URL;
const baseURL = apiUrl ? `${apiUrl.replace(/\/$/, '')}/api/v1` : '/api/v1';

const api = axios.create({
  baseURL,
});

export default api;

export type StructuredChatRequest =
  | {
      message: string;
      session_id?: string;
      history?: schemas.ChatMessage[];
    }
  | schemas.ChatMessage[];

function isObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function normalizeChatMessage(value: unknown): schemas.ChatMessage | null {
  if (!isObject(value)) return null;
  const role = value.role;
  const content = value.content;

  if ((role === 'user' || role === 'assistant') && typeof content === 'string') {
    return {
      role,
      content,
    };
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
    return {
      messages: [],
      missing_required_fields: [],
    };
  }

  const messages = Array.isArray(data.messages)
    ? data.messages.map(normalizeChatMessage).filter(Boolean) as schemas.ChatMessage[]
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

  return {
    messages,
    action,
    booking_draft: bookingDraft,
    missing_required_fields: missingRequiredFields,
    diagnostics,
  };
}

export async function postStructuredChat(
  request: StructuredChatRequest
): Promise<schemas.StructuredChatResponse> {
  try {
    const response = await api.post('/chat/structured', request);
    return normalizeStructuredChatResponse(response.data);
  } catch (error) {
    const axiosError = error as AxiosError<unknown>;

    return {
      messages: [],
      action: undefined,
      booking_draft: null,
      missing_required_fields: [],
      diagnostics: {
        http_status: axiosError.response?.status,
        error:
          axiosError.response?.data &&
          isObject(axiosError.response.data) &&
          typeof axiosError.response.data.detail === 'string'
            ? axiosError.response.data.detail
            : axiosError.message || 'Failed to fetch structured chat response',
      },
    };
  }
}

/*
  Optional legacy helper. Keep it only if your codebase still uses /chat directly.
  It is safe to leave it here for backward compatibility.
*/
export async function postLegacyChat(
  request:
    | {
        message: string;
        session_id?: string;
        history?: schemas.ChatMessage[];
      }
    | schemas.ChatMessage[]
): Promise<schemas.ChatMessage[]> {
  const response = await api.post('/chat', request);
  const data = response.data;

  if (Array.isArray(data)) {
    return data.map(normalizeChatMessage).filter(Boolean) as schemas.ChatMessage[];
  }

  return [];
}
