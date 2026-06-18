export namespace schemas {
  export interface ChatMessage {
    role: 'user' | 'assistant';
    content: string;
  }

  export interface Train {
    train_number: string;
    train_name: string;
    source_station_code: string;
    destination_station_code: string;
  }

  export type TrainSearchResponse = Train[];

  export interface Station {
    station_code: string;
    station_name: string;
    city: string | null;
  }

  export interface Route {
    train_number: string;
    sequence: number;
    station_code: string;
    arrival_time: string | null;
    departure_time: string | null;
    distance_km: number | null;
  }

  export interface Fare {
    train_number: string;
    class_type: string;
    amount: number;
  }

  export interface Booking {
    id: number;
    user_id: number;
    train_number: string;
    passenger_count: number;
    travel_class: string;
    travel_date: string;
    status: string;
    created_at: string;
  }

  export interface BookingConfirmationResponse {
    success: boolean;
    status: string;
    message: string;
    booking?: Booking | null;
    selected_train?: Train | null;
    missing_fields?: string[];
    errors?: string[];
  }

  export type UiAction = 'open_booking_form';

  export interface BookingDraft {
    source?: string;
    destination?: string;
    travel_date?: string;
    travel_class?: string;
    passenger_count?: number;
    train_number?: string;
    time_preference?: string;
    departure_after?: string;
    departure_before?: string;
    berth_preference?: string;
    budget?: number;
    direct_only?: boolean;
    missing_required_fields?: string[];
    ready_for_submit?: boolean;
    [key: string]: unknown;
  }

  export interface ChatDiagnostics {
    route?: string;
    intent?: string;
    llm_attempted?: boolean;
    llm_used?: boolean;
    local_handler_used?: boolean;
    fallback_used?: boolean;
    llm_error?: string | null;
    local_error?: string | null;
    http_status?: number;
    error?: string;
    [key: string]: unknown;
  }

  export interface StructuredChatResponse {
    messages?: ChatMessage[];
    action?: UiAction | string | null;
    booking_draft?: BookingDraft | null;
    missing_required_fields?: string[];
    diagnostics?: ChatDiagnostics | null;
    metadata?: Record<string, unknown>;
  }
}
