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
}