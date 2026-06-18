import React, { useEffect, useMemo, useState } from 'react';
import type { schemas } from '../types';

export interface BookingDrawerProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  bookingDraft?: schemas.BookingDraft | null;
  onSubmit?: (values: {
    source: string;
    destination: string;
    travel_date: string;
    travel_class: string;
    passenger_count: number;
    train_number?: string;
    time_preference?: string;
    budget?: number;
    direct_only?: boolean;
  }) => void | Promise<void>;
}

type BookingFormValues = {
  source: string;
  destination: string;
  travel_date: string;
  travel_class: string;
  passenger_count: string;
  train_number: string;
  time_preference: string;
  budget: string;
  direct_only: boolean;
};

const EMPTY_VALUES: BookingFormValues = {
  source: '',
  destination: '',
  travel_date: '',
  travel_class: '',
  passenger_count: '',
  train_number: '',
  time_preference: '',
  budget: '',
  direct_only: false,
};

function toText(value: unknown): string {
  if (value === null || value === undefined) return '';
  return String(value);
}

function buildInitialValues(draft?: schemas.BookingDraft | null): BookingFormValues {
  if (!draft) return EMPTY_VALUES;

  return {
    source: toText(draft.source),
    destination: toText(draft.destination),
    travel_date: toText(draft.travel_date),
    travel_class: toText(draft.travel_class),
    passenger_count: toText(draft.passenger_count),
    train_number: toText(draft.train_number),
    time_preference: toText(draft.time_preference),
    budget: toText(draft.budget),
    direct_only: Boolean(draft.direct_only),
  };
}

function inputClass(hasError: boolean) {
  return [
    'w-full rounded-lg border px-3 py-2 text-sm outline-none transition bg-white',
    hasError ? 'border-red-400 focus:border-red-500' : 'border-slate-300 focus:border-slate-900',
  ].join(' ');
}

function Field({
  label,
  required,
  error,
  children,
}: {
  label: string;
  required?: boolean;
  error?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="block space-y-1.5">
      <span className="text-sm font-medium text-slate-700">
        {label} {required ? <span className="text-red-600">*</span> : null}
      </span>
      {children}
      {error ? <p className="text-sm text-red-600">{error}</p> : null}
    </label>
  );
}

export default function BookingDrawer({
  open,
  onOpenChange,
  bookingDraft,
  onSubmit,
}: BookingDrawerProps) {
  const [values, setValues] = useState<BookingFormValues>(() => buildInitialValues(bookingDraft));
  const [touched, setTouched] = useState<Record<keyof BookingFormValues, boolean>>({
    source: false,
    destination: false,
    travel_date: false,
    travel_class: false,
    passenger_count: false,
    train_number: false,
    time_preference: false,
    budget: false,
    direct_only: false,
  });
  const [submitAttempted, setSubmitAttempted] = useState(false);

  useEffect(() => {
    if (!open) return;
    setValues(buildInitialValues(bookingDraft));
    setTouched({
      source: false,
      destination: false,
      travel_date: false,
      travel_class: false,
      passenger_count: false,
      train_number: false,
      time_preference: false,
      budget: false,
      direct_only: false,
    });
    setSubmitAttempted(false);
  }, [open, bookingDraft]);

  const errors = useMemo(() => {
    const next: Partial<Record<keyof BookingFormValues, string>> = {};

    if (!values.source.trim()) next.source = 'Source is required.';
    if (!values.destination.trim()) next.destination = 'Destination is required.';
    if (!values.travel_date.trim()) next.travel_date = 'Travel date is required.';
    if (!values.travel_class.trim()) next.travel_class = 'Travel class is required.';

    if (!values.passenger_count.trim()) {
      next.passenger_count = 'Passenger count is required.';
    } else {
      const count = Number(values.passenger_count);
      if (!Number.isInteger(count) || count < 1) {
        next.passenger_count = 'Enter a valid passenger count.';
      }
    }

    if (values.budget.trim()) {
      const budget = Number(values.budget);
      if (Number.isNaN(budget) || budget < 0) {
        next.budget = 'Enter a valid budget.';
      }
    }

    return next;
  }, [values]);

  const isValid = Object.keys(errors).length === 0;

  const setField = <K extends keyof BookingFormValues>(key: K, value: BookingFormValues[K]) => {
    setValues((prev) => ({ ...prev, [key]: value }));
  };

  const showError = (key: keyof BookingFormValues) => Boolean((touched[key] || submitAttempted) && errors[key]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setSubmitAttempted(true);

    if (!onSubmit || !isValid) return;

    await onSubmit({
      source: values.source.trim(),
      destination: values.destination.trim(),
      travel_date: values.travel_date.trim(),
      travel_class: values.travel_class.trim(),
      passenger_count: Number(values.passenger_count),
      train_number: values.train_number.trim() || undefined,
      time_preference: values.time_preference.trim() || undefined,
      budget: values.budget.trim() ? Number(values.budget) : undefined,
      direct_only: values.direct_only,
    });
  };

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-50">
      <button
        type="button"
        aria-label="Close booking drawer"
        className="absolute inset-0 h-full w-full bg-black/40"
        onClick={() => onOpenChange(false)}
      />

      <div className="absolute right-0 top-0 h-full w-full max-w-xl overflow-y-auto bg-white shadow-2xl">
        <div className="flex items-center justify-between border-b px-6 py-4">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Booking form</h2>
            <p className="text-sm text-slate-500">Review and complete the details from chat.</p>
          </div>

          <button
            type="button"
            className="rounded-md px-3 py-2 text-sm text-slate-600 hover:bg-slate-100"
            onClick={() => onOpenChange(false)}
          >
            Close
          </button>
        </div>

        <form className="space-y-5 p-6" onSubmit={handleSubmit} noValidate>
          <Field label="Source" required error={showError('source') ? errors.source : undefined}>
            <input
              value={values.source}
              onChange={(e) => setField('source', e.target.value)}
              onBlur={() => setTouched((prev) => ({ ...prev, source: true }))}
              className={inputClass(showError('source'))}
              placeholder="e.g. Bangalore"
            />
          </Field>

          <Field
            label="Destination"
            required
            error={showError('destination') ? errors.destination : undefined}
          >
            <input
              value={values.destination}
              onChange={(e) => setField('destination', e.target.value)}
              onBlur={() => setTouched((prev) => ({ ...prev, destination: true }))}
              className={inputClass(showError('destination'))}
              placeholder="e.g. Chennai"
            />
          </Field>

          <Field
            label="Travel date"
            required
            error={showError('travel_date') ? errors.travel_date : undefined}
          >
            <input
              type="date"
              value={values.travel_date}
              onChange={(e) => setField('travel_date', e.target.value)}
              onBlur={() => setTouched((prev) => ({ ...prev, travel_date: true }))}
              className={inputClass(showError('travel_date'))}
            />
          </Field>

          <Field
            label="Travel class"
            required
            error={showError('travel_class') ? errors.travel_class : undefined}
          >
            <input
              value={values.travel_class}
              onChange={(e) => setField('travel_class', e.target.value)}
              onBlur={() => setTouched((prev) => ({ ...prev, travel_class: true }))}
              className={inputClass(showError('travel_class'))}
              placeholder="e.g. SL, 3A, 2A"
            />
          </Field>

          <Field
            label="Passenger count"
            required
            error={showError('passenger_count') ? errors.passenger_count : undefined}
          >
            <input
              type="number"
              min={1}
              step={1}
              value={values.passenger_count}
              onChange={(e) => setField('passenger_count', e.target.value)}
              onBlur={() => setTouched((prev) => ({ ...prev, passenger_count: true }))}
              className={inputClass(showError('passenger_count'))}
              placeholder="1"
            />
          </Field>

          <details className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3">
            <summary className="cursor-pointer text-sm font-medium text-slate-700">
              Optional details
            </summary>

            <div className="mt-4 space-y-5">
              <Field label="Train number">
                <input
                  value={values.train_number}
                  onChange={(e) => setField('train_number', e.target.value)}
                  className={inputClass(false)}
                  placeholder="Optional"
                />
              </Field>

              <Field label="Time preference">
                <input
                  value={values.time_preference}
                  onChange={(e) => setField('time_preference', e.target.value)}
                  className={inputClass(false)}
                  placeholder="Morning, evening, 18:00, etc."
                />
              </Field>

              <Field label="Budget" error={showError('budget') ? errors.budget : undefined}>
                <input
                  type="number"
                  min={0}
                  step="0.01"
                  value={values.budget}
                  onChange={(e) => setField('budget', e.target.value)}
                  onBlur={() => setTouched((prev) => ({ ...prev, budget: true }))}
                  className={inputClass(showError('budget'))}
                  placeholder="Optional"
                />
              </Field>

              <label className="flex items-center gap-3 text-sm text-slate-700">
                <input
                  type="checkbox"
                  checked={values.direct_only}
                  onChange={(e) => setField('direct_only', e.target.checked)}
                  className="h-4 w-4 rounded border-slate-300"
                />
                Direct trains only
              </label>
            </div>
          </details>

          <div className="flex items-center justify-end gap-3 border-t pt-4">
            <button
              type="button"
              className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
              onClick={() => onOpenChange(false)}
            >
              Cancel
            </button>

            <button
              type="submit"
              disabled={!onSubmit || !isValid}
              className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
            >
              Submit booking
            </button>
          </div>

          {!onSubmit ? (
            <p className="text-sm text-slate-500">
              This drawer is ready for a submit handler from the parent component.
            </p>
          ) : null}
        </form>
      </div>
    </div>
  );
}
