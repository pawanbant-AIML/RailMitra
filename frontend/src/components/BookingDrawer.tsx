import React, { useEffect, useMemo, useRef, useState } from 'react';
import { searchTrains } from '../api';
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
  submitting?: boolean;
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

type TrainOption = {
  value: string;
  label: string;
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

const TRAVEL_CLASS_OPTIONS = [
  { value: '', label: 'Select a class' },
  { value: 'SL', label: 'SL — Sleeper' },
  { value: '2S', label: '2S — Second Seating' },
  { value: 'CC', label: 'CC — AC Chair Car' },
  { value: '3A', label: '3A — AC 3 Tier' },
  { value: '2A', label: '2A — AC 2 Tier' },
  { value: '1A', label: '1A — AC First Class' },
  { value: '3E', label: '3E — AC 3 Tier Economy' },
  { value: 'EC', label: 'EC — Executive Chair Car' },
  { value: 'FC', label: 'FC — First Class' },
];

const PASSENGER_COUNT_OPTIONS = Array.from({ length: 9 }, (_, index) => {
  const count = String(index + 1);
  return { value: count, label: count };
});

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

function fieldClass(hasError: boolean) {
  return [
    'w-full rounded-lg border px-3 py-2 text-sm outline-none transition-colors',
    'bg-white text-slate-900 placeholder:text-slate-400',
    'border-slate-300 focus:border-slate-900 focus:ring-2 focus:ring-slate-200',
    'dark:bg-slate-900 dark:text-slate-100 dark:placeholder:text-slate-500',
    'dark:border-slate-700 dark:focus:border-slate-400 dark:focus:ring-slate-800',
    hasError ? 'border-red-400 focus:border-red-500 dark:border-red-500 dark:focus:border-red-400' : '',
  ].join(' ');
}

function selectClass(hasError: boolean) {
  return [fieldClass(hasError), 'pr-10', 'appearance-none'].join(' ');
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
      <span className="text-sm font-medium text-slate-700 dark:text-slate-200">
        {label} {required ? <span className="text-red-600 dark:text-red-400">*</span> : null}
      </span>
      {children}
      {error ? <p className="text-sm text-red-600 dark:text-red-400">{error}</p> : null}
    </label>
  );
}

export default function BookingDrawer({
  open,
  onOpenChange,
  bookingDraft,
  onSubmit,
  submitting = false,
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

  const [trainOptions, setTrainOptions] = useState<TrainOption[]>([]);
  const [trainLoading, setTrainLoading] = useState(false);
  const [trainMessage, setTrainMessage] = useState<string>('');
  const requestIdRef = useRef(0);

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
    setTrainOptions([]);
    setTrainLoading(false);
    setTrainMessage('');
    requestIdRef.current += 1;
  }, [open, bookingDraft]);

  const sourceValue = values.source.trim();
  const destinationValue = values.destination.trim();
  const travelDateValue = values.travel_date.trim();

  useEffect(() => {
    if (!open) return;

    const hasSearchInputs = Boolean(sourceValue && destinationValue && travelDateValue);
    const currentRequestId = ++requestIdRef.current;

    if (!hasSearchInputs) {
      setTrainOptions([]);
      setTrainLoading(false);
      setTrainMessage('');
      setValues((prev) => ({
        ...prev,
        train_number: '',
      }));
      return;
    }

    setTrainLoading(true);
    setTrainMessage('');
    setTrainOptions([]);
    setValues((prev) => ({
      ...prev,
      train_number: '',
    }));

    const timer = window.setTimeout(async () => {
      try {
        const trains = await searchTrains({
          from_station: sourceValue,
          to_station: destinationValue,
          date: travelDateValue,
        });

        if (requestIdRef.current !== currentRequestId) return;

        const options: TrainOption[] = (trains ?? []).map((train) => ({
          value: train.train_number,
          label: `${train.train_number} — ${train.train_name}`,
        }));

        setTrainOptions(options);

        if (options.length === 0) {
          setTrainMessage('No trains found for this route and date.');
        }
      } catch {
        if (requestIdRef.current !== currentRequestId) return;
        setTrainOptions([]);
        setTrainMessage('Unable to load trains right now. Please try again.');
      } finally {
        if (requestIdRef.current === currentRequestId) {
          setTrainLoading(false);
        }
      }
    }, 250);

    return () => window.clearTimeout(timer);
  }, [open, sourceValue, destinationValue, travelDateValue]);

  const errors = useMemo(() => {
    const next: Partial<Record<keyof BookingFormValues, string>> = {};

    if (!values.source.trim()) next.source = 'Source is required.';
    if (!values.destination.trim()) next.destination = 'Destination is required.';
    if (!values.travel_date.trim()) next.travel_date = 'Travel date is required.';
    if (!values.travel_class.trim()) next.travel_class = 'Travel class is required.';
    if (!values.train_number.trim()) next.train_number = 'Train selection is required.';

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

      <div className="absolute right-0 top-0 h-full w-full max-w-xl overflow-y-auto border-l border-slate-200 bg-white shadow-2xl dark:border-slate-800 dark:bg-slate-950">
        <div className="flex items-center justify-between border-b border-slate-200 px-6 py-4 dark:border-slate-800">
          <div>
            <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Booking form</h2>
            <p className="text-sm text-slate-500 dark:text-slate-400">
              Review and complete the details from chat.
            </p>
          </div>

          <button
            type="button"
            className="rounded-md px-3 py-2 text-sm text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
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
              className={fieldClass(showError('source'))}
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
              className={fieldClass(showError('destination'))}
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
              className={fieldClass(showError('travel_date'))}
            />
          </Field>

          <Field
            label="Travel class"
            required
            error={showError('travel_class') ? errors.travel_class : undefined}
          >
            <select
              value={values.travel_class}
              onChange={(e) => setField('travel_class', e.target.value)}
              onBlur={() => setTouched((prev) => ({ ...prev, travel_class: true }))}
              className={selectClass(showError('travel_class'))}
            >
              {TRAVEL_CLASS_OPTIONS.map((option) => (
                <option key={option.value || 'placeholder'} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </Field>

          <Field
            label="Train selection"
            required
            error={showError('train_number') ? errors.train_number : undefined}
          >
            {trainOptions.length > 0 ? (
              <select
                value={values.train_number}
                onChange={(e) => setField('train_number', e.target.value)}
                onBlur={() => setTouched((prev) => ({ ...prev, train_number: true }))}
                className={selectClass(showError('train_number'))}
                disabled={trainLoading}
              >
                <option value="">{trainLoading ? 'Searching trains...' : 'Select a train'}</option>
                {trainOptions.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            ) : (
              <input
                value={values.train_number}
                onChange={(e) => setField('train_number', e.target.value)}
                onBlur={() => setTouched((prev) => ({ ...prev, train_number: true }))}
                className={fieldClass(showError('train_number'))}
                placeholder={trainLoading ? 'Searching trains...' : 'Enter train number manually'}
                disabled={trainLoading}
              />
            )}
            <div className="pt-2 text-sm">
              {trainLoading ? (
                <p className="text-slate-500 dark:text-slate-400">Searching trains...</p>
              ) : trainMessage ? (
                <p className="text-slate-500 dark:text-slate-400">{trainMessage}</p>
              ) : null}
            </div>
          </Field>

          <Field
            label="Passenger count"
            required
            error={showError('passenger_count') ? errors.passenger_count : undefined}
          >
            <select
              value={values.passenger_count}
              onChange={(e) => setField('passenger_count', e.target.value)}
              onBlur={() => setTouched((prev) => ({ ...prev, passenger_count: true }))}
              className={selectClass(showError('passenger_count'))}
            >
              <option value="">Select passengers</option>
              {PASSENGER_COUNT_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </Field>

          <details className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 dark:border-slate-800 dark:bg-slate-900/40">
            <summary className="cursor-pointer text-sm font-medium text-slate-700 dark:text-slate-200">
              Optional details
            </summary>

            <div className="mt-4 space-y-5">
              <Field label="Time preference">
                <input
                  value={values.time_preference}
                  onChange={(e) => setField('time_preference', e.target.value)}
                  className={fieldClass(false)}
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
                  className={fieldClass(showError('budget'))}
                  placeholder="Optional"
                />
              </Field>

              <label className="flex items-center gap-3 text-sm text-slate-700 dark:text-slate-200">
                <input
                  type="checkbox"
                  checked={values.direct_only}
                  onChange={(e) => setField('direct_only', e.target.checked)}
                  className="h-4 w-4 rounded border-slate-300 text-slate-900 focus:ring-slate-300 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100 dark:focus:ring-slate-700"
                />
                Direct trains only
              </label>
            </div>
          </details>

          <div className="flex items-center justify-end gap-3 border-t border-slate-200 pt-4 dark:border-slate-800">
            <button
              type="button"
              className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:text-slate-200 dark:hover:bg-slate-800"
              onClick={() => onOpenChange(false)}
            >
              Cancel
            </button>

            <button
              type="submit"
              disabled={!onSubmit || !isValid || submitting}
              className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50 dark:bg-slate-100 dark:text-slate-900 flex items-center gap-2"
            >
              {submitting ? (
                <>
                  <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                  </svg>
                  Confirming...
                </>
              ) : (
                'Submit booking'
              )}
            </button>
          </div>

          {!onSubmit ? (
            <p className="text-sm text-slate-500 dark:text-slate-400">
              This drawer is ready for a submit handler from the parent component.
            </p>
          ) : null}
        </form>
      </div>
    </div>
  );
}
