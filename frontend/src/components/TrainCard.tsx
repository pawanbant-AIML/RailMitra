import React from 'react';

interface Props {
  train: {
    train_number: string;
    train_name: string;
    source_station_code: string;
    destination_station_code: string;
  };
}

const TrainCard: React.FC<Props> = ({ train }) => (
  <div className="glass p-4 flex justify-between items-center">
    <div>
      <h3 className="text-lg font-semibold">{train.train_name}</h3>
      <p>
        {train.source_station_code} → {train.destination_station_code}
      </p>
    </div>
    <div className="text-primary font-bold">{train.train_number}</div>
  </div>
);

export default TrainCard;