import React from 'react';
import TrainCard from './TrainCard';

interface Props {
  trains: any[];
}

const SearchResults: React.FC<Props> = ({ trains }) => (
  <div className="mt-6 grid gap-4">
    {trains.map(train => (
      <TrainCard key={train.id} train={train} />
    ))}
  </div>
);

export default SearchResults;