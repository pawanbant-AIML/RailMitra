import { jsx as _jsx } from "react/jsx-runtime";
import TrainCard from './TrainCard';
const SearchResults = ({ trains }) => (_jsx("div", { className: "mt-6 grid gap-4", children: trains.map(train => (_jsx(TrainCard, { train: train }, train.id))) }));
export default SearchResults;
