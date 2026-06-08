import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
const TrainCard = ({ train }) => (_jsxs("div", { className: "glass p-4 flex justify-between items-center", children: [_jsxs("div", { children: [_jsx("h3", { className: "text-lg font-semibold", children: train.train_name }), _jsxs("p", { children: [train.source_station_code, " \u2192 ", train.destination_station_code] })] }), _jsx("div", { className: "text-primary font-bold", children: train.train_number })] }));
export default TrainCard;
