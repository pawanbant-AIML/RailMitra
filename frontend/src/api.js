import axios from "axios";
const apiUrl = import.meta.env.VITE_API_URL;
const baseURL = apiUrl
    ? `${apiUrl.replace(/\/$/, "")}/api/v1`
    : "/api/v1";
const api = axios.create({
    baseURL,
});
export default api;
