import axios from "axios";

const baseURL = (import.meta.env.VITE_API_URL || "https://railmitra-cazc.onrender.com") + "/api/v1";

const api = axios.create({
  baseURL,
});

export default api;
