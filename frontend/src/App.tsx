import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Forecasting from "./pages/Forecasting";
import Analytics from "./pages/Analytics";
import ModelPerformance from "./pages/ModelPerformance";
import MlopsMonitoring from "./pages/MlopsMonitoring";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="forecasting" element={<Forecasting />} />
        <Route path="analytics" element={<Analytics />} />
        <Route path="model-performance" element={<ModelPerformance />} />
        <Route path="mlops-monitoring" element={<MlopsMonitoring />} />
      </Route>
    </Routes>
  );
}
