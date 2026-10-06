import { useEffect } from "react";
import "./Sidebar.css";

// hidden: page still exists (and can be opened directly) but isn't listed in the menu.
const NAV = [
  { id:"board",     icon:"fa-traffic-light",  label:"Physical Board" },
  { id:"overview",  icon:"fa-th",             label:"Matrix Grid" },
  { id:"map",       icon:"fa-map-marked-alt", label:"Live GIS Map" },
  { id:"emergency", icon:"fa-ambulance",      label:"Green Corridor" },
  { id:"analytics", icon:"fa-chart-line",     label:"Analytics" },
  { id:"incidents", icon:"fa-bell",           label:"Incidents" },
  { id:"livegrid",  icon:"fa-satellite-dish", label:"Live Grid" },
  { id:"cityflow",  icon:"fa-microchip",      label:"City Flow Model" },
  { id:"signalai",  icon:"fa-brain",          label:"Signal AI",        hidden: true },
  { id:"insights",  icon:"fa-diagram-project", label:"Network Insights", hidden: true },
  { id:"webcam",    icon:"fa-video",          label:"Live Webcam" },
];

export default function Sidebar({ open, onClose, currentView, onNav, time, date, stats }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  return (
    <>
      <div className={`sb-backdrop${open ? " open" : ""}`} onClick={onClose} aria-hidden="true" />
      <aside className={`sidebar${open ? " open" : ""}`} aria-hidden={!open} aria-label="Navigation">
        <div className="sb-logo">
          <img className="sb-logo-img" src="/verociti-logo.jpg" alt="VeROCiTI logo" />
          <span className="sb-logo-name">VeRO<span>CiTI</span></span>
          <button className="sb-close-btn" onClick={onClose} aria-label="Close navigation" tabIndex={open ? 0 : -1}>
            <i className="fas fa-xmark" />
          </button>
        </div>

        <div className="sb-status">
          <div className="sb-status-dot" />
          System Online
          <span className="sb-firebase-badge" title="Firebase Real-Time Data Sync Active">
            <i className="fas fa-fire" style={{ color: "#dba53a", marginRight: "4px" }} />
            Firebase Live
          </span>
        </div>

        <nav className="sb-nav">
          {NAV.filter(n => !n.hidden).map(n => (
            <button
              key={n.id}
              data-view={n.id}
              className={`sb-nav-btn${currentView===n.id?" active":""}`}
              onClick={() => onNav(n.id)}
              tabIndex={open ? 0 : -1}
            >
              <i className={`fas ${n.icon}`} />
              <span>{n.label}</span>
              {n.id === "incidents" && stats.criticalCount > 0 && (
                <span className="sb-badge">{stats.criticalCount}</span>
              )}
            </button>
          ))}
        </nav>

        <div className="sb-section">
          <div className="sb-stats">
            <div className="sb-stat-row">
              <span className="sb-stat-label">Active nodes</span>
              <span className="sb-stat-val blue">{stats.totalNodes}</span>
            </div>
            <div className="sb-stat-row">
              <span className="sb-stat-label">Critical</span>
              <span className="sb-stat-val red">{stats.criticalCount}</span>
            </div>
            <div className="sb-stat-row">
              <span className="sb-stat-label">Moderate</span>
              <span className="sb-stat-val amber">{stats.mediumCount}</span>
            </div>
            <div className="sb-stat-row">
              <span className="sb-stat-label">Clear</span>
              <span className="sb-stat-val green">{stats.clearCount}</span>
            </div>
          </div>

          <div className="sb-clock">
            <div className="sb-clock-time">{time}</div>
            <div className="sb-clock-date">{date}</div>
            <div className="sb-clock-city">Bhubaneswar, India</div>
          </div>
        </div>
      </aside>
    </>
  );
}
