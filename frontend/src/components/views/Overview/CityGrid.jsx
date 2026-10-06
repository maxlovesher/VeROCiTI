import { useEffect, useMemo, useRef, useState } from "react";
import GridCell from "./GridCell";
import "./CityGrid.css";

const STATUS_RANK = { critical: 3, medium: 2, low: 1 };
const MAX_SUGGESTIONS = 8;

function Highlight({ text, query }) {
  const i = query ? text.toLowerCase().indexOf(query.toLowerCase()) : -1;
  if (i < 0) return text;
  return (
    <>
      {text.slice(0, i)}
      <strong>{text.slice(i, i + query.length)}</strong>
      {text.slice(i + query.length)}
    </>
  );
}

function SearchBox({ value, onChange, intersections }) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const inputRef = useRef(null);
  const wrapRef = useRef(null);

  // One suggestion per junction name (the matrix repeats names across nodes).
  const suggestions = useMemo(() => {
    const q = value.trim().toLowerCase();
    if (!q) return [];
    const byName = new Map();
    for (const int of intersections) {
      if (!int.name.toLowerCase().includes(q)) continue;
      const s = byName.get(int.name) || { name: int.name, count: 0, total: 0, status: "low" };
      s.count += 1;
      s.total += int.congestionPct;
      if (STATUS_RANK[int.status] > STATUS_RANK[s.status]) s.status = int.status;
      byName.set(int.name, s);
    }
    return [...byName.values()]
      .sort((a, b) => a.name.toLowerCase().indexOf(q) - b.name.toLowerCase().indexOf(q) || a.name.localeCompare(b.name))
      .slice(0, MAX_SUGGESTIONS);
  }, [value, intersections]);

  // "/" jumps to the search box from anywhere on the page (like most search UIs).
  useEffect(() => {
    const onKey = (e) => {
      const tag = document.activeElement?.tagName;
      if (e.key === "/" && tag !== "INPUT" && tag !== "TEXTAREA") {
        e.preventDefault();
        inputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    const onDown = (e) => { if (!wrapRef.current?.contains(e.target)) setOpen(false); };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, []);

  function pick(name) {
    onChange(name);
    setOpen(false);
    setActive(-1);
    inputRef.current?.blur();
  }

  function onKeyDown(e) {
    if (e.key === "ArrowDown" && suggestions.length) {
      e.preventDefault();
      setOpen(true);
      setActive((a) => (a + 1) % suggestions.length);
    } else if (e.key === "ArrowUp" && suggestions.length) {
      e.preventDefault();
      setActive((a) => (a <= 0 ? suggestions.length - 1 : a - 1));
    } else if (e.key === "Enter") {
      if (open && active >= 0 && suggestions[active]) pick(suggestions[active].name);
      else setOpen(false);
    } else if (e.key === "Escape") {
      if (open) setOpen(false);
      else if (value) onChange("");
      else inputRef.current?.blur();
    }
  }

  const showList = open && value.trim() !== "";

  return (
    <div className={`gsearch${showList ? " open" : ""}`} ref={wrapRef}>
      <i className="fas fa-magnifying-glass gsearch-icon" />
      <input
        ref={inputRef}
        type="text"
        className="gsearch-input"
        placeholder="Search junctions"
        value={value}
        onChange={(e) => { onChange(e.target.value); setOpen(true); setActive(-1); }}
        onFocus={() => setOpen(true)}
        onKeyDown={onKeyDown}
        role="combobox"
        aria-expanded={showList}
        aria-controls="gsearch-list"
        aria-autocomplete="list"
        spellCheck={false}
      />
      {value ? (
        <button className="gsearch-clear" onClick={() => { onChange(""); inputRef.current?.focus(); }} aria-label="Clear search">
          <i className="fas fa-xmark" />
        </button>
      ) : (
        <kbd className="gsearch-kbd" title="Press / to search">/</kbd>
      )}

      {showList && (
        <ul className="gsearch-list" id="gsearch-list" role="listbox">
          {suggestions.length === 0 ? (
            <li className="gsearch-empty">No junction matches &ldquo;{value.trim()}&rdquo;</li>
          ) : (
            suggestions.map((s, i) => (
              <li
                key={s.name}
                role="option"
                aria-selected={i === active}
                className={`gsearch-item${i === active ? " active" : ""}`}
                onMouseEnter={() => setActive(i)}
                onMouseDown={(e) => { e.preventDefault(); pick(s.name); }}
              >
                <i className="fas fa-magnifying-glass gsearch-item-icon" />
                <span className="gsearch-item-name"><Highlight text={s.name} query={value.trim()} /></span>
                <span className={`gsearch-dot ${s.status}`} />
                <span className="gsearch-item-meta">
                  {Math.round(s.total / s.count)}% · {s.count} node{s.count === 1 ? "" : "s"}
                </span>
              </li>
            ))
          )}
        </ul>
      )}
    </div>
  );
}

export default function CityGrid({ intersections, onCellClick }) {
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");

  const query = searchTerm.trim().toLowerCase();
  const filteredIntersections = intersections.filter(int => {
    const matchesSearch = int.name.toLowerCase().includes(query);
    const matchesStatus = statusFilter === "all" || int.status === statusFilter;
    return matchesSearch && matchesStatus;
  });
  const narrowed = query !== "" || statusFilter !== "all";

  function clearAll() {
    setSearchTerm("");
    setStatusFilter("all");
  }

  return (
    <div className="city-grid-wrap">
      <div className="city-grid-header">
        <div className="city-grid-title">
          <i className="fas fa-th" />
          Bhubaneswar Intersection Matrix
          {narrowed ? (
            <span className="grid-result-chip">
              {filteredIntersections.length} of {intersections.length} junctions
              <button onClick={clearAll}>Clear</button>
            </span>
          ) : (
            <span className="grid-count">({intersections.length} nodes)</span>
          )}
        </div>

        <div className="grid-controls">
          <SearchBox value={searchTerm} onChange={setSearchTerm} intersections={intersections} />

          <div className="filter-pills">
            <button
              className={`filter-btn ${statusFilter === "all" ? "active" : ""}`}
              onClick={() => setStatusFilter("all")}
            >
              All
            </button>
            <button
              className={`filter-btn ${statusFilter === "critical" ? "active" : ""}`}
              onClick={() => setStatusFilter("critical")}
              style={statusFilter === "critical" ? { color: "var(--red)" } : {}}
            >
              Critical
            </button>
            <button
              className={`filter-btn ${statusFilter === "medium" ? "active" : ""}`}
              onClick={() => setStatusFilter("medium")}
              style={statusFilter === "medium" ? { color: "var(--amber)" } : {}}
            >
              Moderate
            </button>
            <button
              className={`filter-btn ${statusFilter === "low" ? "active" : ""}`}
              onClick={() => setStatusFilter("low")}
              style={statusFilter === "low" ? { color: "var(--green)" } : {}}
            >
              Clear
            </button>
          </div>
        </div>
      </div>

      <div className="city-grid">
        {filteredIntersections.length === 0 ? (
          <div className="no-results-msg">
            <i className="fas fa-magnifying-glass" />
            <span>
              No junctions match
              {query ? <> &ldquo;{searchTerm.trim()}&rdquo;</> : " this filter"}
            </span>
            <button className="no-results-clear" onClick={clearAll}>Show all junctions</button>
          </div>
        ) : (
          filteredIntersections.map(int => (
            <GridCell key={int.id} intersection={int} onClick={onCellClick} />
          ))
        )}
      </div>
    </div>
  );
}
