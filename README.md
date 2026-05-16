# AI Assistant Instructions (GitHub Copilot / Cursor / Coding Agents)

## Project Goal

Build an AI-powered running analytics platform that processes Strava GPX files, visualizes routes as pace heatmaps, and generates machine learning insights from historical running data.

Primary objectives:

1. Parse GPX activity data
2. Calculate pace and movement metrics
3. Generate interactive route visualizations
4. Build predictive ML features
5. Keep everything local-first and free

---

## Development Rules

When generating code:

- Follow existing project structure
- Reuse existing modules before creating new files
- Keep code modular and readable
- Avoid unnecessary complexity
- Prefer maintainable solutions over clever solutions
- Add type hints whenever appropriate
- Avoid duplicate logic

Do not:

- Hardcode values
- Introduce paid APIs
- Add unnecessary dependencies
- mix parsing, visualization, and ML logic in one file

---

## Project Architecture

Maintain this structure:

```bash
src/
├── parser.py
├── pace_calculator.py
├── map_visualizer.py
├── ml_model.py
└── utils.py
```

Responsibilities:

### parser.py

Responsible only for:

- loading GPX files
- extracting coordinates
- extracting timestamps
- extracting elevation data

### pace_calculator.py

Responsible only for:

- distance calculations
- speed calculations
- pace calculations
- moving averages
- smoothing noisy GPS data

### map_visualizer.py

Responsible only for:

- Folium maps
- route rendering
- colour scaling
- pace heatmaps

### ml_model.py

Responsible only for:

- preprocessing
- feature engineering
- model training
- prediction
- evaluation

### utils.py

Responsible only for:

- helper functions
- reusable calculations
- formatting utilities

---

## Coding Standards

Generate:

- descriptive variable names
- docstrings
- comments only where logic is non-obvious
- reusable functions

Prefer:

```python
def calculate_pace(
    distance_m: float,
    time_seconds: float
) -> float:
```

instead of:

```python
def calc(x,y):
```

---

## Testing Requirements

For all new features:

- include example usage
- handle edge cases
- validate input data
- handle missing GPX values
- handle invalid timestamps
- prevent divide-by-zero calculations

Examples of edge cases:

- missing GPS points
- duplicate timestamps
- stationary activities
- corrupted GPX files

---

## Machine Learning Rules

Use:

- scikit-learn
- pandas
- numpy

Preferred model order:

1. Linear Regression
2. Random Forest
3. Gradient Boosting
4. XGBoost (optional)

Feature examples:

Inputs:

- distance
- elevation gain
- average pace
- average heart rate
- temperature
- time of day
- fatigue score

Targets:

- future pace
- finish time
- recovery prediction

Always:

- split train/test datasets
- evaluate model performance
- show metrics
- avoid data leakage

---

## Mapping Rules

Use:

- Folium
- OpenStreetMap only

Colour scale:

- Blue → slower pace
- Green → moderate pace
- Red → faster pace

Maps should:

- auto-center on route
- include tooltips
- support multiple activities
- export as HTML

---

## Workflow for New Features

Before generating code:

1. Read relevant existing files
2. Understand dependencies
3. Identify reusable components
4. Modify minimally

When adding a feature:

1. Implement core functionality
2. Add validation
3. Add example usage
4. Add comments where necessary

---

## Future Expansion Direction

Future additions should prioritize:

- heart-rate analytics
- recovery scoring
- VO2 estimation
- anomaly detection
- AI coaching
- dashboard interfaces
- wearable integrations

Keep all future additions modular and backward compatible.