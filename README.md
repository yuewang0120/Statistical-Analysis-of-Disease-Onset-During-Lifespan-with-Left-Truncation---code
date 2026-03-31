
# Reproducing Figures from the Paper

## Main Text Figures

To reproduce the figures in the **main text**, follow the instructions below.

### Figure 1

In `simulation_F1s.py`, modify the settings as follows and run the script:

```python
from utils import Design1Dataset as dataset
h = 4
n = 1000
```

### Figure 2

- **Panels (a) and (e):**  
  Use the same settings above and run:

  ```bash
  python simulation_F1t.py
  ```

- **Remaining panels:**  
  Use the same settings above and run:

  ```bash
  python simulation_F2s.py
  ```

### Figures 3 and 4

> **Note:** Figures 3 and 4 are based on real-world data that is **not publicly available**, so the corresponding code is **not released**.

---

## Supplementary Material Figures

### Figure 1 (Supplement)

In `simulation_F1s.py`, modify the settings as follows and run the script:

```python
from utils import Design1Dataset as dataset
h = 3
n = 2000
```

### Figure 2 (Supplement)

- **Panels (a) and (e):**  
Use the same settings above and run:

  ```bash
  python simulation_F1t.py
  ```

- **Remaining panels:**  
Use the same settings above and run:

  ```bash
  python simulation_F2s.py
  ```

---

### Figure 3 (Supplement)

In `simulation_F1s.py`, modify the settings as follows and run the script:

```python
from utils import Design2Dataset as dataset
h = 2.6
n = 1000
```

### Figure 4 (Supplement)

- **Panels (a) and (e):**  
Use the same settings above and run:

  ```bash
  python simulation_F1t.py
  ```

- **Remaining panels:**  
Use the same settings above and run:

  ```bash
  python simulation_F2s.py
  ```

---

### Figure 5 (Supplement)

In `simulation_F1s.py`, modify the settings as follows and run the script:

```python
from utils import Design3Dataset as dataset
h = 2.3
n = 1000
```

### Figure 6 (Supplement)

- **Panels (a) and (e):**  
Use the same settings above and run:

  ```bash
  python simulation_F1t.py
  ```

- **Remaining panels:**  
Use the same settings above and run:

  ```bash
  python simulation_F2s.py
  ```

---

## Bandwidth Selection

To select the best bandwidth:

1. Choose the desired dataset design and sample size in `bandwidth_selection.py`.
2. Run the script to evaluate and select an appropriate `h` value.
