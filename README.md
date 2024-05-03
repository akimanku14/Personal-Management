<div align="center"><h1>PersonnelManagement Project Implementation</h1></div>

<h3>The virtual environment:</h3>

**1. Create and activate a virtual environment called *PerMan***

```bash
conda create --name PerMan python=3.10
conda activate PerMan
```

**2. Install the requirements**

```bash
pip install -r requirements.txt
```

**3. Register Your Environment as a Kernel:**

*After installing the packages via the requirements.txt, you still need to register your environment as a kernel manually using the command line as this cannot be done via the requirements file:*

```bash
python -m ipykernel install --user --name PerMan --display-name "Python (PerMan)"
```

**4. Launch Jupyter Notebook:**

*After installing the packages and setting up the kernel, you can start Jupyter Notebook:*

```bash
jupyter notebook
```

> <strong>CAUTION:</strong> Select <em>Python (PerMan)</em> as your kernel in the notebook.

