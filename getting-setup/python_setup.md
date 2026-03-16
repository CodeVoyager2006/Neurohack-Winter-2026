## Foreword:
This setup guide is **NOT REQUIRED.** You may choose to use the provided SURGE computers which come pre-configured with the necessary software. However, if you prefer to set up your own environment on a personal computer, please follow the instructions below.

# Python Setup
This document will guide you through setting up the Python environment for the BrainHACK 2025 hackathon. There are several ways to achieve this:
1. (**Highly Recommended**) Using the provided `brainhack_env.yml` file to create a conda environment with Miniforge.
   - If you have trouble when installing other packages (i.e., those not included by default in the environment), you can try using the `compatibility_brainhack_env.yml` file instead, which uses Python 3.10 instead of 3.13.
2. Creating a virtual environment with `venv` and installing the required packages manually.

This document will only cover the first method, which is the most straightforward and ensures compatibility with the provided starter code and resources.

Keep in mind, **not everyone on your team has to, or even should set up the environment.** Generally, each team should have one or two members who are responsible for setting up the environment and ensuring that the code runs smoothly.

If you don't already have a code editor, we recommend [VSCode](https://code.visualstudio.com/), which is free and works well with Python.

---

# 1. Installing the `Brainhack` Environment with Miniforge

<details>
<summary>Information about packages in the Brainhack environment</summary>

#### `Brainhack_env.yaml` includes:
- Python 3.13: The programming language used for the scripts and notebooks.
- `scipy`: A Python library used for scientific and technical computing.
- `jupyterlab`: An interactive development environment for working with notebooks, code, and data
- `mne`: A Python package for processing and analyzing electrophysiological data, such as EEG and MEG.
- `brainflow`: A library for interfacing with various biosensors and brain-computer interface (BCI) devices.
- `pyserial`: A Python library that encapsulates access to serial ports, useful for communication with hardware devices.

#### `Compatibility_Brainhack_env.yaml` includes:
- Python 3.10: The programming language used for the scripts and notebooks.
- `scipy`: A Python library used for scientific and technical computing.
- `jupyterlab`: An interactive development environment for working with notebooks, code, and data
- `mne`: A Python package for processing and analyzing electrophysiological data, such as EEG and MEG.
- `brainflow`: A library for interfacing with various biosensors and brain-computer interface (BCI) devices.
- `pyserial`: A Python library that encapsulates access to serial ports, useful for communication with hardware devices.

</details>

## Pre-requisites
This guide assumes you have already cloned the repository, or at least have downloaded the `brainhack_env.yml` file from the repository.

### **1️⃣ Install Miniforge**
Miniforge is a minimal conda installer that provides a lightweight environment for Python packages. Download the Miniforge installer for your operating system from the [Miniforge GitHub Releases.](https://conda-forge.org/download/)

### **2️⃣ Create the `Brainhack` Environment**
After installing Miniforge, open a terminal and navigate to the directory with the `brainhack_env.yaml` file. Run the following command to create a new conda environment named `Brainhack` with the necessary packages and Python 3.13:
```bash
conda env create -f brainhack_env.yaml
```
There will be lots of output, and you will eventually be prompted to confirm. Type `y` and press `Enter` to proceed. Package downloading will behin - this process may take a few minutes to complete.

### **3️⃣ Verify Installation and Activate the Environment**
To verify that the environment was created successfully, run:
```bash
conda env list
```
You should see `Brainhack` in the list of environments. Once the environment is created, activate it with:
```bash
conda activate Brainhack
```

You can now use the `Brainhack` environment to run the provided scripts and notebooks in the repository.

## Installing Additional Packages
If you need to install additional packages, you can do so using `conda` or `pip` within the `Brainhack` environment. For example:
```bash
conda activate Brainhack
conda install [package-name]
```

---

# 2. Alternative Method: Using `venv` and `pip`
If you prefer not to use conda, you can create a virtual environment using Python's built-in `venv` module and install the required packages manually. **We can't guarantee compatibility with this method.**

### **1️⃣ Create a Virtual Environment**
Navigate to your project directory and run:
```bash
python3 -m venv brainhack_env
```

### **2️⃣ Activate the Virtual Environment**
- On Windows:
```bash
brainhack_env\Scripts\activate
```
- On macOS/Linux:
```bash
source brainhack_env/bin/activate
```
### **3️⃣ Install Required Packages**
You can install the required packages using `pip`. 
```
pip install scipy jupyerlab mne brainflow pyserial
```
You can manually install other packages you need using:
```bash
pip install [package-name]
```

---

# Troubleshooting

### `conda: command not found` after installing Miniforge
Close and reopen your terminal. If the issue persists, initialize conda manually and restart:
```bash
~/miniforge3/bin/conda init
```
On Windows, use the **Miniforge Prompt** instead of a standard terminal.

### `conda activate Brainhack` has no effect on Windows
Open **Miniforge Prompt** (installed alongside Miniforge) rather than Command Prompt or PowerShell. Alternatively, initialize conda for PowerShell and restart:
```bash
conda init powershell
```

### Environment creation fails with dependency or solver errors
Try the compatibility file, which uses the older Python 3.10 and has broader package compatibility:
```bash
conda env create -f compatibility_brainhack_env.yaml
```

### `import mne` or `import brainflow` fails
Make sure the environment is active before launching Jupyter or running scripts:
```bash
conda activate Brainhack
```
Verify the package is present, then reinstall if missing:
```bash
conda list mne
conda install mne
```

### Serial port permission denied on Linux (BrainFlow / pyserial)
Add your user to the `dialout` group, then log out and back in:
```bash
sudo usermod -aG dialout $USER
```

### Jupyter kernel doesn't show the `Brainhack` environment
Register the environment as a Jupyter kernel:
```bash
conda activate Brainhack
python -m ipykernel install --user --name Brainhack --display-name "Python (Brainhack)"
```