# BCI Basics

## What is a Brain-Computer Interface (BCI)?
A **Brain-Computer Interface (BCI)** allows users to interact with a computer or device using brain activity. Electroencephalography (EEG)-based BCIs measure electrical signals from the scalp and translate them into commands for applications like assistive devices, games, and neurofeedback. 

## Key EEG Features Used in BCIs  
BCI systems rely on various EEG signals/features to interpret brain activity:
- **Event-Related Potentials (ERP)** – Brain responses to specific stimuli (e.g., P300 for spellers).
- **Steady-State Visual Evoked Potentials (SSVEP)** – Brain responses to flickering stimuli at known frequencies.  
- **Motor Imagery (MI)** – Brain patterns associated with imagining movement.

## Typical BCI EEG Signal Pipeline 
BCI systems follow these key steps to convert **raw brain activity into meaningful actions**: 

1. **Signal Acquisition** – EEG electrodes record brain activity.  
2. **Preprocessing** – Noise and artifacts (e.g., eye blinks) are removed through filtering and artifact rejection.
    - Filtering: Band-pass, notch, or spatial filters
    - Artifact Removal: Independent-component analysis (ICA), etc.
    - Epoching/Windowing: Segmentation/epoching around events or stimuli.
    - Baseline Correction: Normalize signals to a reference period
3. **Feature Extraction** – Important signal patterns are identified (e.g., channels, time-points, frequency power, ERP responses).
4. **Classification** – Machine learning or rule-based algorithms map features to specific commands.
5. **Application Control** – The classified signal triggers an action, such as selecting a letter in a speller or moving a game character.  

## Articles and Resources
- **[Article - SSVEP, MI, and P300 BCIs](https://doi.org/10.1093/gigascience/giz002)** - A good primer for different BCI paradigms 
- **[Article - SSVEP Spellers](https://pmc.ncbi.nlm.nih.gov/articles/PMC8065759/)** - In-depth review of SSVEP-based (and hybrid P300 & SSVEP) spellers
- **[Article - P300 Spellers](https://doi.org/10.1111/psyp.13569)** - Great review of the P300 ERP for use in BCIs


---