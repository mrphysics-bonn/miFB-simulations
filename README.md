# Modified interleaved flyback (miFB) simulations
## Supporting Jupyter Notebook for "Echo-planar imaging-based time-of-flight imaging using a modified interleaved flyback trajectory"
Simon Blömer (1,2), Rüdiger Stirnberg (1), Tony Stöcker (1,2)

German Center for Neurodegenerative Diseases (DZNE), Bonn, Germany
Department of Physics and Astronomy, University of Bonn, Bonn, Germany
Magnetic Resonance in Medicine, DATE

DOI: 

## Usage 
This Git repository contains a Python-based simulation of flow effects in Echo Planar Imaging (EPI) and our proposed modified interleaved flyback (miFB) method.

Run the jupyter notebook and use the class **EPISimulator** to simulate a standard EPI sequence and the **FlybackSimulator** for the miFB approach. A preset for the 0.6 mm in-plane isotropic resolution example is provided. 

The following figures are produced:
- Simulated first-order moments from flow along the x-axis for the acquired kinitial, krepeated, and shuffled kcomp and kuncomp k-spaces. 
- Combined projection of the simulated moments along the y-axis. 
- Phase evolution for flow along the y-axis, demonstrating the effects of different combinations of echo train shifting (ETS) and gradient moment smoothing (GMS). 