# lila-for-pycbc
Welcome! This is an active repo containing the progress for lila_for_pycbc, an addition to PyCBC's Detector class that will aim to handle detector-frame simulations for the proposed LILA detector, and will be written with modularity to support other next-generation detectors. Updates to come soon!

Please see func_draft.py for a snapshot into the latest functions I've written for the project [and please ignore my little incomplete sanity checks at the bottom lol].

## abstract
In 1916, along with his formulation of general relativity, Albert Einstein predicted the existence of gravitational waves, which carry energy in a form of radiation akin to electromagnetic radiation. These waves propagate from high-mass high-acceleration systems like binary black hole (BBH) and binary neutron star (BNS) systems and warp our perceived distance between two local points in an effect called strain.

The Laser Interferometer Lunar Antenna (LILA) is a proposed gravitational wave detector that would be stationed on the lunar surface. As opposed to current Earth-based detectors like LIGO, this third-generation detector would cover the decihertz frequency band and in turn provide crucial insight into the early evolution of BNS mergers.

So far, this analysis of LILA has been explored in three crucial steps. Initially it ran a comparison of LILA against Cosmic Explorer, a proposed third-generation Earth-based detector, based on provided noise curves and a randomly sampled population. From there it scoped the time evolution of the Earth-Moon system in the solar system barycentric frame and its influence on calculations. Currently, code is being developed to add a new class to the PyCBC codebase for handling simulated detector responses.

The next step for this analysis is to use this class to simulate the LILA detector response to simulated systems. This will allow for the testing of what LILA’s resolution would look like for a variety of systems, how early it can detect them, and what insights it could provide as opposed to other proposals.

