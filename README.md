# lila-for-pycbc
Welcome! This is an active repo containing the progress for lila_for_pycbc, an addition to PyCBC's Detector class that will aim to handle detector-frame simulations for the proposed LILA detector, and will be written with modularity to support other next-generation detectors. Updates to come soon!

Please see func_draft.py for a snapshot into the latest functions I've written for the project [and please ignore my little incomplete sanity checks at the bottom lol].

## abstract
In 1916, along with his formulation of general relativity, Albert Einstein predicted the existence of gravitational waves that carry energy in the form of radiation, akin to electromagnetic radiation. These waves propagate from high-mass, high-acceleration systems like binary black hole (BBH) and binary neutron star (BNS) systems and contract or expand the distance between two objects in space.

The Laser Interferometer Lunar Antenna (LILA) is a proposed gravitational wave detector that would be stationed on the lunar surface. As opposed to current Earth-based detectors like LIGO, this next-generation detector would cover the decihertz frequency band and in turn provide crucial insight into the early evolution of BNS mergers.

Initially I ran a comparison of LILA against Cosmic Explorer, a proposed next-generation Earth-based detector, using provided noise curves and randomly sampled populations. From there I scoped the time evolution of the Earth-Moon system in solar system barycentric coordinates to determine the best approach for later calculations. Currently, I am developing a new class for the PyCBC codebase for handling simulated LILA detector responses. The main purpose of this class is to efficiently simulate how LILA would respond to a random population of systems, which would in turn provide insight into how effectively LILA could relay data about early BNS inspiral evolutions. Along with this, the way the class is being written aims to generalize PyCBC’s approach for emulating detectors so that the code can be easily manipulated for several other space-based use cases.


