# mixed-mTRF
Work in progress. This repository documents my research project at the 
Neural Dynamics Lab (INI, University of Zurich), supervised by Dr. Tim Piox.

## What this is about
The standard **multivariate Temporal Response Function (mTRF)** is a powerful 
tool for modelling how the brain responds to continuous speech — but it makes 
a fundamental *assumption*: the response has the same shape and timing for every 
word, *every time*. Only the amplitude scales.
That's probably wrong. We know from the N400 literature that predictable words 
are processed faster — the response peaks earlier, not just smaller. Lalor 
(2024) showed this directly using a **Dynamic TRF framework** that lets the TRF 
shape deform word-by-word as a *function of surprisal*. Elegant, but it requires 
*committing to a parametric form* (a Gaussian template + a shallow linear net) 
and *doesn't* give you uncertainty over the TRF shape itself.
The **hypothesis** here is that you can achieve the same thing — and do it more 
principally — by combining the mTRF with a Bayesian framework, specifically 
Gaussian Process priors. Instead of a single fixed TRF, we should get a posterior 
distribution over TRF shapes that can naturally vary with context. No need to 
hard-code the template. No need to assume the relationship between surprisal 
and latency is linear. And we should get honest uncertainty at every lag.

## Context
This sits within a broader question: how does visual speech information 
modulate neural responses over and above the acoustic signal, and does visual 
predictability shift the timing of audiovisual integration the way lexical 
surprisal shifts the N400?

## Literature
Papers I'm reading for this project are tracked here: [[Google Sheets link](https://docs.google.com/spreadsheets/d/1s1F4kjrW1S360toRWma7SZhZf7wZuIoeAkWX1L1ZRWU/edit?usp=sharing)]
