## -*- coding: utf-8 -*-
<%doc>
Shared fragment: LEMS Dimensions & Units
=========================================
Defines all dimensions and units inline so the LEMS file is fully
self-contained — no <Include file="Simulation.xml"/> or similar.
This avoids the jNeuroML double-read bug that occurs when
external NeuroML type files are included.
</%doc>
  <!-- Dimensions -->
  <Dimension name="none"/>
  <Dimension name="time" t="1"/>
  <Dimension name="voltage" m="1" l="2" t="-3" i="-1"/>
  <Dimension name="per_time" t="-1"/>
  <Dimension name="conductance" m="-1" l="-2" t="3" i="2"/>
  <Dimension name="capacitance" m="-1" l="-2" t="4" i="2"/>
  <Dimension name="current" i="1"/>
  <Dimension name="resistance" m="1" l="2" t="-3" i="-2"/>
  <Dimension name="concentration" l="-3" n="1"/>
  <Dimension name="substance" n="1"/>
  <Dimension name="charge" t="1" i="1"/>
  <Dimension name="temperature" k="1"/>

  <!-- Units -->
  <Unit symbol="s" dimension="time" power="0"/>
  <Unit symbol="ms" dimension="time" power="-3"/>
  <Unit symbol="us" dimension="time" power="-6"/>
  <Unit symbol="V" dimension="voltage" power="0"/>
  <Unit symbol="mV" dimension="voltage" power="-3"/>
  <Unit symbol="A" dimension="current" power="0"/>
  <Unit symbol="mA" dimension="current" power="-3"/>
  <Unit symbol="nA" dimension="current" power="-9"/>
  <Unit symbol="pA" dimension="current" power="-12"/>
  <Unit symbol="S" dimension="conductance" power="0"/>
  <Unit symbol="mS" dimension="conductance" power="-3"/>
  <Unit symbol="nS" dimension="conductance" power="-9"/>
  <Unit symbol="F" dimension="capacitance" power="0"/>
  <Unit symbol="uF" dimension="capacitance" power="-6"/>
  <Unit symbol="nF" dimension="capacitance" power="-9"/>
  <Unit symbol="ohm" dimension="resistance" power="0"/>
  <Unit symbol="Mohm" dimension="resistance" power="6"/>
  <Unit symbol="per_s" dimension="per_time" power="0"/>
  <Unit symbol="per_ms" dimension="per_time" power="3"/>
  <Unit symbol="degC" dimension="temperature" offset="273.15"/>
  <Unit symbol="K" dimension="temperature" power="0"/>
