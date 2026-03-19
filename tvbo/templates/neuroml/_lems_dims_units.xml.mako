## -*- coding: utf-8 -*-
<%doc>
Shared fragment: LEMS Dimensions & Units
=========================================
Standard NeuroML dimension and unit declarations used by all LEMS files.
Included via <%include file="_lems_dims_units.xml.mako"/> from parent templates.
</%doc>
  <!-- ════════════════════════════════════════════════════════════════
       Dimensions & Units
       ════════════════════════════════════════════════════════════════ -->
  <Dimension name="voltage"      m="1"  l="2"  t="-3" i="-1"/>
  <Dimension name="time"                        t="1"/>
  <Dimension name="per_time"                    t="-1"/>
  <Dimension name="current"                              i="1"/>
  <Dimension name="conductance"  m="-1" l="-2" t="3"  i="2"/>
  <Dimension name="capacitance"  m="-1" l="-2" t="4"  i="2"/>
  <Dimension name="resistance"   m="1"  l="2"  t="-3" i="-2"/>
  <Dimension name="concentration"       l="-3"               j="1"/>
  <Dimension name="length"              l="1"/>
  <Dimension name="none"/>

  <Unit name="second"      symbol="s"    dimension="time"        power="0"/>
  <Unit name="milliSecond" symbol="ms"   dimension="time"        power="-3"/>
  <Unit name="milliVolt"   symbol="mV"   dimension="voltage"     power="-3"/>
  <Unit name="volt"        symbol="V"    dimension="voltage"     power="0"/>
  <Unit name="milliAmpere" symbol="mA"   dimension="current"     power="-3"/>
  <Unit name="nanoAmpere"  symbol="nA"   dimension="current"     power="-9"/>
  <Unit name="picoAmpere"  symbol="pA"   dimension="current"     power="-12"/>
  <Unit name="siemens"     symbol="S"    dimension="conductance" power="0"/>
  <Unit name="milliSiemens" symbol="mS"  dimension="conductance" power="-3"/>
  <Unit name="nanoSiemens" symbol="nS"   dimension="conductance" power="-9"/>
  <Unit name="microFarad"  symbol="uF"   dimension="capacitance" power="-6"/>
  <Unit name="nanoFarad"   symbol="nF"   dimension="capacitance" power="-9"/>
  <Unit name="picoFarad"   symbol="pF"   dimension="capacitance" power="-12"/>
  <Unit name="ohm"         symbol="ohm"  dimension="resistance"  power="0"/>
  <Unit name="per_second"  symbol="per_s" dimension="per_time"   power="0"/>
  <Unit name="hertz"       symbol="Hz"   dimension="per_time"    power="0"/>
  <Unit name="metre"       symbol="m"    dimension="length"      power="0"/>
  <Unit name="centimetre"  symbol="cm"   dimension="length"      power="-2"/>
  <Unit name="micrometre"  symbol="um"   dimension="length"      power="-6"/>
