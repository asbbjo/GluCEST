# Glutamate weighted protocol according to https://doi.org/10.1002/mrm.27362
# Pipeline from https://github.com/kherz/pulseq-cest-library/tree/master/ 
#
# Tested with pypulseq version 1.3.1post1, bmctool version 0.6.0, and numpy 1.23.5
#
# Asbjørn Bjørkkjær 2025
# asbjorn.bjorkkjar@ntnu.no

from pathlib import Path

import numpy as np
import pypulseq as pp
from bmctool.utils.seq.write import write_seq

# get id of generation file
seqid = Path(__file__).stem + "_id_for_your_file"

# get folder of generation file
folder = Path(__file__).parent

# general settings
AUTHOR = "Asbjorn Bjorkkjar"
FLAG_PLOT_SEQUENCE = False  # plot preparation block?
FLAG_CHECK_TIMING = False  # perform a timing check at the end of the sequence?
FLAG_POST_PREP_SPOIL = True  # add spoiler after preparation block?

# sequence definitions
defs: dict = {}
defs["b1pa"] = 3  # B1 peak amplitude [µT] (b1rms calculated below)
defs["b1rms"] = defs["b1pa"]  # B1 RMS amplitude [µT]
defs["b0"] = 7  # B0 [T]
defs["n_pulses"] = 60  # number of pulses 
defs["tp"] = 50e-3  # pulse duration [s]
defs["td"] = 0.01e-3  # interpulse delay [s]
defs["trec"] = 10  # recovery time [s]
defs["trec_m0"] = 10  # recovery time before M0 [s]
defs["m0_offset"] = -100  # m0 offset [ppm]
defs["offsets_ppm"] = np.append(defs["m0_offset"],np.concatenate(
    [np.arange(-5, 0, 0.2), np.arange(0, 5.2, 0.2)])) # offset vector [ppm])


defs["dcsat"] = (defs["tp"]) / (defs["tp"] + defs["td"])  # duty cycle
defs["num_meas"] = defs["offsets_ppm"].size  # number of repetition
defs["tsat"] = defs["n_pulses"] * (defs["tp"] + defs["td"]) - defs["td"]  # saturation time [s]
defs["seq_id_string"] = seqid  # unique seq id
defs["spoiling"] = "1" if FLAG_POST_PREP_SPOIL else "0"

seq_filename = defs["seq_id_string"] + ".seq"

# scanner limits
sys = pp.Opts(
    max_grad=440,
    grad_unit="mT/m",
    max_slew=3440,
    slew_unit="T/m/s",
    rf_ringdown_time=30e-6,
    rf_dead_time=100e-6,
    rf_raster_time=1e-6,
    gamma=42903623,
)

GAMMA_HZ = sys.gamma * 1e-6
defs["freq"] = defs["b0"] * GAMMA_HZ  # Larmor frequency [Hz]

# ===========
# PREPARATION
# ===========

# spoiler
spoil_amp = 0.8 * sys.max_grad  # Hz/m
rise_time = 1.0e-3  # spoiler rise time in seconds
spoil_dur = 6.5e-3  # complete spoiler duration in seconds

gx_spoil, gy_spoil, gz_spoil = [
    pp.make_trapezoid(channel=c, system=sys, amplitude=spoil_amp, duration=spoil_dur, rise_time=rise_time)
    for c in ["x", "y", "z"]
]

# RF pulses
flip_angle_sat = defs["b1pa"] * GAMMA_HZ * 2 * np.pi * defs["tp"]
sat_pulse = pp.make_block_pulse(flip_angle=flip_angle_sat, duration=defs["tp"], system=sys)

# pseudo ADC event
pseudo_adc = pp.make_adc(num_samples=1, duration=1e-3)

# delays
td_delay = pp.make_delay(defs["td"])
trec_delay = pp.make_delay(defs["trec"])

# Sequence object
seq = pp.Sequence()

# ===
# RUN
# ===

offsets_hz = defs["offsets_ppm"] * defs["freq"]  # convert from ppm to Hz

for m, offset in enumerate(offsets_hz):
    # print progress/offset
    print(f"#{m + 1} / {len(offsets_hz)} : offset {offset / defs['freq']:.2f} ppm ({offset:.3f} Hz)")

    # reset accumulated phase
    accum_phase = 0

    # add delay
    if defs["trec"] > 0:
        seq.add_block(trec_delay)

    # set sat_pulse
    sat_pulse.freq_offset = offset

    for n in range(defs["n_pulses"]):
        sat_pulse.phase_offset = accum_phase % (2 * np.pi)
        seq.add_block(sat_pulse)
        accum_phase = (accum_phase + offset * 2 * np.pi * np.sum(np.abs(sat_pulse.signal) > 0) * 1e-6) % (2 * np.pi)
        if n < defs["n_pulses"] - 1:
            seq.add_block(td_delay)

    if FLAG_POST_PREP_SPOIL:
        seq.add_block(gx_spoil, gy_spoil, gz_spoil)

    seq.add_block(pseudo_adc)

if FLAG_CHECK_TIMING:
    ok, error_report = seq.check_timing()
    if ok:
        print("\nTiming check passed successfully")
    else:
        print("\nTiming check failed! Error listing follows\n")
        print(error_report)

write_seq(seq=seq, seq_defs=defs, filename=folder / seq_filename, author=AUTHOR, use_matlab_names=True)

# plot the sequence
if FLAG_PLOT_SEQUENCE:
    seq.plot()  # to plot all offsets, remove time_range argument