"""
Discrete-event simulation of a mixed-model vehicle assembly line.

This is the measurement instrument for the whole project. It is not a toy
stand-in for a dataset we could not get: it is the only substrate on which a
counterfactual ("how many vehicles would have escaped WITHOUT the twin") can
be measured at all, because history only happens once and no real log records
which station was truly the momentary bottleneck.

What is modelled
----------------
part flow, station sequence, finite buffers, blocking, starving, stochastic
service times, mixed-model variants, breakdown/repair, end-of-line inspection,
rework loops, correlated process tags, injected drift.

What is deliberately skipped
----------------------------
CAD geometry, robot kinematics, weld-point physics, thermal/structural
simulation, supply chain upstream of the line. Fidelity is a cost. We spend it
only where it changes a decision inside one shift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import simpy

from .injectors import (
    BottleneckSchedule,
    FixtureDriftInjector,
    TagGenerator,
    default_drift,
    default_schedule,
    defect_probability_from_gap,
)
from .line_config import (
    BASE_DEFECT_RATE,
    DRIFT_STATION,
    EOL_STATION,
    S08_CORRELATION,
    S08_CORRELATION_DEGRADED,
    TAKT_SECONDS,
    VARIANTS,
    Station,
    build_line,
    variant_multiplier,
)
from .schema import EventType, SensorTier, StationState

# ---------------------------------------------------------------------------
# Shared latent causes
# ---------------------------------------------------------------------------
# The brief names the real root causes as "equipment wear, operator variation,
# upstream part quality, environmental conditions" -- and notes they are hard to
# isolate precisely because they are SHARED across stations. A line where every
# station's readings are independent has none of that difficulty, and nothing
# for a soft sensor to infer from a neighbour.
#
# Two latents, both invisible to every model:
#   batch   -- incoming panel quality, drawn per part, shifts geometry tags
#              across the whole body shop at once
#   ambient -- temperature and humidity, an Ornstein-Uhlenbeck walk over the
#              shift, shifts paint tags and weld current together
#
# Loadings preserve total variance: a tag with loading w takes w of its variance
# from the latent and sqrt(1-w^2) from its own noise, so marginal distributions
# are unchanged and the drift scenario stays exactly as hard as before.
BATCH_LOADING = {"gap_left_mm": 0.55, "gap_right_mm": 0.55, "clamp_force_N": 0.35}
AMBIENT_LOADING = {"booth_temp_C": 0.62, "humidity_pct": 0.58,
                   "film_thickness_um": 0.40, "weld_current_A": 0.30}
AMBIENT_THETA = 0.02        # OU mean reversion per part
AMBIENT_SIGMA = 0.22

# Inspection characteristics at end of line. Not perfect -- some defects
# escape to the customer entirely, which is what a warranty claim is.
EOL_SENSITIVITY = 0.93
EOL_FALSE_POSITIVE = 0.008

# Background defect probability contributed by each non-drift station.
PER_STATION_BACKGROUND = 0.00008

REWORK_BAYS = 2
REWORK_MEAN_SECONDS = 3.0 * TAKT_SECONDS


@dataclass
class Part:
    part_id: int
    variant: str
    t_release: float
    is_defective: bool = False
    cause_station: str | None = None
    cause_mechanism: str | None = None
    severity: float = 0.0
    detected_at: float | None = None
    reworked: bool = False


@dataclass
class Recorder:
    events: list[tuple] = field(default_factory=list)
    tags: list[tuple] = field(default_factory=list)
    states: list[tuple] = field(default_factory=list)

    def event(self, t, part_id, station_id, etype, variant):
        self.events.append((float(t), int(part_id), station_id, etype.value, variant))

    def tag(self, t, part_id, station_id, name, value):
        self.tags.append((float(t), int(part_id), station_id, name, float(value)))

    def state(self, station_id, state, t_start, t_end):
        if t_end - t_start > 1e-9:
            self.states.append((station_id, state.value, float(t_start), float(t_end)))


class AssemblyLineSim:
    def __init__(
        self,
        stations: list[Station] | None = None,
        horizon: float = 3 * 8 * 3600.0,
        seed: int = 7,
        schedule: BottleneckSchedule | None = None,
        drift: FixtureDriftInjector | None = None,
        truth_sample_interval: float = 30.0,
        enable_drift: bool = True,
        enable_bottlenecks: bool = True,
        initial_buffers: dict[str, int] | None = None,
        start_time: float = 0.0,
        release_scale: float = 1.0,
    ):
        # release_scale > 1 lengthens the interval between part releases, i.e.
        # slows the line. This is how a plant actually reduces speed: it releases
        # work more slowly, it does not make machines faster.
        self.release_scale = release_scale
        # initial_buffers primes each downstream buffer with N parts already in
        # it. Without this an MPC rollout starts from an empty line and spends
        # its whole horizon filling up, which would make every intervention look
        # identical. start_time offsets the clock so a frozen degradation
        # schedule is evaluated at the right point.
        self.initial_buffers = initial_buffers or {}
        self.start_time = start_time
        self.stations = stations or build_line()
        self.by_id = {s.station_id: s for s in self.stations}
        self.horizon = horizon
        self.seed = seed
        # SPLIT RNG STREAMS. rng_flow drives everything that determines part
        # movement (variant draw, service times, breakdowns). rng_quality drives
        # everything downstream of movement (defect draws, inspection, rework).
        # Keeping them separate is what makes a paired drift-on / drift-off run
        # produce IDENTICAL part flow, so the counterfactual isolates the drift
        # instead of measuring two different realisations of the same line.
        self.rng = np.random.default_rng(seed)          # flow
        self.rng_q = np.random.default_rng(seed + 90001)  # quality
        self.rng_l = np.random.default_rng(seed + 70001)  # shared latents
        self._ambient = 0.0
        self._ambient_last = None
        self._batch: dict[int, float] = {}
        self.truth_sample_interval = truth_sample_interval

        self.schedule = (
            schedule
            if schedule is not None
            else (default_schedule(self.stations, horizon) if enable_bottlenecks
                  else BottleneckSchedule([], self.stations))
        )
        self.drift = (
            drift if drift is not None
            else (default_drift(t_onset=0.55 * horizon) if enable_drift else None)
        )

        self.rec = Recorder()
        self.parts: dict[int, Part] = {}
        self.truth_bottleneck: list[tuple] = []

        # one generator per instrumented station, seeded independently so that
        # turning drift on/off does not perturb any other station's stream
        self.taggen: dict[str, TagGenerator] = {}
        for i, s in enumerate(self.stations):
            if s.tags:
                gen = TagGenerator(
                    s.tags,
                    S08_CORRELATION if s.station_id == DRIFT_STATION else None,
                    rng=np.random.default_rng(seed * 1000 + i),
                )
                if s.station_id == DRIFT_STATION:
                    gen.set_degraded(S08_CORRELATION_DEGRADED)
                self.taggen[s.station_id] = gen

        self.variant_names = list(VARIANTS.keys())
        self.variant_p = np.array([VARIANTS[v]["mix"] for v in self.variant_names])
        self.variant_p = self.variant_p / self.variant_p.sum()

    # ------------------------------------------------------------------
    # service time
    # ------------------------------------------------------------------

    def service_time(self, s: Station, variant: str, t: float) -> float:
        mean = (
            s.mean_service
            * variant_multiplier(variant, s.zone)
            * self.schedule.multiplier(s.station_id, t)
        )
        cv = s.cv
        # lognormal parameterised to the target mean and cv
        sigma = np.sqrt(np.log(1.0 + cv ** 2))
        mu = np.log(mean) - 0.5 * sigma ** 2
        return float(self.rng.lognormal(mu, sigma))

    # ------------------------------------------------------------------
    # tags and defect assignment
    # ------------------------------------------------------------------

    def ambient_at(self, t: float) -> float:
        """Ornstein-Uhlenbeck walk, advanced lazily to time t."""
        if self._ambient_last is None:
            self._ambient_last = t
        steps = int(max(0, (t - self._ambient_last) // TAKT_SECONDS))
        for _ in range(min(steps, 500)):
            self._ambient += (-AMBIENT_THETA * self._ambient
                              + AMBIENT_SIGMA * self.rng_l.standard_normal())
        if steps:
            self._ambient_last = t
        return float(self._ambient)

    def batch_of(self, part: Part) -> float:
        if part.part_id not in self._batch:
            self._batch[part.part_id] = float(self.rng_l.standard_normal())
        return self._batch[part.part_id]

    def latent_shift(self, s: Station, part: Part, t: float) -> np.ndarray:
        """Per-tag latent contribution, in sigma units."""
        batch, ambient = self.batch_of(part), self.ambient_at(t)
        out = []
        for tag in s.tags:
            suffix = tag.name.split("_", 1)[1] if "_" in tag.name else tag.name
            w_b = BATCH_LOADING.get(suffix, 0.0) if s.zone == "body" else 0.0
            w_a = AMBIENT_LOADING.get(suffix, 0.0)
            out.append(w_b * batch + w_a * ambient)
        return np.array(out)

    def latent_scale(self, s: Station) -> np.ndarray:
        """sqrt(1 - w^2): how much variance the tag keeps for itself."""
        out = []
        for tag in s.tags:
            suffix = tag.name.split("_", 1)[1] if "_" in tag.name else tag.name
            w_b = BATCH_LOADING.get(suffix, 0.0) if s.zone == "body" else 0.0
            w_a = AMBIENT_LOADING.get(suffix, 0.0)
            w2 = min(w_b ** 2 + w_a ** 2, 0.95)
            out.append(float(np.sqrt(1.0 - w2)))
        return np.array(out)

    def emit_tags(self, s: Station, part: Part, t: float) -> dict[str, float]:
        if s.station_id not in self.taggen:
            return {}
        if s.tier is SensorTier.SPARSE and (part.part_id % s.sparse_every) != 0:
            return {}   # irregular sampling: this part simply is not measured

        gen = self.taggen[s.station_id]
        shift, frac = None, 0.0
        if self.drift is not None and s.station_id == self.drift.station_id:
            shift = self.drift.shift_vector(t)
            frac = self.drift.fraction(t)
        values = gen.sample(shift, degrade_frac=frac,
                            latent=self.latent_shift(s, part, t),
                            own_scale=self.latent_scale(s))
        for name, val in values.items():
            self.rec.tag(t, part.part_id, s.station_id, name, val)
        return values

    def assign_defect(self, s: Station, part: Part, values: dict[str, float], t: float):
        """Assign defect causation.

        COMMON RANDOM NUMBERS. Both uniforms are drawn unconditionally, on
        every visit, regardless of outcome. This keeps the random stream
        identical between a drift-on and a drift-off run of the same seed, so
        the two runs produce the same part flow and the counterfactual
        ("how many escaped without the twin") isolates the drift rather than
        measuring divergent noise.
        """
        u_mech = self.rng_q.random()
        u_bg = self.rng_q.random()

        if part.is_defective:
            return

        if s.station_id == DRIFT_STATION and values:
            gl = values.get("s08_gap_left_mm")
            gr = values.get("s08_gap_right_mm")
            if gl is not None and gr is not None:
                p = defect_probability_from_gap(gl, gr, BASE_DEFECT_RATE)
                if u_mech < p:
                    part.is_defective = True
                    part.cause_station = s.station_id
                    part.cause_mechanism = "coupling_loss"
                    part.severity = float(abs(gl - gr))
                    return

        if u_bg < PER_STATION_BACKGROUND:
            part.is_defective = True
            part.cause_station = s.station_id
            part.cause_mechanism = "background"
            part.severity = 0.0

    # ------------------------------------------------------------------
    # processes
    # ------------------------------------------------------------------

    def _source(self, env: simpy.Environment, first_buffer: simpy.Store):
        pid = 0
        while True:
            # release slightly faster than takt so the line, not the source,
            # is the constraint -- body shop always has material
            yield env.timeout(TAKT_SECONDS * 0.92 * self.release_scale)
            variant = self.variant_names[self.rng.choice(len(self.variant_names), p=self.variant_p)]
            part = Part(part_id=pid, variant=variant, t_release=env.now)
            self.parts[pid] = part
            pid += 1
            yield first_buffer.put(part)

    def _station(
        self,
        env: simpy.Environment,
        s: Station,
        in_buf: simpy.Store,
        out_buf: simpy.Store | None,
        rework: simpy.Resource,
    ):
        busy_since_failure = 0.0
        time_to_failure = float(self.rng.exponential(s.mtbf))

        while True:
            t_wait0 = env.now
            part: Part = yield in_buf.get()
            if env.now > t_wait0:
                self.rec.state(s.station_id, StationState.STARVED, t_wait0, env.now)

            # ENTER is emitted the moment the part ARRIVES at the station, i.e.
            # before any repair. Occupancy therefore covers repair + service +
            # blocking, which matches how an MES records station arrival and is
            # what Roser's active-period method assumes: a station under repair
            # is constrained by ITSELF, so it counts as active, not as idle.
            t_enter = env.now
            self.rec.event(t_enter, part.part_id, s.station_id, EventType.ENTER, part.variant)

            if busy_since_failure >= time_to_failure:
                repair = float(self.rng.lognormal(np.log(s.mttr) - 0.125, 0.5))
                self.rec.state(s.station_id, StationState.DOWN, env.now, env.now + repair)
                yield env.timeout(repair)
                busy_since_failure = 0.0
                time_to_failure = float(self.rng.exponential(s.mtbf))

            svc = self.service_time(s, part.variant, env.now)
            self.rec.state(s.station_id, StationState.WORKING, env.now, env.now + svc)
            yield env.timeout(svc)
            busy_since_failure += svc

            values = self.emit_tags(s, part, env.now)
            self.assign_defect(s, part, values, env.now)

            if s.station_id == EOL_STATION:
                # fire-and-forget: a flagged part is pulled OFF-LINE into a
                # rework bay. Yielding here would let rework block the line,
                # which is not how a rework loop works.
                env.process(self._inspect(env, part, rework))

            t_block0 = env.now
            if out_buf is not None:
                yield out_buf.put(part)
                if env.now > t_block0:
                    self.rec.state(s.station_id, StationState.BLOCKED, t_block0, env.now)
            self.rec.event(env.now, part.part_id, s.station_id, EventType.EXIT, part.variant)

    def _inspect(self, env: simpy.Environment, part: Part, rework: simpy.Resource):
        if part.is_defective:
            flagged = self.rng_q.random() < EOL_SENSITIVITY
        else:
            flagged = self.rng_q.random() < EOL_FALSE_POSITIVE
        self.rec.event(env.now, part.part_id, EOL_STATION, EventType.INSPECT, part.variant)
        if flagged:
            part.detected_at = float(env.now)
            self.rec.event(env.now, part.part_id, "REWORK", EventType.REWORK_IN, part.variant)
            with rework.request() as req:
                yield req
                yield env.timeout(float(self.rng_q.exponential(REWORK_MEAN_SECONDS)))
            part.reworked = True
            self.rec.event(env.now, part.part_id, "REWORK", EventType.REWORK_OUT, part.variant)

    def _truth_sampler(self, env: simpy.Environment):
        while True:
            s1, l1, s2, margin = self.schedule.true_bottleneck(env.now)
            self.truth_bottleneck.append((float(env.now), s1, float(l1), s2, float(margin)))
            yield env.timeout(self.truth_sample_interval)

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------

    def run(self) -> dict[str, pd.DataFrame]:
        env = simpy.Environment(initial_time=self.start_time)
        pre = simpy.Store(env, capacity=5)
        buffers = [simpy.Store(env, capacity=s.buffer_out) for s in self.stations]
        rework = simpy.Resource(env, capacity=REWORK_BAYS)

        # prime the line with work in process
        pid = -1
        pid_pool = []
        for s_ in self.stations:
            n = int(self.initial_buffers.get(s_.station_id, 0))
            for _ in range(min(n, s_.buffer_out)):
                v = self.variant_names[self.rng.choice(len(self.variant_names), p=self.variant_p)]
                part = Part(part_id=pid, variant=v, t_release=self.start_time)
                self.parts[pid] = part
                pid_pool.append(part)
                pid -= 1
        pid_pool.reverse()
        for _ in range(5):
            v = self.variant_names[self.rng.choice(len(self.variant_names), p=self.variant_p)]
            part = Part(part_id=pid, variant=v, t_release=self.start_time)
            self.parts[pid] = part
            pre.items.append(part)
            pid -= 1

        env.process(self._source(env, pre))
        env.process(self._truth_sampler(env))

        in_buf = pre
        for i, s in enumerate(self.stations):
            out_buf = buffers[i] if i < len(self.stations) - 1 else None
            n = int(self.initial_buffers.get(s.station_id, 0))
            for _ in range(min(n, s.buffer_out)):
                if pid_pool:
                    buffers[i].items.append(pid_pool.pop())
            env.process(self._station(env, s, in_buf, out_buf, rework))
            in_buf = buffers[i]

        env.run(until=self.start_time + self.horizon)
        return self._frames()

    def _frames(self) -> dict[str, pd.DataFrame]:
        events = pd.DataFrame(
            self.rec.events, columns=["t", "part_id", "station_id", "event_type", "variant"]
        ).sort_values("t", kind="stable").reset_index(drop=True)

        tags = pd.DataFrame(
            self.rec.tags, columns=["t", "part_id", "station_id", "tag", "value"]
        ).sort_values("t", kind="stable").reset_index(drop=True)

        states = pd.DataFrame(
            self.rec.states, columns=["station_id", "state", "t_start", "t_end"]
        ).sort_values(["station_id", "t_start"], kind="stable").reset_index(drop=True)

        truth_bn = pd.DataFrame(
            self.truth_bottleneck,
            columns=["t", "true_bottleneck", "true_load", "runner_up", "margin"],
        )

        truth_def = pd.DataFrame(
            [
                (
                    p.part_id, p.is_defective, p.cause_station, p.cause_mechanism,
                    p.severity, p.detected_at, p.reworked, p.variant, p.t_release,
                )
                for p in self.parts.values()
            ],
            columns=[
                "part_id", "is_defective", "cause_station", "cause_mechanism",
                "severity", "detected_at", "reworked", "variant", "t_release",
            ],
        )

        truth_drift = pd.DataFrame(
            []
            if self.drift is None
            else [(
                self.drift.station_id, self.drift.t_onset, self.drift.t_full,
                self.drift.mechanism, ",".join(sorted(self.drift.peak_shift)),
            )],
            columns=["station_id", "t_onset", "t_full", "mechanism", "affected_tags"],
        )

        truth_ep = pd.DataFrame(
            [(e.station_id, e.t_start, e.t_start + e.t_ramp,
              e.t_start + e.t_ramp + e.t_hold,
              e.t_start + e.t_ramp + e.t_hold + e.t_recover,
              e.severity, e.label)
             for e in self.schedule.episodes],
            columns=["station_id", "t_start", "t_hold_start", "t_hold_end",
                     "t_end", "severity", "label"],
        ).sort_values("t_start").reset_index(drop=True)

        return {
            "events": events,
            "truth_episodes": truth_ep,
            "tags": tags,
            "states": states,
            "truth_bottleneck": truth_bn,
            "truth_defects": truth_def,
            "truth_drift": truth_drift,
        }
