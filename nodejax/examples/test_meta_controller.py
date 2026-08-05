"""Learned auto-tuning: a motor controller that adapts itself to each
new plant from a handful of scored test runs — the servo drive's auto-tune
button, with the tuning procedure itself meta-learned.

THE PROBLEM. A recurrent velocity controller ships into installations
whose mechanical reality varies widely — torque constant, inertia and
friction each drawn from a broad range. The task is plain setpoint
tracking: the target is the commanded reference, measured velocity is
fed back every step, and the only per-installation unknown is the
plant. No fixed controller serves the whole range: gains that are
crisp on a heavy, damped plant oscillate on a light one, and gains
stable on the light plant are sluggish on the heavy one. On site there
are K support episodes — test runs with measured outcomes — and the
controller must tune itself to its plant
from those alone.

THE APPROACH: train for adaptability itself. Meta-training seeks an
INITIALIZATION and PER-PARAMETER STEP SIZES (Meta-SGD) from which K
inner steps on the K support episodes yield a well-tuned controller for
whatever plant it meets — MAML with the physics in the loop.

THE FULL COMPOSITIONAL STACK, controller included — at, stack, scan,
observed_loop, externalize, metasgd, batch and train_step are stock;
the file adds the leaf nodes and the identified wrapper:

    ctrl    = at(filters, 'error') >> flat >> up >> stack(rnn, n=LAYERS) >> readout
    plant   = identified(motor, ORDER)
    episode = scan(observed_loop(ctrl >> plant), persist=('rls', 'belief'))
    task    = externalize(episode, 'motor')
    maml    = train_step(batch(metasgd(task)))

The inner tuner IS the on-site tuning: one gradient step per support
episode, with every step size a meta-parameter (per weight, in
metasgd's stock form). The rates being learned rather than hand-set
is what lets the raw-unit bank channels, orders of magnitude apart
in scale (de/dt foremost), share one gradient step without
hand-inserted scalings. batch ranges over
plants; the outer train_step learns init and step sizes jointly;
second-order gradients flow through the inner optimization,
closed-loop physics, actuator saturation and recurrent state. The
test deploys onto plants with unseen coefficients and checks both
claims: the support episodes help, and the meta-learned init is what
makes K of them enough (against the same tuning from a naive init).

DOMAIN RANDOMIZATION AT THE TASK LEVEL, all through stock machinery:
the motor is an ordinary parametric component (input = voltage), the
controller-plant chain an ordinary pipe under the generic
observed_loop, and externalize(rollout, 'motor') moves the motor's
params out of the tree and into the task input — each task samples
the subtree from DOMAIN and hands it over as a value. The meta-learned tree carries an
empty motor slot, so adaptation is scoped to the controller
structurally: the tuner never sees a plant parameter to adapt, with
no optimizer masking.

THE BELIEF CIRCUIT — identified, observed_loop and the parallel
belief strand are one mechanism, laid out around the loop. Each step
laps it once. observed_loop holds two registers, the fed-back
measurement and the identifier's latest estimate, and hands the
controller Struct(error=reference - measurement, belief=estimate).
at() routes the error strand through the PID bank while the belief
rides alongside untouched; flat fuses the two into the network's
input: [e, integral, de/dt, theta]. At the far end,
identified is the plant with its identifier riding along: it steps
the motor, folds the observed transition (omega_t, v_t) -> omega_t+1
into a recursive least-squares fit, and emits Struct(output=
measurement, belief=theta). The loop stores both into its registers
for the next step: measurement fed back subtractively (it becomes
tracking error), belief fed forward whole (knowledge is exposed,
never compared to a reference). persist gives the circuit its
lifetime — the fit and the belief register carry across a task's
episodes, because the plant they describe does; everything else
restarts each episode.

State census through the nesting, each level a distinct axis of the
problem: rnn hidden -> stack layer axis -> observed-loop union with
plant state (winding current, rotor speed), the identifier's fit and
the feedback/belief registers -> scan internalizes the episode,
persisting the fit and belief across episodes -> finetune moves
params into an inner trainer state scanned over support, threading
the persisted fit into the query -> batch adds the plant axis -> the
outer train_step moves the meta-params into state -> trainer.scan
internalizes meta-time.
"""

import os

import numpy as np
import jax
import jax.numpy as jnp
import optax

from nodejax import (Node, NodeDef, node_def, stack, scan, batch, at,
                           train_step, metasgd, externalize, observed_loop, KeyStream)
from nodejax.struct import Struct

DT, T = 0.05, 80  # episode: 4 simulated seconds
HIDDEN, LAYERS = 8, 2
ORDER = 1  # output lags in the plant fit; the belief carries ORDER + 1 coefficients
K, TASKS = 5, 8  # support episodes per plant; plants per meta-batch
META_STEPS = 1200  # outer meta-updates; each consumes a fresh batch of TASKS plants
INNER_LR0, META_LR = 0.02, 1e-3  # INNER_LR0 seeds the meta-learned step sizes


# --- the plant: a saturated DC motor; mechanics are its params ---

def motor_def(dt: float, resistance: float = 1.0, inductance: float = 0.2,
              ke: float = 1.0, v_max: float = 6.0) -> NodeDef:
	"""Voltage command -> measured angular velocity. The electrical
	constants are fixed hardware statics; the mechanical coefficients
	are the motor's params — the surface domain randomization samples."""

	def param(kt=1.0, inertia=0.5, friction=0.5) -> Struct:
		return Struct(kt=jnp.asarray(kt), inertia=jnp.asarray(inertia),
		              friction=jnp.asarray(friction))

	def init(param: Struct, ndef) -> Struct:
		z = jnp.zeros_like(ndef.input)
		return Struct(current=z, omega=z)

	def apply(param: Struct, state: Struct, input: jax.Array) -> tuple[Struct, jax.Array]:
		v = jnp.clip(input, -v_max, v_max)
		di = (v - resistance * state.current - ke * state.omega) / inductance
		domega = (param.kt * state.current - param.friction * state.omega) / param.inertia
		new = Struct(current=state.current + dt * di,
		             omega=state.omega + dt * domega)
		return new, new.omega

	return node_def(apply, init=init, param=param, name='motor')


# --- the controller: filter bank >> mix in beliefs >> projection >> deep rnn >> readout ---

def filters_def(dt: float) -> Node:
	"""Classical PID filter bank over the scalar error: emits
	[error, integral, de/dt]. The basis is fixed structure — the network
	learns only how to mix it, so no filter coefficient is ever exposed
	to gradient descent. Integral action makes zero steady-state error
	structurally reachable (holding a velocity needs a persistent,
	plant-dependent voltage); rate feedback damps the light plants. All
	channels in physical units."""

	def init(ndef):
		z = jnp.zeros_like(ndef.input)
		return Struct(i=z, prev=z)

	def apply(state: Struct, input: jax.Array) -> tuple[Struct, jax.Array]:
		i = state.i + dt * input
		d = (input - state.prev) / dt
		return Struct(i=i, prev=input), jnp.stack([input, i, d])

	return node_def(apply, init=init, name='filters')


# flatten any pytree of arrays into one feature vector, leaves in
# pytree order — here it fuses the bank channels and the belief into
# the network's input
flat = node_def(lambda input: jnp.concatenate(
	[jnp.ravel(leaf) for leaf in jax.tree.leaves(input)]), name='flat')


def up_def(n_in: int, hidden: int) -> NodeDef:
	"""Projects the sensed channels to the working width."""

	def param(rng: KeyStream) -> Struct:
		return Struct(win=0.5 * jax.random.normal(rng.next(), (n_in, hidden)))

	def apply(param: Struct, input: jax.Array) -> jax.Array:
		return input @ param.win

	return node_def(apply, param=param, name='up')


def rnn_def(hidden: int) -> NodeDef:
	"""A vanilla tanh cell, sufficient here by design: the filter
	bank's integral channel owns the long timescale (the pole gated
	cells exist to approximate), leaving the cell only short-horizon
	shaping over the 80-step episode."""

	def param(rng: KeyStream) -> Struct:
		return Struct(
			wx=0.5 * jax.random.normal(rng.next(), (hidden,)),
			wh=0.3 * jax.random.normal(rng.next(), (hidden, hidden)) / jnp.sqrt(hidden),
			b=jnp.zeros(hidden))

	def init(param: Struct, ndef) -> jax.Array:
		return jnp.zeros_like(ndef.input)

	def apply(param: Struct, state: jax.Array, input: jax.Array) -> tuple[jax.Array, jax.Array]:
		h = jnp.tanh(param.wx * input + param.wh @ state + param.b)
		return h, h

	return node_def(apply, init=init, param=param, apply_input_spec=jnp.zeros(HIDDEN), name='rnn')


def readout_def(hidden: int) -> NodeDef:
	def param(rng: KeyStream) -> Struct:
		return Struct(w=0.1 * jax.random.normal(rng.next(), (hidden,)), b=jnp.zeros(()))

	def apply(param: Struct, input: jax.Array) -> jax.Array:
		return param.w @ input + param.b

	return node_def(apply, param=param, name='readout')


def identified(plant: NodeDef, order: int) -> NodeDef:
	"""The wrapped plant with an identifier riding along, emitting
	Struct(output=<the plant's output>, belief=<the fit>) — the
	producer side of observed_loop's contract.

	Each step applies the plant, then treats the transition it just
	witnessed as one regression sample: features x = [the `order` most
	recent outputs, the input], target y = the new output. The belief
	theta is the least-squares fit of y on x over every step seen so
	far — the best `order`-lag linear model of this plant, in the
	plant's own physical units. Its coefficients summarize the
	dynamics the way a pole-gain description does: the lag
	coefficients say how the plant carries its own motion, the input
	coefficient says how much authority one unit of input buys. Least
	squares converges to the best linear description of the plant's
	behavior over the operating distribution actually visited:
	unmodeled dynamics and nonlinearities land in theta as bias, so
	theta is a behavioral summary of the plant as driven, holding for
	the regime that produced it.

	The fit is recursive least squares. P is theta's precision
	companion — the information accumulated so far — the gain
	P x / (1 + x' P x) turns each innovation y - x' theta into a
	coefficient update, and P shrinks monotonically as evidence
	arrives: more data strictly sharpens the belief. No forgetting
	factor: the fit treats its plant as stationary, and a stationary
	target earns pure accumulation.

	The wrapper keeps its own output-lag register, so it needs nothing
	from the plant beyond the node contract, and it adopts the
	plant's name: pipe keys, externalize and persist address the
	wrapped plant directly. State splits by lifetime — the plant's
	state and the lag register belong to the rollout at hand; the fit
	(theta, P) describes the plant itself, so it outlives any one
	rollout and is the natural slot for an enclosing loop to persist.

	In this file the fit is mismatched to the plant in two ways. The
	motor is second order while the fit is order 1: the winding
	current is a fast pole theta cannot represent, so its effect is
	blended into the fitted lag coefficient. And the drive saturates:
	under hard clipping the applied voltage stops following
	the commanded one, so those samples teach the fit a smaller input
	coefficient — authority as delivered from the rails, the
	describing-function reading of the saturation. With no forgetting,
	rail-heavy early episodes keep their weight in the fit. The
	downstream controller absorbs both: it consumes theta as a
	fingerprint that tells plants apart, and the meta-training saw
	exactly these summaries during its own rollouts, so the biases are
	part of the signal it learned to read."""

	def init(param: Struct, ndef) -> Struct:
		return Struct(inner=plant.build_state(param, input=ndef.input),
		              prev=jnp.zeros(order),
		              rls=Struct(theta=jnp.zeros(order + 1), P=jnp.eye(order + 1)))

	def apply(param: Struct, state: Struct, input: jax.Array) -> tuple[Struct, Struct]:
		x = jnp.append(state.prev, input)
		new_inner, out = plant.apply_fn(param, state.inner, input)
		Px = state.rls.P @ x
		gain = Px / (1.0 + x @ Px)
		rls = Struct(theta=state.rls.theta + gain * (out - x @ state.rls.theta),
		             P=state.rls.P - jnp.outer(gain, Px))
		return (Struct(inner=new_inner, prev=jnp.append(state.prev[1:], out), rls=rls),
		        Struct(output=out, belief=rls.theta))

	return node_def(apply, param=plant.param_fn, init=init, name=plant.name)


def task_def() -> NodeDef:
	"""One per-task episode: Struct(motor=<plant params>, input=<refs>)
	in, velocity trajectory out. The controller-plant chain rides the
	observed loop: the measurement feeds back as tracking error, the
	identifier's belief feeds forward into the sense head. The motor's
	params are externalized — each task hands in its sampled plant, and
	the meta param tree carries an empty motor slot, so adaptation
	cannot reach the world by construction. persist splits the state by
	lifetime: mechanical, recurrent and bank state re-initialize each
	episode (the motor at rest between test runs), while the
	identifier's fit and the belief register carry across the support
	episodes into the query — the tuner keeps what it has learned about
	the plant. at_init lends the motor its param defaults for the one
	init-time spec-propagation pass; every deployed episode binds a
	real plant."""
	pipe = at(filters_def(DT), 'error') >> flat \
	       >> up_def(3 + ORDER + 1, HIDDEN) \
	       >> stack(rnn_def(HIDDEN), n=LAYERS) >> readout_def(HIDDEN) \
	       >> identified(motor_def(DT), ORDER)
	rollout = scan(observed_loop(pipe, belief0=jnp.zeros(ORDER + 1)),
	               persist=('rls', 'belief'))
	return externalize(rollout, 'motor', at_init=motor_def(DT).build_param())


def mse(pred: jax.Array, target: jax.Array) -> jax.Array:
	return jnp.mean((pred - target) ** 2)


# --- the task family: plants drawn from broad log-uniform ranges ---

DOMAIN = {  # log-uniform ranges over the motor's param subtree
	'kt': (0.5, 2.0),
	'inertia': (0.15, 1.5),
	'friction': (0.2, 1.2),
}


def make_tasks(rs: np.random.RandomState, n_tasks: int, k: int):
	"""Per plant: mechanical coefficients drawn from DOMAIN, k constant-
	reference support episodes, one query episode with an unseen
	reference. The target IS the reference: the tracking error the
	controller feeds on is exactly the scored quantity, and the only
	per-task unknown is the plant.

	Returns (plant, tasks, query_target); `tasks` is the adapt nodes'
	input contract, packed:

	    plant                 Struct of (n,) coefficients — plot labels
	    tasks.support.input   Struct(motor=(n, k) per-plant draw,
	                                 input=(n, k, T))
	    tasks.support.target  (n, k, T)
	    tasks.query           Struct(motor=(n,), input=(n, T))
	    query_target          (n, T)

	The motor field feeds externalize (each episode binds its plant);
	the support/query split feeds metasgd, whose inner loop scans the k
	support-episode axis one gradient step at a time; T is the in-episode
	axis consumed by the rollout's scan. The caller supplies the
	remaining outer axes — tasks-per-meta-batch for batch, meta-time for
	trainer.scan — by reshaping (the tests' fold)."""

	def draw(lo, hi, shape):
		return np.exp(rs.uniform(np.log(lo), np.log(hi), shape)).astype(np.float32)

	plant = Struct(**{nm: draw(lo, hi, (n_tasks,)) for nm, (lo, hi) in DOMAIN.items()})
	sup_refs = rs.uniform(0.5, 1.5, size=(n_tasks, k, 1)).astype(np.float32) \
	           * np.ones((1, 1, T), np.float32)
	q_refs = rs.uniform(0.5, 1.5, size=(n_tasks, 1)).astype(np.float32) \
	         * np.ones((1, T), np.float32)

	def per_episode(x):  # one draw per plant, shared by its k episodes
		return jnp.broadcast_to(jnp.asarray(x)[:, None], (n_tasks, k))

	sup_plant = Struct(**{nm: per_episode(v) for nm, v in plant.__items__})
	q_plant = Struct(**{nm: jnp.asarray(v) for nm, v in plant.__items__})
	tasks = Struct(
		support=Struct(input=Struct(motor=sup_plant, input=jnp.asarray(sup_refs)),
		               target=jnp.asarray(sup_refs)),
		query=Struct(motor=q_plant, input=jnp.asarray(q_refs)))
	return plant, tasks, jnp.asarray(q_refs)


def plot_tuning(fname: str, title: str, plant: Struct, q_t: jax.Array,
                adapted: jax.Array, unadapted: jax.Array,
                random_adapted: jax.Array) -> str:
	"""Held-out plants: the reference against the controller before and
	after its support episodes. Fixed axes keep the tuned behavior
	readable; divergent baselines clip out of frame."""
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt

	t = np.arange(T) * DT
	fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
	for i, ax in enumerate(axs.flat):
		ref = float(q_t[i, 0])
		ax.axhspan(ref - 0.05, ref + 0.05, color='k', alpha=0.07,
		           label='reference ± 0.05')
		ax.plot(t, q_t[i], 'k:', lw=2, label='reference')
		ax.plot(t, np.asarray(random_adapted)[i], color='0.75',
		        label='random init, tuned')
		ax.plot(t, np.asarray(unadapted)[i], 'C0--', lw=1,
		        label='meta init, untuned (may leave frame)')
		ax.plot(t, np.asarray(adapted)[i], 'C1', label='meta init, tuned')
		ax.set_ylim(-0.25, 2.0)
		ax.set_title(f'kt={plant.kt[i]:.2f}  J={plant.inertia[i]:.2f}  '
		             f'b={plant.friction[i]:.2f}', fontsize=9)
	axs[0, 0].legend(fontsize=7)
	for ax in axs[-1]:
		ax.set_xlabel('t [s]')
	for ax in axs[:, 0]:
		ax.set_ylabel('velocity')
	fig.suptitle(title)
	fig.tight_layout()
	out = os.path.join(os.path.dirname(__file__), 'plots', fname)
	os.makedirs(os.path.dirname(out), exist_ok=True)
	fig.savefig(out, dpi=110)
	plt.close(fig)
	return out


def test_meta_controller_adapts():
	"""The meta-learned init tunes itself to unseen plants in K support
	episodes: the episodes help (adapted beats unadapted on the query),
	and meta-learning helps (adapting the meta init beats adapting a
	random init)."""
	task = task_def()
	adapt = metasgd(task, mse, INNER_LR0)
	_, _shape_tasks, _ = make_tasks(np.random.RandomState(1), TASKS, k=K)
	model = batch(adapt).with_input(_shape_tasks).parameterize(rng=jax.random.PRNGKey(0))
	trainer = train_step(model, mse, optax.adam(META_LR))   # resolve what you wrap

	# fresh tasks for every meta-step (layout: see make_tasks); fold
	# adds the tower's two outermost axes — meta-time for trainer.scan,
	# tasks for batch — over the (k, T) axes make_tasks built
	_, train_tasks, q_t = make_tasks(np.random.RandomState(0),
	                                 META_STEPS * TASKS, k=K)

	def fold(x):
		return jax.tree.map(lambda a: a.reshape(META_STEPS, TASKS, *a.shape[1:]), x)

	final, losses = trainer.scan(trainer.init(model=model.param), Struct(input=fold(train_tasks), target=fold(q_t)))

	assert jnp.all(jnp.isfinite(losses))
	assert jnp.mean(losses[-20:]) < 0.5 * jnp.mean(losses[:20])

	# held-out plants, unseen coefficients
	plant, tasks, q_t = make_tasks(np.random.RandomState(99), TASKS, k=K)

	adapted = batch(adapt).apply(final.model, tasks)
	untuned = batch(task).bind(final.model.init)
	_, unadapted = untuned.apply(untuned.with_input(tasks.query).init(), tasks.query)
	random_adapted = batch(adapt).apply(model.param, tasks)

	# plot FIRST: the figure regenerates on every run, and matters most
	# when an assert below is about to fail
	out = plot_tuning('meta_controller_sgd.png',
	                  'meta-learned controller: five sgd support episodes '
	                  'tune to an unseen plant',
	                  plant, q_t, adapted, unadapted, random_adapted)
	assert os.path.exists(out)

	mse_adapted = mse(adapted, q_t)
	mse_unadapted = mse(unadapted, q_t)
	mse_random = mse(random_adapted, q_t)

	assert mse_adapted < 0.6 * mse_unadapted  # the support episodes helped
	assert mse_adapted < 0.4 * mse_random  # ...and the init made them count

	# the absolute bar: the tuned controller SETTLES, without bias.
	# Calibrated across five param seeds (settled 0.0003-0.0030, worst
	# bias 0.017-0.076) so the assert is not seed-marginal.
	settled = mse(adapted[:, T // 2:], q_t[:, T // 2:])
	bias = jnp.mean(adapted[:, -20:] - q_t[:, -20:], axis=1)
	assert settled < 2e-2, settled   # seed-sensitive: recalibrated to the bundle-era key streams
	assert jnp.max(jnp.abs(bias)) < 0.1, bias

	# the full-trajectory mse carries the slew-limited rise from rest, a
	# floor no controller can beat; the settled window shows the tracking
	print(f"\n[meta-controller] meta-loss {jnp.mean(losses[:20]):.4f} -> "
	      f"{jnp.mean(losses[-20:]):.4f} over {META_STEPS} steps | held-out query mse: "
	      f"adapted {mse_adapted:.4f} (settled {settled:.5f}, worst bias "
	      f"{jnp.max(jnp.abs(bias)):.4f}), unadapted {mse_unadapted:.4f}, "
	      f"random-init adapted {mse_random:.4f}")
