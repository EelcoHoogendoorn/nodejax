"""Stock recurrent cells under the ordinary NodeJAX transforms."""

import jax
import jax.numpy as jnp

from nodejax import batch, nn, scanned, stack


KEY = jax.random.PRNGKey(0)


def test_recurrent_cells_infer_fan_in_and_own_hidden_width() -> None:
    input = jnp.ones(3)

    rnn = nn.RNN(5).with_input(input).parameterize(rng=KEY)
    assert rnn.param.wx.shape == (3, 5)
    assert rnn.param.wh.shape == (5, 5)
    assert rnn.init().shape == (5,)
    _, rnn_output = rnn.apply(rnn.init(), input)
    assert rnn_output.shape == (5,)

    gru = nn.GRU(5).with_input(input).parameterize(rng=KEY)
    assert gru.param.update_weight.shape == (8, 5)
    assert gru.param.candidate_weight.shape == (8, 5)
    gru_state = gru.init()
    assert gru_state.shape == (5,)
    gru_next, gru_output = gru.apply(gru_state, input)
    assert gru_output.shape == (5,)

    gate_input = jnp.concatenate((input, gru_state))
    update = jax.nn.sigmoid(
        gate_input @ gru.param.update_weight + gru.param.update_bias)
    reset = jax.nn.sigmoid(
        gate_input @ gru.param.reset_weight + gru.param.reset_bias)
    candidate_input = jnp.concatenate((input, reset * gru_state))
    candidate = jnp.tanh(
        candidate_input @ gru.param.candidate_weight
        + gru.param.candidate_bias)
    assert jnp.allclose(gru_next, (1.0 - update) * candidate
                        + update * gru_state)
    assert jnp.allclose(gru_output, gru_next)

    lstm = nn.LSTM(5).with_input(input).parameterize(rng=KEY)
    assert lstm.param.weight.shape == (8, 20)
    lstm_state = lstm.init()
    assert jnp.allclose(lstm_state.hidden, 0.0)
    assert jnp.allclose(lstm_state.cell, 0.0)
    _, lstm_output = lstm.apply(lstm_state, input)
    assert lstm_output.shape == (5,)


def test_recurrent_cells_scan_batch_and_differentiate() -> None:
    input = jnp.arange(24.0).reshape(2, 4, 3) / 24.0

    for cell in (nn.RNN(5), nn.GRU(5), nn.MinGRU(5), nn.LSTM(5)):
        model = batch(scanned(cell)).with_input(input).parameterize(rng=KEY)
        output = model.apply(input)
        assert output.shape == (2, 4, 5)

        def loss(model) -> jax.Array:
            return jnp.sum(model.apply(input) ** 2)

        gradients = jax.grad(loss)(model)
        assert all(jnp.all(jnp.isfinite(value))
                   for value in jax.tree.leaves(gradients.param))


def test_mingru_transport_is_linear_diagonal_and_contractive() -> None:
    """The state transition is diag(1 - update): independent of the state,
    with every factor inside (0, 1)."""
    input = jnp.array([0.3, -0.8, 0.5])
    cell = nn.MinGRU(5).with_input(input).parameterize(rng=KEY)
    assert cell.param.update_weight.shape == (3, 5)
    assert cell.init().shape == (5,)
    state = jnp.linspace(-1.0, 1.0, 5)

    update = jax.nn.sigmoid(
        input @ cell.param.update_weight + cell.param.update_bias)
    candidate = input @ cell.param.candidate_weight + cell.param.candidate_bias
    next_state, output = cell.apply(state, input)
    assert jnp.allclose(next_state, (1.0 - update) * state + update * candidate)
    assert jnp.allclose(output, next_state)

    transport = jax.jacobian(
        lambda state: cell.apply(state, input)[0])(state)
    assert jnp.allclose(transport, jnp.diag(1.0 - update))
    assert jnp.all((jnp.diag(transport) > 0.0) & (jnp.diag(transport) < 1.0))


def test_lstm_state_stacks_over_depth() -> None:
    cell = stack(nn.LSTM(4), n=3).with_input(jnp.ones(4)).parameterize(rng=KEY)
    state = cell.init()
    assert state.hidden.shape == (3, 4)
    assert state.cell.shape == (3, 4)

    state, output = cell.apply(state, jnp.ones(4))
    assert state.hidden.shape == (3, 4)
    assert output.shape == (4,)
