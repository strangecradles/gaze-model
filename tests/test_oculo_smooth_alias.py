import numpy as np

import oculo_smooth as osm


RATE = 15000.0


def test_repair_alias_excursion_removes_short_jump_return():
    x = np.zeros(200)
    x[50:57] = 3.0
    valid = np.ones_like(x, dtype=bool)

    repaired = osm.repair_alias_excursions(x, RATE, valid)

    assert np.nanmax(np.abs(repaired[50:57])) < 0.5
    assert np.allclose(repaired[:45], x[:45])
    assert np.allclose(repaired[65:], x[65:])


def test_repair_alias_excursion_preserves_smooth_microsaccade_ramp():
    x = np.zeros(260)
    ramp = np.linspace(0.0, 3.0, 150)
    x[50:200] = ramp
    x[200:] = ramp[-1]
    valid = np.ones_like(x, dtype=bool)

    repaired = osm.repair_alias_excursions(x, RATE, valid)

    assert np.allclose(repaired, x)


def test_repair_alias_excursion_respects_event_mask():
    x = np.zeros(200)
    x[50:57] = 3.0
    valid = np.ones_like(x, dtype=bool)
    event = np.zeros_like(valid)
    event[45:65] = True

    repaired = osm.repair_alias_excursions(x, RATE, valid, event_mask=event)

    assert np.allclose(repaired, x)

