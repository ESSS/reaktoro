# Ignore the tests below because they require pytest-lazy-fixture, which is abandoned and does not support pytest 8:
# https://github.com/TvoroG/pytest-lazy-fixture/issues/65
# We decided to ignore these files for now until reaktoro itself drops pytest-lazy-fixture:
# https://zulip.esss.co/#narrow/stream/12-souring/topic/reaktoro.20.2B.20lazy-fixtures/near/2098586
collect_ignore = (
    "test_equilibrium_solver.py",
    "test_equilibrium_utils.py",
    "test_kinetic_solver.py",
)
