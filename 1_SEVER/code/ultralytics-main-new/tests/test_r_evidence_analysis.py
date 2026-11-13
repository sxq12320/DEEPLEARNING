import pytest

from ANALYZE_R_EVIDENCE import ARMS, contrasts


def synthetic_rows():
    rows = []
    for (a, b, c), name in ARMS.items():
        value = 10 + 2*a + 3*b + 4*c + 5*a*b
        rows.append(dict(seed='42', model=name, targets='1181', mask_ap50=value,
                         mask_ap5095=value, recall=value, r_at_p90=value))
    return rows


def test_known_main_and_interaction_effects():
    results = contrasts(synthetic_rows())
    found = {r['contrast']: r['delta_pp'] for r in results if r['metric'] == 'mask_ap5095'}
    assert found['ABC_minus_baseline'] == 14
    assert found['AB_interaction_C0'] == 5
    assert found['A_marginal_mean'] == 4.5
    assert found['BC_interaction_A0'] == 0


def test_partial_data_does_not_fabricate_effects():
    assert contrasts(synthetic_rows()[:4]) == []


def test_different_population_is_rejected():
    rows = synthetic_rows()
    rows[-1]['targets'] = '1077'
    with pytest.raises(ValueError, match='Different validation target'):
        contrasts(rows)
