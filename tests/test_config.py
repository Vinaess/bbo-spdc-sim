from bbospdc.config import SIM_PARAMS

def test_params_loaded():
    """YAML dosyasının doğru okunup okunmadığını test eder."""
    assert isinstance(SIM_PARAMS, dict), "Parametreler bir dictionary olmalı"
    
    # default.yaml içindeki yeni Type-II değerlerinin doğru gelip gelmediğini kontrol et
    assert "crystal" in SIM_PARAMS
    assert SIM_PARAMS["crystal"]["material"] == "BBO"
    assert SIM_PARAMS["crystal"]["type"] == 2
    assert SIM_PARAMS["crystal"]["thickness_mm"] == 2.00
    assert SIM_PARAMS["crystal"]["cut_angle_deg"] == 41.8
    assert SIM_PARAMS["pump"]["wavelength_nm"] == 405