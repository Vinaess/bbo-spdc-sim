import yaml
from pathlib import Path

# Proje ana dizinini bul (src/bbospdc/config.py dosyasının 3 üst klasörü)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PARAMS_FILE = BASE_DIR / "params" / "default.yaml"

def load_params(file_path=PARAMS_FILE):
    """YAML dosyasından simülasyon parametrelerini okur ve dictionary olarak döndürür."""
    if not file_path.exists():
        raise FileNotFoundError(f"Parametre dosyası bulunamadı: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    
    return params

# Modül import edildiğinde parametreleri otomatik yükle
SIM_PARAMS = load_params()

if __name__ == "__main__":
    # Bu dosyayı doğrudan çalıştırırsak (test için) parametreleri ekrana yazdırır
    print("Yüklenen Parametreler:")
    print(yaml.dump(SIM_PARAMS, allow_unicode=True))