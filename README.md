# Seasonal-Discount-Effective-Turkcell-Project

Bu proje, mevsimsel indirimlerin satışlar üzerindeki etkisini analiz etmek ve tahmin etmek için geliştirilmiştir.

## Proje Yapısı

```
├── data/

│   ├── raw/              # Ham veri

│   └── processed/        # İşlenmiş veri

├── models/               # Eğitilmiş modeller

├── reports/              # Raporlar ve grafikler

│   └── figures/

├── src/

│   ├── api/             # FastAPI uygulaması

│   ├── data/            # Veri işleme kodları

│   └── model/           # Model eğitimi ve değerlendirme

├── .env                 # Çevre değişkenleri

├── requirements.txt     # Bağımlılıklar

└── README.md         
```

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Veri dosyalarını `data/raw/` klasörüne yerleştirin:
- `tabledf.csv`

3. Çevre değişkenlerini ayarlayın:
- `.env` dosyasını düzenleyin

## Kullanım

### Veri İşleme ve Model Eğitimi

1. Veri ön işleme:
```bash
python src/data/pre.py
python src/data/preproccessing.py
```

2. Model eğitimi:
```bash
python src/model/train_best_model.py
```

### API Kullanımı

1. API'yi başlatın:
```bash
uvicorn src.api.app:app --reload
```

2. API'yi test edin:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "product_name": "Ürün Adı",
           "city": "Şehir",
           "category_name": "Kategori",
           "yearquarter": "2023 Q4",
           "discount": 0.2
         }'
```

## Örnek Senaryo

1. Kullanıcı bir ürün için indirim planlıyor
2. API'ye ürün bilgilerini gönderiyor
3. API, eğitilmiş modeli kullanarak indirimin etkisini tahmin ediyor
4. Sonuç olarak indirimin başarılı olup olmayacağı bildiriliyor

## Model Performansı

- ROC-AUC: 0.85
- F1-Score: 0.82
- Accuracy: 0.81

## Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın. 
