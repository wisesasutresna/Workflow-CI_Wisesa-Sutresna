# Pipeline CI/CD: Prediksi Diabetes

Repositori ini difokuskan untuk otomatisasi pelatihan model (Retraining) dan Deployment menggunakan GitHub Actions.
Model klasifikasi Gradient Boosting dilatih secara otomatis, dilacak metriknya menggunakan MLflow (melalui DagsHub), lalu di-build menjadi Docker Image dan dipush ke Docker Hub.

## Fitur Utama
- **Automated Retraining**: Menjalankan run MLflow menggunakan spesifikasi `MLProject`.
- **DagsHub Tracking**: Tracking akurasi dan parameter model secara terpusat.
- **Docker Hub Deployment**: Build dan push otomatis image container saat ada perubahan pada branch utama.