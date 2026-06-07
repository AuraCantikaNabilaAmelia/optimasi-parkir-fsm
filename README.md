# Optimasi Pencarian Slot Parkir FSM UNDIP

Implementasi dan perbandingan lima algoritma pencarian lintasan untuk menemukan
slot parkir tersedia terdekat (berbiaya minimum) di area Fakultas Sains dan
Matematika (FSM) Universitas Diponegoro, dengan bobot sisi yang memperhitungkan
kepadatan kendaraan.

Repositori ini berisi kode pendukung makalah proyek mata kuliah
**Analisis dan Strategi Algoritma (ASA)**, Semester Genap 2025/2026.

**Penulis:** Aura Cantika Nabila Amelia — 24060123140161

## Ringkasan

Area parkir dimodelkan sebagai graf berbobot (49 simpul, 57 sisi) yang terdiri
atas gerbang masuk, simpang jalur, dan slot parkir. Bobot tiap sisi dihitung
dengan `w(u,v) = d(u,v) · (1 + α·ρ)`, dengan `d` jarak dasar, `ρ` tingkat
kepadatan ruas, dan `α = 1.5`. Lima algoritma dibandingkan pada tiga skenario
kepadatan (rendah, sedang, tinggi):

| Algoritma | Optimal | Heuristik | Catatan |
|-----------|---------|-----------|---------|
| Uniform Cost Search (UCS) | Ya | Tidak | Prioritas biaya kumulatif g(n) |
| Greedy Best First Search (GBFS) | Tidak | Ya | Prioritas heuristik h(n) saja |
| A* | Ya | Ya | Prioritas f(n) = g(n) + h(n) |
| Dynamic Programming (DAG) | Ya | Tidak | Satu sapuan urutan topologi |
| Bellman-Ford | Ya | Tidak | Relaksasi sisi |V|−1 kali |

## Kebutuhan

- Python 3.9+
- matplotlib
- numpy

Instalasi dependensi:

```bash
pip install matplotlib numpy
```

## Cara Menjalankan

```bash
python optimasi_parkir_fsm.py
```

Program akan:

1. Mencetak ringkasan hasil (biaya, slot tujuan, jumlah operasi, memori, waktu)
   untuk setiap algoritma pada ketiga skenario ke terminal.
2. Menampilkan tiga visualisasi Matplotlib:
   - Model graf area parkir,
   - Perbandingan lintasan optimal (A*) vs Greedy BFS,
   - Perbandingan metrik (biaya, jumlah operasi, waktu eksekusi).

## Struktur Graf

- 49 simpul: 1 gerbang, 16 simpang (grid 4×4), dan 32 slot parkir.
- 57 sisi: 1 sisi gerbang, 24 sisi grid, dan 32 sisi penghubung slot.
- UCS, GBFS, A*, dan Bellman-Ford berjalan pada graf dua arah (bersiklus);
  DP berjalan pada orientasi DAG (sirkulasi satu arah).

## Catatan

- Nilai biaya, jumlah operasi, dan memori bersifat deterministik (selalu sama
  setiap dijalankan).
- Nilai waktu eksekusi dapat sedikit berbeda tiap kali dijalankan karena
  bergantung pada beban CPU dan interpreter Python.
