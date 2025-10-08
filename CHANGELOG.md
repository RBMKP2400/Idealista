# CHANGELOG.md

## 0.2.0 (2025-10-08)

Features
- **Property classification and scoring** -> [hash](https://github.com/RBMKP2400/Idealista/commit/9172798c4302b5afd8186031b1e4df2bf8d40481)  
  - Added model-based renovation prediction and score normalization by neighborhood.  
  - Integrated fine-tuning with pseudo-labeling for “Reformado” detection.  
  - Updated `main.py` and database cleanup logic.

- **USE_MODEL trigger** -> [hash](https://github.com/RBMKP2400/Idealista/commit/5ba59381538f0fd46be1a37a6e39971ad0e016e5)  
  - Allows switching between trained model (`True`) or pseudo-label pipeline (`False`).

- **'Reformado' labeling pipeline** -> [hash](https://github.com/RBMKP2400/Idealista/commit/fab30fd2ea2303f8cf0951323100569888150cb0)  
  - Hybrid system using lemmatization and embeddings.  
  - Docker limited to `USE_MODEL=False`.

Chores
- **Docker network mode set to host** -> [hash](https://github.com/RBMKP2400/Idealista/commit/645427e2fb7b4914896e499627f72953226276b3)

---

## 0.1.0 (2025-09-11)

Features:
  - add pagination to Idealista data extraction -> [hash](https://github.com/RBMKP2400/Idealista/commit/96eaf91d625b86715c070cb8a1d715c5d1a9eed6)
  - add price history per property and keep only the latest record -> [hash](https://github.com/RBMKP2400/Idealista/commit/671578f3cb8fffd8f705851ba772dbb812977a6b)

Chores:
  - initial project setup with Docker, config templates and main scripts -> [hash](https://github.com/RBMKP2400/Idealista/commit/62166ac1b98859abf24492818e245b9405e0b47f)

Bugfix:
  - correct Idealista API pagination and price history logic -> [hash](https://github.com/RBMKP2400/Idealista/commit/0a2462cdf7f4c5fb4912b4807a10b93bee1e3f71)