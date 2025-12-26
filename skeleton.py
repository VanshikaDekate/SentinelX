
backend/
    uvicorn app.main:app --reload
/
        main.py
        models/
              predictor.py
              __init__.py
        schemas/
            predict_request.py
            predict_response.py
            __init__.py
        utils/
            severity.py
            preprocessing.py
            dem_preprocessing.py
            ndwi.py
            __init__.py
        routers/
            predict_router.py
            risk_map_router.py
            __init__.py
        __init__.py

    model/
        lstm_model.h5

    requirements.txt
    start.sh

ml/
    notebooks/
        ndwi_analysis.ipynb
        dem_slope.ipynb
        exploratory_data.ipynb
        
    data_processing/
        compute_ndwi.py
        compute_slope.py
        cloud_masking.py
        dataset_builder.py
        __init__.py

    models/
        lstm_train.py          # Fixed typo: was "ltsm_train.py"
        random_forest_train.py
        __init__.py
    
    datasets/
        (empty - your GeoTIFF, DEM files go here)
    README.md

dashboard/
    public/
    src/
        components/
            MapView.jsx
            RiskSidebar.jsx
            AlertsPanel.jsx
        api/
            predict.js
            riskMap.js
            alerts.js
        pages/
            Home.jsx
            Dashboard.jsx
        App.jsx
        index.js
    package.json
    vite.config.js

mobile_app/
    lib/
        main.dart
        screens/
            Home_screen.dart
            reports_screen.dart
            alerts_screen.dart
        services/
            api_service.dart
            telegram_alert.dart
        components/
            risk_card.dart
            report_input.dart
    pubspec.yaml
    android/ + ios/ (auto created)

docs/
    architecture.md
    api_reference.md
    data_flow.md
    model_pipeline.png
    system_design.png
