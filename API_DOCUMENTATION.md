# RaithaMitra Backend API Documentation
**AI-Powered Crop Advisory & Farmer Distress Early Warning System**  
**Major Project BAD685 — Department of AI & Data Science, KSSEM Bengaluru**  
**Version:** 1.0.0 | **Protocol:** REST JSON over HTTP | **Base Port:** `5000`

---

## 1. API Purpose & Architecture Overview
RaithaMitra exposes a lightweight, non-blocking HTTP REST API adapter designed for frontend and mobile website integration. The API forwards requests into an integrated, offline-first multi-module agricultural AI pipeline:

$$\text{Farmer Query (KN)} \longrightarrow \text{NLLB Translation} \longrightarrow \text{Canonical Crop Identity} \longrightarrow \text{Location / Weather / Soil / Schemes / Mandi} \longrightarrow \text{Dhenu LLM} \longrightarrow \text{Kannada Advisory}$$

---

## 2. How to Start the Backend Server

### Prerequisites
Activate the project's Python virtual environment:
```powershell
# Windows
.\.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

### Running the API Server
```powershell
python -m api.app
# OR
python api/app.py
```
By default, the server starts on `http://127.0.0.1:5000` (`http://localhost:5000`).

---

## 3. Base URL & Endpoints Summary

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Lightweight service health check (does NOT load heavy models) | No |
| `GET` | `/api/v1/version` | Returns API version and commit hash | No |
| `POST` | `/api/v1/advisory` | Primary agricultural advisory generation endpoint (JSON text query) | No |
| `POST` | `/api/v1/advisory/audio` | Voice advisory generation endpoint (Multipart audio upload + optional TTS) | No |
| `GET` | `/api/v1/advisory/audio/download` | Download generated spoken audio WAV file (`?file=<filename>`) | No |


---

## 4. Endpoint Specifications

### `GET /api/v1/health`
Lightweight health probe for container orchestrators, load balancers, and frontend ping checks.
* **HTTP Status:** `200 OK`
* **Response Payload:**
```json
{
  "status": "ok",
  "service": "RaithaMitra"
}
```

---

### `GET /api/v1/version`
Returns build metadata.
* **HTTP Status:** `200 OK`
* **Response Payload:**
```json
{
  "service": "RaithaMitra",
  "version": "1.0.0",
  "commit": "ff9e04c",
  "status": "operational"
}
```

---

### `POST /api/v1/advisory`
Primary endpoint for farmer advisory queries.

#### Request Headers
```http
Content-Type: application/json
Accept: application/json
```

#### Request JSON Schema
```json
{
  "query": "ನನ್ನ ರಾಗಿ ಬೆಳೆ ಒಣಗುತ್ತಿದೆ",
  "district": "Mandya",
  "taluk": "Pandavapura",
  "village": "Melukote",
  "crop": "ragi",
  "language": "kn"
}
```

#### Request Fields

| Field | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `query` | `string` | **Yes** | — | Farmer's question in Kannada or English (non-empty). |
| `district` | `string` | No | `null` | Karnataka district (e.g., `"Mandya"`, `"Belagavi"`, `"Udupi"`). |
| `taluk` | `string` | No | `null` | Karnataka taluk within the specified district (e.g., `"Pandavapura"`). |
| `village` | `string` | No | `null` | Karnataka village/gram panchayat (e.g., `"Melukote"`). |
| `crop` | `string` | No | `null` | Optional explicit crop override (e.g., `"ragi"`, `"tomato"`). |
| `language`| `string` | No | `"kn"` | Output language code (`"kn"` for Kannada, `"en"` for English). |

---

#### Success Response (`200 OK`)
```json
{
  "success": true,
  "language": "kn",
  "canonical_crop": "ragi",
  "answer": "ರಾಗಿ ಬೆಳೆಗೆ ನೀರಿನ ಕೊರತೆಯಾದಾಗ ರಕ್ಷಣಾತ್ಮಕ ನೀರಾವರಿ ನೀಡಿ ಮತ್ತು ತೇವಾಂಶ ಸಂರಕ್ಷಣೆಗೆ ಮಣ್ಣಿನ ಹೊದಿಕೆ ಮಾಡಿ.",
  "location": {
    "district": "Mandya",
    "district_kn": "ಮಂಡ್ಯ",
    "taluk": "Pandavapura",
    "taluk_kn": "ಪಾಂಡವಪುರ",
    "village": "Melukote",
    "village_kn": "ಮೇಲುಕೋಟೆ",
    "latitude": 12.6625,
    "longitude": 76.6542,
    "state": "Karnataka",
    "state_kn": "ಕರ್ನಾಟಕ",
    "source": "Local Government Directory (LGD), Ministry of Panchayati Raj, Government of India & KSRSAC"
  },
  "weather": {
    "available": true,
    "current": {
      "temperature_c": 27.5,
      "humidity_percent": 68.0,
      "precipitation_mm": 0.2,
      "wind_speed_kmh": 9.5,
      "weather_condition": "Partly cloudy"
    },
    "forecast": {
      "precipitation_next_24h_mm": 2.4,
      "precipitation_next_3_days_mm": 8.4,
      "temperature_max_next_24h_c": 29.0,
      "temperature_min_next_24h_c": 21.0
    },
    "source": "Open-Meteo"
  },
  "soil": {
    "available": true,
    "district": "Mandya",
    "taluk": "Pandavapura",
    "village": "Melukote",
    "soil_order": "Alfisols",
    "dominant_soil_types": [
      "Red sandy loam",
      "Red loamy soil"
    ],
    "texture": "Coarse to medium textured, light to medium loamy",
    "typical_ph_range": "6.2 - 7.5 (Slightly acidic to neutral)",
    "agro_climatic_zone": "Southern Dry Zone (Zone 6)",
    "management_recommendations": "Add Farm Yard Manure (FYM) or compost to improve water-retention in red sandy soils.",
    "is_measured_data": false,
    "source_authority": "ICAR-NBSS&LUP Regional Centre Bengaluru & KSDA"
  },
  "schemes": [
    {
      "id": "karnataka_raita_siri",
      "name_en": "Karnataka Raita Siri (Millet Incentive Scheme)",
      "name_kn": "ಕರ್ನಾಟಕ ರೈತ ಸಿರಿ ಯೋಜನೆ (ಸಿರಿಧಾನ್ಯ ಪ್ರೋತ್ಸಾಹಧನ)",
      "benefit_summary": "Direct cash incentive of ₹10,000 per hectare (up to 2 ha maximum, ₹20,000 max limit) directly to Aadhaar-linked bank accounts via DBT.",
      "application_portal": "https://raitamitra.karnataka.gov.in",
      "source_authority": "Department of Agriculture, Government of Karnataka"
    }
  ],
  "market": {
    "available": true,
    "is_today_data": true,
    "records": [
      {
        "commodity": "Ragi",
        "market_name": "Mandya",
        "district": "Mandya",
        "market_date": "2026-08-19",
        "min_price": 2800.0,
        "max_price": 3400.0,
        "modal_price": 3200.0,
        "unit": "₹/quintal",
        "arrivals": 45.0,
        "source_authority": "AGMARKNET / Directorate of Marketing & Inspection, GoI & KSAMB"
      }
    ]
  },
  "metadata": {
    "model": "dhenu2-in-1b",
    "backend": "mock",
    "rag_enabled": true,
    "retrieved_documents_count": 2,
    "retrieval_time_seconds": 0.0012,
    "translation_in_time_seconds": 0.0004,
    "generation_time_seconds": 0.0008,
    "translation_out_time_seconds": 0.0003,
    "processing_time_seconds": 0.0031
  }
}
```

---

### `POST /api/v1/advisory/audio`
Voice-enabled agricultural advisory generation endpoint.
Accepts multipart form-data with an uploaded audio query (e.g. farmer speaking Kannada), transcribes it via Whisper Kannada ASR, retrieves grounding contexts, executes local Dhenu LLM inference, and optionally synthesizes the response into spoken Kannada audio via Neural TTS.

* **HTTP Status:** `200 OK`
* **Content-Type:** `multipart/form-data`
* **Form Parameters:**

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `audio` (or `file`) | Binary File | **Yes** | Audio recording (`.wav`, `.mp3`, `.ogg`, `.flac`, `.m4a`, max 5 MB). |
| `district` | string | Optional | Karnataka district name (e.g. `Mandya`). |
| `taluk` | string | Optional | Karnataka taluk name (e.g. `Pandavapura`). |
| `village` | string | Optional | Karnataka village name (e.g. `Melukote`). |
| `crop` | string | Optional | Explicit crop override (e.g. `ragi`). |
| `language` | string | Optional | Target output language (`kn` default, or `en`). |
| `synthesize_audio` | string / bool | Optional | Set to `"true"` or `"1"` to trigger spoken Kannada TTS audio generation. |

* **Success Response Payload:**
```json
{
  "success": true,
  "language": "kn",
  "canonical_crop": "ragi",
  "answer": "1. ಪ್ರಸ್ತುತ ಒಣಗಿದ ಅವಧಿಯನ್ನು ಗಮನಿಸಿದರೆ ನಿಮಗೆ ತಕ್ಷಣದ ನೀರಾವರಿ ಬೆಂಬಲ ಬೇಕೇ ಇಲ್ಲವೇ ಎಂಬುದನ್ನು ನಿರ್ಣಯಿಸಿ...",
  "asr": {
    "transcript": "ನನ್ನ ರಾಗಿ ಬೆಳೆಗೆ ಮಳೆ ಸರಿಯಾಗಿ ಆಗದೆ ಒಣಗುತ್ತಿದೆ ಏನು ಮಾಡಬೇಕು",
    "audio_duration_seconds": 6.62,
    "asr_processing_time_seconds": 19.76,
    "asr_model": "vasista22/whisper-kannada-small",
    "asr_device": "cpu"
  },
  "audio": {
    "available": true,
    "audio_path": "/path/to/outputs/kannada_advisory_xyz.wav",
    "duration_seconds": 34.63,
    "sample_rate": 24000,
    "format": "wav",
    "voice": "kn-IN-GaganNeural",
    "latency_seconds": 8.66
  },
  "location": {
    "district": "Mandya",
    "taluk": "Pandavapura",
    "village": "Melukote",
    "hierarchy_label": "Melukote, Pandavapura, Mandya, Karnataka"
  },
  "weather": {
    "available": true,
    "current": {
      "temperature_c": 27.5,
      "weather_condition": "Partly cloudy"
    }
  },
  "soil": {
    "available": true,
    "soil_order": "Alfisols"
  },
  "schemes": [],
  "market": null,
  "metadata": {
    "model": "KissanAI/Dhenu2-In-Llama3.2-1B-Instruct",
    "backend": "dhenu",
    "rag_enabled": true,
    "voice_pipeline_total_time_seconds": 105.59
  }
}
```

---

### `GET /api/v1/advisory/audio/download`
Downloads or streams generated spoken Kannada WAV audio files.
* **HTTP Status:** `200 OK`
* **Content-Type:** `audio/wav`
* **Query Parameters:**

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `file` | string | **Yes** | Basename of audio file generated in `outputs/` (e.g. `kannada_advisory_xyz.wav`). |

* **Example Request:**
```bash
curl -O "http://127.0.0.1:5000/api/v1/advisory/audio/download?file=raithamitra_kannada_advisory.wav"
```

---


## 5. Error Schema & HTTP Status Codes

### Standard Error JSON Format
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Field 'query' is required."
  }
}
```

### Supported HTTP Status Codes & Error Codes

| Status Code | Error Code | Cause / Meaning |
| :--- | :--- | :--- |
| `200 OK` | — | Successful advisory generation. |
| `400 Bad Request` | `VALIDATION_ERROR` | Missing query, empty query, whitespace query, wrong data types, or malformed JSON. |
| `404 Not Found` | `LOCATION_NOT_FOUND` | Specified district/taluk/village is not in the official Karnataka location registry. |
| `405 Method Not Allowed` | `METHOD_NOT_ALLOWED` | Using GET/PUT/DELETE on POST-only endpoints. |
| `413 Payload Too Large` | `PAYLOAD_TOO_LARGE` | Request payload exceeds 2 MB limit. |
| `500 Internal Error` | `INTERNAL_ERROR` | Unhandled backend exception (sanitized, zero stack-trace leakage). |
| `503 Unavailable` | `SERVICE_UNAVAILABLE` | Advisory backend service temporarily down. |

---

## 6. Supported Canonical Crops (11 Crops)
The API automatically normalizes Kannada and English crop names:
* `ragi` (`ರಾಗಿ` / Finger Millet)
* `paddy` (`ಭತ್ತ` / Rice)
* `maize` (`ಮೆಕ್ಕೆಜೋಳ` / Corn)
* `groundnut` (`ಕಡಲೆಕಾಯಿ` / Peanut)
* `sugarcane` (`ಕಬ್ಬು`)
* `cotton` (`ಹತ್ತಿ`)
* `chilli` (`ಮೆಣಸಿನಕಾಯಿ` / Green/Red Chilli)
* `onion` (`ಈರುಳ್ಳಿ`)
* `potato` (`ಆಲೂಗಡ್ಡೆ`)
* `banana` (`ಬಾಳೆ`)
* `tomato` (`ಟೊಮ್ಯಾಟೊ`)

---

## 7. Special Query Handling & Edge Cases

1. **Single-Word Crop Name (e.g. `{"query": "ರಾಗಿ"}`):**  
   The API asks for clarification rather than hallucinating random disease treatments:  
   *"ನಿಮ್ಮ ಬೆಳೆಯಲ್ಲಿ ನಿಮಗೆ ಯಾವ ಸಮಸ್ಯೆ ಇದೆ? ಎಲೆಗಳ ಲಕ್ಷಣ, ಕೀಟ, ರೋಗ, ನೀರಾವರಿ, ಗೊಬ್ಬರ/ಮಣ್ಣು, ಮಾರುಕಟ್ಟೆ ಬೆಲೆ ಅಥವಾ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳ ಬಗ್ಗೆ ಮಾಹಿತಿ ಬೇಕೇ ಎಂದು ದಯವಿಟ್ಟು ತಿಳಿಸಿ."*

2. **Non-Agricultural Query (e.g. `{"query": "ನನ್ನ ಲ್ಯಾಪ್ಟಾಪ್ ಹೇಗೆ ಸರಿಪಡಿಸುವುದು?"}`):**  
   Returns a polite domain disclaimer (`canonical_crop: null`, 0 RAG docs, 0 schemes, `market: null`):  
   *"ರೈತಮಿತ್ರ (RaithaMitra) ಕೃಷಿ ಸಲಹಾ ವ್ಯವಸ್ಥೆಯಾಗಿದ್ದು, ಬೆಳೆಗಳ ಆರೋಗ್ಯ, ಹವಾಮಾನ, ಮಣ್ಣು, ಕೃಷಿ ಯೋಜನೆಗಳು ಮತ್ತು ಮಾರುಕಟ್ಟೆ ಬೆಲೆಗಳ ಬಗ್ಗೆ ಮಾತ್ರ ಮಾಹಿತಿ ನೀಡಬಲ್ಲದು. ದಯವಿಟ್ಟು ಕೃಷಿ ಸಂಬಂಧಿತ ಪ್ರಶ್ನೆಗಳನ್ನು ಕೇಳಿ."*

3. **Unsupported / Exotic Crop (e.g. Saffron / `{"query": "ನಾನು ಕೇಸರಿ ಬೆಳೆಯಲು ಬಯಸುತ್ತೇನೆ."}`):**  
   Directs the farmer to the nearest Krishi Vigyan Kendra (KVK) without fabricating Karnataka practices:  
   *"ಈ ಬೆಳೆಯ ಕುರಿತು ಕರ್ನಾಟಕ ಕೃಷಿ ಮಾಹಿತಿ ಕೋಶದಲ್ಲಿ ನಿರ್ದಿಷ್ಟ ಸ್ಥಳೀಯ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ. ಹೆಚ್ಚಿನ ಮಾಹಿತಿಗಾಗಿ ಹತ್ತಿರದ ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರ (KVK) ಅಥವಾ ಕೃಷಿ ವಿಶ್ವವಿದ್ಯಾಲಯವನ್ನು ಸಂಪರ್ಕಿಸಿ."*

4. **External Provider Outage (Weather / APMC / Network failure):**  
   The pipeline degrades gracefully: `weather.available = false` or `market.available = false`. The API returns HTTP 200 with the remaining verified knowledge without fabricating values.

---

## 8. Client Code Examples

### A. JavaScript `fetch()` (Async/Await)
```javascript
/**
 * Calls RaithaMitra Advisory API
 * @param {string} query - Farmer query in Kannada or English
 * @param {string} district - Karnataka district name
 * @param {string} taluk - Karnataka taluk name
 * @param {string} village - Karnataka village name
 */
async function getFarmerAdvisory(query, district = null, taluk = null, village = null) {
  const API_URL = "http://127.0.0.1:5000/api/v1/advisory";

  const payload = {
    query: query,
    district: district,
    taluk: taluk,
    village: village,
    language: "kn"
  };

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Accept": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok || !data.success) {
      console.error("Advisory API Error:", data.error?.message || "Unknown error");
      return null;
    }

    console.log("Canonical Crop:", data.canonical_crop);
    console.log("Advisory Answer (Kannada):", data.answer);
    console.log("Weather:", data.weather);
    console.log("Soil:", data.soil);
    console.log("Market Prices:", data.market);
    console.log("Schemes:", data.schemes);

    return data;
  } catch (error) {
    console.error("Network / Connection failure:", error);
    throw error;
  }
}

// Example usage:
getFarmerAdvisory("ನನ್ನ ರಾಗಿ ಬೆಳೆ ಒಣಗುತ್ತಿದೆ", "Mandya", "Pandavapura", "Melukote");
```

---

### B. cURL Request Examples

#### 1. Standard Advisory Request (Mandya Ragi)
```bash
curl -X POST http://127.0.0.1:5000/api/v1/advisory \
  -H "Content-Type: application/json" \
  -d '{
    "query": "ನನ್ನ ರಾಗಿ ಬೆಳೆ ಒಣಗುತ್ತಿದೆ",
    "district": "Mandya",
    "taluk": "Pandavapura",
    "village": "Melukote"
  }'
```

#### 2. Mandi Market Price Query
```bash
curl -X POST http://127.0.0.1:5000/api/v1/advisory \
  -H "Content-Type: application/json" \
  -d '{
    "query": "ಮಂಡ್ಯದಲ್ಲಿ ರಾಗಿ ಬೆಲೆ ಎಷ್ಟು?",
    "district": "Mandya"
  }'
```

#### 3. Health Check
```bash
curl -X GET http://127.0.0.1:5000/api/v1/health
```

---

## 9. CORS & Local Frontend Integration
The backend is pre-configured with `Flask-CORS` enabling `*` origins for local development servers (e.g. React/Vite on `http://localhost:5173`, Next.js on `http://localhost:3000`, Vue/Angular on `http://localhost:8080`).

---

## 10. Performance & Model Inference Latency Expectations
* **Local In-Memory Orchestration:** $< 5\text{ ms}$ (Retrieval, Location, Soil, Market, Schemas).
* **Live Weather API Latency:** $150\text{ ms} - 400\text{ ms}$ (Open-Meteo).
* **CPU-Based Dhenu / NLLB Inference:** When running full 7B/1B transformer model weights locally on CPU without CUDA, generation can take between $10\text{s} - 30\text{s}$. Frontend applications should display an agricultural loading spinner / progress indicator while awaiting response.
