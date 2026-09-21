# Repository Review – ticket-priority-ml-service

## 1. Executive Summary

Dieses Repository demonstriert ein gut abgegrenztes, durchgängiges ML-MVP für die Triage mehrsprachiger Support-Tickets: zwei getrennte TF-IDF/LinearSVC-Klassifikatoren sagen `queue` und `priority` voraus; FastAPI, Streamlit, Docker, Tests und feste Modellartefakte machen die Demo greifbar. Der Code trennt Training, Vorverarbeitung, Evaluation, Tracking und Serving nachvollziehbar. Besonders überzeugend ist, dass die Evaluation nicht nur Accuracy, sondern Macro-F1, Klassen- und Sprachmetriken sowie Confusion Matrices erzeugt und die öffentliche Demo auf explizit ausgewählte Artefakte zeigt.

Die dokumentierten Ergebnisse sind für ein synthetisches Demonstrationsdataset plausibel und transparent eingeordnet: Queue Macro-F1 0,6854, Priority Macro-F1 0,7108; die erhebliche Schwäche für deutsche Tickets wird nicht verborgen (`README.md:19-27`, `docs/experiments.md:56-61`). Die CV-Implementierung passt die TF-IDF- und Label-Transformation pro Fold nur auf dem Trainingsanteil an; ein offensichtliches Preprocessing-Leakage ist im geprüften Pfad nicht erkennbar (`src/classification.py:165-189`, `src/preprocessing.py:245-279`).

Die wichtigste offene methodische Frage ist nicht ein Codefehler, sondern die Aussagekraft der Erfolgszahlen nach Modellwahl: Die Dokumentation wählt die „best overall“-Kandidaten anhand derselben CV-artigen Vergleichsergebnisse aus, die anschließend als Headline-Resultate gezeigt werden; ein einmaliger, unangetasteter Testsplit oder Nested CV ist nicht dokumentiert (`docs/experiments.md:20-40`). Bei synthetischen Tickets sollte zudem geprüft werden, ob nahe Varianten/Generator-Templates über Row-wise Folds in Train und Test geraten können. Das ist ein sinnvoller Sol-High-Deep-Dive, keine Behauptung eines nachgewiesenen Leaks.

Als Portfolio-Projekt ist es stark, weil es bewusst ein überzeugendes, lokal lauffähiges MVP baut statt einen unnötigen Produktionsstack zu imitieren. Der nächste Qualitätsgewinn liegt in einer belastbareren Evaluation/Provenance, nicht in Transformern, Kubernetes oder einem komplexen Model-Registry-System.

## 2. Standard-Check

| Kriterium | Status | Evidenz | Kommentar |
|---|---|---|---|
| Problem & Ziel | Green | `README.md:1-15`; `docs/experiments.md:3-10` | Operatives Routing (`queue`) und Business-Dringlichkeit (`priority`) sind klar getrennt und als End-to-End-Demo eingegrenzt. |
| Methoden | Green | `src/classification.py:17-43, 135-154`; `src/preprocessing.py:57-151, 296-333` | Sinnvolle, erklärbare Baseline-Familie: sprachbewusste Normalisierung, Wort-TF-IDF 1–3-Gramme, klassengewichtete LinearSVC; beide Zielaufgaben haben eigene Hyperparameter. |
| Daten | Yellow | `README.md:82-94, 111-120`; `docs/experiments.md:10, 69-71` | Quelle, erwarteter Dateiname, Umfang und synthetischer Charakter sind genannt; CSV selbst, maschinenprüfbare Version/Hash, Schema und Generator-/Duplikat-Analyse fehlen. |
| Evaluation | Yellow | `src/evaluation.py:28-117, 120-223, 288-363`; `docs/experiments.md:12-67` | 5-fold CV, Macro-F1, Accuracy, je Klasse/Language und Confusion Matrices sind substanziell. Unabhängige abschließende Evaluation nach der Modellwahl bzw. Nested CV ist nicht belegt. |
| Methodische Sauberkeit | Yellow | `train.py:101-141`; `src/training_utils.py:45-74`; `src/classification.py:157-220`; `src/preprocessing.py:245-279` | Folds entstehen vor task-spezifischem Fitting und der Vectorizer wird nur auf `train_df` gefittet: gut. Splitting ist aber zeilenbasiert und nicht gruppen-/template-/sprachstratifiziert; das Risiko hängt von der (nicht im Repo enthaltenen) Datenstruktur ab. |
| Reproduzierbarkeit | Yellow | `README.md:29-109`; `requirements*.txt`; `.gitattributes:1`; `serving_assets/promoted_models.json:1-105` | Demo ist mit gepinnten Abhängigkeiten, LFS-Artefakten und Docker gut reproduzierbar. Vollständiges Retraining benötigt externes Kaggle-CSV; dessen konkrete Version/Hash und die erzeugenden Originalruns liegen nicht bei. |
| Code-Struktur | Green | `train.py`; `src/preprocessing.py`; `src/classification.py`; `src/evaluation.py`; `app/service.py:95-188` | Klare Schichten und kleine, testbare Komponenten. Training und Serving sind sinnvoll entkoppelt. |
| Tests / Validierung | Yellow | `tests/test_preprocessing.py`; `tests/test_evaluation.py`; `tests/test_training_tracking.py:24-167`; `.github/workflows/ci.yml:1-32` | Gute Unit-, API- und Trainings-Smoke-Tests; CI installiert Abhängigkeiten inkl. NLTK. Keine Prüfung eines vollständigen Retrainings mit Originaldaten, kein lint/format-/Coverage-Gate – für das Portfolio kein Muss, aber der Teststatus ohne CI-Badge/Run-Link bleibt nicht sichtbar. |
| Ergebnisse | Green | `README.md:17-27`; `docs/experiments.md:20-67`; `serving_assets/promoted_models.json:17-21, 68-72` | Konkrete, aufgeschlüsselte Metriken, Modellvergleich und Fehlerbilder statt bloßer Erfolgsbehauptung. Die Werte sind als CV-/synthetische-Daten-Ergebnis zu lesen. |
| Limitationen | Green | `README.md:111-116`; `docs/experiments.md:69-71` | Sprachlücke, lineares Bag-of-words-Modell, verwechslungsanfällige Klassen und fehlende Trainingsdaten sind explizit benannt. |
| Repo-Hygiene | Green | `.gitignore:1-62`; `.gitattributes:1`; `LICENSE`; `README.md:118-122` | MIT-Code-Lizenz, separater Daten-Lizenzhinweis, ignorierte lokale Artefakte und Git-LFS für Modelle sind angemessen. |
| Portfolio-Wert | Green | `README.md:1-15`; `app/api.py:14-83`; `app/ui.py:46-74` | Zeigt Data/ML, Evaluation, Experimenttracking und eine sichtbar laufende Produktoberfläche – deutlich stärker als ein Trainingsnotebook allein. |
| Dokumentation | Green | `README.md:29-122`; `docs/experiments.md` | Setup, Testweg, Kennzahlen, Modellwahl und Grenzen sind gut auffindbar. |
| CI/CD | Green | `.github/workflows/ci.yml:1-32`; `Dockerfile:1-21` | Schlanke CI für Tests und ein nachvollziehbares Docker-Demo-Image. Eine Deployment-Pipeline wäre für dieses Showcase nicht erforderlich. |
| Overengineering | Green | `src/tracking.py:47-146`; `app/service.py:103-146`; `tools/prepare_serving_assets.py:120-200` | MLflow-Artefakte und feste Promoted Assets lösen ein echtes Demo-/Nachvollziehbarkeitsproblem; kein unnötiger Registry-, Cloud- oder Microservice-Overhead. |

## 3. Stärken

1. **Echte End-to-End-Kohärenz:** Training produziert versionierte MLflow-Artefakte (`train.py:163-185`), `tools/prepare_serving_assets.py:137-200` überführt explizit gewählte Runs in servingfähige Assets, und `app/service.py:103-146` lädt diese deterministisch.
2. **Saubere fold-lokale Vorverarbeitung:** `_evaluate_split` trainiert den Preprocessor auf `split.train_df` und transformiert erst dann `split.test_df` (`src/classification.py:165-189`); TF-IDF wird in `fit_transform` separat gefittet (`src/preprocessing.py:245-279`).
3. **Evaluation über die Headline-Metrik hinaus:** `evaluate_fold` berechnet Macro-F1, Klassenmetriken und Confusion Counts (`src/evaluation.py:43-117`), zusätzlich werden Sprachgruppen pro Fold aggregiert (`src/evaluation.py:288-363`).
4. **Ehrliche, nützliche Ergebnisnarrative:** Im Experimentbericht steht sowohl die Klassenimbalance als Begründung für Macro-F1 (`docs/experiments.md:12-16`) als auch die EN/DE-Lücke (`docs/experiments.md:56-61`).
5. **Portfolio-taugliche Bedienbarkeit:** Das UI zeigt Prediction, Runner-up und ausdrücklich nur relative Margin-Information (`app/ui.py:46-56`), statt eine unkalibrierte SVC-Marge als Wahrscheinlichkeit auszugeben.

## 4. Schwächen / Lücken

1. **Keine unabhängige Schluss-Evaluation dokumentiert:** Die Modellwahl und die publizierten CV-Kennzahlen sind nicht methodisch getrennt (`docs/experiments.md:20-40`).
2. **Datenprovenance ist nur teilweise reproduzierbar:** Dataset-Link/Dateiname reichen nicht für exakt denselben Retrain; Version, Prüfsumme, Schema und Duplikat-/Template-Befund fehlen (`README.md:82-94`).
3. **Row-wise Splitting kann bei synthetischen Varianten zu optimistischen Zahlen führen:** `StratifiedKFold.split(frame, stratify_labels)` kennt keine Gruppierung (`src/training_utils.py:61-74`); ob das tatsächlich greift, ist ohne CSV offen.
4. **Sprach-Fairness wird gemessen, aber nicht als Split-/Entscheidungsregel behandelt:** Language ist nicht in `STRATIFY_TARGET_COLUMNS` (`train.py:32-34`); trotz sichtbarer Differenz gibt es keine klar dokumentierte Mindestleistung oder Follow-up-Entscheidung für Deutsch.
5. **Lokale Testausführung ist nicht out-of-the-box verifiziert:** Im Audit-Umfeld fehlten die gepinnten Python-Pakete; `python3 -m unittest discover -s tests -v` endete daher mit vier Importfehlern und acht übersprungenen Tests. Das ist ein Umgebungsbefund, kein nachgewiesener Fehler im Repo; CI installiert die Anforderungen vorab (`.github/workflows/ci.yml:23-32`).

## 5. Technische oder methodische Findings

### F1 – Modellwahl und Bericht verwenden offenbar dieselbe CV-Evidenz

- **Severity:** Important
- **Evidenz:** Die Vergleichstabellen deklarieren die Stufen 4 jeweils als „best overall … promoted“ und berichten ihre Macro-F1/Accuracy (`docs/experiments.md:20-40`). `train.py:130-141` evaluiert und fitttet danach das finale Modell auf alle Daten; dokumentiert ist kein unbeeinflusster Testsplit/Nested-CV-Pfad.
- **Warum relevant:** Bei mehreren Varianten und C-Werten ist die beste CV-Zahl tendenziell leicht optimistisch. Die Zahlen bleiben als Explorations-/CV-Resultate nützlich, sollten aber nicht wie eine finale Generalisierungszusage wirken.
- **Deep Dive nötig:** Yes – den tatsächlichen Experimentverlauf und die Datenstruktur prüfen; dann entweder Nested CV oder einen klar reservierten, nach der Auswahl einmalig verwendeten Testsplit wählen.

### F2 – Mögliches Template-/Near-Duplicate-Leakage ist nicht ausgeschlossen

- **Severity:** Important
- **Evidenz:** Der Bericht bezeichnet die 28.587 Tickets als synthetisch (`docs/experiments.md:10, 69-71`). Die Folds werden zeilenweise mit kombinierten `queue`/`priority`-Labels erzeugt (`src/training_utils.py:20-74`), nicht nach Ticketfamilie, Quelle, Generator-Seed oder Textnähe gruppiert.
- **Warum relevant:** Falls ein Generator semantisch oder wortgleich nahe Varianten hervorbringt, können Train/Test-Ähnlichkeiten die TF-IDF-CV zu positiv machen. Das ist nur ein Risikoindikator, weil die CSV bewusst nicht im Repository liegt.
- **Deep Dive nötig:** Yes – exakte/normalisierte Duplikate, Text-Ähnlichkeit und mögliche Gruppenfelder (etwa `version`) auswerten; nur bei Befund GroupKFold/GroupShuffleSplit einsetzen.

### F3 – Retraining-Provenance ist nicht vollständig eingefroren

- **Severity:** Important
- **Evidenz:** Das Training verweist auf ein externes Kaggle-CSV mit festem lokalen Namen (`README.md:82-99`); Tracking speichert Dateipfad, Zeilenzahl und Dateistamm (`src/tracking.py:47-52, 116-145`), aber keinen Datei-Hash, Dataset-Release oder Schema-Snapshot.
- **Warum relevant:** Ein später heruntergeladenes CSV mit gleichem Namen kann andere Zeilen oder Labels enthalten; dann sind die bereits eingecheckten Modellmetriken nicht exakt nachstellbar.
- **Deep Dive nötig:** No – eine kleine Dataset-Card plus SHA-256, Spaltenliste und bezogene Dataset-Version genügt.

### F4 – Deutsche Leistung ist sichtbar, aber operativ nicht eingeordnet

- **Severity:** Minor
- **Evidenz:** Queue Macro-F1 fällt von 0,7841 (EN) auf 0,5341 (DE), Priority von 0,7951 auf 0,5960 (`README.md:24-27`; `docs/experiments.md:56-61`). `evaluate_fold` kann Sprachmetriken erzeugen (`src/evaluation.py:288-335`), aber Training stratifiziert nur `queue` und `priority` (`train.py:32-34, 103-109`).
- **Warum relevant:** Als Demo ist das transparent; als Triage-Unterstützung braucht es zumindest eine dokumentierte Aussage, ob Deutsch „exploratory only“ ist, oder eine gruppenspezifische Akzeptanzgrenze.
- **Deep Dive nötig:** Yes – nach F1/F2, mit Klassen- und Sprachverteilungen pro Fold sowie einem kleinen Fehlerkatalog statt sofortigem Modellwechsel.

### F5 – Modell-Marge wird verantwortungsvoll bezeichnet, aber nicht kalibriert

- **Severity:** Minor
- **Evidenz:** `LoadedTaskModel.predict` berechnet den Abstand zweier `decision_function`-Scores (`app/service.py:33-57`); das UI nennt ihn ausdrücklich „relative signal, not an absolute confidence score“ (`app/ui.py:52-56`).
- **Warum relevant:** Kein irreführendes Wahrscheinlichkeitsversprechen – positiv. Sollte die Demo später Schwellenwerte/Auto-Routing darauf stützen, wären Kalibrierung und eine selektive Review-Regel nötig.
- **Deep Dive nötig:** No – für den aktuellen Demo-Scope korrekt ausreichend.

## 6. Overengineering

**Beibehalten:** Die zwei klaren Modelle, MLflow für reproduzierbare Trainingsartefakte, feste in Git-LFS versionierte Serving-Assets und das kleine API/UI-Paar. Insbesondere verhindert die explizite Konfiguration „latest run wins“-Zufälligkeit (`README.md:107-109`; `serving_assets/serving_config.json:1-16`).

**Nicht weiter ausbauen:** Kein Transformer-Finetuning, Feature Store, Kubernetes, asynchrone Queue oder Cloud-Model-Registry, bevor die Daten-/Evaluationsfrage geklärt ist. Der aktuelle TF-IDF/LinearSVC-Ansatz ist für einen nachvollziehbaren Portfolio-MVP angemessen und leichter erklärbar.

**Einfach halten:** Falls eine Unsicherheitsregel ergänzt wird, reicht zunächst eine dokumentierte Margin-basierte *Manual-review*-Heuristik, validiert auf einem sauberen Holdout. Eine produktionsähnliche Monitoring-Plattform ist dafür nicht nötig.

## 7. Reproduzierbarkeit

- **Demo:** Gut. README dokumentiert Docker und lokalen Start (`README.md:29-80`); Modellartefakte sind LFS-tracked (`.gitattributes:1`), Konfiguration und Run-IDs liegen in `serving_assets/`.
- **Dependencies:** Gut. Exakte Versionen sind in `requirements.txt:1-12` und der schlankeren Serving-Liste gepinnt; das Dockerfile installiert Stopwords (`Dockerfile:10-14`).
- **Training:** Teilweise. `train.py` akzeptiert Datenpfad, Foldzahl, Seed, Algorithmus und Tracking-Ziel (`train.py:36-89`). Die Daten müssen jedoch extern bezogen werden; der Repo-Stand enthält weder CSV noch Dataset-Manifest/Hash.
- **Run path:** Für einen neuen Durchlauf vorhanden (`README.md:82-105`). Für eine bit-identische Wiederholung der promoteten Ergebnisse fehlen Dataset-Release/Hash und die vollständigen historischen `mlruns` (das Promotion-Tool erwartet sie lokal unter `mlruns/<experiment>/<run>/...`, `tools/prepare_serving_assets.py:18-31, 137-143`).
- **Prüfung im Audit:** Der vollständige Testlauf konnte im Audit-Container nicht validiert werden, weil `pandas`, `joblib`, FastAPI/HTTPX und MLflow nicht installiert waren; ein bereits beigestelltes Setup wäre nötig. Statisches Test-/CI-Design wurde geprüft; keine Repo-Datei wurde verändert.

## 8. Portfolio-Wirkung

- **Nachgewiesene Skills:** Textklassifikation, sprachbewusste Vorverarbeitung, klassengewichtete Modelle, CV und Fehleranalyse, MLflow-Experimentprovenance, serialisierte Modellartefakte, FastAPI/Streamlit, Docker und CI.
- **Recruiter-Sicht:** Ein deutlich sichtbares, ausführbares Projekt mit klarer Nutzerhandlung und Ergebnis statt eines unsichtbaren Modellskripts. README und Screenshot erleichtern eine schnelle Bewertung.
- **Technische Interviewer-Sicht:** Die gute Frage ist nicht „Warum kein LLM?“, sondern ob CV, Imbalance, Sprachlücke, Modellpromotion und Serving-Traceability verstanden wurden. Der Code liefert dafür solide Gesprächsanker.
- **Unangenehme Rückfrage:** „Wie stellst du sicher, dass deine 0,71 Macro-F1 nach der Modellauswahl und bei synthetischen Ticketvarianten wirklich generalisiert – besonders auf Deutsch?“ Eine kurze, belastbare Antwort braucht den F1/F2-Deep-Dive.

## 9. Deep-Dive-Kandidaten

1. **P1 – Sol High:** CV-/Modellwahl-Audit: alle Kandidaten, Suchraum und Auswahlprozess rekonstruieren; finalen Testsplit oder Nested-CV-Design festlegen (F1).
2. **P1 – Sol High:** Data-Leakage-Audit auf Original-CSV: Duplikate, Near Duplicates, Template-/`version`-Gruppen und gruppierte Gegen-Evaluation prüfen (F2).
3. **P2 – Sol High:** Language slice: pro Sprache × Klasse Supports, Varianz und Fehlermuster ermitteln; eine ehrliche Demogrenze oder Verbesserungshypothese formulieren (F4).
4. **P2 – Terra High:** Dataset-Card/Manifest: Kaggle-Release, Lizenz, Schema, Zeilenzahl und SHA-256 in die Trainingsprovenance integrieren (F3).
5. **P3 – Sol High:** Nur falls eine automatische Routing-Schwelle vorgesehen ist: Margin-Kalibrierung/Review-Policy anhand sauberen Holdouts prüfen (F5).

## 10. Top-5-Maßnahmen

| Priorität | Maßnahme | Aufwand | Begründung / konkreter Zielzustand |
|---|---|---:|---|
| P1 | Modellwahl von finaler Evidenz trennen | M | Nach dem Sol-High-Audit: Nested CV **oder** ein einmalig eingefrorener Testsplit. README/Experiments sollen „selection CV“ und „final test“ klar ausweisen. |
| P1 | Duplikat-/Template-Risiko messen | M | Auf der Original-CSV exakte/normalisierte Duplikate und Gruppenfelder prüfen. Nur bei Befund gruppiert splitten und beide Ergebnisse transparent vergleichen. |
| P2 | Kleine Dataset-Card plus Hash ergänzen | S | Quelle/Release, Lizenz, erwartete Spalten, Zeilenzahl, SHA-256 und Zugriffsweg festhalten; `build_dataset_metadata` kann den Hash loggen. |
| P2 | Sprach-Slice operationalisieren | S | EN/DE × Klasse und Mindest-Support dokumentieren; im README klar sagen, ob das Modell für Deutsch nur demonstrativ ist oder welche Review-Regel gilt. |
| P3 | CI-Ergebnis sichtbar machen | S | README-Badge/Link zum bestehenden Workflow und optional ein schneller Syntax-/Formatcheck. Kein schweres Release-System erforderlich. |
