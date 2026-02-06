# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-02-06

### Added

- `SbVoiceDb` class constructor to take predownloaded `data.zip` via `data_zip_file`
              argument

### Fixed

- a bug in displaying gender info in `RecordingSummaryTableModel._compose_tooltip()`

## [0.5.0] - 2026-02-06

### Added

- summary metafile `summary.csv` is added to the package
- [`qt.RecordingSummaryTableWidget`] added `recordingCount()` and `iterRecordings()` 
  methods to `RecordingSummaryTableWidget` class

### Fixed

- a bug in displaying gender info in `RecordingSummaryTableModel._compose_tooltip()`

## [0.4.0] - 2025-12-02

### Added

- 3 summary views `pathology_summary`, `recording_session_summary`, and `recording_summary`
- `SbVoiceDb` methods to fetch these views (the same name as the views)
- `qt` (independent) submodule
- `SbVoiceDb.get_pathology()`
- Default database directory using `platformdirs`


## [0.3.0] - 2025-09-12

- Changed on-the-fly filter arguments
- Removed `*_index()` and `*_id()` methods

## [0.2.0] - 2025-09-11

- Complete rebuild of the package with new API

## [0.1.0.dev7] - 2023-03-10

### Added

- Initial release.
