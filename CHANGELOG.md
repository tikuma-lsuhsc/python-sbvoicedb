# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
