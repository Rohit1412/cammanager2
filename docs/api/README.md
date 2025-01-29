# Camera Manager API Documentation

## Overview
This document describes the REST API endpoints available in the Camera Manager system.

## Base URL
All endpoints are relative to: `http://<server>:<port>/`

## Authentication
Currently, the API does not require authentication.

## Endpoints

### Camera Management

#### Add Camera
- **POST** `/add_camera`
- **Description**: Add a new camera to the system
- **Request Body**:
  ```json
  {
    "source": "string or integer",
    "name": "string (optional)",
    "max_fps": "integer (optional)"
  }
  ```
- **Response**: Camera ID and status

#### Remove Camera
- **POST** `/remove_camera/<camera_id>`
- **Description**: Remove a camera from the system
- **Response**: Success/failure status

#### List Cameras
- **GET** `/list_cameras`
- **Description**: Get list of all cameras
- **Response**: Array of camera details

### Recording Control

#### Start Recording
- **POST** `/start_recording`
- **Request Body**:
  ```json
  {
    "camera_id": "integer"
  }
  ```
- **Response**: Recording status and path

#### Stop Recording
- **POST** `/stop_recording`
- **Request Body**:
  ```json
  {
    "camera_id": "integer"
  }
  ```
- **Response**: Success/failure status

### Quality Control

#### Set Camera Quality
- **POST** `/camera/<camera_id>/quality`
- **Description**: Set camera quality preset
- **Request Body**:
  ```json
  {
    "preset": "string (low/medium/high)"
  }
  ```
- **Response**: Success/failure status

#### Set Camera Codec
- **POST** `/camera/<camera_id>/codec`
- **Description**: Set recording codec
- **Request Body**:
  ```json
  {
    "codec": "string (mp4v/avc1/hevc/mjpg)"
  }
  ```
- **Response**: Success/failure status

### System Management

#### Health Check
- **GET** `/health`
- **Description**: Get system health status
- **Response**: System and camera health details

#### Check Available Cameras
- **GET** `/check_cameras`
- **Description**: Scan for available camera devices
- **Response**: List of available cameras and their properties 