from app.main import app


def test_app_includes_api_routes_and_upload_mount():
    """The FastAPI app should expose the API router and uploads mount."""
    route_paths = {route.path for route in app.routes}

    assert "/ingest" in route_paths
    assert "/query" in route_paths
    assert "/uploads" in route_paths


def test_app_has_cors_middleware_configured():
    """CORS middleware should be present on the application."""
    middleware_classes = {middleware.cls.__name__ for middleware in app.user_middleware}

    assert "CORSMiddleware" in middleware_classes
